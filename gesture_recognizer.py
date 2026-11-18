"""Recognizes a closed fist and one-to-five extended fingers from MediaPipe landmarks.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from color_themes import THEMES, ColorTheme


@dataclass
class FingerState:
    extended: bool
    curled: bool
    ratio: float
    tip_dist: float
    pip_angle: float


@dataclass
class RecognitionResult:
    number: int  # -1 fist, 0 no pose, or 1 to 5 extended fingers
    confidence: float
    theme_name: str
    theme_color_bgr: Tuple[int, int, int]
    finger_states: dict
    wrist: Tuple[float, float, float]
    palm_center: Tuple[float, float, float]
    hand_size: float
    yaw: float
    pitch: float
    roll: float


class GestureSmoother:
    """Requires a pose to remain stable before accepting a gesture."""

    HOLD_SECONDS = 0.4

    def __init__(self):
        self.current_number: int = 0
        self.last_confirmed_number: int = 0
        self.candidate_number: int = 0
        self.candidate_since: Optional[float] = None
        self.smoothed_x: float = 0.5
        self.smoothed_y: float = 0.5
        self.smoothed_z: float = 0.0
        self.alpha_pos: float = 0.28  # EMA smoothing factor for position

    def update_position(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        self.smoothed_x += self.alpha_pos * (x - self.smoothed_x)
        self.smoothed_y += self.alpha_pos * (y - self.smoothed_y)
        self.smoothed_z += self.alpha_pos * (z - self.smoothed_z)
        return self.smoothed_x, self.smoothed_y, self.smoothed_z

    def update_number(self, raw_number: int, confidence: float) -> int:
        # A continuous hold prevents transient poses from switching the scene.
        if raw_number == 0 or raw_number < -1 or raw_number > 5:
            self.candidate_number = 0
            self.candidate_since = None
            self.current_number = 0
            return 0

        now = time.monotonic()
        if raw_number != self.candidate_number:
            self.candidate_number = raw_number
            self.candidate_since = now
            self.current_number = 0
            return 0

        if self.candidate_since is not None and now - self.candidate_since >= self.HOLD_SECONDS:
            self.current_number = raw_number
            self.last_confirmed_number = raw_number
        return self.current_number


class GestureRecognizer:
    """Classifies a fist or the count of extended fingers from 1 through 5."""

    def __init__(self):
        self.smoother = GestureSmoother()

    @staticmethod
    def _to_numpy(landmarks) -> np.ndarray:
        pts = np.empty((21, 3), dtype=np.float32)
        for i in range(21):
            pt = landmarks[i]
            if hasattr(pt, "x"):
                pts[i] = (pt.x, pt.y, pt.z if hasattr(pt, "z") else 0.0)
            else:
                pts[i] = pt[:3]
        return pts

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos_val = np.dot(v1, v2) / (n1 * n2)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))

    def evaluate(self, raw_landmarks, handedness: Optional[str] = None) -> RecognitionResult:
        if raw_landmarks is None or len(raw_landmarks) < 21:
            self.smoother.update_number(0, 0.0)
            return RecognitionResult(
                number=0,
                confidence=0.0,
                theme_name="None",
                theme_color_bgr=(180, 180, 180),
                finger_states={},
                wrist=(0.5, 0.5, 0.0),
                palm_center=(0.5, 0.5, 0.0),
                hand_size=0.1,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
            )

        pts = self._to_numpy(raw_landmarks)
        wrist = pts[0]
        thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = pts[1], pts[2], pts[3], pts[4]
        index_mcp, index_pip, index_dip, index_tip = pts[5], pts[6], pts[7], pts[8]
        mid_mcp, mid_pip, mid_dip, mid_tip = pts[9], pts[10], pts[11], pts[12]
        ring_mcp, ring_pip, ring_dip, ring_tip = pts[13], pts[14], pts[15], pts[16]
        pinky_mcp, pinky_pip, pinky_dip, pinky_tip = pts[17], pts[18], pts[19], pts[20]

        palm_center = (wrist + index_mcp + mid_mcp + pinky_mcp) * 0.25
        hand_span = self._dist(index_mcp, pinky_mcp)
        hand_length = self._dist(wrist, mid_mcp)
        hand_size = max(hand_span, hand_length, 0.05)

        # Coordinate axes of the hand for orientation
        up_vec = mid_mcp - wrist
        up_norm = np.linalg.norm(up_vec)
        if up_norm > 1e-5:
            up_vec /= up_norm
        else:
            up_vec = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        side_vec = index_mcp - pinky_mcp
        side_norm = np.linalg.norm(side_vec)
        if side_norm > 1e-5:
            side_vec /= side_norm
        else:
            side_vec = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        normal_vec = np.cross(up_vec, side_vec)
        norm_len = np.linalg.norm(normal_vec)
        if norm_len > 1e-5:
            normal_vec /= norm_len
        else:
            normal_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        yaw = float(np.degrees(np.arctan2(up_vec[0], -up_vec[1])))
        pitch = float(np.degrees(np.arcsin(np.clip(normal_vec[1], -1.0, 1.0))))
        roll = float(np.degrees(np.arctan2(side_vec[1], side_vec[0])))

        # Finger extension analysis
        def analyze_finger(mcp, pip, dip, tip):
            d_tip = self._dist(tip, mcp)
            d_pip = self._dist(pip, mcp)
            ratio = d_tip / max(d_pip, 1e-5)
            d_wrist_tip = self._dist(tip, wrist)
            d_wrist_pip = self._dist(pip, wrist)
            wrist_ratio = d_wrist_tip / max(d_wrist_pip, 1e-5)
            angle = self._angle(mcp, pip, tip)

            # Decide independently for each finger using both wrist reach and joint shape.
            extended = wrist_ratio > 1.12 and ratio > 1.20 and angle > 135.0
            curled = not extended
            return FingerState(extended=extended, curled=curled, ratio=ratio, tip_dist=d_tip, pip_angle=angle)

        index_state = analyze_finger(index_mcp, index_pip, index_dip, index_tip)
        mid_state = analyze_finger(mid_mcp, mid_pip, mid_dip, mid_tip)
        ring_state = analyze_finger(ring_mcp, ring_pip, ring_dip, ring_tip)
        pinky_state = analyze_finger(pinky_mcp, pinky_pip, pinky_dip, pinky_tip)

        # The thumb bends sideways, so compare it with the index base and use the
        # tracker's handedness for the expected lateral direction.
        thumb_tip_index_mcp = self._dist(thumb_tip, index_mcp) / hand_size
        if handedness == "Right":
            thumb_side_delta = index_mcp[0] - thumb_tip[0]
        elif handedness == "Left":
            thumb_side_delta = thumb_tip[0] - index_mcp[0]
        else:
            thumb_side_delta = 0.0
        thumb_ip_angle = self._angle(thumb_mcp, thumb_ip, thumb_tip)
        thumb_by_handedness = (
            thumb_tip_index_mcp > 0.55
            and (handedness not in ("Left", "Right") or thumb_side_delta > hand_size * 0.20)
            and thumb_ip_angle > 120.0
        )
        four_fingers_extended = all(
            state.extended for state in (index_state, mid_state, ring_state, pinky_state)
        )
        fingertips_spread = self._dist(index_tip, pinky_tip) / hand_size > 0.55
        # Handedness labels can reverse with palm orientation or mirrored camera
        # input. A clearly spread four-finger fan plus an abducted, straight thumb
        # is an open hand even when its horizontal direction disagrees with the label.
        thumb_open_palm_fallback = (
            four_fingers_extended
            and fingertips_spread
            and thumb_tip_index_mcp > 0.40
            and thumb_ip_angle > 105.0
        )
        thumb_extended = thumb_by_handedness or thumb_open_palm_fallback
        thumb_curled = not thumb_extended
        thumb_state = FingerState(
            extended=thumb_extended,
            curled=thumb_curled,
            ratio=thumb_tip_index_mcp,
            tip_dist=self._dist(thumb_tip, thumb_mcp),
            pip_angle=thumb_ip_angle
        )

        finger_states = {
            "thumb": thumb_state,
            "index": index_state,
            "middle": mid_state,
            "ring": ring_state,
            "pinky": pinky_state,
        }

        extended_count = sum(
            int(state.extended)
            for state in (thumb_state, index_state, mid_state, ring_state, pinky_state)
        )
        if extended_count == 0:
            raw_number = -1
        elif extended_count == 5 and not fingertips_spread:
            raw_number = 0
        else:
            raw_number = extended_count
        confidence = 0.95 if raw_number in (-1, 5) else 0.90

        # Debounce number with temporal smoothing
        final_number = self.smoother.update_number(raw_number, confidence)

        theme_info = THEMES.get(final_number, THEMES[5])
        theme_name = theme_info.name if final_number in THEMES else "None"
        theme_bgr = theme_info.bgr_color if final_number in THEMES else (180, 180, 180)

        sx, sy, sz = self.smoother.update_position(float(wrist[0]), float(wrist[1]), float(wrist[2]))

        return RecognitionResult(
            number=final_number,
            confidence=confidence,
            theme_name=theme_name,
            theme_color_bgr=theme_bgr,
            finger_states=finger_states,
            wrist=(sx, sy, sz),
            palm_center=(float(palm_center[0]), float(palm_center[1]), float(palm_center[2])),
            hand_size=float(hand_size),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
