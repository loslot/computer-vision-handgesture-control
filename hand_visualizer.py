"""Clean hand tracking lines and floating 3D typography number on camera feed."""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from gesture_recognizer import RecognitionResult


class HandVisualizer:
    """Renders clean tracking lines along the fingers on the real webcam feed
    and displays a floating number near the hand without blocking the camera.
    """

    # Finger bone chains from wrist/palm to fingertips
    FINGER_CHAINS = [
        # Thumb: wrist -> CMC -> MCP -> IP -> TIP
        [0, 1, 2, 3, 4],
        # Index: wrist -> MCP -> PIP -> DIP -> TIP
        [0, 5, 6, 7, 8],
        # Middle: wrist -> MCP -> PIP -> DIP -> TIP
        [0, 9, 10, 11, 12],
        # Ring: wrist -> MCP -> PIP -> DIP -> TIP
        [0, 13, 14, 15, 16],
        # Pinky: wrist -> MCP -> PIP -> DIP -> TIP
        [0, 17, 18, 19, 20],
    ]

    # Knuckle bridge across palm
    PALM_BRIDGE = [5, 9, 13, 17]

    def __init__(self):
        self.num_x: float = 0.5
        self.num_y: float = 0.5
        self.num_alpha: float = 0.0
        self.target_alpha: float = 0.0
        self.last_number: int = 0
        self.change_time: float = 0.0
        self.last_update = time.perf_counter()

    def draw_skeleton(self, frame: np.ndarray, raw_landmarks, theme_bgr: Tuple[int, int, int] = (80, 230, 120)) -> None:
        """Draws clean, elegant tracking lines directly on the fingers."""
        if raw_landmarks is None or len(raw_landmarks) < 21:
            return

        h, w = frame.shape[:2]
        pts_2d = []
        for i in range(21):
            pt = raw_landmarks[i]
            px = int(np.clip(pt.x if hasattr(pt, "x") else pt[0], 0.0, 1.0) * w)
            py = int(np.clip(pt.y if hasattr(pt, "y") else pt[1], 0.0, 1.0) * h)
            pts_2d.append((px, py))

        # 1. Draw smooth lines along each finger
        for chain in self.FINGER_CHAINS:
            for k in range(len(chain) - 1):
                p1, p2 = pts_2d[chain[k]], pts_2d[chain[k + 1]]
                # Clean anti-aliased line along the finger
                cv2.line(frame, p1, p2, theme_bgr, 2, cv2.LINE_AA)

        # 2. Draw knuckle bridge across palm
        for k in range(len(self.PALM_BRIDGE) - 1):
            p1, p2 = pts_2d[self.PALM_BRIDGE[k]], pts_2d[self.PALM_BRIDGE[k + 1]]
            cv2.line(frame, p1, p2, theme_bgr, 1, cv2.LINE_AA)

        # 3. Clean compact joint dots on the fingers
        for i, pt in enumerate(pts_2d):
            if i in (4, 8, 12, 16, 20):
                # Fingertips
                cv2.circle(frame, pt, 4, theme_bgr, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Joints
                cv2.circle(frame, pt, 2, theme_bgr, -1, cv2.LINE_AA)

    def draw_floating_badge(self, frame: np.ndarray, result: RecognitionResult) -> None:
        """Renders the recognized number floating near the hand with glowing 3D typography,
        without any opaque box blocking the camera.
        """
        now = time.perf_counter()
        dt = max(1e-4, now - self.last_update)
        self.last_update = now

        h, w = frame.shape[:2]

        if 1 <= result.number <= 5:
            self.target_alpha = 1.0
            # Anchor near index MCP / wrist
            target_x = result.wrist[0] * w + 30
            target_y = result.wrist[1] * h - 35

            # Smooth movement to eliminate any jitter
            alpha_pos = min(1.0, dt * 11.0)
            self.num_x += alpha_pos * (target_x - self.num_x)
            self.num_y += alpha_pos * (target_y - self.num_y)

            if result.number != self.last_number:
                self.last_number = result.number
                self.change_time = now
        else:
            self.target_alpha = 0.0

        # Fade in/out
        alpha_spd = 8.0 if self.target_alpha > self.num_alpha else 4.0
        self.num_alpha += (self.target_alpha - self.num_alpha) * min(1.0, dt * alpha_spd)

        if self.num_alpha < 0.05 or self.last_number == 0:
            return

        # Pop scale on number change
        t_since = now - self.change_time
        scale_pop = 1.0
        if t_since < 0.25:
            scale_pop = 1.0 + 0.25 * math.sin((t_since / 0.25) * math.pi)

        theme_bgr = result.theme_color_bgr

        # Floating 3D Typography Number
        num_str = str(self.last_number)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.55 * scale_pop
        thick = 2

        t_size = cv2.getTextSize(num_str, font, font_scale, thick)[0]
        nx = int(np.clip(self.num_x, 15, w - t_size[0] - 15))
        ny = int(np.clip(self.num_y, t_size[1] + 15, h - 20))

        # 3D Glowing Text Layers:
        # Layer 1: Dark drop shadow
        cv2.putText(frame, num_str, (nx + 2, ny + 2), font, font_scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        # Layer 2: Theme color glow
        cv2.putText(frame, num_str, (nx, ny), font, font_scale, theme_bgr, thick + 2, cv2.LINE_AA)
        # Layer 3: Crisp bright white core
        cv2.putText(frame, num_str, (nx, ny), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

        # Small clean theme label below number
        theme_str = result.theme_name.upper()
        s_font = cv2.FONT_HERSHEY_SIMPLEX
        s_scale = 0.35 * scale_pop
        s_size = cv2.getTextSize(theme_str, s_font, s_scale, 1)[0]
        sx = nx
        sy = ny + s_size[1] + 6
        if sy < h - 5:
            cv2.putText(frame, theme_str, (sx + 1, sy + 1), s_font, s_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, theme_str, (sx, sy), s_font, s_scale, theme_bgr, 1, cv2.LINE_AA)
