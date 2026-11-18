"""OpenCV webcam capture, MediaPipe hand tracking, and fist / 1-5 gesture recognition."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from gesture_recognizer import GestureRecognizer, RecognitionResult
from hand_visualizer import HandVisualizer


@dataclass
class HandState:
    number: int = 0
    detected: bool = False
    theme_name: str = "None"
    x: float = 0.5
    y: float = 0.5
    inference_ms: float = 0.0
    confidence: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    recognition: Optional[RecognitionResult] = None
    is_manual: bool = False
    handedness: Optional[str] = None


class HandTracker:
    """Captures real upright webcam frames, tracks hand landmarks with MediaPipe,
    classifies a fist and one-to-five extended fingers without automatic self-changing,
    and draws clean lines along the fingers.
    """

    def __init__(self, camera_index: int = 0, model_path: str = "hand_landmarker.task"):
        self.camera_index = camera_index
        self.model_path = model_path
        self.camera_available = False
        self.cap = None

        # Attempt to open webcam cleanly
        for idx in (camera_index, 0, 1, 2):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.cap = cap
                        self.camera_available = True
                        break
                    else:
                        cap.release()
            except Exception:
                pass

        self.recognizer = GestureRecognizer()
        self.visualizer = HandVisualizer()
        self.state = HandState()

        self.mode = "tasks"
        self.landmarker = None
        self.hands = None
        try:
            import mediapipe as mp
            if hasattr(mp, "tasks"):
                self._init_tasks(mp, model_path)
            else:
                self._init_legacy(mp)
        except Exception as exc:
            print(f"[HandTracker] MediaPipe init warning: {exc}.", flush=True)

        self.manual_number: Optional[int] = None

    def _init_tasks(self, mp, model_path: str) -> None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not os.path.exists(model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                print(f"Downloading hand landmarker model to {model_path}...")
                urllib.request.urlretrieve(url, model_path)
            except Exception as exc:
                raise RuntimeError(f"Could not download hand model: {exc}") from exc

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _init_legacy(self, mp) -> None:
        self.mode = "legacy"
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50
        )

    def set_manual_gesture(self, number: Optional[int]) -> None:
        """Sets a manual gesture (1 to 5) when user presses keys 1-5."""
        self.manual_number = number

    def _create_synthetic_landmarks(self, number: int) -> list:
        """Creates accurate synthetic landmarks for manual testing (keys 1-5)."""
        cx, cy = 0.50, 0.55
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = [cx, cy + 0.20, 0.0]  # Wrist

        pts[1] = [cx - 0.06, cy + 0.13, 0.0]
        pts[2] = [cx - 0.09, cy + 0.07, 0.0]
        pts[5] = [cx - 0.05, cy - 0.02, 0.0]
        pts[9] = [cx, cy - 0.04, 0.0]
        pts[13] = [cx + 0.05, cy - 0.02, 0.0]
        pts[17] = [cx + 0.10, cy + 0.01, 0.0]

        ext_map = {
            1: (False, True, False, False, False),
            2: (False, True, True, False, False),
            3: (False, True, True, True, False),
            4: (False, True, True, True, True),
            5: (True, True, True, True, True),
        }
        th_ext, id_ext, mi_ext, ri_ext, pi_ext = ext_map.get(number, (False, False, False, False, False))

        if th_ext:
            pts[3] = [cx - 0.15, cy + 0.02, 0.0]
            pts[4] = [cx - 0.22, cy - 0.04, 0.0]
        else:
            pts[3] = [cx - 0.06, cy + 0.05, 0.0]
            pts[4] = [cx - 0.02, cy + 0.03, 0.0]

        for b_idx, mcp_pos, ext in [
            (5, pts[5], id_ext),
            (9, pts[9], mi_ext),
            (13, pts[13], ri_ext),
            (17, pts[17], pi_ext)
        ]:
            if ext:
                pts[b_idx + 1] = mcp_pos + [0, -0.08, 0]
                pts[b_idx + 2] = mcp_pos + [0, -0.15, 0]
                pts[b_idx + 3] = mcp_pos + [0, -0.22, 0]
            else:
                pts[b_idx + 1] = mcp_pos + [0, -0.05, 0.03]
                pts[b_idx + 2] = mcp_pos + [0, -0.01, 0.05]
                pts[b_idx + 3] = mcp_pos + [0, 0.04, 0.04]

        landmarks = []
        for p in pts:
            class LandmarkObj:
                pass
            lo = LandmarkObj()
            lo.x, lo.y, lo.z = float(p[0]), float(p[1]), float(p[2])
            landmarks.append(lo)
        return landmarks

    def read(self) -> Tuple[Optional[np.ndarray], HandState]:
        start_time = time.perf_counter()
        frame = None
        raw_landmarks = None
        handedness = None

        if self.camera_available and self.cap is not None:
            ok, raw_frame = self.cap.read()
            if ok and raw_frame is not None:
                # Upright real camera: standard horizontal mirror so moving right moves right on screen
                # NO cv2.ROTATE_180 - upright orientation ensures hand going up goes UP!
                frame = cv2.flip(raw_frame, 1)

                if self.mode == "tasks" and self.landmarker is not None:
                    import mediapipe as mp
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ms = int(time.monotonic() * 1000)
                    result = self.landmarker.detect_for_video(mp_image, ms)
                    if result.hand_landmarks:
                        raw_landmarks = result.hand_landmarks[0]
                        if result.handedness and result.handedness[0]:
                            handedness = result.handedness[0][0].category_name
                elif self.hands is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = self.hands.process(rgb)
                    if res.multi_hand_landmarks:
                        raw_landmarks = res.multi_hand_landmarks[0].landmark
                        if res.multi_handedness:
                            handedness = res.multi_handedness[0].classification[0].label
        else:
            # Standby camera frame when camera hardware is disconnected or in use
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (18, 22, 28)
            cv2.putText(frame, "Webcam Standby (Show hand or press keys 1-5)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1, cv2.LINE_AA)

        # If user explicitly pressed a key 1-5 for testing
        used_manual_gesture = raw_landmarks is None and self.manual_number is not None and 1 <= self.manual_number <= 5
        if used_manual_gesture:
            raw_landmarks = self._create_synthetic_landmarks(self.manual_number)
            handedness = "Right"

        inference_ms = (time.perf_counter() - start_time) * 1000.0

        # A fist is -1, no pose is 0, and extended-finger counts are 1 to 5.
        rec_res = self.recognizer.evaluate(raw_landmarks, handedness)

        # Draw clean lines directly on the fingers and floating number
        if frame is not None and raw_landmarks is not None:
            theme_color = rec_res.theme_color_bgr if rec_res.number > 0 else (80, 230, 120)
            self.visualizer.draw_skeleton(frame, raw_landmarks, theme_color)
            if rec_res.number > 0:
                self.visualizer.draw_floating_badge(frame, rec_res)

        self.state = HandState(
            number=rec_res.number,
            detected=raw_landmarks is not None,
            theme_name=rec_res.theme_name,
            x=rec_res.wrist[0],
            y=rec_res.wrist[1],
            inference_ms=inference_ms,
            confidence=rec_res.confidence,
            yaw=rec_res.yaw,
            pitch=rec_res.pitch,
            roll=rec_res.roll,
            recognition=rec_res,
            is_manual=used_manual_gesture,
            handedness=handedness
        )

        return frame, self.state

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
            self.hands = None
