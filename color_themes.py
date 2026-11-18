"""The 5 canonical color themes for hand gestures 1 to 5:
1. Electric Blue
2. Emerald Green
3. Royal Purple
4. Crimson Red
5. Gold
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ColorTheme:
    number: int
    name: str
    primary_rgb: Tuple[float, float, float]
    accent_rgb: Tuple[float, float, float]
    glow_rgb: Tuple[float, float, float]
    key_light: Tuple[float, float, float, float]
    fill_light: Tuple[float, float, float, float]
    bgr_color: Tuple[int, int, int]  # OpenCV BGR format


THEMES: Dict[int, ColorTheme] = {
    1: ColorTheme(
        number=1,
        name="Electric Blue",
        primary_rgb=(0.05, 0.48, 1.00),
        accent_rgb=(0.45, 0.82, 1.00),
        glow_rgb=(0.10, 0.65, 1.00),
        key_light=(0.20, 0.65, 1.00, 1.0),
        fill_light=(0.05, 0.20, 0.75, 1.0),
        bgr_color=(255, 125, 12),
    ),
    2: ColorTheme(
        number=2,
        name="Emerald Green",
        primary_rgb=(0.08, 0.90, 0.42),
        accent_rgb=(0.50, 1.00, 0.70),
        glow_rgb=(0.12, 0.96, 0.52),
        key_light=(0.20, 0.95, 0.50, 1.0),
        fill_light=(0.02, 0.45, 0.20, 1.0),
        bgr_color=(100, 230, 20),
    ),
    3: ColorTheme(
        number=3,
        name="Royal Purple",
        primary_rgb=(0.65, 0.15, 0.96),
        accent_rgb=(0.88, 0.50, 1.00),
        glow_rgb=(0.72, 0.25, 1.00),
        key_light=(0.75, 0.35, 1.00, 1.0),
        fill_light=(0.35, 0.05, 0.65, 1.0),
        bgr_color=(245, 40, 165),
    ),
    4: ColorTheme(
        number=4,
        name="Crimson Red",
        primary_rgb=(0.96, 0.08, 0.22),
        accent_rgb=(1.00, 0.45, 0.55),
        glow_rgb=(1.00, 0.15, 0.30),
        key_light=(1.00, 0.25, 0.35, 1.0),
        fill_light=(0.50, 0.02, 0.10, 1.0),
        bgr_color=(55, 20, 245),
    ),
    5: ColorTheme(
        number=5,
        name="Gold",
        primary_rgb=(1.00, 0.82, 0.18),
        accent_rgb=(1.00, 0.95, 0.60),
        glow_rgb=(1.00, 0.85, 0.25),
        key_light=(1.00, 0.88, 0.45, 1.0),
        fill_light=(0.55, 0.40, 0.05, 1.0),
        bgr_color=(45, 210, 255),
    ),
}


class ColorThemeManager:
    """Manages smooth transitions between the 5 color themes across the 3D environment."""

    def __init__(self, initial_number: int = 5):
        self.active_number = initial_number
        self.current_theme = THEMES.get(initial_number, THEMES[5])
        self.target_theme = self.current_theme

        self.current_primary = np.array(self.current_theme.primary_rgb, dtype=np.float32)
        self.target_primary = self.current_primary.copy()

        self.current_key_light = np.array(self.current_theme.key_light, dtype=np.float32)
        self.target_key_light = self.current_key_light.copy()

        self.current_fill_light = np.array(self.current_theme.fill_light, dtype=np.float32)
        self.target_fill_light = self.current_fill_light.copy()

        self.is_transitioning = False
        self.transition_time = 0.0
        self.transition_duration = 1.35  # Duration for smooth color shift
        self.transition_progress = 1.0
        self.ambient_time = 0.0

    def set_theme(self, number: int) -> bool:
        """Sets a new theme (1 to 5). Returns True if theme was changed;
        returns False if already active or outside 1..5.
        """
        if number < 1 or number > 5:
            return False
        if number == self.active_number:
            return False

        self.active_number = number
        self.target_theme = THEMES[number]
        self.target_primary = np.array(self.target_theme.primary_rgb, dtype=np.float32)
        self.target_key_light = np.array(self.target_theme.key_light, dtype=np.float32)
        self.target_fill_light = np.array(self.target_theme.fill_light, dtype=np.float32)

        self.is_transitioning = True
        self.transition_time = 0.0
        self.transition_progress = 0.0
        return True

    def update(self, dt: float) -> None:
        dt = float(np.clip(dt, 0.001, 0.050))
        self.ambient_time += dt

        if self.is_transitioning:
            self.transition_time += dt
            t = min(1.0, self.transition_time / self.transition_duration)
            self.transition_progress = t

            lerp_speed = min(1.0, dt * 5.0)
            self.current_primary += (self.target_primary - self.current_primary) * lerp_speed
            self.current_key_light += (self.target_key_light - self.current_key_light) * lerp_speed
            self.current_fill_light += (self.target_fill_light - self.current_fill_light) * lerp_speed

            if t >= 1.0:
                self.is_transitioning = False
                self.current_theme = self.target_theme
                self.current_primary = self.target_primary.copy()
                self.current_key_light = self.target_key_light.copy()
                self.current_fill_light = self.target_fill_light.copy()
        else:
            self.current_theme = self.target_theme
            self.current_primary = self.target_primary.copy()

    def get_particle_colors(
        self,
        positions: np.ndarray,
        base_radiance: np.ndarray
    ) -> np.ndarray:
        n = len(positions)
        cols = np.empty((n, 4), dtype=np.float32)
        cols[:, 3] = 0.95

        prim = self.current_primary
        cols[:, :3] = prim[None, :] * base_radiance[:, None]

        # Specular flare during transitions
        if self.is_transitioning:
            pulse = math.sin(self.transition_progress * math.pi) * 0.25
            cols[:, :3] = np.clip(cols[:, :3] + pulse, 0.0, 1.0)

        return cols
