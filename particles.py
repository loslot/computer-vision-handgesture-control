"""Realistic 3D particle morphing and simulation engine:
Transforms from Angkor Wat (Initial Object) into 3D Numbers 1, 2, 3, 4, and 5
with dynamic 5-color theme transitions and cinematic particle animations.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from color_themes import ColorThemeManager, THEMES
from models_3d import Model3D, get_model


def _ease_in_out_cubic(x: float) -> float:
    return 4.0 * x * x * x if x < 0.5 else 1.0 - math.pow(-2.0 * x + 2.0, 3) / 2.0


class ParticleSystem:
    """Manages 50,000+ particles with cinematic 3D shape transformations:
    Starts with Angkor Wat and transforms into 3D Numbers 1, 2, 3, 4, and 5.
    """

    TRANSITION_DURATION = 1.65  # Seconds for complete shape transformation

    def __init__(self, count: int = 50_000, seed: int = 42):
        self.count = count
        self.rng = np.random.default_rng(seed)

        # Pre-generate all 6 models:
        # Index 0: Angkor Wat (Initial Object)
        # Index 1: 3D Number 1
        # Index 2: 3D Number 2
        # Index 3: 3D Number 3
        # Index 4: 3D Number 4
        # Index 5: 3D Number 5
        self.models: list[Model3D] = [
            get_model(0, count),  # Angkor Wat (Initial)
            get_model(1, count),  # 3D Number 1
            get_model(2, count),  # 3D Number 2
            get_model(3, count),  # 3D Number 3
            get_model(4, count),  # 3D Number 4
            get_model(5, count),  # 3D Number 5
        ]

        # Initial state: ALWAYS Angkor Wat (Index 0)
        self.current_model_idx = 0
        self.target_model_idx = 0

        initial_model = self.models[0]
        self.positions = initial_model.points.copy()
        self.normals = initial_model.normals.copy()
        self.velocities = np.zeros_like(self.positions)

        # Source and destination buffers for morphing
        self.source_positions = self.positions.copy()
        self.dest_positions = self.positions.copy()
        self.source_normals = self.normals.copy()
        self.dest_normals = self.normals.copy()

        # Random per-particle parameters for organic flow
        self.particle_phases = self.rng.uniform(0.0, 2.0 * np.pi, (count, 1)).astype(np.float32)
        self.particle_speeds = self.rng.uniform(0.8, 1.3, (count, 1)).astype(np.float32)
        self.swirl_axes = self.rng.normal(0.0, 1.0, (count, 3)).astype(np.float32)
        self.swirl_axes /= (np.linalg.norm(self.swirl_axes, axis=1, keepdims=True) + 1e-6)
        self.base_radiance = self.rng.uniform(0.75, 1.05, count).astype(np.float32)

        # 5 Color Themes Manager (Initial: Theme 5 Gold for Angkor Wat)
        self.theme_mgr = ColorThemeManager(initial_number=5)
        self.colors = self.theme_mgr.get_particle_colors(self.positions, self.base_radiance)

        # Morphing state
        self.is_transitioning = False
        self.transition_time = 0.0
        self.transition_progress = 1.0
        self.ambient_timer = 0.0

    @property
    def current_model(self) -> Model3D:
        return self.models[self.target_model_idx]

    @property
    def active_number(self) -> int:
        return self.theme_mgr.active_number

    @property
    def current_theme_name(self) -> str:
        return self.theme_mgr.current_theme.name

    def transform_to_number(self, number: int) -> bool:
        """Transforms 3D object to the specified 3D number (1 to 5) and activates its color theme.
        Returns True if a transformation was initiated; returns False if already at this target.
        """
        if number < 1 or number > 5:
            return False

        # If already transitioning to or currently at this target number, do NOT re-trigger!
        if number == self.target_model_idx:
            return False

        self.current_model_idx = self.target_model_idx
        self.target_model_idx = number

        # Snapshot current positions as source
        self.source_positions = self.positions.copy()
        self.source_normals = self.normals.copy()

        dest_model = self.models[self.target_model_idx]
        self.dest_positions = dest_model.points.copy()
        self.dest_normals = dest_model.normals.copy()

        # Update color theme
        self.theme_mgr.set_theme(number)

        # Initial outward velocity burst
        tangent = np.cross(self.source_positions, self.swirl_axes)
        self.velocities = tangent * 1.6 + self.rng.normal(0.0, 0.35, self.positions.shape).astype(np.float32)

        self.is_transitioning = True
        self.transition_time = 0.0
        self.transition_progress = 0.0
        return True

    def reset_to_angkor_wat(self) -> None:
        """Resets 3D object back to Angkor Wat (Model 0)."""
        # Also guard while the morph is in progress: the reset gesture is reported
        # every frame, but should initiate the reset only once.
        if self.target_model_idx == 0:
            return

        self.current_model_idx = self.target_model_idx
        self.target_model_idx = 0
        self.source_positions = self.positions.copy()
        self.source_normals = self.normals.copy()

        angkor = self.models[0]
        self.dest_positions = angkor.points.copy()
        self.dest_normals = angkor.normals.copy()

        self.theme_mgr.set_theme(5)  # Gold theme for Angkor Wat

        self.is_transitioning = True
        self.transition_time = 0.0
        self.transition_progress = 0.0

    def update(self, dt: float, hand_y: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advances simulation, particle morphing, and color transitions by dt seconds."""
        dt = float(np.clip(dt, 0.001, 0.050))
        self.ambient_timer += dt

        # Update color transitions
        self.theme_mgr.update(dt)

        if self.is_transitioning:
            self.transition_time += dt
            raw_t = min(1.0, self.transition_time / self.TRANSITION_DURATION)
            self.transition_progress = raw_t

            # Smooth ease-in-out cubic morph curve
            s = _ease_in_out_cubic(raw_t)

            # 1. Base shape interpolation
            pos_interp = (1.0 - s) * self.source_positions + s * self.dest_positions

            # 2. Dynamic 3D Spatial Migration and Cosmic Vortex
            flight_envelope = math.sin(raw_t * math.pi)
            vortex_angle = raw_t * 3.2 * math.pi * self.particle_speeds
            cos_a = np.cos(vortex_angle)
            sin_a = np.sin(vortex_angle)

            px = pos_interp[:, 0:1]
            pz = pos_interp[:, 2:3]
            swirl_x = px * cos_a - pz * sin_a - px
            swirl_z = px * sin_a + pz * cos_a - pz

            wave_y = np.sin(pos_interp[:, 0:1] * 3.0 + self.particle_phases + self.ambient_timer * 4.0) * 0.20
            wave_x = np.cos(pos_interp[:, 1:2] * 2.5 + self.particle_phases) * 0.16

            disp_x = (swirl_x * 0.40 + wave_x) * flight_envelope
            disp_y = wave_y * flight_envelope
            disp_z = (swirl_z * 0.40) * flight_envelope

            self.positions = pos_interp + np.column_stack([disp_x, disp_y, disp_z])
            self.normals = (1.0 - s) * self.source_normals + s * self.dest_normals

            if raw_t >= 1.0:
                self.is_transitioning = False
                self.positions = self.dest_positions.copy()
                self.normals = self.dest_normals.copy()
                self.velocities.fill(0.0)

        else:
            # Steady-state with subtle organic breathing oscillation
            dest = self.dest_positions.copy()
            dir_norm = dest / (np.linalg.norm(dest, axis=1, keepdims=True) + 1e-5)
            breath = 0.018 * np.sin(self.ambient_timer * 1.8 + self.particle_phases)
            target_pos = dest + dir_norm * breath

            spring = (target_pos - self.positions) * 22.0
            damping = -self.velocities * 7.5
            self.velocities += (spring + damping) * dt
            self.positions += self.velocities * dt
            self.normals = self.dest_normals

        # Compute dynamic particle colors matching the active theme
        self.colors = self.theme_mgr.get_particle_colors(self.positions, self.base_radiance)

        return self.positions, self.normals, self.colors
