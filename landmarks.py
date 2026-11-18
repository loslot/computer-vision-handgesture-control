"""Landmark definitions and compatibility access."""
from __future__ import annotations

import numpy as np

from models_3d import get_model, Model3D

LANDMARKS = ("ប្រាសាទអង្គរវត្ត (Angkor Wat)", "លេខ ១ (3D Number 1)", "លេខ ២ (3D Number 2)", "លេខ ៣ (3D Number 3)", "លេខ ៤ (3D Number 4)", "លេខ ៥ (3D Number 5)")


def make_landmark(index: int, count: int = 50_000, seed: int = 7) -> np.ndarray:
    """Return a normalized (count, 3) float32 point-cloud target from high-detail 3D models."""
    model = get_model(index % len(LANDMARKS), count)
    return model.points.copy()
