"""Modern OpenGL renderer with theme-driven dynamic lighting,
ground pedestal reflections, and restored original webcam layout.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective


class Renderer:
    """Renders 3D particle objects with realistic materials,
    dynamic lighting matching active color themes 1-10, and original webcam display.
    """

    def __init__(self, width: int = 1280, height: int = 800):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

        # Deep cosmic navy background
        glClearColor(0.015, 0.020, 0.038, 1.0)
        glClearDepth(1.0)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Dynamic OpenGL Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)  # Dynamic Key Light
        glEnable(GL_LIGHT1)  # Dynamic Fill Light
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Default Key Light: Electric Blue
        glLightfv(GL_LIGHT0, GL_POSITION, (3.0, 4.5, 4.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.15, 0.15, 0.20, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.20, 0.65, 1.00, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.70, 0.90, 1.00, 1.0))

        # Default Fill Light
        glLightfv(GL_LIGHT1, GL_POSITION, (-4.0, -1.5, 3.0, 1.0))
        glLightfv(GL_LIGHT1, GL_AMBIENT, (0.05, 0.08, 0.12, 1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.05, 0.20, 0.75, 1.0))
        glLightfv(GL_LIGHT1, GL_SPECULAR, (0.40, 0.60, 0.90, 1.0))

        # Specular material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.90, 0.90, 0.90, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 48.0)

        self.pedestal_color = (0.2, 0.6, 1.0)

    def resize(self, width: int, height: int):
        self.width, self.height = max(1, width), max(1, height)
        glViewport(0, 0, self.width, self.height)

    def set_theme_lighting(self, key_rgba: Tuple[float, float, float, float], fill_rgba: Tuple[float, float, float, float], primary_rgb: Tuple[float, float, float]):
        """Updates OpenGL lights to smoothly reflect the current active color theme."""
        glLightfv(GL_LIGHT0, GL_DIFFUSE, key_rgba)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, fill_rgba)
        self.pedestal_color = primary_rgb

    def _draw_ground_pedestal(self, rot_angle: float):
        """Draws glowing geometric ground rings tinted by active theme color."""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        glPushMatrix()
        glTranslatef(0.0, -1.30, 0.0)
        glRotatef(90.0, 1.0, 0.0, 0.0)
        glRotatef(rot_angle * 0.4, 0.0, 0.0, 1.0)

        pr, pg, pb = self.pedestal_color

        # Outer ring
        glColor4f(pr, pg, pb, 0.35)
        glLineWidth(1.8)
        segs = 64
        glBegin(GL_LINE_LOOP)
        for i in range(segs):
            theta = 2.0 * math.pi * i / segs
            glVertex2f(1.85 * math.cos(theta), 1.85 * math.sin(theta))
        glEnd()

        # Inner ring
        glRotatef(-rot_angle * 0.8, 0.0, 0.0, 1.0)
        glColor4f(min(1.0, pr + 0.3), min(1.0, pg + 0.3), min(1.0, pb + 0.3), 0.22)
        glLineWidth(1.2)
        glBegin(GL_LINE_LOOP)
        for i in range(segs):
            theta = 2.0 * math.pi * i / segs
            glVertex2f(1.45 * math.cos(theta), 1.45 * math.sin(theta))
        glEnd()

        glPopMatrix()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)

    def draw(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
        mesh_lines: Optional[np.ndarray],
        yaw: float,
        pitch: float,
        show_overlay: bool = True,
        translate_x: float = 0.0,
        translate_y: float = 0.0
    ):
        """Renders 3D scene with theme-colored lighting, particles, and wireframes."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.width / max(1, self.height), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3.95)
        glTranslatef(translate_x, translate_y, 0.0)
        glRotatef(pitch, 1.0, 0.0, 0.0)
        glRotatef(yaw, 0.0, 1.0, 0.0)

        # 1. Ground Pedestal
        self._draw_ground_pedestal(yaw)

        # 2. Structural Wireframe Overlay (if enabled)
        if show_overlay and mesh_lines is not None and len(mesh_lines) > 0:
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)

            pr, pg, pb = self.pedestal_color
            glColor4f(pr, pg, pb, 0.32)
            glLineWidth(1.3)

            flat_lines = np.ascontiguousarray(mesh_lines.reshape(-1, 3), dtype=np.float32)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, flat_lines)
            glDrawArrays(GL_LINES, 0, len(flat_lines))
            glDisableClientState(GL_VERTEX_ARRAY)

            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LIGHTING)

        # 3. 50,000 Dynamic Particles with Normals and Theme Colors
        glEnable(GL_LIGHTING)
        glPointSize(2.8)

        pts_contiguous = np.ascontiguousarray(points, dtype=np.float32)
        norms_contiguous = np.ascontiguousarray(normals, dtype=np.float32)
        cols_contiguous = np.ascontiguousarray(colors, dtype=np.float32)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, pts_contiguous)
        glNormalPointer(GL_FLOAT, 0, norms_contiguous)
        glColorPointer(4, GL_FLOAT, 0, cols_contiguous)

        glDrawArrays(GL_POINTS, 0, len(points))

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def draw_pixels(self, image: np.ndarray, x: int, y: int, width: int, height: int):
        """Draw an RGB image in window coordinates (origin at lower left).
        Restores the authentic, unadorned layout and performance of the previous webcam style.
        """
        if image is None or image.size == 0:
            return
        from cv2 import resize, INTER_AREA
        image = resize(image, (width, height), interpolation=INTER_AREA)
        image = np.ascontiguousarray(image)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glWindowPos2i(x, y)
        glPixelZoom(width / image.shape[1], height / image.shape[0])
        glDrawPixels(image.shape[1], image.shape[0], GL_RGB, GL_UNSIGNED_BYTE, image)
        glPixelZoom(1, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
