"""Interactive 3D Hand Gesture Viewer (Numbers 1 to 4, plus open hand).
Initial Object: Angkor Wat (ប្រាសាទអង្គរវត្ត).
Gestures 1-4 transform Angkor Wat into volumetric 3D Numbers 1-4 with color themes:
1. Electric Blue
2. Emerald Green
3. Royal Purple
4. Crimson Red
Keyboard 5 still selects 3D Number 5.
"""
from __future__ import annotations

import sys
import time

if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    print(
        f"This dependency set requires Python 3.10, 3.11, or 3.12. "
        f"You are running Python {sys.version.split()[0]}.",
        file=sys.stderr
    )
    raise SystemExit(1)

try:
    import cv2
    import numpy as np
    import pygame
    from OpenGL.GL import *
except ImportError as exc:
    missing = exc.name or "a required package"
    print(f"Missing dependency: {missing}. Install with: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc

from color_themes import THEMES
from hand_tracking import HandTracker
from particles import ParticleSystem
from renderer import Renderer

WIDTH, HEIGHT, PARTICLES = 1280, 800, 50_000
# Significantly bigger webcam dimensions (380 x 260)
CAM_W, CAM_H = 380, 260

# Keep every user-facing HUD label here so the interface can be translated in one place.
HUD_STRINGS = {
    "title": "3D OBJECT: {shape} / [ THEME {number}: {theme} ]",
    "title_angkor": "3D OBJECT: ANGKOR WAT",
    "gesture_ready": "Ready: Show Fingers (1 to 5)  |  Fist = Angkor Wat",
    "diagnostics": "FPS {fps:5.1f}   Tracking {ms:4.1f} ms",
    "particles": "Particles {count:,}   Shape: {shape}",
    "transforming": "TRANSFORMING: {source} -> {target} [{percent}%]",
    "gesture_help": "Gestures: Fist.Angkor Wat  1.Blue  2.Green  3.Purple  4.Red  5.Gold",
    "key_help": "Keys: 1-5 Number | 0/Tab/R Reset | F Flip X | V Flip Y | Esc Quit",
}


def _apply_hand_gesture(system: ParticleSystem, hand) -> None:
    """Apply a stable recognized pose; no detected pose leaves the scene alone."""
    if hand.is_manual:
        if 1 <= hand.number <= 5:
            system.transform_to_number(hand.number)
    elif hand.number == -1:
        system.reset_to_angkor_wat()
    elif 1 <= hand.number <= 5:
        system.transform_to_number(hand.number)


def _draw_text(
    text: str,
    x: int,
    top: int,
    font: pygame.font.Font,
    win_height: int,
    color=(225, 238, 250),
    background=(10, 18, 32, 210)
):
    """Draws clean HUD text overlay matching the original text presentation."""
    surface = font.render(text, True, color)
    rgba = pygame.Surface((surface.get_width() + 14, surface.get_height() + 8), pygame.SRCALPHA)
    rgba.fill(background)
    rgba.blit(surface, (7, 4))
    raw = pygame.image.tostring(rgba, "RGBA", True)

    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glWindowPos2i(x, max(0, win_height - top - rgba.get_height()))
    glDrawPixels(rgba.get_width(), rgba.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT),
        pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    )
    pygame.display.set_caption("3D Hand Gesture Viewer | Angkor Wat ➔ 3D Numbers 1-5")

    win_w, win_h = WIDTH, HEIGHT
    renderer = Renderer(win_w, win_h)

    latin_font_names = ["Share Tech Mono", "Consolas", "monospace"]
    font = pygame.font.SysFont(latin_font_names, 17)
    bold_font = pygame.font.SysFont(latin_font_names, 19, bold=True)
    small = pygame.font.SysFont(latin_font_names, 14)

    try:
        tracker = HandTracker()
    except Exception as exc:
        pygame.quit()
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    # Initial 3D Object is ALWAYS Angkor Wat (Model 0)
    system = ParticleSystem(PARTICLES)
    overlay = True
    help_visible = True
    preview_flip_horizontal = False
    preview_flip_vertical = False

    yaw = 0.0
    pitch = 0.0
    follow_x = 0.0
    follow_y = 0.0
    last = time.perf_counter()
    fps = 60.0
    running = True

    try:
        while running:
            now = time.perf_counter()
            dt = now - last
            last = now
            fps = fps * 0.9 + (1.0 / max(dt, 1e-5)) * 0.1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    win_w, win_h = event.size
                    renderer.resize(win_w, win_h)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif pygame.K_1 <= event.key <= pygame.K_5:
                        num = event.key - pygame.K_0
                        tracker.set_manual_gesture(num)
                        system.transform_to_number(num)
                    elif event.key == pygame.K_0 or event.key == pygame.K_TAB:
                        tracker.set_manual_gesture(None)
                        system.reset_to_angkor_wat()
                    elif event.key == pygame.K_r:
                        yaw = pitch = 0.0
                        tracker.set_manual_gesture(None)
                        system.reset_to_angkor_wat()
                    elif event.key == pygame.K_o:
                        overlay = not overlay
                    elif event.key == pygame.K_h:
                        help_visible = not help_visible
                    elif event.key == pygame.K_f:
                        preview_flip_horizontal = not preview_flip_horizontal
                    elif event.key == pygame.K_v:
                        preview_flip_vertical = not preview_flip_vertical

            # Read real webcam frame and hand tracking state
            frame, hand = tracker.read()

            # Keyboard 1-5 is kept distinct from the real fist reset gesture.
            _apply_hand_gesture(system, hand)

            # Keep particle control aligned with the displayed preview when its adjustment keys are used.
            control_x = 1.0 - hand.x if preview_flip_horizontal else hand.x
            control_y = 1.0 - hand.y if preview_flip_vertical else hand.y

            # Angkor Wat follows the tracked wrist position; number shapes keep
            # their existing hand-driven rotation behavior.
            if system.target_model_idx == 0 and hand.detected:
                target_follow_x = (control_x - 0.5) * 2.0
                target_follow_y = (0.5 - control_y) * 1.4
                follow_x += (target_follow_x - follow_x) * min(1.0, dt * 8.0)
                follow_y += (target_follow_y - follow_y) * min(1.0, dt * 8.0)
                yaw += (0.0 - yaw) * min(1.0, dt * 4.0)
                pitch += (0.0 - pitch) * min(1.0, dt * 4.0)
            elif hand.detected and hand.number > 0:
                target_yaw = (control_x - 0.5) * 110.0
                # When hand moves up (hand.y < 0.5), model tilts UP:
                target_pitch = (control_y - 0.5) * 60.0
                yaw += (target_yaw - yaw) * min(1.0, dt * 6.0)
                pitch += (target_pitch - pitch) * min(1.0, dt * 6.0)

            # Update particle simulation & color transitions
            points, normals, colors = system.update(dt, control_y)

            # Update dynamic OpenGL lighting
            tm = system.theme_mgr
            renderer.set_theme_lighting(
                tuple(tm.current_key_light),
                tuple(tm.current_fill_light),
                tuple(tm.current_primary)
            )

            # Render 3D Scene
            renderer.draw(
                points,
                normals,
                colors,
                system.current_model.mesh_lines,
                yaw,
                pitch,
                overlay,
                follow_x,
                follow_y
            )

            # Render significantly bigger webcam feed (380 x 260) with original clean layout
            cam_x = 18
            cam_y = win_h - CAM_H - 18
            if frame is not None:
                preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # OpenGL pixel coordinates start at the lower-left; flip rows for upright display.
                preview = cv2.flip(preview, 0)
                if preview_flip_horizontal:
                    preview = cv2.flip(preview, 1)
                if preview_flip_vertical:
                    preview = cv2.flip(preview, 0)
                renderer.draw_pixels(preview, cam_x, cam_y, CAM_W, CAM_H)

            # --- HUD OVERLAYS ---
            active_num = system.active_number
            active_theme = THEMES.get(active_num, THEMES[5])
            pr, pg, pb = active_theme.primary_rgb
            theme_color_255 = (int(pr * 255), int(pg * 255), int(pb * 255))

            # Title card
            current_shape_name = system.current_model.name_en
            title_text = (
                HUD_STRINGS["title_angkor"]
                if system.target_model_idx == 0
                else HUD_STRINGS["title"].format(
                    shape=current_shape_name.upper(), number=active_num,
                    theme=active_theme.name.upper()
                )
            )
            _draw_text(
                title_text,
                18, 16, bold_font, win_h,
                color=theme_color_255
            )

            # Gesture status below enlarged webcam
            hud_y_offset = 18 + CAM_H + 16
            num_label = HUD_STRINGS["gesture_ready"]
            _draw_text(num_label, 18, hud_y_offset, font, win_h, color=theme_color_255)

            # Diagnostics
            _draw_text(HUD_STRINGS["diagnostics"].format(fps=fps, ms=hand.inference_ms), 18, hud_y_offset + 36, small, win_h)
            _draw_text(HUD_STRINGS["particles"].format(count=PARTICLES, shape=system.current_model.name_en), 18, hud_y_offset + 66, small, win_h)

            # Transformation progress banner
            if system.is_transitioning:
                pct = int(system.transition_progress * 100)
                from_shape = system.models[system.current_model_idx].name_en
                to_shape = system.models[system.target_model_idx].name_en
                _draw_text(
                    HUD_STRINGS["transforming"].format(source=from_shape, target=to_shape, percent=pct),
                    18, hud_y_offset + 96, small, win_h,
                    color=(255, 240, 120)
                )

            # Bottom help controls
            if help_visible:
                _draw_text(HUD_STRINGS["gesture_help"], 18, win_h - 36, small, win_h)
                _draw_text(HUD_STRINGS["key_help"], 18, win_h - 64, small, win_h)

            pygame.display.flip()
            pygame.time.wait(1)

    finally:
        tracker.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
