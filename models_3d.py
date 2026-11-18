"""High-detail procedural 3D models for:
0. Angkor Wat (ប្រាសាទអង្គរវត្ត) - Initial iconic Cambodian temple with 5 lotus-bud towers
1. 3D Number 1 (Volumetric beveled Roman numeral 1)
2. 3D Number 2 (Volumetric sculpted numeral 2 with upper arch and diagonal stroke)
3. 3D Number 3 (Volumetric double-lobed numeral 3 with central cusp)
4. 3D Number 4 (Volumetric open-top numeral 4 with crossbar)
5. 3D Number 5 (Volumetric top-flagged and curved-bowl numeral 5)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Model3D:
    name_kh: str
    name_en: str
    points: np.ndarray      # (N, 3) float32 positions
    normals: np.ndarray     # (N, 3) float32 surface normals
    colors: np.ndarray      # (N, 4) float32 RGBA colors
    mesh_lines: np.ndarray  # (M, 2, 3) float32 wireframe segments for structural overlay


def _normalize_vectors(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norms > 1e-6, v / norms, np.array([0.0, 1.0, 0.0], dtype=np.float32))


def _sample_box(
    n: int,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    color: Tuple[float, float, float, float],
    rng: np.random.Generator,
    lines_list: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points on the 6 outer faces of a 3D box, with exact normals."""
    cx, cy, cz = center
    sx, sy, sz = size
    half_x, half_y, half_z = sx * 0.5, sy * 0.5, sz * 0.5

    if lines_list is not None:
        corners = np.array([
            [cx - half_x, cy - half_y, cz - half_z],
            [cx + half_x, cy - half_y, cz - half_z],
            [cx + half_x, cy - half_y, cz + half_z],
            [cx - half_x, cy - half_y, cz + half_z],
            [cx - half_x, cy + half_y, cz - half_z],
            [cx + half_x, cy + half_y, cz - half_z],
            [cx + half_x, cy + half_y, cz + half_z],
            [cx - half_x, cy + half_y, cz + half_z],
        ], dtype=np.float32)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for e1, e2 in edges:
            lines_list.append(np.array([corners[e1], corners[e2]], dtype=np.float32))

    area_x = sy * sz
    area_y = sx * sz
    area_z = sx * sy
    total_area = 2 * (area_x + area_y + area_z)
    probs = np.array([area_x, area_x, area_y, area_y, area_z, area_z], dtype=np.float64) / max(total_area, 1e-6)
    probs /= probs.sum()

    face_assignments = rng.choice(6, size=n, p=probs)
    pts = np.empty((n, 3), dtype=np.float32)
    norms = np.empty((n, 3), dtype=np.float32)

    u = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    v = rng.uniform(-1.0, 1.0, n).astype(np.float32)

    # Face 0: +X
    m0 = face_assignments == 0
    pts[m0] = np.column_stack([np.full(np.sum(m0), cx + half_x), cy + u[m0] * half_y, cz + v[m0] * half_z])
    norms[m0] = [1.0, 0.0, 0.0]

    # Face 1: -X
    m1 = face_assignments == 1
    pts[m1] = np.column_stack([np.full(np.sum(m1), cx - half_x), cy + u[m1] * half_y, cz + v[m1] * half_z])
    norms[m1] = [-1.0, 0.0, 0.0]

    # Face 2: +Y
    m2 = face_assignments == 2
    pts[m2] = np.column_stack([cx + u[m2] * half_x, np.full(np.sum(m2), cy + half_y), cz + v[m2] * half_z])
    norms[m2] = [0.0, 1.0, 0.0]

    # Face 3: -Y
    m3 = face_assignments == 3
    pts[m3] = np.column_stack([cx + u[m3] * half_x, np.full(np.sum(m3), cy - half_y), cz + v[m3] * half_z])
    norms[m3] = [0.0, -1.0, 0.0]

    # Face 4: +Z
    m4 = face_assignments == 4
    pts[m4] = np.column_stack([cx + u[m4] * half_x, cy + v[m4] * half_y, np.full(np.sum(m4), cz + half_z)])
    norms[m4] = [0.0, 0.0, 1.0]

    # Face 5: -Z
    m5 = face_assignments == 5
    pts[m5] = np.column_stack([cx + u[m5] * half_x, cy + v[m5] * half_y, np.full(np.sum(m5), cz - half_z)])
    norms[m5] = [0.0, 0.0, -1.0]

    base_col = np.array(color, dtype=np.float32)
    variation = rng.uniform(0.90, 1.10, (n, 1)).astype(np.float32)
    cols = np.tile(base_col, (n, 1))
    cols[:, :3] = np.clip(cols[:, :3] * variation, 0.0, 1.0)
    return pts, norms, cols


def _sample_angled_box(
    n: int,
    start_pt: Tuple[float, float, float],
    end_pt: Tuple[float, float, float],
    thickness: float,
    depth: float,
    color: Tuple[float, float, float, float],
    rng: np.random.Generator,
    lines_list: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1 = np.array(start_pt, dtype=np.float32)
    p2 = np.array(end_pt, dtype=np.float32)
    delta = p2 - p1
    length = float(np.linalg.norm(delta[:2]))
    angle = math.atan2(float(delta[1]), float(delta[0]))
    center = (p1 + p2) * 0.5

    pts, norms, cols = _sample_box(n, (0, 0, center[2]), (length, thickness, depth), color, rng)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    x = pts[:, 0] * cos_a - pts[:, 1] * sin_a + center[0]
    y = pts[:, 0] * sin_a + pts[:, 1] * cos_a + center[1]
    z = pts[:, 2]

    nx = norms[:, 0] * cos_a - norms[:, 1] * sin_a
    ny = norms[:, 0] * sin_a + norms[:, 1] * cos_a
    nz = norms[:, 2]

    if lines_list is not None:
        hw, ht, hd = length * 0.5, thickness * 0.5, depth * 0.5
        box_corners = np.array([
            [-hw, -ht, -hd], [hw, -ht, -hd], [hw, -ht, hd], [-hw, -ht, hd],
            [-hw,  ht, -hd], [hw,  ht, -hd], [hw,  ht, hd], [-hw,  ht, hd]
        ], dtype=np.float32)
        rx = box_corners[:, 0] * cos_a - box_corners[:, 1] * sin_a + center[0]
        ry = box_corners[:, 0] * sin_a + box_corners[:, 1] * cos_a + center[1]
        rz = box_corners[:, 2]
        trans_corners = np.column_stack([rx, ry, rz])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for e1, e2 in edges:
            lines_list.append(np.array([trans_corners[e1], trans_corners[e2]], dtype=np.float32))

    return np.column_stack([x, y, z]), np.column_stack([nx, ny, nz]), cols


def _sample_curved_bar(
    n: int,
    center: Tuple[float, float, float],
    radius: float,
    width: float,
    depth: float,
    theta_start: float,
    theta_end: float,
    color: Tuple[float, float, float, float],
    rng: np.random.Generator,
    lines_list: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points along a 3D curved volumetric arch with front, back, outer, and inner faces."""
    cx, cy, cz = center
    arc_len = abs(theta_end - theta_start) * radius
    area_front = arc_len * width
    area_back = arc_len * width
    area_outer = arc_len * depth
    area_inner = arc_len * depth
    tot = area_front + area_back + area_outer + area_inner
    probs = np.array([area_front, area_back, area_outer, area_inner], dtype=np.float64) / max(tot, 1e-6)

    face = rng.choice(4, size=n, p=probs)
    theta = rng.uniform(theta_start, theta_end, size=n).astype(np.float32)
    u = rng.uniform(-width * 0.5, width * 0.5, size=n).astype(np.float32)
    z = rng.uniform(-depth * 0.5, depth * 0.5, size=n).astype(np.float32)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    pts = np.empty((n, 3), dtype=np.float32)
    norms = np.empty((n, 3), dtype=np.float32)

    # Front (+Z)
    m0 = face == 0
    pts[m0] = np.column_stack([cx + (radius + u[m0]) * cos_t[m0], cy + (radius + u[m0]) * sin_t[m0], np.full(np.sum(m0), cz + depth * 0.5)])
    norms[m0] = [0, 0, 1]

    # Back (-Z)
    m1 = face == 1
    pts[m1] = np.column_stack([cx + (radius + u[m1]) * cos_t[m1], cy + (radius + u[m1]) * sin_t[m1], np.full(np.sum(m1), cz - depth * 0.5)])
    norms[m1] = [0, 0, -1]

    # Outer (+R)
    m2 = face == 2
    pts[m2] = np.column_stack([cx + (radius + width * 0.5) * cos_t[m2], cy + (radius + width * 0.5) * sin_t[m2], cz + z[m2]])
    norms[m2] = np.column_stack([cos_t[m2], sin_t[m2], np.zeros(np.sum(m2))])

    # Inner (-R)
    m3 = face == 3
    pts[m3] = np.column_stack([cx + (radius - width * 0.5) * cos_t[m3], cy + (radius - width * 0.5) * sin_t[m3], cz + z[m3]])
    norms[m3] = np.column_stack([-cos_t[m3], -sin_t[m3], np.zeros(np.sum(m3))])

    cols = np.tile(np.array(color, dtype=np.float32), (n, 1))

    if lines_list is not None:
        segs = 12
        ths = np.linspace(theta_start, theta_end, segs)
        for r_edge in (radius - width * 0.5, radius + width * 0.5):
            for z_edge in (cz - depth * 0.5, cz + depth * 0.5):
                for i in range(segs - 1):
                    p1 = [cx + r_edge * np.cos(ths[i]), cy + r_edge * np.sin(ths[i]), z_edge]
                    p2 = [cx + r_edge * np.cos(ths[i+1]), cy + r_edge * np.sin(ths[i+1]), z_edge]
                    lines_list.append(np.array([p1, p2], dtype=np.float32))

    return pts, norms, cols


def _sample_lotus_tower(
    n: int,
    center: Tuple[float, float, float],
    base_radius: float,
    height: float,
    color: Tuple[float, float, float, float],
    rng: np.random.Generator,
    lines_list: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample an iconic Angkor Wat lotus-bud prasat tower."""
    cx, cy, cz = center
    t = rng.uniform(0.0, 1.0, n).astype(np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)

    tier_freq = 6.0
    tier_bulge = 0.08 * np.sin(tier_freq * t * 2.0 * np.pi)
    profile = (1.0 - t ** 1.8) * (1.0 + tier_bulge)
    radius = base_radius * np.maximum(profile, 0.02)
    flute = 1.0 + 0.04 * np.cos(12.0 * theta)
    radius *= flute

    x = cx + radius * np.cos(theta)
    y = cy + t * height
    z = cz + radius * np.sin(theta)
    pts = np.column_stack([x, y, z]).astype(np.float32)

    nx = np.cos(theta)
    ny = np.full(n, 0.25, dtype=np.float32)
    nz = np.sin(theta)
    norms = _normalize_vectors(np.column_stack([nx, ny, nz]))

    if lines_list is not None:
        for ring_t in (0.2, 0.45, 0.7, 0.9):
            r_ring = base_radius * (1.0 - ring_t ** 1.8)
            y_ring = cy + ring_t * height
            segs = 16
            angs = np.linspace(0, 2 * np.pi, segs, endpoint=False)
            for i in range(segs):
                a1, a2 = angs[i], angs[(i + 1) % segs]
                p1 = [cx + r_ring * np.cos(a1), y_ring, cz + r_ring * np.sin(a1)]
                p2 = [cx + r_ring * np.cos(a2), y_ring, cz + r_ring * np.sin(a2)]
                lines_list.append(np.array([p1, p2], dtype=np.float32))

    base_col = np.array(color, dtype=np.float32)
    cols = np.tile(base_col, (n, 1))
    cols[:, :3] *= (0.85 + 0.35 * t[:, None])
    cols[:, :3] = np.clip(cols[:, :3], 0.0, 1.0)
    return pts, norms, cols


def _assemble_model(
    pts_list: List[np.ndarray],
    norms_list: List[np.ndarray],
    cols_list: List[np.ndarray],
    lines_list: List[np.ndarray],
    total_count: int,
    name_kh: str,
    name_en: str,
    rng: np.random.Generator,
    target_scale: float = 1.15
) -> Model3D:
    all_pts = np.concatenate(pts_list, axis=0)
    all_norms = np.concatenate(norms_list, axis=0)
    all_cols = np.concatenate(cols_list, axis=0)

    if len(all_pts) > total_count:
        all_pts = all_pts[:total_count]
        all_norms = all_norms[:total_count]
        all_cols = all_cols[:total_count]
    elif len(all_pts) < total_count:
        rem = total_count - len(all_pts)
        idx = rng.choice(len(all_pts), rem)
        all_pts = np.concatenate([all_pts, all_pts[idx]], axis=0)
        all_norms = np.concatenate([all_norms, all_norms[idx]], axis=0)
        all_cols = np.concatenate([all_cols, all_cols[idx]], axis=0)

    all_pts -= all_pts.mean(axis=0, keepdims=True)
    scale = target_scale / max(float(np.abs(all_pts).max()), 1e-4)
    all_pts *= scale
    mesh_arr = np.array(lines_list, dtype=np.float32) * scale if lines_list else np.zeros((0, 2, 3), dtype=np.float32)

    return Model3D(
        name_kh=name_kh,
        name_en=name_en,
        points=all_pts.astype(np.float32),
        normals=all_norms.astype(np.float32),
        colors=all_cols.astype(np.float32),
        mesh_lines=mesh_arr
    )


# -------------------------------------------------------------
# 0. Initial Model: Angkor Wat (ប្រាសាទអង្គរវត្ត)
# -------------------------------------------------------------
def build_angkor_wat(total_count: int = 50_000, seed: int = 7) -> Model3D:
    """Builds detailed 3D Angkor Wat with 5 lotus-bud towers and concentric stepped galleries."""
    rng = np.random.default_rng(seed)
    lines: List[np.ndarray] = []
    pts_list, norms_list, cols_list = [], [], []

    c_base = (0.78, 0.68, 0.48, 0.95)
    c_gallery = (0.85, 0.74, 0.52, 0.95)
    c_tower = (0.95, 0.80, 0.40, 0.96)
    c_gold = (1.00, 0.86, 0.25, 0.98)

    # Concentric stepped galleries
    p1, n1, c1 = _sample_box(int(total_count * 0.14), (0, -0.68, 0), (2.85, 0.16, 2.15), c_base, rng, lines)
    p2, n2, c2 = _sample_box(int(total_count * 0.12), (0, -0.48, 0), (2.25, 0.18, 1.68), c_gallery, rng, lines)
    p3, n3, c3 = _sample_box(int(total_count * 0.10), (0, -0.26, 0), (1.65, 0.22, 1.25), c_gallery, rng, lines)
    pts_list.extend([p1, p2, p3])
    norms_list.extend([n1, n2, n3])
    cols_list.extend([c1, c2, c3])

    # Central causeway
    pcw, ncw, ccw = _sample_box(int(total_count * 0.06), (0, -0.62, 1.25), (0.45, 0.12, 0.85), c_base, rng, lines)
    pts_list.append(pcw); norms_list.append(ncw); cols_list.append(ccw)

    # Five lotus-bud towers (Quincunx)
    p_center, n_center, c_center = _sample_lotus_tower(int(total_count * 0.28), (0, -0.15, 0), 0.38, 1.25, c_gold, rng, lines)
    pts_list.append(p_center); norms_list.append(n_center); cols_list.append(c_center)

    corner_coords = [(-0.62, -0.45), (0.62, -0.45), (-0.62, 0.45), (0.62, 0.45)]
    pts_per_corner = int(total_count * 0.30) // 4
    for cx, cz in corner_coords:
        p_c, n_c, c_c = _sample_lotus_tower(pts_per_corner, (cx, -0.20, cz), 0.24, 0.82, c_tower, rng, lines)
        pts_list.append(p_c); norms_list.append(n_c); cols_list.append(c_c)

    return _assemble_model(pts_list, norms_list, cols_list, lines, total_count, "ប្រាសាទអង្គរវត្ត", "Angkor Wat", rng)


# -------------------------------------------------------------
# 1. 3D Number 1
# -------------------------------------------------------------
def build_number_1(total_count: int = 50_000, seed: int = 101) -> Model3D:
    """Builds a realistic 3D volumetric Numeral 1."""
    rng = np.random.default_rng(seed)
    lines: List[np.ndarray] = []
    pts_list, norms_list, cols_list = [], [], []
    c_gold = (1.0, 0.85, 0.25, 0.98)

    # Main vertical pillar
    p_stem, n_stem, c_stem = _sample_box(int(total_count * 0.60), (0.04, 0.02, 0.0), (0.30, 1.50, 0.38), c_gold, rng, lines)
    # Diagonal top serif / beak
    p_flag, n_flag, c_flag = _sample_angled_box(int(total_count * 0.22), (0.04, 0.72, 0.0), (-0.42, 0.38, 0.0), 0.24, 0.38, c_gold, rng, lines)
    # Base plinth
    p_base, n_base, c_base = _sample_box(int(total_count * 0.18), (0.04, -0.74, 0.0), (0.85, 0.22, 0.45), c_gold, rng, lines)

    pts_list.extend([p_stem, p_flag, p_base])
    norms_list.extend([n_stem, n_flag, n_base])
    cols_list.extend([c_stem, c_flag, c_base])

    return _assemble_model(pts_list, norms_list, cols_list, lines, total_count, "លេខ ១", "Number 1", rng)


# -------------------------------------------------------------
# 2. 3D Number 2
# -------------------------------------------------------------
def build_number_2(total_count: int = 50_000, seed: int = 102) -> Model3D:
    """Builds a realistic 3D volumetric Numeral 2."""
    rng = np.random.default_rng(seed)
    lines: List[np.ndarray] = []
    pts_list, norms_list, cols_list = [], [], []
    c_gold = (1.0, 0.85, 0.25, 0.98)

    # Upper curved arch
    p_arc, n_arc, c_arc = _sample_curved_bar(int(total_count * 0.42), (0.0, 0.34, 0.0), 0.40, 0.24, 0.36, -0.35, math.pi, c_gold, rng, lines)
    # Diagonal neck stroke
    p_diag, n_diag, c_diag = _sample_angled_box(int(total_count * 0.36), (0.34, 0.20, 0.0), (-0.38, -0.66, 0.0), 0.24, 0.36, c_gold, rng, lines)
    # Horizontal bottom base
    p_base, n_base, c_base = _sample_box(int(total_count * 0.22), (0.0, -0.75, 0.0), (0.86, 0.22, 0.40), c_gold, rng, lines)

    pts_list.extend([p_arc, p_diag, p_base])
    norms_list.extend([n_arc, n_diag, n_base])
    cols_list.extend([c_arc, c_diag, c_base])

    return _assemble_model(pts_list, norms_list, cols_list, lines, total_count, "លេខ ២", "Number 2", rng)


# -------------------------------------------------------------
# 3. 3D Number 3
# -------------------------------------------------------------
def build_number_3(total_count: int = 50_000, seed: int = 103) -> Model3D:
    """Builds a geometric flat-top 3 from uniform-width extruded strokes."""
    rng = np.random.default_rng(seed)
    lines: List[np.ndarray] = []
    pts_list, norms_list, cols_list = [], [], []
    c_gold = (1.0, 0.85, 0.25, 0.98)

    stroke = 0.20
    depth = 0.36

    # Flat, square-ended top bar. The diagonal descends from its right end
    # toward the open left side of the lower bowl.
    p_top, n_top, c_top = _sample_box(
        int(total_count * 0.30), (0.0, 0.70, 0.0), (1.30, stroke, depth), c_gold, rng, lines
    )
    p_diag, n_diag, c_diag = _sample_angled_box(
        int(total_count * 0.20), (0.58, 0.70, 0.0), (0.0, 0.15, 0.0), stroke, depth, c_gold, rng, lines
    )

    # One continuous lower bowl: it turns right at the middle, rounds through
    # the bottom, then ends at the lower left. The short final bar makes a flat cut.
    bowl_center = (0.0, -0.27, 0.0)
    bowl_radius = 0.45
    tail_angle = -2.45
    p_bowl, n_bowl, c_bowl = _sample_curved_bar(
        int(total_count * 0.42), bowl_center, bowl_radius, stroke, depth,
        tail_angle + 2.0 * math.pi, math.pi / 2.0 + 2.0 * math.pi,
        c_gold, rng, lines
    )
    tail_x = bowl_center[0] + bowl_radius * math.cos(tail_angle)
    tail_y = bowl_center[1] + bowl_radius * math.sin(tail_angle)
    tail_direction = np.array([math.sin(tail_angle), -math.cos(tail_angle)], dtype=np.float32)
    tail_direction /= np.linalg.norm(tail_direction)
    tail_start = np.array([tail_x, tail_y], dtype=np.float32) - tail_direction * 0.14
    p_tail, n_tail, c_tail = _sample_angled_box(
        int(total_count * 0.08),
        (float(tail_start[0]), float(tail_start[1]), 0.0),
        (tail_x, tail_y, 0.0), stroke, depth, c_gold, rng, lines
    )

    pts_list.extend([p_top, p_diag, p_bowl, p_tail])
    norms_list.extend([n_top, n_diag, n_bowl, n_tail])
    cols_list.extend([c_top, c_diag, c_bowl, c_tail])

    return _assemble_model(pts_list, norms_list, cols_list, lines, total_count, "លេខ ៣", "Number 3", rng)


# -------------------------------------------------------------
# 4. 3D Number 4
# -------------------------------------------------------------
def build_number_4(total_count: int = 50_000, seed: int = 104) -> Model3D:
    """Builds a realistic 3D volumetric Numeral 4."""
    rng = np.random.default_rng(seed)
    lines: List[np.ndarray] = []
    pts_list, norms_list, cols_list = [], [], []
    c_gold = (1.0, 0.85, 0.25, 0.98)

    # Main vertical spine
    p_stem, n_stem, c_stem = _sample_box(int(total_count * 0.48), (0.24, 0.0, 0.0), (0.24, 1.65, 0.36), c_gold, rng, lines)
    # Diagonal left beam
    p_diag, n_diag, c_diag = _sample_angled_box(int(total_count * 0.30), (0.24, 0.80, 0.0), (-0.46, -0.15, 0.0), 0.22, 0.36, c_gold, rng, lines)
    # Horizontal crossbar
    p_cross, n_cross, c_cross = _sample_box(int(total_count * 0.22), (-0.08, -0.15, 0.0), (0.88, 0.20, 0.36), c_gold, rng, lines)

    pts_list.extend([p_stem, p_diag, p_cross])
    norms_list.extend([n_stem, n_diag, n_cross])
    cols_list.extend([c_stem, c_diag, c_cross])

    return _assemble_model(pts_list, norms_list, cols_list, lines, total_count, "លេខ ៤", "Number 4", rng)


# -------------------------------------------------------------
# 5. 3D Number 5
# -------------------------------------------------------------
def build_number_5(total_count: int = 50_000, seed: int = 105) -> Model3D:
    """Builds a realistic 3D volumetric Numeral 5."""
    rng = np.random.default_rng(seed)
    lines: List[np.ndarray] = []
    pts_list, norms_list, cols_list = [], [], []
    c_gold = (1.0, 0.85, 0.25, 0.98)

    # Top horizontal bar
    p_top, n_top, c_top = _sample_box(int(total_count * 0.26), (0.02, 0.74, 0.0), (0.76, 0.20, 0.36), c_gold, rng, lines)
    # Vertical upper-left stem
    p_stem, n_stem, c_stem = _sample_box(int(total_count * 0.24), (-0.28, 0.38, 0.0), (0.20, 0.56, 0.36), c_gold, rng, lines)
    # Lower rounded bowl
    p_bowl, n_bowl, c_bowl = _sample_curved_bar(int(total_count * 0.50), (-0.02, -0.22, 0.0), 0.46, 0.22, 0.36, -1.25 * math.pi, 0.45 * math.pi, c_gold, rng, lines)

    pts_list.extend([p_top, p_stem, p_bowl])
    norms_list.extend([n_top, n_stem, n_bowl])
    cols_list.extend([c_top, c_stem, c_bowl])

    return _assemble_model(pts_list, norms_list, cols_list, lines, total_count, "លេខ ៥", "Number 5", rng)


# List of all 6 models: 0 is Angkor Wat, 1-5 are 3D Numbers 1-5
MODELS_GENERATORS = [
    build_angkor_wat,  # 0: Initial state
    build_number_1,    # 1: 3D Number 1
    build_number_2,    # 2: 3D Number 2
    build_number_3,    # 3: 3D Number 3
    build_number_4,    # 4: 3D Number 4
    build_number_5,    # 5: 3D Number 5
]


def get_model(index: int, total_count: int = 50_000) -> Model3D:
    gen = MODELS_GENERATORS[index % len(MODELS_GENERATORS)]
    return gen(total_count)
