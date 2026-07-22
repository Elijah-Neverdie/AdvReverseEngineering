# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""拟合外轮廓曲线的拆分与贝塞尔重拟合算法。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .region_fit import (
    RegionFitError,
    build_quad_patch,
    build_triangular_patch,
    fit_cubic_bezier_controls,
    make_segment_color,
    polyline_length,
    polyline_parameters,
    resample_closed_polyline,
    resample_polyline,
    sample_cubic_bezier,
)


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def turn_angles_deg(points: np.ndarray, cyclic: bool) -> np.ndarray:
    """折线各顶点转向角（度）：0=共线，越大折角越尖。"""
    pts = _as_float_array(points)
    count = len(pts)
    angles = np.zeros(count, dtype=np.float64)
    if count < 3:
        return angles
    for index in range(count):
        if not cyclic and (index == 0 or index == count - 1):
            continue
        prev_pt = pts[(index - 1) % count]
        curr_pt = pts[index]
        next_pt = pts[(index + 1) % count]
        incoming = curr_pt - prev_pt
        outgoing = next_pt - curr_pt
        in_len = float(np.linalg.norm(incoming))
        out_len = float(np.linalg.norm(outgoing))
        if in_len < 1e-12 or out_len < 1e-12:
            continue
        cos_v = float(
            np.clip(
                np.dot(incoming, outgoing) / (in_len * out_len),
                -1.0,
                1.0,
            )
        )
        angles[index] = float(np.degrees(np.arccos(cos_v)))
    return angles


def find_break_indices(
    points: np.ndarray,
    angle_threshold_deg: float,
    cyclic: bool,
    min_separation: int = 2,
) -> list[int]:
    """转向角 >= 阈值的顶点作为断开处（局部非极大抑制）。"""
    pts = _as_float_array(points)
    count = len(pts)
    if count < 3:
        return []
    angles = turn_angles_deg(pts, cyclic=cyclic)
    threshold = float(max(angle_threshold_deg, 1.0))
    candidates = [
        index
        for index in range(count)
        if angles[index] >= threshold
        and (cyclic or 0 < index < count - 1)
    ]
    if not candidates:
        return []

    candidates.sort(key=lambda i: float(angles[i]), reverse=True)
    kept: list[int] = []
    sep = max(int(min_separation), 1)
    for index in candidates:
        if any(_circular_distance(index, other, count) < sep for other in kept):
            continue
        kept.append(index)
    return sorted(kept)


def _circular_distance(a: int, b: int, count: int) -> int:
    raw = abs(int(a) - int(b))
    return min(raw, count - raw)


def split_polyline_at_breaks(
    points: np.ndarray,
    break_indices: Sequence[int],
    cyclic: bool,
) -> list[np.ndarray]:
    """按断开下标切成连续折线段；闭环 N 个断点 -> N 段。"""
    pts = _as_float_array(points)
    count = len(pts)
    breaks = sorted({int(i) % count for i in break_indices})
    if count < 2:
        return []
    if not breaks:
        return [pts.copy()]

    if not cyclic:
        cuts = [0] + breaks + [count - 1]
        segments: list[np.ndarray] = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            if end <= start:
                continue
            segment = pts[start : end + 1]
            if len(segment) >= 2:
                segments.append(segment.copy())
        return segments

    if len(breaks) == 1:
        start = breaks[0]
        ordered = np.vstack((pts[start:], pts[: start + 1]))
        return [ordered.copy()]

    segments = []
    for index, start in enumerate(breaks):
        end = breaks[(index + 1) % len(breaks)]
        if end > start:
            segment = pts[start : end + 1]
        else:
            segment = np.vstack((pts[start:], pts[: end + 1]))
        if len(segment) >= 2:
            segments.append(segment.copy())
    return segments


def segment_colors_for_count(count: int) -> list[tuple[float, float, float, float]]:
    return [
        make_segment_color(0xC0FFEE00 ^ (i * 0x9E3779B9)) for i in range(max(count, 0))
    ]


def point_at_arc_length(points: np.ndarray, arc: float) -> np.ndarray:
    """开环折线上按弧长取点。"""
    pts = _as_float_array(points)
    if len(pts) == 0:
        raise RegionFitError("空折线")
    if len(pts) == 1:
        return pts[0].copy()
    params = polyline_parameters(pts)
    total = float(polyline_length(pts))
    if total < 1e-12:
        return pts[0].copy()
    t = float(np.clip(arc / total, 0.0, 1.0))
    result = np.empty(3, dtype=np.float64)
    for axis in range(3):
        result[axis] = float(np.interp(t, params, pts[:, axis]))
    return result


def extract_subpolyline_by_arc(
    points: np.ndarray,
    arc0: float,
    arc1: float,
    samples: int = 24,
) -> np.ndarray:
    """开环折线在弧长 [arc0, arc1] 上重采样。"""
    pts = _as_float_array(points)
    total = float(polyline_length(pts))
    if total < 1e-12:
        return np.repeat(pts[:1], max(samples, 2), axis=0)
    a0 = float(np.clip(min(arc0, arc1), 0.0, total))
    a1 = float(np.clip(max(arc0, arc1), 0.0, total))
    count = max(int(samples), 2)
    arcs = np.linspace(a0, a1, count, dtype=np.float64)
    return np.vstack([point_at_arc_length(pts, float(a)) for a in arcs])


def fit_bezier_n_controls(
    points: np.ndarray,
    control_count: int,
    cyclic: bool = False,
) -> list[dict]:
    """
    将折线拟合成 n 个贝塞尔锚点（Blender BEZIER spline）。

    返回每项: {co, handle_left, handle_right}。
    """
    pts = _as_float_array(points)
    n = max(int(control_count), 2)
    if len(pts) < 2:
        raise RegionFitError("折线点数不足，无法拟合贝塞尔")

    if cyclic:
        if len(pts) < 3:
            raise RegionFitError("闭环点数不足")
        sample_n = max(len(pts), n * 8)
        loop = resample_closed_polyline(pts, sample_n)
        open_loop = np.vstack((loop, loop[:1]))
        total = float(polyline_length(open_loop))
        anchor_arcs = [total * i / n for i in range(n)]
        anchors = [point_at_arc_length(open_loop, a) for a in anchor_arcs]

        bezier_points: list[dict] = [
            {
                "co": anchors[i].copy(),
                "handle_left": anchors[i].copy(),
                "handle_right": anchors[i].copy(),
            }
            for i in range(n)
        ]
        for i in range(n):
            a0 = anchor_arcs[i]
            a1 = anchor_arcs[(i + 1) % n]
            if a1 <= a0:
                part_a = extract_subpolyline_by_arc(open_loop, a0, total, 16)
                part_b = extract_subpolyline_by_arc(open_loop, 0.0, a1, 16)
                span_pts = np.vstack((part_a[:-1], part_b))
            else:
                span_pts = extract_subpolyline_by_arc(open_loop, a0, a1, 24)
            controls = fit_cubic_bezier_controls(span_pts)
            bezier_points[i]["co"] = controls[0].copy()
            bezier_points[i]["handle_right"] = controls[1].copy()
            j = (i + 1) % n
            bezier_points[j]["co"] = controls[3].copy()
            bezier_points[j]["handle_left"] = controls[2].copy()
        return bezier_points

    dense = resample_polyline(pts, max(len(pts), n * 8))
    total = float(polyline_length(dense))
    arcs = [total * i / (n - 1) for i in range(n)]
    anchors = [point_at_arc_length(dense, a) for a in arcs]
    bezier_points = [
        {
            "co": anchors[i].copy(),
            "handle_left": anchors[i].copy(),
            "handle_right": anchors[i].copy(),
        }
        for i in range(n)
    ]
    for i in range(n - 1):
        span_pts = extract_subpolyline_by_arc(dense, arcs[i], arcs[i + 1], 24)
        controls = fit_cubic_bezier_controls(span_pts)
        bezier_points[i]["co"] = controls[0].copy()
        bezier_points[i]["handle_right"] = controls[1].copy()
        bezier_points[i + 1]["co"] = controls[3].copy()
        bezier_points[i + 1]["handle_left"] = controls[2].copy()
    return bezier_points


def sample_polyline_uniform(
    points: np.ndarray,
    count: int,
    cyclic: bool,
) -> np.ndarray:
    pts = _as_float_array(points)
    target = max(int(count), 3)
    if cyclic:
        return resample_closed_polyline(pts, target)
    return resample_polyline(pts, target)


def best_open_alignment(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, float]:
    """开环采样对齐：仅尝试是否反向。"""
    a = _as_float_array(src)
    b = _as_float_array(dst)
    if len(a) != len(b) or len(a) < 2:
        raise RegionFitError("开环对齐需要等长采样点")
    err = float(np.mean(np.sum((b - a) ** 2, axis=1)))
    rev = b[::-1].copy()
    err_r = float(np.mean(np.sum((rev - a) ** 2, axis=1)))
    if err_r < err:
        return rev, err_r
    return b.copy(), err


def order_open_curves_as_closed_loop(
    polylines: Sequence[np.ndarray],
    max_gap_frac: float = 0.08,
    *,
    allow_large_gaps: bool = False,
) -> tuple[list[int], list[bool], float] | None:
    """
    将开环折线排成闭合环（端点近乎相接）。

    返回 (顺序下标, 是否反向, 最大接缝距离)；无法成环则 None。
    支持 3 或 4 条（三边/四边合成区面）。
    allow_large_gaps=True 时忽略间隙上限（供「缝合开口」先成环再延伸）。
    """
    curves = [_as_float_array(p) for p in polylines]
    count = len(curves)
    if count not in (3, 4):
        return None
    if any(len(p) < 2 for p in curves):
        return None

    lengths = [float(polyline_length(p)) for p in curves]
    ref_len = float(max(sum(lengths) / len(lengths), 1e-6))
    max_gap = ref_len * float(max(max_gap_frac, 1e-4))
    if allow_large_gaps:
        max_gap = float("inf")

    ends = [(p[0].copy(), p[-1].copy()) for p in curves]
    used = [False] * count
    order = [0]
    flipped = [False]
    used[0] = True
    tip = ends[0][1].copy()
    worst_gap = 0.0

    for _ in range(count - 1):
        best_i = -1
        best_flip = False
        best_d = float("inf")
        best_tip = tip
        for index in range(count):
            if used[index]:
                continue
            start_pt, end_pt = ends[index]
            d0 = float(np.linalg.norm(start_pt - tip))
            d1 = float(np.linalg.norm(end_pt - tip))
            if d0 < best_d:
                best_d = d0
                best_i = index
                best_flip = False
                best_tip = end_pt
            if d1 < best_d:
                best_d = d1
                best_i = index
                best_flip = True
                best_tip = start_pt
        if best_i < 0 or best_d > max_gap:
            return None
        used[best_i] = True
        order.append(best_i)
        flipped.append(best_flip)
        tip = best_tip.copy()
        worst_gap = max(worst_gap, best_d)

    close_d = float(np.linalg.norm(tip - ends[0][0]))
    if close_d > max_gap:
        return None
    worst_gap = max(worst_gap, close_d)
    return order, flipped, worst_gap


def opposite_edge_pairs(count: int = 4) -> list[tuple[int, int]]:
    """有序闭环上的对边下标对：0-2、1-3。"""
    if int(count) != 4:
        return []
    return [(0, 2), (1, 3)]


def closest_points_on_rays(
    origin_a: np.ndarray,
    dir_a: np.ndarray,
    origin_b: np.ndarray,
    dir_b: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """
    两条射线（半直线）的最近点对中点。

    要求参数 ta>=0、tb>=0（向外延伸）。返回 (交点近似, ta, tb)；失败 None。
    """
    p1 = _as_float_array(origin_a).reshape(3)
    p2 = _as_float_array(origin_b).reshape(3)
    d1 = _as_float_array(dir_a).reshape(3)
    d2 = _as_float_array(dir_b).reshape(3)
    n1 = float(np.linalg.norm(d1))
    n2 = float(np.linalg.norm(d2))
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    d1 = d1 / n1
    d2 = d2 / n2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, r))
    e = float(np.dot(d2, r))
    denom = a * c - b * b
    if abs(denom) < 1e-14:
        # 近似平行：取中点投影
        ta = 0.0
        tb = float(np.dot(p1 - p2, d2))
        if tb < 0.0:
            return None
        qa = p1
        qb = p2 + tb * d2
        return 0.5 * (qa + qb), ta, tb
    ta = (b * e - c * d) / denom
    tb = (a * e - b * d) / denom
    if ta < -1e-6 or tb < -1e-6:
        return None
    ta = max(ta, 0.0)
    tb = max(tb, 0.0)
    qa = p1 + ta * d1
    qb = p2 + tb * d2
    return 0.5 * (qa + qb), ta, tb


def stitch_oriented_loop_polylines(
    loop_sides: Sequence[np.ndarray],
    *,
    gap_frac: float = 0.015,
    max_extend_frac: float = 3.0,
) -> tuple[list[np.ndarray], int]:
    """
    环向折线缝合开口：对间隙过大的接缝，沿两端切向延伸至交点。

    返回 (新折线列表, 缝合接缝数)。
    """
    sides = [_as_float_array(s).copy() for s in loop_sides]
    n = len(sides)
    if n < 2:
        return sides, 0
    lengths = [float(max(polyline_length(s), 1e-6)) for s in sides]
    mean_len = float(sum(lengths) / n)
    gap_tol = mean_len * float(max(gap_frac, 1e-5))
    max_extend = mean_len * float(max(max_extend_frac, 0.1))
    stitched = 0

    for index in range(n):
        a = sides[index]
        b = sides[(index + 1) % n]
        if len(a) < 2 or len(b) < 2:
            continue
        end_a = a[-1]
        start_b = b[0]
        gap = float(np.linalg.norm(end_a - start_b))
        if gap <= gap_tol:
            continue
        dir_a = end_a - a[-2]
        dir_b = start_b - b[1]  # 向环外延伸（与 b 行进反向）
        hit = closest_points_on_rays(end_a, dir_a, start_b, dir_b)
        if hit is None:
            continue
        point, ta, tb = hit
        if ta > max_extend or tb > max_extend:
            continue
        # 延伸：在端点外侧追加交点（已在端点则仅钉齐）
        if ta > 1e-8:
            sides[index] = np.vstack((a, point.reshape(1, 3)))
        else:
            sides[index] = a.copy()
            sides[index][-1] = point
        if tb > 1e-8:
            sides[(index + 1) % n] = np.vstack((point.reshape(1, 3), b))
        else:
            sides[(index + 1) % n] = b.copy()
            sides[(index + 1) % n][0] = point
        stitched += 1

    return sides, stitched


def sample_bezier_anchor_chain(
    bezier_points: Sequence[dict],
    cyclic: bool = False,
    samples_per_span: int = 24,
) -> np.ndarray:
    """将贝塞尔锚点链（含手柄）采样为开/闭环折线。"""
    if not bezier_points:
        return np.zeros((0, 3), dtype=np.float64)
    n = len(bezier_points)
    span_count = n if cyclic else max(n - 1, 0)
    if span_count <= 0:
        co = _as_float_array(bezier_points[0]["co"]).reshape(3)
        return co.reshape(1, 3)
    parts: list[np.ndarray] = []
    per = max(int(samples_per_span), 2)
    for i in range(span_count):
        a = bezier_points[i]
        b = bezier_points[(i + 1) % n]
        controls = np.vstack(
            (
                _as_float_array(a["co"]).reshape(3),
                _as_float_array(a["handle_right"]).reshape(3),
                _as_float_array(b["handle_left"]).reshape(3),
                _as_float_array(b["co"]).reshape(3),
            )
        )
        sampled = sample_cubic_bezier(controls, per)
        if i > 0:
            sampled = sampled[1:]
        parts.append(sampled)
    return np.vstack(parts)


def orient_loop_polylines(
    polylines: Sequence[np.ndarray],
    max_gap_frac: float = 0.08,
) -> list[np.ndarray] | None:
    """排成闭合环并统一方向；失败返回 None。"""
    ordered = order_open_curves_as_closed_loop(polylines, max_gap_frac=max_gap_frac)
    if ordered is None:
        return None
    order, flipped, _gap = ordered
    result: list[np.ndarray] = []
    for index, reverse in zip(order, flipped):
        pts = _as_float_array(polylines[index])
        if reverse:
            pts = pts[::-1].copy()
        result.append(pts)
    return result


def prepare_triangular_sides_from_loop(
    loop_sides: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从三边闭环取 (side_a, side_b, base)：
    a/b 从底角指向 tip，base 为最短边。
    """
    sides = [_as_float_array(s) for s in loop_sides]
    if len(sides) != 3:
        raise RegionFitError("三边合成需要恰好三条边")
    lengths = [float(polyline_length(s)) for s in sides]
    base_i = int(np.argmin(lengths))
    base = sides[base_i]
    right = sides[(base_i + 1) % 3]  # base 终点 → tip
    left_rev = sides[(base_i + 2) % 3]  # tip → base 起点
    left = left_rev[::-1].copy()
    return left, right, base


def default_surface_segments(
    length_u: float,
    length_v: float,
    controls_u: int,
    controls_v: int,
    *,
    seg_min: int = 1,
    seg_max: int = 64,
) -> tuple[int, int]:
    """
    默认曲面细分：短边方向 = 该边控制点数；
    长边 = 短边控制点数 × (长边长/短边长)；对边共用同一细分。
    """
    lu = max(float(length_u), 1e-12)
    lv = max(float(length_v), 1e-12)
    cu = max(int(controls_u), 2)
    cv = max(int(controls_v), 2)
    lo = max(int(seg_min), 1)
    hi = max(int(seg_max), lo)

    def _clamp(value: int) -> int:
        return int(max(lo, min(hi, int(value))))

    if lu <= lv:
        seg_u = cu
        seg_v = max(lo, int(round(cu * (lv / lu))))
    else:
        seg_v = cv
        seg_u = max(lo, int(round(cv * (lu / lv))))
    return _clamp(seg_u), _clamp(seg_v)


def compose_patch_from_boundary_polylines(
    polylines: Sequence[np.ndarray],
    *,
    segments_u: int = 12,
    segments_v: int = 12,
    max_gap_frac: float = 0.12,
) -> tuple[np.ndarray, list[tuple[int, ...]], str]:
    """
    用 3/4 条边界折线合成区面网格。

    返回 (vertices_world, faces, kind) ，kind 为 \"TRI\" 或 \"QUAD\"。
    """
    vertices, faces, kind, _loop = compose_patch_from_boundary_polylines_ex(
        polylines,
        segments_u=segments_u,
        segments_v=segments_v,
        max_gap_frac=max_gap_frac,
    )
    return vertices, faces, kind


def compose_patch_from_boundary_polylines_ex(
    polylines: Sequence[np.ndarray],
    *,
    segments_u: int = 12,
    segments_v: int = 12,
    max_gap_frac: float = 0.12,
) -> tuple[np.ndarray, list[tuple[int, ...]], str, list[np.ndarray]]:
    """
    同 ``compose_patch_from_boundary_polylines``，额外返回定向焊接后的边界边列表。
    """
    curves = [_as_float_array(p) for p in polylines]
    if len(curves) not in (3, 4):
        raise RegionFitError("合成区面需要选中 3 或 4 条曲线")
    loop = orient_loop_polylines(curves, max_gap_frac=max_gap_frac)
    if loop is None:
        raise RegionFitError("曲线端点未近乎闭合，无法排成封闭边界")
    # 接缝共点，利于 Coons 角点
    for index in range(len(loop)):
        nxt = (index + 1) % len(loop)
        mid = 0.5 * (loop[index][-1] + loop[nxt][0])
        loop[index][-1] = mid
        loop[nxt][0] = mid

    seg_u = max(int(segments_u), 1)
    seg_v = max(int(segments_v), 1)
    if len(loop) == 4:
        vertices, faces = build_quad_patch(loop, seg_u, seg_v)
        return vertices, faces, "QUAD", loop

    left, right, base = prepare_triangular_sides_from_loop(loop)
    vertices, faces = build_triangular_patch(
        left, right, base, segments_long=seg_v, segments_base=seg_u
    )
    return vertices, faces, "TRI", loop


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _polyline_end_tangent(points: np.ndarray, at_start: bool) -> np.ndarray:
    pts = _as_float_array(points)
    if len(pts) < 2:
        return np.zeros(3, dtype=np.float64)
    if at_start:
        return _normalize_vec(pts[1] - pts[0])
    return _normalize_vec(pts[-1] - pts[-2])


def _edge_alignment_cost(a: np.ndarray, b: np.ndarray) -> float:
    """两端点距离和；越小越同向对齐。"""
    return float(
        np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[-1] - b[-1])
    )


def _orient_edge_to_match(ref: np.ndarray, edge: np.ndarray) -> np.ndarray:
    pts = _as_float_array(edge).copy()
    if _edge_alignment_cost(ref, pts) <= _edge_alignment_cost(ref, pts[::-1]):
        return pts
    return pts[::-1].copy()


def _pick_facing_edge_pair(
    sides_a: Sequence[np.ndarray],
    sides_b: Sequence[np.ndarray],
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    在两组边界边中选最适合桥接的一对对边。

    评分：中点近、方向近平行、长度接近。
    """
    best = None
    best_score = float("inf")
    for i, ea in enumerate(sides_a):
        ea = _as_float_array(ea)
        if len(ea) < 2:
            continue
        mid_a = ea.mean(axis=0)
        dir_a = _normalize_vec(ea[-1] - ea[0])
        len_a = max(float(polyline_length(ea)), 1e-9)
        for j, eb in enumerate(sides_b):
            eb = _as_float_array(eb)
            if len(eb) < 2:
                continue
            mid_b = eb.mean(axis=0)
            dir_b = _normalize_vec(eb[-1] - eb[0])
            len_b = max(float(polyline_length(eb)), 1e-9)
            gap = float(np.linalg.norm(mid_a - mid_b))
            parallel = abs(float(np.dot(dir_a, dir_b)))
            length_ratio = max(len_a, len_b) / min(len_a, len_b)
            # 距离为主，平行加分，长度比惩罚
            score = gap * (1.35 - parallel) * (1.0 + 0.25 * (length_ratio - 1.0))
            if score < best_score:
                best_score = score
                best = (i, j, ea, eb)
    if best is None:
        raise RegionFitError("无法在两曲面边界上找到可桥接的对边")
    i, j, ea, eb = best
    eb = _orient_edge_to_match(ea, eb)
    return i, j, ea, eb


def _bezier_connector(
    p0: np.ndarray,
    t0: np.ndarray,
    p1: np.ndarray,
    t1: np.ndarray,
    samples: int,
) -> np.ndarray:
    """用端点切向构造三次贝塞尔连接线（沿曲率桥接）。"""
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    gap = float(np.linalg.norm(p1 - p0))
    handle = max(gap * 0.38, 1e-6)
    t0n = _normalize_vec(t0)
    t1n = _normalize_vec(t1)
    gap_dir = _normalize_vec(p1 - p0)
    if float(np.linalg.norm(t0n)) < 1e-9:
        t0n = gap_dir
    if float(np.linalg.norm(t1n)) < 1e-9:
        t1n = gap_dir
    # 切向朝向缺口
    if float(np.dot(t0n, gap_dir)) < 0.0:
        t0n = -t0n
    if float(np.dot(t1n, gap_dir)) < 0.0:
        t1n = -t1n
    # 与缺口方向混合，避免手柄过度弯曲
    t0n = _normalize_vec(0.55 * t0n + 0.45 * gap_dir)
    t1n = _normalize_vec(0.55 * t1n + 0.45 * gap_dir)
    controls = np.vstack(
        (
            p0,
            p0 + t0n * handle,
            p1 - t1n * handle,
            p1,
        )
    )
    return sample_cubic_bezier(controls, max(int(samples), 2))


def bridge_fit_surface_boundaries(
    sides_a: Sequence[np.ndarray],
    sides_b: Sequence[np.ndarray],
    *,
    segments_u: int = 12,
    segments_v: int = 12,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """
    两张拟合曲面边界之间做 Coons 桥接。

    选取最近对边作为上下轨，两端用贝塞尔连接线（继承邻边切向曲率）。
    """
    if len(sides_a) < 3 or len(sides_b) < 3:
        raise RegionFitError("桥接需要两侧至少各有 3 条边界边")
    ia, ib, edge_a, edge_b = _pick_facing_edge_pair(sides_a, sides_b)
    # 邻边：用于端点切向（沿原拟合贝塞尔曲率延续）
    prev_a = _as_float_array(sides_a[(ia - 1) % len(sides_a)])
    next_a = _as_float_array(sides_a[(ia + 1) % len(sides_a)])
    prev_b = _as_float_array(sides_b[(ib - 1) % len(sides_b)])
    next_b = _as_float_array(sides_b[(ib + 1) % len(sides_b)])

    # edge_b 已与 edge_a 同向；两侧邻边也按端点就近匹配
    # A 起点邻边应终于 A[0]；若 next 终于起点则对调语义
    def _tangent_at_corner(
        edge: np.ndarray,
        prev_edge: np.ndarray,
        next_edge: np.ndarray,
        at_start: bool,
    ) -> np.ndarray:
        corner = edge[0] if at_start else edge[-1]
        # 选端点更靠近 corner 的邻边，取离开曲面进入缺口的切向
        candidates = []
        for adj in (prev_edge, next_edge):
            d0 = float(np.linalg.norm(adj[0] - corner))
            d1 = float(np.linalg.norm(adj[-1] - corner))
            if d0 <= d1:
                tan = _polyline_end_tangent(adj, at_start=True)
                # 从邻边起点离开：若邻边从 corner 出发，切向指向邻边内部；
                # 桥接要离开 corner 进入缺口，取反
                candidates.append((-tan, d0))
            else:
                tan = _polyline_end_tangent(adj, at_start=False)
                candidates.append((tan, d1))
        candidates.sort(key=lambda item: item[1])
        return candidates[0][0]

    # 若 B 边相对 A 翻转过，邻边索引仍对应原 ib；端点已随 edge_b 翻转对齐
    # 重新按端点距离决定 B 侧邻边切向即可
    t_a0 = _tangent_at_corner(edge_a, prev_a, next_a, at_start=True)
    t_a1 = _tangent_at_corner(edge_a, prev_a, next_a, at_start=False)
    t_b0 = _tangent_at_corner(edge_b, prev_b, next_b, at_start=True)
    t_b1 = _tangent_at_corner(edge_b, prev_b, next_b, at_start=False)

    seg_u = max(int(segments_u), 1)
    seg_v = max(int(segments_v), 1)
    # 对边采样点数一致
    bottom = resample_polyline(edge_a, seg_u + 1)
    top = resample_polyline(edge_b, seg_u + 1)
    left = _bezier_connector(bottom[0], t_a0, top[0], t_b0, seg_v + 1)
    right = _bezier_connector(bottom[-1], t_a1, top[-1], t_b1, seg_v + 1)
    # build_quad_patch 期望 sides=[bottom, right, top, left]，且会反向 top/left
    # 这里 top 与 bottom 同向，需传入 top[::-1] 作为 sides[2] 前的原始 top
    # build_quad_patch: top = resample(sides[2][::-1])，故 sides[2] 应与 bottom 反向
    sides = [bottom, right, top[::-1].copy(), left[::-1].copy()]
    return build_quad_patch(sides, seg_u, seg_v)


def pack_boundary_sides(sides: Sequence[np.ndarray]) -> tuple[list[float], list[int]]:
    """把边界边折线打包为扁平 float 列表与每边点数。"""
    flat: list[float] = []
    counts: list[int] = []
    for side in sides:
        pts = _as_float_array(side)
        counts.append(int(len(pts)))
        flat.extend(float(v) for v in pts.reshape(-1))
    return flat, counts


def unpack_boundary_sides(
    flat,
    counts,
) -> list[np.ndarray]:
    """还原边界边折线列表。"""
    values = np.asarray(list(flat), dtype=np.float64)
    counts_list = [int(c) for c in list(counts)]
    sides: list[np.ndarray] = []
    offset = 0
    for count in counts_list:
        n = max(int(count), 0)
        need = n * 3
        chunk = values[offset : offset + need]
        offset += need
        if n < 2 or len(chunk) < need:
            continue
        sides.append(chunk.reshape(n, 3))
    return sides


def extract_mesh_boundary_loop_sides(
    vertices: np.ndarray,
    faces: Sequence[Sequence[int]],
    *,
    corner_angle_deg: float = 55.0,
) -> list[np.ndarray]:
    """
    从三角/四边形网格提取开边界环，并按折角拆成多条边。

    用于旧拟合曲面未持久化边界时的回退。
    """
    verts = _as_float_array(vertices)
    edge_faces: dict[tuple[int, int], int] = {}
    for face in faces:
        ids = [int(i) for i in face]
        if len(ids) < 3:
            continue
        for i in range(len(ids)):
            a = ids[i]
            b = ids[(i + 1) % len(ids)]
            key = (a, b) if a < b else (b, a)
            edge_faces[key] = edge_faces.get(key, 0) + 1
    boundary_adj: dict[int, list[int]] = {}
    for (a, b), count in edge_faces.items():
        if count != 1:
            continue
        boundary_adj.setdefault(a, []).append(b)
        boundary_adj.setdefault(b, []).append(a)
    if not boundary_adj:
        raise RegionFitError("网格没有可桥接的开放边界")

    # 取最长边界环
    unused = set(boundary_adj.keys())
    loops: list[list[int]] = []
    while unused:
        start = unused.pop()
        loop = [start]
        prev = None
        cur = start
        guard = 0
        while guard < len(boundary_adj) + 2:
            guard += 1
            nbrs = boundary_adj.get(cur, [])
            nxt = None
            for cand in nbrs:
                if cand != prev:
                    nxt = cand
                    break
            if nxt is None:
                break
            if nxt == start:
                break
            loop.append(nxt)
            unused.discard(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(loop)
    if not loops:
        raise RegionFitError("无法提取网格边界环")
    loops.sort(
        key=lambda ids: float(
            sum(
                np.linalg.norm(verts[ids[i]] - verts[ids[(i + 1) % len(ids)]])
                for i in range(len(ids))
            )
        ),
        reverse=True,
    )
    ring = loops[0]
    pts = verts[np.asarray(ring, dtype=np.int64)]
    # 闭环折角拆边
    angles = turn_angles_deg(pts, cyclic=True)
    breaks = [
        i
        for i, ang in enumerate(angles)
        if float(ang) >= float(corner_angle_deg)
    ]
    if len(breaks) < 3:
        # 均匀拆成 4 段回退
        n = len(pts)
        step = max(n // 4, 1)
        breaks = sorted({(i * step) % n for i in range(4)})
    breaks = sorted(set(int(b) % len(pts) for b in breaks))
    sides: list[np.ndarray] = []
    for i, b0 in enumerate(breaks):
        b1 = breaks[(i + 1) % len(breaks)]
        if b1 > b0:
            seg = pts[b0 : b1 + 1]
        else:
            seg = np.vstack((pts[b0:], pts[: b1 + 1]))
        if len(seg) >= 2:
            sides.append(seg.copy())
    if len(sides) < 3:
        raise RegionFitError("边界折角不足，无法拆成桥接边")
    return sides


def weld_bezier_loop_endpoints(
    loop_beziers: Sequence[Sequence[dict]],
) -> list[list[dict]]:
    """
    四段开环贝塞尔首尾共点，保持封闭。

    每个接缝取相邻端点中点，手柄随锚点平移。
    """
    result = [
        [
            {
                "co": np.asarray(bp["co"], dtype=np.float64).copy(),
                "handle_left": np.asarray(bp["handle_left"], dtype=np.float64).copy(),
                "handle_right": np.asarray(
                    bp["handle_right"], dtype=np.float64
                ).copy(),
            }
            for bp in bezier
        ]
        for bezier in loop_beziers
    ]
    n = len(result)
    if n < 2:
        return result
    for index in range(n):
        a = result[index]
        b = result[(index + 1) % n]
        if not a or not b:
            continue
        end_bp = a[-1]
        start_bp = b[0]
        mid = 0.5 * (end_bp["co"] + start_bp["co"])
        delta_end = mid - end_bp["co"]
        delta_start = mid - start_bp["co"]
        end_bp["co"] = mid.copy()
        end_bp["handle_left"] = end_bp["handle_left"] + delta_end
        end_bp["handle_right"] = end_bp["handle_right"] + delta_end
        start_bp["co"] = mid.copy()
        start_bp["handle_left"] = start_bp["handle_left"] + delta_start
        start_bp["handle_right"] = start_bp["handle_right"] + delta_start
    return result


def best_closed_alignment(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, float]:
    """搜索最佳循环位移与是否反向，使 RMSE 最小。"""
    a = _as_float_array(src)
    b = _as_float_array(dst)
    if len(a) != len(b) or len(a) < 3:
        raise RegionFitError("相似对齐需要等长采样点")
    n = len(a)
    best_rmse = float("inf")
    best = b.copy()
    for reversed_flag in (False, True):
        cand = b[::-1].copy() if reversed_flag else b.copy()
        for shift in range(n):
            rolled = np.roll(cand, shift, axis=0)
            diff = rolled - a
            rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
            if rmse < best_rmse:
                best_rmse = rmse
                best = rolled
    return best, best_rmse


def estimate_similarity_transform(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama：s * R @ src + t ≈ dst。返回 (scale, R 3x3, t)。"""
    p = _as_float_array(src)
    q = _as_float_array(dst)
    if len(p) != len(q) or len(p) < 3:
        raise RegionFitError("相似变换需要至少 3 组对应点")
    n = len(p)
    mu_p = p.mean(axis=0)
    mu_q = q.mean(axis=0)
    pc = p - mu_p
    qc = q - mu_q
    var_p = float(np.sum(pc * pc) / n)
    if var_p < 1e-16:
        return 1.0, np.eye(3, dtype=np.float64), (mu_q - mu_p)
    cov = (qc.T @ pc) / n
    u, singular, vt = np.linalg.svd(cov)
    d = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    scale = float(np.sum(singular * d) / var_p)
    if scale < 1e-12:
        scale = 1.0
    translation = mu_q - scale * (rotation @ mu_p)
    return scale, rotation, translation


def estimate_open_directed_similarity(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    开环有向相似变换：严格按采样顺序对应（src[i]→dst[i]）。

    对闭合四边形的对边必须使用本函数，禁止再做反向对齐；
    否则会把环向首尾对调，变换后手柄扭曲。
    """
    return estimate_similarity_transform(src, dst)


def snap_bezier_endpoints(
    bezier_points: Sequence[dict],
    start: np.ndarray,
    end: np.ndarray,
) -> list[dict]:
    """将首尾锚点钉到指定端点，手柄随锚点平移。"""
    if not bezier_points:
        return []
    result = [
        {
            "co": np.asarray(bp["co"], dtype=np.float64).copy(),
            "handle_left": np.asarray(bp["handle_left"], dtype=np.float64).copy(),
            "handle_right": np.asarray(
                bp["handle_right"], dtype=np.float64
            ).copy(),
        }
        for bp in bezier_points
    ]
    start_pt = _as_float_array(start).reshape(3)
    end_pt = _as_float_array(end).reshape(3)
    delta0 = start_pt - result[0]["co"]
    result[0]["co"] = start_pt.copy()
    result[0]["handle_left"] = result[0]["handle_left"] + delta0
    result[0]["handle_right"] = result[0]["handle_right"] + delta0
    delta1 = end_pt - result[-1]["co"]
    result[-1]["co"] = end_pt.copy()
    result[-1]["handle_left"] = result[-1]["handle_left"] + delta1
    result[-1]["handle_right"] = result[-1]["handle_right"] + delta1
    return result


def opposite_pair_colors() -> list[tuple[float, float, float, float]]:
    """对边预览色：组0 / 组1。"""
    return [
        (1.0, 0.28, 0.22, 1.0),
        (0.15, 0.85, 1.0, 1.0),
    ]


def apply_similarity(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    pts = _as_float_array(points)
    return (float(scale) * (pts @ _as_float_array(rotation).T)) + _as_float_array(
        translation
    ).reshape(3)


def transform_bezier_points(
    bezier_points: Sequence[dict],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[dict]:
    result = []
    for item in bezier_points:
        result.append(
            {
                "co": apply_similarity(
                    item["co"].reshape(1, 3), scale, rotation, translation
                )[0],
                "handle_left": apply_similarity(
                    item["handle_left"].reshape(1, 3),
                    scale,
                    rotation,
                    translation,
                )[0],
                "handle_right": apply_similarity(
                    item["handle_right"].reshape(1, 3),
                    scale,
                    rotation,
                    translation,
                )[0],
            }
        )
    return result


__all__ = (
    "RegionFitError",
    "apply_similarity",
    "best_closed_alignment",
    "best_open_alignment",
    "bridge_fit_surface_boundaries",
    "closest_points_on_rays",
    "compose_patch_from_boundary_polylines",
    "compose_patch_from_boundary_polylines_ex",
    "default_surface_segments",
    "estimate_open_directed_similarity",
    "estimate_similarity_transform",
    "extract_mesh_boundary_loop_sides",
    "extract_subpolyline_by_arc",
    "find_break_indices",
    "fit_bezier_n_controls",
    "opposite_edge_pairs",
    "opposite_pair_colors",
    "order_open_curves_as_closed_loop",
    "orient_loop_polylines",
    "pack_boundary_sides",
    "point_at_arc_length",
    "prepare_triangular_sides_from_loop",
    "sample_bezier_anchor_chain",
    "sample_polyline_uniform",
    "segment_colors_for_count",
    "snap_bezier_endpoints",
    "split_polyline_at_breaks",
    "stitch_oriented_loop_polylines",
    "transform_bezier_points",
    "turn_angles_deg",
    "unpack_boundary_sides",
    "weld_bezier_loop_endpoints",
)
