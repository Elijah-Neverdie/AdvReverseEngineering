# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""拟合外轮廓曲线的拆分与贝塞尔重拟合算法。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .region_fit import (
    RegionFitError,
    fit_cubic_bezier_controls,
    make_segment_color,
    polyline_length,
    polyline_parameters,
    resample_closed_polyline,
    resample_polyline,
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
    n = max(int(control_count), 3)
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
    "estimate_similarity_transform",
    "extract_subpolyline_by_arc",
    "find_break_indices",
    "fit_bezier_n_controls",
    "point_at_arc_length",
    "sample_polyline_uniform",
    "segment_colors_for_count",
    "split_polyline_at_breaks",
    "transform_bezier_points",
    "turn_angles_deg",
)
