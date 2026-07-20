# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""领域三边/四边规则曲面拟合（纯 NumPy，可脱离 Blender 测试）。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


DEFAULT_TRIANGLE_RATIO = 0.15
DEFAULT_CORNER_ANGLE_DEG = 35.0
DEFAULT_SEG_U = 4
DEFAULT_SEG_V = 4
MIN_SEGMENTS = 1
MAX_SEGMENTS = 64
# 边界闭环重采样数量范围：角点检测在均匀弧长采样上进行
_BOUNDARY_SAMPLES_MIN = 96
_BOUNDARY_SAMPLES_MAX = 384
# 先多取角点候选，再按最短边合并到三/四边，避免尖角被 NMS 挤掉后并成长边
_MAX_CORNER_CANDIDATES = 12
_STRONG_CONCAVE_ANGLE_DEG = 35.0
# 调试边：按折角拆分时允许的最大角点数（不再压成 3/4 边）
_MAX_SPLIT_CORNERS = 64
# 相对闭环周长过短的碎段（常见于 T 接缝锯齿）并入邻边
_MIN_SEGMENT_LOOP_FRAC = 0.025
# 多岛：线/点邻近判定相对包围盒对角线的比例
_MULTI_ISLAND_PROXIMITY_FRAC = 0.05
# 多岛融并外轮廓：仅间隙小于该比例才合并；更大则分岛，禁止弦线桥接
_MULTI_ISLAND_MERGE_GAP_FRAC = 0.08
# 单环自贴近（深凹口/内缝）判定：非邻接点间距相对包围盒对角线
_LOOP_CORRIDOR_GAP_FRAC = 0.06
# 近平行判定：方向点积绝对值下限
_MULTI_ISLAND_PARALLEL_DOT = 0.85
# 投影重叠占较长段比例下限
_MULTI_ISLAND_OVERLAP_FRAC = 0.15
# 锯齿边界：隔 2/4/8/16 点取弦，多尺度共识判定真尖角
_CORNER_SAMPLE_STRIDES = (2, 4, 8, 16)
_CORNER_STRIDE_MIN_VOTES = 2
# 凹折角判定阈值（度）；控制手柄偏离边线超过该比例视为离散噪声并忽略
_CONCAVE_FOLD_ANGLE_DEG = 35.0
_HANDLE_OUTLIER_FRAC = 0.03
_MAX_CONCAVE_FOLDS = 32


class RegionFitError(ValueError):
    """领域拟合失败。"""


@dataclass
class RegionFitResult:
    """规则拟合网格结果。"""

    vertices: np.ndarray
    faces: list[tuple[int, ...]]
    topology: str  # "TRI" | "QUAD"
    segments_u: int
    segments_v: int
    side_lengths: tuple[float, ...]
    warnings: list[str] = field(default_factory=list)


def _as_int_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vector / length


def polyline_length(points: np.ndarray) -> float:
    """折线总弧长。"""
    pts = _as_float_array(points)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def polyline_parameters(points: np.ndarray) -> np.ndarray:
    """累积弦长参数，归一化到 [0, 1]。"""
    pts = _as_float_array(points)
    if len(pts) == 0:
        return np.empty(0, dtype=np.float64)
    if len(pts) == 1:
        return np.array([0.0], dtype=np.float64)
    distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total = float(cumulative[-1])
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(pts), dtype=np.float64)
    return cumulative / total


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """按弧长重采样折线，端点固定。"""
    pts = _as_float_array(points)
    target = max(int(count), 2)
    if len(pts) == 0:
        raise RegionFitError("空折线无法重采样")
    if len(pts) == 1:
        return np.repeat(pts, target, axis=0)
    params = polyline_parameters(pts)
    sample_t = np.linspace(0.0, 1.0, target, dtype=np.float64)
    result = np.empty((target, pts.shape[1]), dtype=np.float64)
    for axis in range(pts.shape[1]):
        result[:, axis] = np.interp(sample_t, params, pts[:, axis])
    return result


def resample_closed_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """闭环折线按弧长均匀重采样，返回不含重复终点的 count 个点。"""
    pts = _as_float_array(points)
    if len(pts) < 3:
        raise RegionFitError("闭环点数不足，无法重采样")
    closed = np.vstack((pts, pts[:1]))
    sampled = resample_polyline(closed, int(count) + 1)
    return sampled[:-1]


def _normalize_target_ids(target_id) -> list[int]:
    """把单个编号或编号集合规整为有序去重列表。"""
    if isinstance(target_id, (int, np.integer)):
        ids = [int(target_id)]
    else:
        ids = sorted({int(value) for value in target_id})
    if not ids or any(value < 0 for value in ids):
        raise RegionFitError("拟合领域编号无效")
    return ids


def _target_label(target_ids: list[int]) -> str:
    return "+".join(str(value) for value in target_ids)


def extract_region_boundary_loops(
    region_ids: np.ndarray,
    target_id,
    loop_start: np.ndarray,
    loop_total: np.ndarray,
    loop_vertex_indices: np.ndarray,
) -> list[list[int]]:
    """
    从 polygon loops 提取目标领域（可为多个领域的并集）的有序边界闭环。

    通过取消并集内部成对有向半边保留外轮廓与跨领域边；
    相邻被选领域之间的公共边会自动消去。
    """
    ids = _as_int_array(region_ids)
    starts = _as_int_array(loop_start)
    totals = _as_int_array(loop_total)
    loops = _as_int_array(loop_vertex_indices)
    if len(ids) == 0:
        return []
    targets = _normalize_target_ids(target_id)
    label = _target_label(targets)
    mask = np.isin(ids, np.asarray(targets, dtype=np.int32))
    if not np.any(mask):
        raise RegionFitError(f"领域 {label} 不存在")

    halfedge_count: dict[tuple[int, int], int] = defaultdict(int)
    face_indices = np.flatnonzero(mask)
    for face_index in face_indices.tolist():
        start = int(starts[face_index])
        total = int(totals[face_index])
        if total < 3:
            continue
        for offset in range(total):
            v0 = int(loops[start + offset])
            v1 = int(loops[start + ((offset + 1) % total)])
            if v0 == v1:
                continue
            halfedge_count[(v0, v1)] += 1

    boundary_edges: list[tuple[int, int]] = []
    for (v0, v1), count in halfedge_count.items():
        opposite = halfedge_count.get((v1, v0), 0)
        keep = int(count) - int(opposite)
        for _ in range(max(keep, 0)):
            boundary_edges.append((v0, v1))

    if not boundary_edges:
        raise RegionFitError(f"领域 {label} 没有可提取的外边界")

    outgoing: dict[int, list[int]] = defaultdict(list)
    for v0, v1 in boundary_edges:
        outgoing[v0].append(v1)

    for vertex, neighbors in outgoing.items():
        if len(neighbors) != 1:
            # 分支/非流形：暂时仍尝试追踪，但若失败会报错
            if len(neighbors) == 0:
                raise RegionFitError(
                    f"领域 {label} 边界在顶点 {vertex} 处中断"
                )

    unused: dict[tuple[int, int], int] = defaultdict(int)
    for edge in boundary_edges:
        unused[edge] += 1

    closed_loops: list[list[int]] = []
    for start_edge in boundary_edges:
        if unused[start_edge] <= 0:
            continue
        v_start, v_next = start_edge
        unused[start_edge] -= 1
        loop = [v_start]
        current = v_next
        guard = 0
        max_steps = len(boundary_edges) + 2
        while current != v_start:
            loop.append(current)
            candidates = [
                nxt
                for nxt in outgoing.get(current, [])
                if unused[(current, nxt)] > 0
            ]
            if not candidates:
                raise RegionFitError(
                    f"领域 {label} 边界不闭合（顶点 {current}）"
                )
            # 多分支时选第一条未使用出边
            nxt = candidates[0]
            unused[(current, nxt)] -= 1
            current = nxt
            guard += 1
            if guard > max_steps:
                raise RegionFitError(f"领域 {label} 边界追踪溢出")
        if len(loop) >= 3:
            closed_loops.append(loop)

    if not closed_loops:
        raise RegionFitError(f"领域 {label} 未形成有效闭环边界")
    return closed_loops


def build_edge_face_adjacency(
    loop_start: np.ndarray,
    loop_total: np.ndarray,
    loop_vertex_indices: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    """无向边 (min,max) -> 入射面索引列表。"""
    starts = _as_int_array(loop_start)
    totals = _as_int_array(loop_total)
    loops = _as_int_array(loop_vertex_indices)
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index in range(len(starts)):
        start = int(starts[face_index])
        total = int(totals[face_index])
        if total < 3:
            continue
        for offset in range(total):
            v0 = int(loops[start + offset])
            v1 = int(loops[start + ((offset + 1) % total)])
            if v0 == v1:
                continue
            key = (v0, v1) if v0 < v1 else (v1, v0)
            edge_faces[key].append(int(face_index))
    return edge_faces


def boundary_loop_neighbor_ids(
    loop: Sequence[int],
    region_ids: np.ndarray,
    edge_faces: dict[tuple[int, int], list[int]],
    target_id,
) -> np.ndarray:
    """
    环上每条边 (loop[i] -> loop[i+1]) 的外侧邻域编号。

    无外侧面（网格外边界）记为 -1。
    """
    ids = _as_int_array(region_ids)
    targets = set(_normalize_target_ids(target_id))
    count = len(loop)
    neighbors = np.full(count, -1, dtype=np.int32)
    for index in range(count):
        v0 = int(loop[index])
        v1 = int(loop[(index + 1) % count])
        key = (v0, v1) if v0 < v1 else (v1, v0)
        outside: list[int] = []
        for face_index in edge_faces.get(key, []):
            rid = int(ids[int(face_index)])
            if rid not in targets:
                outside.append(rid)
        if outside:
            neighbors[index] = int(max(set(outside), key=outside.count))
    return neighbors


def neighbor_change_vertex_indices(neighbor_ids: np.ndarray) -> list[int]:
    """邻域 ID 在顶点处发生变化的下标（T 接缝切点）。"""
    values = np.asarray(neighbor_ids, dtype=np.int32)
    count = len(values)
    if count < 2:
        return []
    changes: list[int] = []
    for index in range(count):
        prev = int(values[(index - 1) % count])
        curr = int(values[index])
        if prev != curr:
            changes.append(int(index))
    return changes


def map_loop_indices_to_resampled(
    loop_points: np.ndarray,
    vertex_indices: Sequence[int],
    sample_count: int,
) -> list[int]:
    """按弧长把原始环顶点下标映射到重采样环下标。"""
    pts = _as_float_array(loop_points)
    if len(pts) < 2 or int(sample_count) < 2:
        return []
    closed = np.vstack((pts, pts[:1]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cum[-1])
    if total < 1e-12:
        return [0]
    mapped: list[int] = []
    n_samples = int(sample_count)
    for vertex_index in vertex_indices:
        idx = int(vertex_index) % len(pts)
        param = float(cum[idx]) / total
        mapped.append(int(round(param * n_samples)) % n_samples)
    return sorted(set(mapped))


def select_primary_boundary_loop(
    loops: Sequence[Sequence[int]],
    vertices: np.ndarray,
) -> list[int]:
    """选择弧长最长的边界环作为外轮廓。"""
    if not loops:
        raise RegionFitError("没有边界环可选")
    verts = _as_float_array(vertices)
    best_loop: list[int] | None = None
    best_length = -1.0
    for loop in loops:
        pts = verts[np.asarray(loop, dtype=np.int32)]
        closed = np.vstack((pts, pts[:1]))
        length = polyline_length(closed)
        if length > best_length:
            best_length = length
            best_loop = list(int(v) for v in loop)
    if best_loop is None:
        raise RegionFitError("无法选择主边界环")
    return best_loop


# 相对最长边界环周长的比例：更短的碎环视为离散极小值，不参与边拟合
_MIN_ISLAND_PERIMETER_FRAC = 0.12


def filter_significant_boundary_loops(
    loops: Sequence[Sequence[int]],
    vertices: np.ndarray,
    min_perimeter_frac: float = _MIN_ISLAND_PERIMETER_FRAC,
) -> list[list[int]]:
    """
    过滤离散极小边界环（碎岛 / 噪声环）。

    仅保留周长 ≥ 最长环 * min_perimeter_frac 的闭环，避免面内漂浮的极小拟合段。
    """
    if not loops:
        return []
    verts = _as_float_array(vertices)
    scored: list[tuple[float, list[int]]] = []
    for loop in loops:
        indices = [int(v) for v in loop]
        if len(indices) < 3:
            continue
        pts = verts[np.asarray(indices, dtype=np.int32)]
        closed = np.vstack((pts, pts[:1]))
        length = float(polyline_length(closed))
        if length <= 1e-12:
            continue
        scored.append((length, indices))
    if not scored:
        return []
    max_length = max(length for length, _ in scored)
    threshold = max_length * float(max(min_perimeter_frac, 0.0))
    kept = [loop for length, loop in scored if length >= threshold]
    if not kept:
        # 极端情况：全部被滤掉时至少保留最长环
        kept = [max(scored, key=lambda item: item[0])[1]]
    return kept


def _fit_circle_2d(points_2d: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Kasa 最小二乘圆拟合，返回 (圆心(2,), 半径)；退化时返回 None。"""
    pts = _as_float_array(points_2d)
    if len(pts) < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    matrix = np.column_stack((x, y, np.ones(len(pts), dtype=np.float64)))
    rhs = x * x + y * y
    try:
        coeffs, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    center = np.array(
        [coeffs[0] * 0.5, coeffs[1] * 0.5],
        dtype=np.float64,
    )
    radius_sq = float(coeffs[2] + center[0] ** 2 + center[1] ** 2)
    if radius_sq <= 0.0:
        return None
    return center, float(np.sqrt(radius_sq))


def _band_parameterization(
    points_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    为包络分箱选择参数化：返回 (param_u, param_v)。

    平直条带用 PCA 主轴坐标；弯曲成弧的条带（点集贴近圆环）
    改用绕拟合圆心的极角 + 半径，避免同一主轴坐标截到弧的两条臂
    导致上下包络在两臂之间跳变（拟合面扭曲交叉的根源）。
    """
    pts = _as_float_array(points_2d)
    param_u = pts[:, 0]
    param_v = pts[:, 1]

    circle = _fit_circle_2d(pts)
    if circle is None:
        return param_u, param_v
    center, _radius = circle
    radial = pts - center[None, :]
    radii = np.linalg.norm(radial, axis=1)
    r_mean = float(radii.mean())
    if r_mean < 1e-9:
        return param_u, param_v
    # 半径分布集中（环带特征）才启用极角参数化
    if float(radii.std()) / r_mean >= 0.25:
        return param_u, param_v

    angles = np.arctan2(radial[:, 1], radial[:, 0])
    # 以最大空缺角为分支切口，保证条带角度连续不跨越 ±π
    order = np.argsort(angles)
    sorted_angles = angles[order]
    gaps = np.diff(sorted_angles)
    wrap_gap = float(sorted_angles[0] + 2.0 * np.pi - sorted_angles[-1])
    if len(gaps) == 0 or wrap_gap >= float(gaps.max()):
        cut_angle = float(sorted_angles[-1]) + wrap_gap * 0.5
    else:
        widest = int(np.argmax(gaps))
        cut_angle = float(
            (sorted_angles[widest] + sorted_angles[widest + 1]) * 0.5
        )
    param_u = np.mod(angles - cut_angle, 2.0 * np.pi)
    return param_u, radii


def _polyline_self_cross_count_2d(points_2d: np.ndarray) -> int:
    """闭环折线在 2D 上的真自交段数（不含相邻边）。"""
    xy = _as_float_array(points_2d)
    count = len(xy)
    if count < 4:
        return 0
    crosses = 0

    def _orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _proper_cross(a, b, c, d) -> bool:
        o1 = _orient(a, b, c)
        o2 = _orient(a, b, d)
        o3 = _orient(c, d, a)
        o4 = _orient(c, d, b)
        return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

    for i in range(count):
        a = xy[i]
        b = xy[(i + 1) % count]
        for j in range(i + 2, count if i > 0 else count - 1):
            if i == 0 and j == count - 1:
                continue
            if (j + 1) % count == i:
                continue
            if _proper_cross(a, b, xy[j], xy[(j + 1) % count]):
                crosses += 1
    return crosses


def _hermite_fill_missing_bins(
    values: np.ndarray,
    valid: np.ndarray,
    first: int,
    last: int,
) -> np.ndarray:
    """用邻近端点切向补齐空分箱，使 island 之间沿边沿方向顺延。"""
    result = values.copy()
    valid_order = [int(v) for v in valid.tolist() if first <= int(v) <= last]
    if len(valid_order) < 2:
        return result

    for pos in range(len(valid_order) - 1):
        left = valid_order[pos]
        right = valid_order[pos + 1]
        if right <= left + 1:
            continue

        span = float(right - left)
        p0 = result[left].copy()
        p1 = result[right].copy()
        prev_index = valid_order[pos - 1] if pos > 0 else None
        next_index = valid_order[pos + 2] if pos + 2 < len(valid_order) else None
        if prev_index is not None:
            m0 = (p0 - result[prev_index]) * (span / max(left - prev_index, 1))
        else:
            m0 = p1 - p0
        if next_index is not None:
            m1 = (result[next_index] - p1) * (
                span / max(next_index - right, 1)
            )
        else:
            m1 = p1 - p0

        chord = float(np.linalg.norm(p1 - p0))
        if chord > 1e-9:
            for tangent in (m0, m1):
                length = float(np.linalg.norm(tangent))
                if length > chord * 2.5:
                    tangent *= (chord * 2.5) / length

        for index in range(left + 1, right):
            t = (index - left) / span
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            result[index] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    return result


def _band_envelope_from_samples(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    force_axis: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    由采样点构建条带外包络。

    返回 (envelope_3d, structural_corners_3d[4], envelope_2d, band_sides)。
    band_sides 为 [下链, 末端短边, 上链(反向), 首端短边]，两条长边共享
    同一 u 分箱下标，供 Coons 按索引重采样，避免弧长参数错位扭曲。
    """
    pts3 = _as_float_array(points_3d)
    pts2 = _as_float_array(points_2d)
    if force_axis:
        param_u = pts2[:, 0]
        param_v = pts2[:, 1]
    else:
        param_u, param_v = _band_parameterization(pts2)

    u_min = float(param_u.min())
    u_max = float(param_u.max())
    u_span = u_max - u_min
    if u_span < 1e-12:
        raise RegionFitError("多个 island 沿主方向跨度过小，无法合并")

    bin_count = int(np.clip(len(pts3) // 4, 48, _BOUNDARY_SAMPLES_MAX // 2))
    bin_ids = np.floor(
        (param_u - u_min) / u_span * (bin_count - 1)
    ).astype(np.int32)
    bin_ids = np.clip(bin_ids, 0, bin_count - 1)

    lower3 = np.full((bin_count, 3), np.nan, dtype=np.float64)
    upper3 = np.full((bin_count, 3), np.nan, dtype=np.float64)
    lower2 = np.full((bin_count, 2), np.nan, dtype=np.float64)
    upper2 = np.full((bin_count, 2), np.nan, dtype=np.float64)
    for bin_index in range(bin_count):
        members = np.flatnonzero(bin_ids == bin_index)
        if len(members) == 0:
            continue
        member_v = param_v[members]
        lo = int(members[int(np.argmin(member_v))])
        hi = int(members[int(np.argmax(member_v))])
        lower3[bin_index] = pts3[lo]
        upper3[bin_index] = pts3[hi]
        lower2[bin_index] = pts2[lo]
        upper2[bin_index] = pts2[hi]

    valid = np.flatnonzero(np.isfinite(lower3[:, 0]))
    if len(valid) < 2:
        raise RegionFitError("多个 island 无法形成连续包络")
    first = int(valid[0])
    last = int(valid[-1])
    lower3 = _hermite_fill_missing_bins(lower3, valid, first, last)
    upper3 = _hermite_fill_missing_bins(upper3, valid, first, last)
    lower2 = _hermite_fill_missing_bins(lower2, valid, first, last)
    upper2 = _hermite_fill_missing_bins(upper2, valid, first, last)

    lower_u = lower3[first : last + 1]
    upper_u = upper3[first : last + 1]  # 与 lower 共享同一 u 分箱
    lower_u2 = lower2[first : last + 1]
    upper_u2 = upper2[first : last + 1]

    # 仅修剪近乎零宽的端箱，避免压扁 Coons；阈值放宽以免裁掉角部延伸
    widths = np.linalg.norm(upper_u - lower_u, axis=1)
    positive = widths[widths > 1e-9]
    if len(positive) == 0:
        raise RegionFitError("多个 island 条带宽度过小，无法合并")
    width_min = max(float(np.median(positive)) * 0.05, 1e-6)
    usable = np.flatnonzero(widths >= width_min)
    if len(usable) < 2:
        usable = np.flatnonzero(widths > 1e-9)
    if len(usable) < 2:
        raise RegionFitError("多个 island 条带有效跨度过小，无法合并")
    lo = int(usable[0])
    hi = int(usable[-1])
    lower_u = lower_u[lo : hi + 1]
    upper_u = upper_u[lo : hi + 1]
    lower_u2 = lower_u2[lo : hi + 1]
    upper_u2 = upper_u2[lo : hi + 1]

    upper_chain = upper_u[::-1]
    envelope = np.vstack((lower_u, upper_chain))
    envelope_2d = np.vstack((lower_u2, upper_u2[::-1]))
    n_lower = len(lower_u)
    raw_corners = [0, n_lower - 1, n_lower, 2 * n_lower - 1]

    keep = np.ones(len(envelope), dtype=bool)
    if len(envelope) > 1:
        keep[1:] = (
            np.linalg.norm(np.diff(envelope, axis=0), axis=1) > 1e-10
        )
    for index in raw_corners:
        keep[index] = True
    envelope = envelope[keep]
    envelope_2d = envelope_2d[keep]
    old_to_new = np.cumsum(keep) - 1
    corner_indices = [int(old_to_new[i]) for i in raw_corners]
    count = len(envelope)
    unique: list[int] = []
    for index in corner_indices:
        candidate = int(index) % count
        for _ in range(count):
            if candidate not in unique:
                break
            candidate = (candidate + 1) % count
        unique.append(candidate)
    corners = envelope[np.asarray(unique, dtype=np.int32)]
    if len(envelope) < 4:
        raise RegionFitError("合并后的 island 包络点数不足")

    # 端帽：在端部窗口内取最靠外的上下极值点，再连线——
    # 比「当前分箱直线」更能盖住角部，又比折线端帽更不易扭曲。
    u_start = u_min + (first + lo) / max(bin_count - 1, 1) * u_span
    u_end = u_min + (first + hi) / max(bin_count - 1, 1) * u_span
    # 端部窗口加宽，优先吃进角部/island 断口上的采样
    window = max(u_span * 0.14, 1e-9)

    def _extreme_end_pair(u0: float, at_start: bool) -> tuple[np.ndarray, np.ndarray]:
        if at_start:
            mask = param_u <= u0 + window
        else:
            mask = param_u >= u0 - window
        members = np.flatnonzero(mask)
        if len(members) == 0:
            return lower_u[0 if at_start else -1], upper_u[0 if at_start else -1]
        # 取更靠端部的点，再在其中取 v 极值以延伸角部
        u_vals = param_u[members]
        if at_start:
            keep_u = members[u_vals <= float(np.percentile(u_vals, 35))]
        else:
            keep_u = members[u_vals >= float(np.percentile(u_vals, 65))]
        if len(keep_u) < 2:
            keep_u = members
        lo_i = int(keep_u[int(np.argmin(param_v[keep_u]))])
        hi_i = int(keep_u[int(np.argmax(param_v[keep_u]))])
        return pts3[lo_i], pts3[hi_i]

    start_lo, start_hi = _extreme_end_pair(u_start, True)
    end_lo, end_hi = _extreme_end_pair(u_end, False)
    cap_count = int(np.clip(n_lower // 8, 2, 16))
    t = np.linspace(0.0, 1.0, cap_count, dtype=np.float64)[:, None]
    start_cap = start_hi[None, :] + t * (start_lo - start_hi)
    end_cap = end_lo[None, :] + t * (end_hi - end_lo)
    lower_u = lower_u.copy()
    upper_u = upper_u.copy()
    lower_u[0] = start_lo
    upper_u[0] = start_hi
    lower_u[-1] = end_lo
    upper_u[-1] = end_hi
    upper_chain = upper_u[::-1]
    band_sides = [lower_u, end_cap, upper_chain, start_cap]
    return envelope, corners, envelope_2d, band_sides


def combine_boundary_islands(
    loops: Sequence[Sequence[int]],
    vertices: np.ndarray,
    interior_points: np.ndarray | None = None,
    prefer_outer_envelope: bool = False,
) -> tuple[np.ndarray, list[np.ndarray] | None, dict | None]:
    """
    将同一领域内互不连通的 island 边界合成为一个连续外包络。

    所有环先投影到共同 PCA 平面，按条带参数化（平直用主轴坐标、
    弯弧用极角）分箱提取两侧包络；island 之间无采样的区间按邻近
    边沿切向补齐，模拟边沿曲率顺延后相接。

    interior_points（通常为领域面心）会并入分箱极值，使包络向外
    延伸覆盖角部/凹口。

    prefer_outer_envelope=True 时：
      - 单环也走条带外包络（消去深凹口/内缝）
      - 条带不对称时仍返回外包络，不退回最长单环

    返回 (envelope, band_sides, band_samples)：
      - envelope: 有序 3D 闭环（不重复首点）
      - band_sides: 形成条带包络时为共享 u 参数的四边；否则为 None
      - band_samples: 条带包络时含 points/param_u/param_v；否则为 None
    """
    if not loops:
        raise RegionFitError("没有边界环可合并")
    verts = _as_float_array(vertices)
    if len(loops) == 1 and not prefer_outer_envelope:
        return verts[np.asarray(loops[0], dtype=np.int32)].copy(), None, None

    loop_points = [
        verts[np.asarray(loop, dtype=np.int32)]
        for loop in loops
        if len(loop) >= 3
    ]
    if not loop_points:
        raise RegionFitError("所有 island 边界均无效")

    all_points = np.vstack(loop_points)
    if interior_points is not None and len(interior_points) > 0:
        interior = _as_float_array(interior_points)
        if interior.ndim == 1:
            interior = interior.reshape(1, 3)
        all_points = np.vstack((all_points, interior))
    else:
        interior = None

    centroid = all_points.mean(axis=0)
    _, _, vh = np.linalg.svd(all_points - centroid, full_matrices=False)
    axis_u = _normalize(vh[0])
    axis_v = _normalize(vh[1])

    # 每个 island 独立按弧长加密，避免原扫描边密度差异造成包络空洞。
    samples_3d: list[np.ndarray] = []
    samples_2d: list[np.ndarray] = []
    for points in loop_points:
        sample_count = int(
            np.clip(len(points) * 2, 64, _BOUNDARY_SAMPLES_MAX)
        )
        sampled = resample_closed_polyline(points, sample_count)
        samples_3d.append(sampled)
        samples_2d.append(
            project_points_to_plane(sampled, centroid, axis_u, axis_v)
        )

    # 面心等内部点并入分箱，向外延伸覆盖角部
    if interior is not None and len(interior) > 0:
        if len(interior) > 8000:
            step = int(np.ceil(len(interior) / 8000))
            interior = interior[::step]
        samples_3d.append(interior)
        samples_2d.append(
            project_points_to_plane(interior, centroid, axis_u, axis_v)
        )

    points_3d = np.vstack(samples_3d)
    points_2d = np.vstack(samples_2d)

    force_axis = False
    envelope, _corners, envelope_2d, band_sides = _band_envelope_from_samples(
        points_3d, points_2d, force_axis=False
    )
    # 极角包络若自交，回退主轴分箱；仍自交则保留自交较少者
    if _polyline_self_cross_count_2d(envelope_2d) > 0:
        alt_env, _alt_c, alt_2d, alt_sides = _band_envelope_from_samples(
            points_3d, points_2d, force_axis=True
        )
        if _polyline_self_cross_count_2d(alt_2d) <= _polyline_self_cross_count_2d(
            envelope_2d
        ):
            envelope, band_sides = alt_env, alt_sides
            force_axis = True

    # 对边长弧长度相差过大，或 island 过于碎散：说明并非同一条带，
    # 强行并包络会产生扭曲；曲面拟合可回退最长单环。
    # 调试外轮廓必须保留包络，否则会「只剩一部分」。
    if (
        not prefer_outer_envelope
        and band_sides is not None
        and len(band_sides) == 4
    ):
        long_a = polyline_length(band_sides[0])
        long_b = polyline_length(band_sides[2])
        shorter = min(long_a, long_b)
        longer = max(long_a, long_b)
        too_asymmetric = shorter < 1e-9 or longer > 1.6 * shorter
        too_fragmented = len(loop_points) >= 10
        if too_asymmetric or too_fragmented:
            primary = max(
                loop_points,
                key=lambda pts: polyline_length(
                    np.vstack((pts, pts[:1]))
                ),
            )
            return primary.copy(), None, None

    if force_axis:
        param_u = points_2d[:, 0]
        param_v = points_2d[:, 1]
    else:
        param_u, param_v = _band_parameterization(points_2d)
    band_samples = {
        "points": points_3d,
        "param_u": param_u,
        "param_v": param_v,
    }
    return envelope, band_sides, band_samples


def loop_has_internal_corridor(
    points: np.ndarray,
    gap_frac: float = _LOOP_CORRIDOR_GAP_FRAC,
) -> bool:
    """
    判断闭环是否存在深凹口/内缝：非邻接边界点彼此贴近。

    此类环若直接按原边界拆边，会把内缝当成外轮廓的一部分画出来。
    """
    pts = _as_float_array(points)
    if len(pts) < 8:
        return False
    sample_n = int(np.clip(len(pts), 32, 96))
    rs = resample_closed_polyline(pts, sample_n)
    extent = float(np.linalg.norm(rs.max(axis=0) - rs.min(axis=0)))
    if extent < 1e-12:
        return False
    gap = max(extent * float(gap_frac), 1e-6)
    # 沿环至少隔开 1/6 周长，避免把相邻折角当成内缝
    min_sep = max(sample_n // 6, 3)
    max_sep = sample_n - min_sep
    for i in range(sample_n):
        for offset in range(min_sep, max_sep + 1):
            j = (i + offset) % sample_n
            if j <= i:
                continue
            if float(np.linalg.norm(rs[i] - rs[j])) <= gap:
                return True
    return False


def approx_closed_polyline_gap(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    """两条闭环折线之间的近似最小间隙。"""
    a = _as_float_array(points_a)
    b = _as_float_array(points_b)
    if len(a) < 3 or len(b) < 3:
        return float("inf")
    sa = resample_closed_polyline(a, int(np.clip(len(a), 16, 48)))
    sb = resample_closed_polyline(b, int(np.clip(len(b), 16, 48)))
    closed_a = np.vstack((sa, sa[:1]))
    closed_b = np.vstack((sb, sb[:1]))
    step_a = max(1, len(sa) // 24)
    step_b = max(1, len(sb) // 24)
    d_ab = min(
        point_to_polyline_distance(point, closed_b) for point in sa[::step_a]
    )
    d_ba = min(
        point_to_polyline_distance(point, closed_a) for point in sb[::step_b]
    )
    return float(min(d_ab, d_ba))


def cluster_nearby_boundary_loops(
    loops: Sequence[Sequence[int]],
    vertices: np.ndarray,
    max_gap: float,
) -> list[list[int]]:
    """按间隙把邻近边界环聚成簇（将融并为同一外轮廓）。"""
    verts = _as_float_array(vertices)
    count = len(loops)
    if count == 0:
        return []
    if count == 1:
        return [[0]]
    samples: list[np.ndarray | None] = []
    bboxes: list[tuple[np.ndarray, np.ndarray] | None] = []
    for loop in loops:
        pts = verts[np.asarray(loop, dtype=np.int32)]
        if len(pts) < 3:
            samples.append(None)
            bboxes.append(None)
            continue
        # 聚类只需稀疏采样，避免卡顿
        samples.append(resample_closed_polyline(pts, int(np.clip(len(pts), 12, 24))))
        bboxes.append((pts.min(axis=0), pts.max(axis=0)))

    parent = list(range(count))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    gap_limit = float(max(max_gap, 0.0))
    for i in range(count):
        if samples[i] is None or bboxes[i] is None:
            continue
        amin, amax = bboxes[i]
        for j in range(i + 1, count):
            if samples[j] is None or bboxes[j] is None:
                continue
            bmin, bmax = bboxes[j]
            # 包围盒分离过远则跳过精确间隙
            sep = np.maximum(amin - bmax, bmin - amax)
            sep = np.maximum(sep, 0.0)
            if float(np.linalg.norm(sep)) > gap_limit:
                continue
            if approx_closed_polyline_gap(samples[i], samples[j]) <= gap_limit:
                _union(i, j)

    clusters: dict[int, list[int]] = {}
    for index in range(count):
        if samples[index] is None:
            continue
        root = _find(index)
        clusters.setdefault(root, []).append(index)
    return [sorted(members) for members in clusters.values()]


def _segment_facing_score(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    proximity: float,
) -> float:
    """
    两条线段若近平行、投影重叠且垂直间隙小，返回间隙；否则 +inf。

    环绕向不一致时对边可能同向，故用 |ua·ub| 判定平行（不强制反向）。
    间隙用垂直于边方向的距离，避免采样相位错位时中点欧氏距离虚高。
    间距极近时放宽平行阈值，避免弯曲条带缝两侧微折导致漏消。
    """
    da = a1 - a0
    db = b1 - b0
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    if la < 1e-12 or lb < 1e-12:
        return float("inf")
    ua = da / la
    ub = db / lb
    mid_a = 0.5 * (a0 + a1)
    mid_b = 0.5 * (b0 + b1)
    delta = mid_b - mid_a
    # 沿边方向的错位不计入间隙
    perp = delta - ua * float(delta.dot(ua))
    gap = float(np.linalg.norm(perp))
    prox = float(proximity)
    if gap > prox * 1.35:
        return float("inf")
    abs_dot = abs(float(ua.dot(ub)))
    # 极近缝：允许更不平行；一般缝：需明显平行
    parallel_need = (
        _MULTI_ISLAND_PARALLEL_DOT * 0.55
        if gap <= prox * 0.55
        else _MULTI_ISLAND_PARALLEL_DOT * 0.85
    )
    if abs_dot < parallel_need:
        return float("inf")
    # 投影到 a 方向看重叠
    t0 = float((b0 - a0).dot(ua))
    t1 = float((b1 - a0).dot(ua))
    lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
    overlap = min(la, hi) - max(0.0, lo)
    overlap_need = _MULTI_ISLAND_OVERLAP_FRAC * (0.5 if gap <= prox * 0.55 else 1.0)
    if overlap < min(la, lb) * overlap_need:
        return float("inf")
    return gap


def _decimate_closed_loop_ids(loop: Sequence[int], max_verts: int) -> list[int]:
    """闭环节流：按步长抽稀，保留原始顶点索引（不造新点）。"""
    ids = [int(v) for v in loop]
    count = len(ids)
    if count <= max_verts:
        return ids
    step = int(np.ceil(count / float(max_verts)))
    kept = ids[::step]
    if kept[0] != ids[0]:
        kept.insert(0, ids[0])
    if len(kept) < 3:
        return ids[:max_verts]
    return kept


def outer_contour_by_dissolving_facing_edges(
    loops: Sequence[Sequence[int]],
    vertices: np.ndarray,
    proximity: float,
) -> np.ndarray | None:
    """
    消去邻近孤岛之间的对向内边，用原始 3D 顶点拼出真正外轮廓。

    为避免卡死：环顶点数设上限、空间哈希配对、有限次环追踪；
    桥接只连对边端点最近邻。失败返回 None，由调用方分岛处理。
    """
    verts = _as_float_array(vertices)
    raw_loops = [list(int(v) for v in loop) for loop in loops if len(loop) >= 3]
    if len(raw_loops) < 2:
        return None

    _MAX_LOOP_VERTS = 48
    _MAX_SEGMENTS = 480
    _MAX_TRACE_STARTS = 12

    max_verts = _MAX_LOOP_VERTS
    segments: list[tuple[int, np.ndarray, np.ndarray]] = []
    while True:
        sample_loops = [
            _decimate_closed_loop_ids(loop, max_verts) for loop in raw_loops
        ]
        segments = []
        for loop_index, ids in enumerate(sample_loops):
            count = len(ids)
            for index in range(count):
                segments.append(
                    (
                        loop_index,
                        verts[ids[index]].copy(),
                        verts[ids[(index + 1) % count]].copy(),
                    )
                )
        if len(segments) <= _MAX_SEGMENTS or max_verts <= 12:
            break
        max_verts = max(12, max_verts // 2)
    n_seg = len(segments)
    if n_seg > _MAX_SEGMENTS or n_seg < 3:
        return None

    prox = float(max(proximity, 1e-6))
    # 空间哈希：只与邻近桶内线段比对面
    cell = max(prox * 0.9, 1e-6)

    def _mid_key(point: np.ndarray) -> tuple[int, int, int]:
        return (
            int(np.floor(point[0] / cell)),
            int(np.floor(point[1] / cell)),
            int(np.floor(point[2] / cell)),
        )

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    mids = np.zeros((n_seg, 3), dtype=np.float64)
    for index, (_li, pa, pb) in enumerate(segments):
        mid = 0.5 * (pa + pb)
        mids[index] = mid
        buckets[_mid_key(mid)].append(index)

    # 按垂直间隙从小到大配对，每段最多取消一次
    pair_candidates: list[tuple[float, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for i in range(n_seg):
        li, pa, pb = segments[i]
        base = _mid_key(mids[i])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in buckets.get(
                        (base[0] + dx, base[1] + dy, base[2] + dz),
                        (),
                    ):
                        if j <= i:
                            continue
                        lj, pc, pd = segments[j]
                        if li == lj:
                            continue
                        key = (i, j)
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        score = _segment_facing_score(pa, pb, pc, pd, prox)
                        if score < float("inf"):
                            pair_candidates.append((score, i, j))

    pair_candidates.sort(key=lambda item: item[0])
    cancelled = np.zeros(n_seg, dtype=bool)
    bridges: list[tuple[np.ndarray, np.ndarray]] = []
    for _score, i, j in pair_candidates:
        if cancelled[i] or cancelled[j]:
            continue
        cancelled[i] = True
        cancelled[j] = True
        _li, pa, pb = segments[i]
        _lj, pc, pd = segments[j]
        # 每端点只连对边上最近端点，避免密布短桥成梯子
        for p in (pa, pb):
            q = pc if float(np.linalg.norm(p - pc)) <= float(
                np.linalg.norm(p - pd)
            ) else pd
            dist = float(np.linalg.norm(p - q))
            if 1e-9 < dist <= prox * 1.15:
                bridges.append((p.copy(), q.copy()))

    kept = [segments[i] for i in range(n_seg) if not cancelled[i]]
    if len(kept) < 3:
        return None

    # 仅焊接重合端点，禁止把岛间隙吸附成同一点
    weld = max(min(prox * 0.05, cell * 0.05), 1e-9)
    node_points: list[np.ndarray] = []
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def _weld_key(point: np.ndarray) -> tuple[int, int, int]:
        return (
            int(np.floor(point[0] / max(weld, 1e-12))),
            int(np.floor(point[1] / max(weld, 1e-12))),
            int(np.floor(point[2] / max(weld, 1e-12))),
        )

    def _register(point: np.ndarray) -> int:
        base = _weld_key(point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for index in grid.get(
                        (base[0] + dx, base[1] + dy, base[2] + dz),
                        (),
                    ):
                        if float(np.linalg.norm(node_points[index] - point)) <= weld:
                            return index
        index = len(node_points)
        node_points.append(point.copy())
        grid[base].append(index)
        return index

    edges: list[tuple[int, int]] = []
    for _li, pa, pb in kept:
        na = _register(pa)
        nb = _register(pb)
        if na != nb:
            edges.append((na, nb))
    for pa, pb in bridges:
        na = _register(pa)
        nb = _register(pb)
        if na != nb:
            edges.append((na, nb))

    if len(edges) < 3:
        return None

    outgoing: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        outgoing[a].append(b)
        outgoing[b].append(a)

    def _trace_from(start_a: int, start_b: int) -> list[int] | None:
        unused: dict[tuple[int, int], int] = defaultdict(int)
        for a, b in edges:
            unused[(a, b)] += 1
            unused[(b, a)] += 1
        if unused[(start_a, start_b)] <= 0:
            return None
        loop_nodes = [start_a]
        current = start_b
        prev = start_a
        unused[(start_a, start_b)] -= 1
        unused[(start_b, start_a)] -= 1
        max_steps = len(edges) + 2
        for _ in range(max_steps):
            loop_nodes.append(current)
            if current == start_a and len(loop_nodes) > 3:
                return loop_nodes[:-1]
            candidates = [
                nxt
                for nxt in outgoing.get(current, [])
                if nxt != prev and unused[(current, nxt)] > 0
            ]
            if not candidates:
                return None
            nxt = candidates[0]
            if len(candidates) > 1:
                inbound = node_points[current] - node_points[prev]
                in_norm = float(np.linalg.norm(inbound))
                best_dot = -1e30
                if in_norm > 1e-12:
                    inbound = inbound / in_norm
                for cand in candidates:
                    outbound = node_points[cand] - node_points[current]
                    out_norm = float(np.linalg.norm(outbound))
                    if out_norm < 1e-12:
                        continue
                    # 优先直行/外轮廓（转向最小），避免钻进短桥死胡同
                    dot = float(inbound.dot(outbound / out_norm))
                    if dot > best_dot:
                        best_dot = dot
                        nxt = cand
            unused[(current, nxt)] -= 1
            unused[(nxt, current)] -= 1
            prev, current = current, nxt
        return None

    # 有限起点：优先度数异常节点，再补若干边
    degree = {node: len(neigh) for node, neigh in outgoing.items()}
    start_edges: list[tuple[int, int]] = []
    for a, b in edges:
        if degree.get(a, 0) != 2 or degree.get(b, 0) != 2:
            start_edges.append((a, b))
    for a, b in edges:
        if (a, b) not in start_edges and (b, a) not in start_edges:
            start_edges.append((a, b))
        if len(start_edges) >= _MAX_TRACE_STARTS:
            break

    best_nodes: list[int] | None = None
    best_len = -1.0
    for start_a, start_b in start_edges[:_MAX_TRACE_STARTS]:
        nodes = _trace_from(start_a, start_b)
        if nodes is None or len(nodes) < 3:
            continue
        pts = np.asarray([node_points[i] for i in nodes], dtype=np.float64)
        length = float(
            np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1))
        )
        if length > best_len:
            best_len = length
            best_nodes = nodes

    if best_nodes is None:
        return None
    result = np.asarray([node_points[i] for i in best_nodes], dtype=np.float64)
    if len(result) < 3:
        return None
    origin, axis_u, axis_v = _loop_pca_basis(result)
    if _signed_area_2d(
        project_points_to_plane(result, origin, axis_u, axis_v)
    ) < 0.0:
        result = result[::-1]
    return result


_ULTRA_REFLEX_INTERIOR_DEG = 355.0


def collect_interior_angle_labels(
    points: np.ndarray,
    *,
    min_deviation_deg: float = 4.0,
    lift: float = 0.0,
    normal: np.ndarray | None = None,
) -> list[dict]:
    """
    为闭环折线各显著内角生成标注数据。

    points 与返回的 world_co / face_center 使用同一坐标空间
    （拟合流程里来自 extract_mesh_data，已是世界坐标）。
    返回元素含: text, angle_deg, hairpin_deg, world_co, face_center, normal。
    """
    pts = _as_float_array(points)
    if len(pts) < 3:
        return []
    if float(np.linalg.norm(pts[0] - pts[-1])) < 1e-9:
        pts = pts[:-1].copy()
    if len(pts) < 3:
        return []

    origin, axis_u, axis_v = _loop_pca_basis(pts)
    pts2 = project_points_to_plane(pts, origin, axis_u, axis_v)
    if _signed_area_2d(pts2) < 0.0:
        pts = pts[::-1].copy()
        pts2 = pts2[::-1].copy()

    if normal is None:
        normal_vec = _normalize(np.cross(axis_u, axis_v))
        if float(np.linalg.norm(normal_vec)) < 1e-12:
            normal_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal_vec = _normalize(_as_float_array(normal))
        if float(np.linalg.norm(normal_vec)) < 1e-12:
            normal_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    # 仅沿法线略抬，避免平面偏移把标签甩出边界
    lift_dist = float(max(lift, extent * 0.015, 1e-4))
    min_dev = float(max(min_deviation_deg, 0.0))

    labels: list[dict] = []
    count = len(pts)
    for index in range(count):
        local = _vertex_interior_angle_deg_2d(pts2, index)
        hairpin = _vertex_hairpin_interior_angle_deg_2d(pts2, index)
        angle = float(max(local, hairpin))
        if abs(angle - 180.0) < min_dev and abs(local - 180.0) < min_dev:
            continue

        face_center = pts[index].copy()
        world_co = face_center + normal_vec * lift_dist
        labels.append(
            {
                "text": f"{int(round(angle))}",
                "angle_deg": angle,
                "local_deg": float(local),
                "hairpin_deg": float(hairpin),
                "world_co": world_co,
                "face_center": face_center,
                "normal": normal_vec.copy(),
            }
        )
    return labels


def _vertex_interior_angle_deg_2d(pts2: np.ndarray, index: int) -> float:
    """逆时针闭环上顶点的内角（度）。"""
    count = len(pts2)
    if count < 3:
        return 180.0
    prev_p = pts2[(index - 1) % count]
    cur_p = pts2[index]
    next_p = pts2[(index + 1) % count]
    vin = cur_p - prev_p
    vout = next_p - cur_p
    in_len = float(np.linalg.norm(vin))
    out_len = float(np.linalg.norm(vout))
    if in_len < 1e-12 or out_len < 1e-12:
        return 360.0
    cross = float(vin[0] * vout[1] - vin[1] * vout[0])
    cos_angle = float(np.clip(vin.dot(vout) / (in_len * out_len), -1.0, 1.0))
    turn = float(np.degrees(np.arccos(cos_angle)))
    if cross >= 0.0:
        return 180.0 - turn
    return 180.0 + turn


def _skip_collinear_neighbor_2d(
    pts2: np.ndarray,
    tip_index: int,
    direction: int,
    collinear_tol_deg: float = 8.0,
) -> int:
    """沿环跳过近共线茎点，取能代表尖刺走向的邻点。"""
    count = len(pts2)
    tip = pts2[int(tip_index) % count]
    direction = 1 if direction >= 0 else -1
    cursor = (int(tip_index) + direction) % count
    max_steps = max(count // 3, 2)
    for _ in range(max_steps):
        nxt = (cursor + direction) % count
        if nxt == int(tip_index) % count:
            return cursor
        vin = pts2[cursor] - tip
        vout = pts2[nxt] - pts2[cursor]
        in_len = float(np.linalg.norm(vin))
        out_len = float(np.linalg.norm(vout))
        if in_len < 1e-12:
            cursor = nxt
            continue
        if out_len < 1e-12:
            return cursor
        cos_angle = float(
            np.clip(vin.dot(vout) / (in_len * out_len), -1.0, 1.0)
        )
        turn = float(np.degrees(np.arccos(cos_angle)))
        # 茎点近似直线前进；一旦明显转向则停在当前茎端
        if turn > float(collinear_tol_deg):
            return cursor
        cursor = nxt
    return cursor


def _vertex_hairpin_interior_angle_deg_2d(pts2: np.ndarray, index: int) -> float:
    """
    尖刺几何内角：跳过两侧近共线茎点后再量角。

    圆滑尖端的相邻折线角常只有 340°~354°，但跳过茎点后接近 360°。
    """
    count = len(pts2)
    if count < 3:
        return 180.0
    index = int(index) % count
    back = _skip_collinear_neighbor_2d(pts2, index, -1)
    fwd = _skip_collinear_neighbor_2d(pts2, index, 1)
    if back == index or fwd == index or back == fwd:
        return _vertex_interior_angle_deg_2d(pts2, index)
    tip = pts2[index]
    vin = tip - pts2[back]
    vout = pts2[fwd] - tip
    in_len = float(np.linalg.norm(vin))
    out_len = float(np.linalg.norm(vout))
    if in_len < 1e-12 or out_len < 1e-12:
        return _vertex_interior_angle_deg_2d(pts2, index)
    cross = float(vin[0] * vout[1] - vin[1] * vout[0])
    cos_angle = float(np.clip(vin.dot(vout) / (in_len * out_len), -1.0, 1.0))
    turn = float(np.degrees(np.arccos(cos_angle)))
    if cross >= 0.0:
        return 180.0 - turn
    return 180.0 + turn


def _is_spike_stem_interior(interior_deg: float, interior_threshold_deg: float) -> bool:
    """尖刺链上的顶点：超锐内角或近乎直线（~180°）。"""
    if interior_deg > interior_threshold_deg:
        return True
    return abs(interior_deg - 180.0) < 8.0


def _spike_junction_index_2d(
    pts2: np.ndarray,
    tip_index: int,
    direction: int,
    interior_threshold_deg: float,
) -> int:
    """沿尖刺走向寻找上一级分岔交点（T 形接头）。"""
    count = len(pts2)
    tip_index = int(tip_index) % count
    direction = 1 if direction >= 0 else -1
    cursor = (tip_index + direction) % count
    for _step in range(max(count // 2, 2)):
        if cursor == tip_index:
            break
        # 茎点用局部角；分岔用跳过共线后的几何角
        local = _vertex_interior_angle_deg_2d(pts2, cursor)
        hairpin = _vertex_hairpin_interior_angle_deg_2d(pts2, cursor)
        if _is_spike_stem_interior(local, interior_threshold_deg) or (
            hairpin > interior_threshold_deg
        ):
            cursor = (cursor + direction) % count
            continue
        return cursor
    return (tip_index + direction) % count


def collapse_ultra_reflex_spike_vertices_closed_loop(
    points: np.ndarray,
    interior_threshold_deg: float = _ULTRA_REFLEX_INTERIOR_DEG,
    max_passes: int = 32,
) -> np.ndarray:
    """
    内角超过阈值的尖刺：删去尖刺链，保留上一级交点（肩部）。

    适用于拟合边界上几乎 360° 的内角（细刺/毛刺），
    将尖点及其同侧细段收束到分岔处的交点。
    圆滑尖端用「跳过共线茎点」后的几何内角判定，避免局部折线角略低于阈值而漏收。
    """
    pts = _as_float_array(points).copy()
    if len(pts) < 4:
        return pts

    threshold = float(min(max(interior_threshold_deg, 180.0), 359.9))
    origin, axis_u, axis_v = _loop_pca_basis(pts)
    pts2 = project_points_to_plane(pts, origin, axis_u, axis_v)
    if _signed_area_2d(pts2) < 0.0:
        pts = pts[::-1].copy()
        pts2 = pts2[::-1].copy()

    for _ in range(int(max(max_passes, 1))):
        count = len(pts)
        if count < 4:
            break

        tip_index = -1
        best_interior = threshold
        for index in range(count):
            local = _vertex_interior_angle_deg_2d(pts2, index)
            hairpin = _vertex_hairpin_interior_angle_deg_2d(pts2, index)
            interior = max(local, hairpin)
            if interior > best_interior:
                best_interior = interior
                tip_index = index
        if tip_index < 0:
            break

        junction_back = _spike_junction_index_2d(
            pts2, tip_index, -1, threshold
        )
        junction_fwd = _spike_junction_index_2d(
            pts2, tip_index, 1, threshold
        )

        remove_set: set[int] = set()
        if junction_back == junction_fwd:
            remove_set.add(tip_index)
        else:
            cursor = (junction_back + 1) % count
            while cursor != junction_fwd:
                remove_set.add(cursor)
                cursor = (cursor + 1) % count
                if len(remove_set) >= count - 3:
                    break
            if not remove_set:
                remove_set.add(tip_index)

        keep = [i for i in range(count) if i not in remove_set]
        if len(keep) < 3:
            break
        if len(keep) == count:
            # 未删点则强制去掉尖点，避免死循环
            keep = [i for i in range(count) if i != tip_index]
            if len(keep) < 3:
                break
        pts = pts[np.asarray(keep, dtype=np.int32)].copy()
        pts2 = project_points_to_plane(pts, origin, axis_u, axis_v)

    if _signed_area_2d(pts2) < 0.0:
        pts = pts[::-1].copy()
    return pts


def bridge_reentrant_corners_closed_loop(
    points: np.ndarray,
    angle_threshold_deg: float = _CONCAVE_FOLD_ANGLE_DEG,
    max_span_frac: float = 0.22,
    max_passes: int = 12,
) -> np.ndarray:
    """
    缝合后外轮廓上的内阴角：用弦切除凹口，使相邻孤岛接缝成连续外缘。

    仅处理逆时针环上的凹折（叉积为负）；弦长超过包围盒对角线比例则跳过，
    避免把整条内弧裁掉。远岛弦线桥接不在此处理。先收束内角>355°的尖刺。
    """
    pts = collapse_ultra_reflex_spike_vertices_closed_loop(points)
    pts = _as_float_array(pts).copy()
    if len(pts) < 4:
        return pts

    origin, axis_u, axis_v = _loop_pca_basis(pts)
    pts2 = project_points_to_plane(pts, origin, axis_u, axis_v)
    if _signed_area_2d(pts2) < 0.0:
        pts = pts[::-1].copy()
        pts2 = pts2[::-1].copy()

    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    max_span = max(extent * float(max_span_frac), 1e-6)
    threshold = float(max(angle_threshold_deg, 1.0))

    def _local_concave_turns() -> list[tuple[float, int]]:
        """稀疏折线也能检出的局部凹折：(转角, 下标)。"""
        count = len(pts2)
        scored: list[tuple[float, int]] = []
        for index in range(count):
            prev_p = pts2[(index - 1) % count]
            cur_p = pts2[index]
            next_p = pts2[(index + 1) % count]
            vin = cur_p - prev_p
            vout = next_p - cur_p
            in_len = float(np.linalg.norm(vin))
            out_len = float(np.linalg.norm(vout))
            if in_len < 1e-12 or out_len < 1e-12:
                continue
            cross = float(vin[0] * vout[1] - vin[1] * vout[0])
            if cross >= 0.0:
                continue
            cos_angle = float(
                np.clip(vin.dot(vout) / (in_len * out_len), -1.0, 1.0)
            )
            turn = float(np.degrees(np.arccos(cos_angle)))
            if turn >= threshold:
                scored.append((turn, index))
        scored.sort(reverse=True)
        return scored

    for _ in range(int(max(max_passes, 1))):
        scored = _local_concave_turns()
        if not scored:
            break
        _best_turn, best_index = scored[0]
        count = len(pts2)

        # 向两侧扩展到转角变缓处
        i0 = best_index
        i1 = best_index
        for _step in range(max(count // 8, 3)):
            left = (i0 - 1) % count
            right = (i1 + 1) % count
            expanded = False
            for candidate, toward_left in ((left, True), (right, False)):
                prev_p = pts2[(candidate - 1) % count]
                cur_p = pts2[candidate]
                next_p = pts2[(candidate + 1) % count]
                vin = cur_p - prev_p
                vout = next_p - cur_p
                in_len = float(np.linalg.norm(vin))
                out_len = float(np.linalg.norm(vout))
                if in_len < 1e-12 or out_len < 1e-12:
                    continue
                cross = float(vin[0] * vout[1] - vin[1] * vout[0])
                if cross >= 0.0:
                    continue
                cos_angle = float(
                    np.clip(vin.dot(vout) / (in_len * out_len), -1.0, 1.0)
                )
                turn = float(np.degrees(np.arccos(cos_angle)))
                if turn >= threshold * 0.5:
                    if toward_left:
                        i0 = left
                    else:
                        i1 = right
                    expanded = True
            span = (i1 - i0) % count
            if not expanded or span <= 1 or span >= count - 3:
                break

        # 锚点取凹折两侧邻点（肩部），切除中间尖底
        a_idx = (best_index - 1) % count
        b_idx = (best_index + 1) % count
        # 若扩展后有更宽凹口，用扩展端点
        span = (i1 - i0) % count
        if span >= 2 and span < count - 2:
            a_idx = i0
            b_idx = i1

        chord_len = float(np.linalg.norm(pts2[b_idx] - pts2[a_idx]))
        if chord_len < 1e-9 or chord_len > max_span:
            # 无法弦切：至少丢掉最尖点，避免死循环
            keep = [i for i in range(count) if i != best_index]
            if len(keep) < 4:
                break
            pts = pts[np.asarray(keep, dtype=np.int32)].copy()
            pts2 = pts2[np.asarray(keep, dtype=np.int32)].copy()
            continue

        chain_idx = []
        cursor = a_idx
        for _ in range(count):
            chain_idx.append(cursor)
            if cursor == b_idx:
                break
            cursor = (cursor + 1) % count
        if len(chain_idx) < 3:
            break

        chord = pts2[b_idx] - pts2[a_idx]
        left_n = np.array([-chord[1], chord[0]], dtype=np.float64)
        left_n /= max(float(np.linalg.norm(left_n)), 1e-12)
        segment = pts2[np.asarray(chain_idx, dtype=np.int32)]
        deviation = float(np.mean((segment - pts2[a_idx]) @ left_n))
        if deviation <= chord_len * 0.008:
            # 外凸：丢掉当前尖点再试其余
            keep = [i for i in range(count) if i != best_index]
            if len(keep) < 4:
                break
            pts = pts[np.asarray(keep, dtype=np.int32)].copy()
            pts2 = pts2[np.asarray(keep, dtype=np.int32)].copy()
            continue

        drop_set = set(chain_idx[1:-1])
        keep = [i for i in range(count) if i not in drop_set]
        if len(keep) < 4:
            break
        pts = pts[np.asarray(keep, dtype=np.int32)].copy()
        pts2 = pts2[np.asarray(keep, dtype=np.int32)].copy()

    if _signed_area_2d(pts2) < 0.0:
        pts = pts[::-1].copy()
    return pts


def _loop_corner_indices_3d(
    points: np.ndarray,
    angle_threshold_deg: float,
) -> list[int]:
    """在闭环上检测折角下标（含直角）；短环用局部转角，长环用多尺度检测。"""
    pts = _as_float_array(points)
    count = len(pts)
    if count < 3:
        return []
    origin, axis_u, axis_v = _loop_pca_basis(pts)
    pts2 = project_points_to_plane(pts, origin, axis_u, axis_v)
    threshold = float(max(angle_threshold_deg, 1.0))
    if count <= 32:
        corners: list[int] = []
        for index in range(count):
            prev_p = pts2[(index - 1) % count]
            cur_p = pts2[index]
            next_p = pts2[(index + 1) % count]
            vin = cur_p - prev_p
            vout = next_p - cur_p
            in_len = float(np.linalg.norm(vin))
            out_len = float(np.linalg.norm(vout))
            if in_len < 1e-12 or out_len < 1e-12:
                continue
            cos_angle = float(
                np.clip(vin.dot(vout) / (in_len * out_len), -1.0, 1.0)
            )
            turn = float(np.degrees(np.arccos(cos_angle)))
            if turn >= threshold:
                corners.append(index)
        if len(corners) >= 2:
            return corners
        step = max(count // 4, 1)
        return sorted({(i * step) % count for i in range(4)})

    corners = detect_corner_indices(
        pts2,
        threshold,
        max_corners=_MAX_SPLIT_CORNERS,
        convex_only=False,
        include_strong_concave=True,
    )
    return list(corners) if corners else [0, count // 2]


def _closed_arc_indices(count: int, start: int, end: int, forward: bool) -> list[int]:
    """闭环从 start 到 end 的下标链（含两端）。"""
    if count <= 0:
        return []
    start = int(start) % count
    end = int(end) % count
    indices = [start]
    if start == end:
        return indices
    step = 1 if forward else -1
    cursor = start
    for _ in range(count):
        cursor = (cursor + step) % count
        indices.append(cursor)
        if cursor == end:
            break
    return indices


def _mean_polyline_gap(a: np.ndarray, b: np.ndarray) -> float:
    """两条开链之间的平均间隙（双向采样）。"""
    pa = _as_float_array(a)
    pb = _as_float_array(b)
    if len(pa) < 2 or len(pb) < 2:
        return float("inf")
    closed_b = np.vstack((pb, pb[:1]))
    closed_a = np.vstack((pa, pa[:1]))
    step_a = max(1, len(pa) // 16)
    step_b = max(1, len(pb) // 16)
    d_ab = [
        point_to_polyline_distance(point, closed_b) for point in pa[::step_a]
    ]
    d_ba = [
        point_to_polyline_distance(point, closed_a) for point in pb[::step_b]
    ]
    return float(0.5 * (float(np.mean(d_ab)) + float(np.mean(d_ba))))


def stitch_two_loops_by_corner_seams(
    points_a: np.ndarray,
    points_b: np.ndarray,
    max_gap: float,
    corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
) -> tuple[np.ndarray, list[np.ndarray], float] | None:
    """
    用邻近折角定义缝合曲线并合并两岛外轮廓。

    1) 检测各岛折角
    2) 配对邻近折角，连线距离取平均
    3) 均值 <= max_gap 时，折角之间相向边链视为缝合曲线并删除
    4) 剩余边链在折角处相接 → 外轮廓
    """
    pts_a = _as_float_array(points_a)
    pts_b = _as_float_array(points_b)
    if len(pts_a) < 3 or len(pts_b) < 3:
        return None

    corners_a = _loop_corner_indices_3d(pts_a, corner_angle_deg)
    corners_b = _loop_corner_indices_3d(pts_b, corner_angle_deg)
    if len(corners_a) < 2 or len(corners_b) < 2:
        return None

    prox = float(max(max_gap, 1e-6))
    pair_candidates: list[tuple[float, int, int]] = []
    for ia in corners_a:
        for ib in corners_b:
            dist = float(np.linalg.norm(pts_a[ia] - pts_b[ib]))
            if dist <= prox * 1.35:
                pair_candidates.append((dist, int(ia), int(ib)))
    if len(pair_candidates) < 2:
        return None
    pair_candidates.sort(key=lambda item: item[0])

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for dist, ia, ib in pair_candidates:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        matched.append((ia, ib, dist))
    if len(matched) < 2:
        return None

    avg_corner_gap = float(np.mean([item[2] for item in matched]))
    if avg_corner_gap > prox:
        return None

    n_a = len(pts_a)
    n_b = len(pts_b)
    ordered = sorted(matched, key=lambda item: item[0])

    best = None
    best_score = float("inf")
    for index in range(len(ordered)):
        ia0, ib0, _d0 = ordered[index]
        ia1, ib1, _d1 = ordered[(index + 1) % len(ordered)]
        if ia0 == ia1:
            continue
        for fwd_a in (True, False):
            arc_a_idx = _closed_arc_indices(n_a, ia0, ia1, fwd_a)
            if len(arc_a_idx) < 2 or len(arc_a_idx) >= n_a:
                continue
            arc_a = pts_a[np.asarray(arc_a_idx, dtype=np.int32)]
            for fwd_b in (True, False):
                arc_b_idx = _closed_arc_indices(n_b, ib0, ib1, fwd_b)
                if len(arc_b_idx) < 2 or len(arc_b_idx) >= n_b:
                    continue
                arc_b = pts_b[np.asarray(arc_b_idx, dtype=np.int32)]
                mean_gap = _mean_polyline_gap(arc_a, arc_b)
                if mean_gap > prox * 1.25:
                    continue
                score = mean_gap + 0.01 * (
                    polyline_length(arc_a) + polyline_length(arc_b)
                )
                if score < best_score:
                    best_score = score
                    best = (
                        ia0,
                        ia1,
                        ib0,
                        ib1,
                        fwd_a,
                        fwd_b,
                        arc_a_idx,
                        arc_b_idx,
                        mean_gap,
                    )

    if best is None:
        return None

    ia0, ia1, ib0, ib1, fwd_a, fwd_b, arc_a_idx, arc_b_idx, mean_gap = best
    # 外轮廓 = 去掉缝合弧后的互补弧：沿同一绕向从缝合终点走回起点
    outer_a_idx = _closed_arc_indices(n_a, ia1, ia0, forward=fwd_a)
    # B 侧从配对起点 ib0 走到 ib1 的非缝合弧（与缝合方向相反）
    outer_b_idx = _closed_arc_indices(n_b, ib0, ib1, forward=not fwd_b)
    # 外弧至少要有一个中间点（三点含端点），且不能与整环重合于「未删缝」
    if len(outer_a_idx) < 3 or len(outer_b_idx) < 3:
        return None
    if len(arc_a_idx) >= n_a or len(arc_b_idx) >= n_b:
        return None
    if len(arc_a_idx) < 2 or len(arc_b_idx) < 2:
        return None

    outer_a = pts_a[np.asarray(outer_a_idx, dtype=np.int32)]
    outer_b = pts_b[np.asarray(outer_b_idx, dtype=np.int32)]
    # 端点对齐：outer_a 终点应靠近 outer_b 起点
    if float(np.linalg.norm(outer_a[-1] - outer_b[0])) > float(
        np.linalg.norm(outer_a[-1] - outer_b[-1])
    ):
        outer_b = outer_b[::-1].copy()

    parts = [outer_a]
    if float(np.linalg.norm(outer_a[-1] - outer_b[0])) > 1e-9:
        parts.append(np.asarray([outer_a[-1], outer_b[0]], dtype=np.float64))
    parts.append(outer_b[1:] if len(outer_b) > 1 else outer_b)
    if float(np.linalg.norm(parts[-1][-1] - outer_a[0])) > 1e-9:
        parts.append(
            np.asarray([parts[-1][-1], outer_a[0]], dtype=np.float64)
        )
    outer = np.vstack(parts)
    if len(outer) >= 2 and float(np.linalg.norm(outer[0] - outer[-1])) < 1e-9:
        outer = outer[:-1]
    if len(outer) < 3:
        return None

    stitch_seams = [
        pts_a[np.asarray(arc_a_idx, dtype=np.int32)].copy(),
        pts_b[np.asarray(arc_b_idx, dtype=np.int32)].copy(),
    ]
    outer = bridge_reentrant_corners_closed_loop(outer)
    return outer, stitch_seams, float(mean_gap)


def _outer_contour_for_nearby_cluster(
    cluster_loops: Sequence[Sequence[int]],
    vertices: np.ndarray,
    max_gap: float,
    allow_band_fallback: bool = False,
    corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """
    邻近簇外轮廓：优先折角缝合曲线；失败再消对向内边。

    返回 (envelope, stitch_seams)。
    """
    verts = _as_float_array(vertices)
    loops = [list(int(v) for v in loop) for loop in cluster_loops if len(loop) >= 3]
    empty: list[np.ndarray] = []
    if len(loops) < 2:
        return None, empty

    working: list[np.ndarray] = [
        verts[np.asarray(loop, dtype=np.int32)].copy() for loop in loops
    ]
    all_seams: list[np.ndarray] = []

    def _finalize(envelope: np.ndarray | None) -> np.ndarray | None:
        if envelope is None or len(envelope) < 3:
            return None
        return bridge_reentrant_corners_closed_loop(_as_float_array(envelope))

    max_rounds = max(len(working) - 1, 0)
    for _ in range(max_rounds):
        if len(working) < 2:
            break
        best_pair: tuple[int, int] | None = None
        best_gap = float("inf")
        for i in range(len(working)):
            for j in range(i + 1, len(working)):
                gap = approx_closed_polyline_gap(working[i], working[j])
                if gap < best_gap:
                    best_gap = gap
                    best_pair = (i, j)
        if best_pair is None or best_gap > float(max_gap) * 1.35:
            break
        i, j = best_pair
        corner_hit = stitch_two_loops_by_corner_seams(
            working[i],
            working[j],
            max_gap=float(max_gap),
            corner_angle_deg=corner_angle_deg,
        )
        if corner_hit is not None:
            envelope, seams, _avg = corner_hit
            all_seams.extend(seams)
            working = [
                pts
                for index, pts in enumerate(working)
                if index not in best_pair
            ] + [envelope.copy()]
            continue

        base = len(verts)
        merged_verts = np.vstack((verts, working[i], working[j]))
        n_a = len(working[i])
        n_b = len(working[j])
        pair_loops = [
            list(range(base, base + n_a)),
            list(range(base + n_a, base + n_a + n_b)),
        ]
        pair_env = outer_contour_by_dissolving_facing_edges(
            pair_loops,
            merged_verts,
            proximity=float(max_gap),
        )
        pair_env = _finalize(pair_env)
        if pair_env is None or len(pair_env) < 3:
            break
        working = [
            pts
            for index, pts in enumerate(working)
            if index not in best_pair
        ] + [pair_env.copy()]
        verts = merged_verts

    if len(working) == 1 and len(working[0]) >= 3:
        return _finalize(working[0]), all_seams

    if len(loops) == 2:
        direct = stitch_two_loops_by_corner_seams(
            verts[np.asarray(loops[0], dtype=np.int32)],
            verts[np.asarray(loops[1], dtype=np.int32)],
            max_gap=float(max_gap),
            corner_angle_deg=corner_angle_deg,
        )
        if direct is not None:
            envelope, seams, _avg = direct
            return _finalize(envelope), seams

    envelope = outer_contour_by_dissolving_facing_edges(
        loops,
        verts,
        proximity=float(max_gap),
    )
    finalized = _finalize(envelope)
    if finalized is not None:
        return finalized, all_seams

    if not allow_band_fallback:
        return None, all_seams

    try:
        envelope, _, _ = combine_boundary_islands(
            loops,
            vertices,
            prefer_outer_envelope=True,
        )
    except RegionFitError:
        return None, all_seams
    return _finalize(envelope), all_seams


def merge_nearby_loops_to_outer_contours(
    loops: Sequence[Sequence[int]],
    vertices: np.ndarray,
    max_gap: float,
    corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
) -> list[dict]:
    """
    邻近孤岛融并为新孤岛（折角缝合曲线优先）；远岛保持独立。

    返回工作项: points / loop_ids / merged_from / stitch_seams。
    """
    verts = _as_float_array(vertices)
    if not loops:
        return []
    clusters = cluster_nearby_boundary_loops(loops, verts, max_gap=max_gap)
    items: list[dict] = []
    for cluster in clusters:
        cluster_loops = [list(int(v) for v in loops[index]) for index in cluster]
        if len(cluster_loops) == 1:
            ids = cluster_loops[0]
            items.append(
                {
                    "points": verts[np.asarray(ids, dtype=np.int32)].copy(),
                    "loop_ids": ids,
                    "merged_from": 1,
                    "stitch_seams": [],
                }
            )
            continue
        envelope, seams = _outer_contour_for_nearby_cluster(
            cluster_loops,
            verts,
            max_gap=float(max_gap),
            allow_band_fallback=False,
            corner_angle_deg=corner_angle_deg,
        )
        if envelope is None or len(envelope) < 3:
            for ids in cluster_loops:
                items.append(
                    {
                        "points": verts[np.asarray(ids, dtype=np.int32)].copy(),
                        "loop_ids": ids,
                        "merged_from": 1,
                        "stitch_seams": [],
                    }
                )
            continue
        items.append(
            {
                "points": _as_float_array(envelope).copy(),
                "loop_ids": None,
                "merged_from": int(len(cluster_loops)),
                "stitch_seams": seams,
            }
        )
    return items


def outer_contour_from_boundary_loops(
    loops: Sequence[Sequence[int]],
    vertices: np.ndarray,
    bridge_gap_frac: float = 0.05,
    grid_size: int = 192,
) -> np.ndarray:
    """
    多岛外轮廓：优先消对向内边；失败再退回分岛首环（不造弦线桥）。
    """
    _ = bridge_gap_frac, grid_size
    verts = _as_float_array(vertices)
    valid = [list(int(v) for v in loop) for loop in loops if len(loop) >= 3]
    if not valid:
        raise RegionFitError("没有可提取外轮廓的边界环")
    if len(valid) == 1:
        return verts[np.asarray(valid[0], dtype=np.int32)].copy()

    all_pts = np.vstack([verts[np.asarray(loop, dtype=np.int32)] for loop in valid])
    extent = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))
    proximity = max(extent * _MULTI_ISLAND_MERGE_GAP_FRAC, 1e-4)
    dissolved = outer_contour_by_dissolving_facing_edges(
        valid,
        verts,
        proximity=proximity,
    )
    if dissolved is not None and len(dissolved) >= 3:
        return dissolved
    # 不桥接远岛：返回最长单环网格轮廓
    primary = max(
        valid,
        key=lambda loop: polyline_length(
            np.vstack(
                (
                    verts[np.asarray(loop, dtype=np.int32)],
                    verts[np.asarray(loop[0], dtype=np.int32)],
                )
            )
        ),
    )
    return verts[np.asarray(primary, dtype=np.int32)].copy()


def _label_face_components(
    face_indices: np.ndarray,
    adjacency_offsets: np.ndarray,
    adjacency_indices: np.ndarray,
) -> dict[int, int]:
    """将面集合按共享边邻接划分为连通块，返回 face -> component_id。"""
    faces = [int(f) for f in face_indices]
    index = {f: i for i, f in enumerate(faces)}
    seen = np.zeros(len(faces), dtype=bool)
    comp_of: dict[int, int] = {}
    comp_id = 0
    off = adjacency_offsets
    adj = adjacency_indices
    for i, start in enumerate(faces):
        if seen[i]:
            continue
        stack = [start]
        seen[i] = True
        while stack:
            cur = stack.pop()
            comp_of[cur] = comp_id
            a = int(off[cur])
            b = int(off[cur + 1])
            for n in adj[a:b]:
                n = int(n)
                j = index.get(n)
                if j is None or seen[j]:
                    continue
                seen[j] = True
                stack.append(n)
        comp_id += 1
    return comp_of


def select_bridgeable_region_mask(
    region_ids: np.ndarray,
    target_id,
    adjacency_offsets: np.ndarray,
    adjacency_indices: np.ndarray,
    corridor_hops: int = 80,
) -> tuple[np.ndarray, int, int]:
    """
    选取同编号下「可通过 -1 碎面互相到达」的最大 island 簇。

    无法桥接的远岛若强行并包络，延长段会变成空间弦线、视觉上不相接。
    返回 (face_mask, kept_component_count, dropped_component_count)。
    """
    ids = _as_int_array(region_ids)
    targets = _normalize_target_ids(target_id)
    mask = np.isin(ids, np.asarray(targets, dtype=np.int32))
    if not np.any(mask):
        return mask, 0, 0

    off = _as_int_array(adjacency_offsets)
    adj = _as_int_array(adjacency_indices)
    target_faces = np.flatnonzero(mask)
    comp_of = _label_face_components(target_faces, off, adj)
    if not comp_of:
        return mask, 0, 0

    faces_by: dict[int, list[int]] = {}
    for face, cid in comp_of.items():
        faces_by.setdefault(cid, []).append(face)
    comp_ids = list(faces_by.keys())
    if len(comp_ids) == 1:
        return mask, 1, 0

    parent = {cid: cid for cid in comp_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hop_limit = max(int(corridor_hops), 1)
    for i, ca in enumerate(comp_ids):
        for cb in comp_ids[i + 1 :]:
            seeds = faces_by[ca]
            goal = set(faces_by[cb])
            queue: list[tuple[int, int]] = [(s, 0) for s in seeds]
            seen = set(seeds)
            head = 0
            linked = False
            while head < len(queue):
                face, hops = queue[head]
                head += 1
                if face in goal and hops > 0:
                    linked = True
                    break
                if hops >= hop_limit:
                    continue
                a = int(off[face])
                b = int(off[face + 1])
                for n in adj[a:b]:
                    n = int(n)
                    if n in seen:
                        continue
                    nid = int(ids[n])
                    if nid < 0 or bool(mask[n]):
                        seen.add(n)
                        queue.append((n, hops + 1))
            if linked:
                unite(ca, cb)

    groups: dict[int, list[int]] = {}
    for cid in comp_ids:
        groups.setdefault(find(cid), []).append(cid)

    def group_face_count(members: list[int]) -> int:
        return sum(len(faces_by[c]) for c in members)

    best_root = max(groups.keys(), key=lambda r: group_face_count(groups[r]))
    keep_comps = set(groups[best_root])
    keep_faces = []
    for cid in keep_comps:
        keep_faces.extend(faces_by[cid])
    kept_mask = np.zeros(len(ids), dtype=bool)
    kept_mask[np.asarray(keep_faces, dtype=np.int32)] = True
    dropped = len(comp_ids) - len(keep_comps)
    return kept_mask, len(keep_comps), dropped


def collect_island_bridge_interiors(
    region_ids: np.ndarray,
    target_id,
    face_centers: np.ndarray,
    adjacency_offsets: np.ndarray | None = None,
    adjacency_indices: np.ndarray | None = None,
    max_hops: int = 12,
    corridor_hops: int = 80,
) -> np.ndarray:
    """
    收集用于 island 延伸的内部采样点：目标领域面心 + 桥接碎面心。

    两类 -1 面会并入包络，使孤岛之间能靠「延长段」相接：
      1) 近邻层：从目标面有限跳数 BFS（角部/薄断口）
      2) 走廊层：可被两个及以上同编号 island 共同到达的 -1 面
         （跨岛空隙主路径，跳数上限更大）
    """
    ids = _as_int_array(region_ids)
    targets = _normalize_target_ids(target_id)
    mask = np.isin(ids, np.asarray(targets, dtype=np.int32))
    centers = _as_float_array(face_centers)
    parts = [centers[mask]]
    hop_limit = max(int(max_hops), 0)
    corridor_limit = max(int(corridor_hops), hop_limit)

    if (
        hop_limit > 0
        and adjacency_offsets is not None
        and adjacency_indices is not None
        and len(adjacency_offsets) > 1
        and np.any(mask)
    ):
        off = _as_int_array(adjacency_offsets)
        adj = _as_int_array(adjacency_indices)
        target_faces = np.flatnonzero(mask)
        comp_of = _label_face_components(target_faces, off, adj)
        n_comp = (max(comp_of.values()) + 1) if comp_of else 0

        # 近邻层：任意目标出发、跳数受限
        near_ids: set[int] = set()
        best_hop: dict[int, int] = {}
        queue: list[tuple[int, int]] = [(int(f), 0) for f in target_faces]
        head = 0
        while head < len(queue):
            f, hops = queue[head]
            head += 1
            if hops >= hop_limit:
                continue
            a = int(off[f])
            b = int(off[f + 1])
            for n in adj[a:b]:
                n = int(n)
                if int(ids[n]) >= 0:
                    continue
                next_hops = hops + 1
                prev = best_hop.get(n)
                if prev is not None and prev <= next_hops:
                    continue
                best_hop[n] = next_hops
                near_ids.add(n)
                queue.append((n, next_hops))

        # 走廊层：只从「贴着 -1」的目标边界面出发，记录 -1 被哪些
        # island 分量触及；被 ≥2 个分量触及的即为跨岛延长走廊。
        reach: dict[int, set[int]] = {}
        if n_comp >= 2:
            seeds_by_comp: dict[int, list[int]] = {}
            for seed, cid in comp_of.items():
                a = int(off[seed])
                b = int(off[seed + 1])
                if any(int(ids[int(n)]) < 0 for n in adj[a:b]):
                    seeds_by_comp.setdefault(cid, []).append(seed)

            for cid, seeds in seeds_by_comp.items():
                q2: list[tuple[int, int]] = [(s, 0) for s in seeds]
                seen_neg: dict[int, int] = {}
                h = 0
                while h < len(q2):
                    f, hops = q2[h]
                    h += 1
                    if hops >= corridor_limit:
                        continue
                    a = int(off[f])
                    b = int(off[f + 1])
                    for n in adj[a:b]:
                        n = int(n)
                        if int(ids[n]) >= 0:
                            continue
                        next_hops = hops + 1
                        prev = seen_neg.get(n)
                        if prev is not None and prev <= next_hops:
                            continue
                        seen_neg[n] = next_hops
                        bucket = reach.setdefault(n, set())
                        bucket.add(cid)
                        q2.append((n, next_hops))

        corridor_ids = {
            face for face, comps in reach.items() if len(comps) >= 2
        }
        bridge_ids = near_ids | corridor_ids
        if bridge_ids:
            parts.append(
                centers[np.asarray(sorted(bridge_ids), dtype=np.int32)]
            )

    return np.vstack(parts) if parts else np.empty((0, 3), dtype=np.float64)


def resample_polyline_by_index(points: np.ndarray, count: int) -> np.ndarray:
    """按下标均匀重采样（非弧长）。用于共享参数域的条带对边。"""
    pts = _as_float_array(points)
    target = max(int(count), 2)
    if len(pts) == 0:
        raise RegionFitError("空折线无法重采样")
    if len(pts) == 1:
        return np.repeat(pts, target, axis=0)
    positions = np.linspace(0.0, len(pts) - 1, target, dtype=np.float64)
    result = np.empty((target, pts.shape[1]), dtype=np.float64)
    base = np.arange(len(pts), dtype=np.float64)
    for axis in range(pts.shape[1]):
        result[:, axis] = np.interp(positions, base, pts[:, axis])
    return result


def compute_region_frame(
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    region_ids: np.ndarray,
    target_id,
    face_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (origin, normal, axis_u, axis_v)；target_id 可为编号集合。"""
    ids = _as_int_array(region_ids)
    targets = _normalize_target_ids(target_id)
    mask = np.isin(ids, np.asarray(targets, dtype=np.int32))
    if not np.any(mask):
        raise RegionFitError(f"领域 {_target_label(targets)} 不存在")
    normals = _as_float_array(face_normals)[mask]
    areas = np.maximum(_as_float_array(face_areas)[mask], 1e-12)
    centers = _as_float_array(face_centers)[mask]
    weighted = (normals * areas[:, None]).sum(axis=0)
    normal = _normalize(weighted)
    if float(np.linalg.norm(normal)) < 1e-8:
        # 回退 PCA
        centroid = centers.mean(axis=0)
        cov = np.cov((centers - centroid).T)
        _, _, vh = np.linalg.svd(cov)
        normal = _normalize(vh[-1])
    origin = (centers * areas[:, None]).sum(axis=0) / areas.sum()
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(normal.dot(helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_u = _normalize(np.cross(helper, normal))
    axis_v = _normalize(np.cross(normal, axis_u))
    return origin, normal, axis_u, axis_v


def project_points_to_plane(
    points: np.ndarray,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    """世界坐标投影到局部 2D。"""
    pts = _as_float_array(points)
    delta = pts - _as_float_array(origin)
    u = delta @ _as_float_array(axis_u)
    v = delta @ _as_float_array(axis_v)
    return np.column_stack((u, v))


def unproject_points_from_plane(
    points_2d: np.ndarray,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    """局部 2D 还原到世界坐标（平面上）。"""
    pts = _as_float_array(points_2d)
    return (
        _as_float_array(origin)
        + pts[:, 0:1] * _as_float_array(axis_u)
        + pts[:, 1:2] * _as_float_array(axis_v)
    )


def detect_corner_indices(
    points_2d: np.ndarray,
    angle_threshold_deg: float = DEFAULT_CORNER_ANGLE_DEG,
    window: int | None = None,
    max_corners: int = 4,
    min_separation_frac: float = 0.025,
    convex_only: bool = True,
    include_strong_concave: bool = True,
    sample_strides: Sequence[int] | None = None,
) -> list[int]:
    """
    在均匀采样的闭环折线上检测角点。

    默认隔 2/4/8/16 个点取弦做多重采样：真尖角在多个尺度都呈大转角，
    锯齿噪声通常只在小尺度突出。以中位数转角 + 尺度投票抑制假角，
    再按强度做非极大值抑制。
    要求闭环为逆时针方向（凸/凹判定依赖叉积符号）。
    """
    pts = _as_float_array(points_2d)
    count = len(pts)
    if count < 3:
        return []

    max_span = max((count - 1) // 2, 1)
    if window is not None:
        strides = [max(1, min(int(window), max_span))]
    elif sample_strides is not None:
        strides = sorted(
            {
                max(1, min(int(stride), max_span))
                for stride in sample_strides
                if int(stride) > 0
            }
        )
    else:
        strides = [
            stride
            for stride in _CORNER_SAMPLE_STRIDES
            if stride <= max_span
        ]
        if not strides:
            strides = [1]

    indices = np.arange(count)
    angle_stack: list[np.ndarray] = []
    cross_stack: list[np.ndarray] = []
    for stride in strides:
        prev_pts = pts[(indices - stride) % count]
        next_pts = pts[(indices + stride) % count]
        incoming = pts - prev_pts
        outgoing = next_pts - pts
        in_len = np.linalg.norm(incoming, axis=1)
        out_len = np.linalg.norm(outgoing, axis=1)
        valid = (in_len > 1e-12) & (out_len > 1e-12)
        cos_angle = np.zeros(count, dtype=np.float64)
        cos_angle[valid] = np.clip(
            np.einsum("ij,ij->i", incoming[valid], outgoing[valid])
            / (in_len[valid] * out_len[valid]),
            -1.0,
            1.0,
        )
        angles_w = np.degrees(np.arccos(cos_angle))
        angles_w[~valid] = 0.0
        cross_w = (
            incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
        )
        cross_w[~valid] = 0.0
        angle_stack.append(angles_w)
        cross_stack.append(cross_w)

    angle_mat = np.vstack(angle_stack)
    cross_mat = np.vstack(cross_stack)
    # 中位数抑制单尺度锯齿尖峰；大尺度也参与投票
    robust_angles = np.median(angle_mat, axis=0)
    threshold = float(max(angle_threshold_deg, 1.0))
    votes = np.sum(angle_mat >= threshold, axis=0)
    min_votes = min(int(_CORNER_STRIDE_MIN_VOTES), len(strides))
    # 大步长必须全部过阈值：真尖角在 8/16 仍接近直角，锯齿通常只在小尺度尖
    large_strides = [
        index
        for index, stride in enumerate(strides)
        if stride >= 8
    ]
    if large_strides:
        large_ok = np.all(
            angle_mat[np.asarray(large_strides, dtype=np.int32)] >= threshold,
            axis=0,
        )
    else:
        large_ok = np.ones(count, dtype=bool)

    candidates = (robust_angles >= threshold) & (votes >= min_votes) & large_ok
    # 忽略离散极值尖峰（框中那类孤立噪声）
    candidates = _suppress_discrete_local_extrema(
        robust_angles,
        candidates,
        half_width=max(2, (strides[-1] // 4) if strides else 2),
        neighbor_ratio=0.55,
    )

    # 叉积取「过阈值尺度」的符号和，稳定凸/凹判定
    pass_mask = angle_mat >= threshold
    cross_votes = np.sum(
        np.where(pass_mask, np.sign(cross_mat), 0.0),
        axis=0,
    )
    if convex_only:
        keep = cross_votes > 0.0
        if include_strong_concave:
            keep |= (cross_votes < 0.0) & (
                robust_angles
                >= float(max(threshold, _STRONG_CONCAVE_ANGLE_DEG))
            )
        candidates &= keep

    candidate_idx = np.flatnonzero(candidates)
    if len(candidate_idx) == 0:
        return []

    # 强度：中位转角 + 大尺度加成，真直角优先于局部锯齿
    strength = robust_angles.copy()
    if large_strides:
        strength += 0.25 * np.max(
            angle_mat[np.asarray(large_strides, dtype=np.int32)],
            axis=0,
        )
    order = candidate_idx[np.argsort(strength[candidate_idx])[::-1]]
    min_sep = max(1, int(round(count * float(min_separation_frac))))
    kept: list[int] = []
    for index in order.tolist():
        conflict = False
        for existing in kept:
            distance = abs(index - existing)
            if min(distance, count - distance) < min_sep:
                conflict = True
                break
        if not conflict:
            kept.append(int(index))
        if len(kept) >= int(max_corners):
            break
    return sorted(kept)



def _multi_stride_turn_fields(
    points_2d: np.ndarray,
    sample_strides: Sequence[int] | None = None,
    closed: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """计算多尺度转角 / 叉积，返回 (angle_mat, cross_mat, robust_angles, strides)。"""
    pts = _as_float_array(points_2d)
    count = len(pts)
    if count < 3:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, empty, np.zeros(0, dtype=np.float64), []

    max_span = max((count - 1) // 2, 1)
    if sample_strides is not None:
        strides = sorted(
            {
                max(1, min(int(stride), max_span))
                for stride in sample_strides
                if int(stride) > 0
            }
        )
    else:
        strides = [
            stride for stride in _CORNER_SAMPLE_STRIDES if stride <= max_span
        ]
    if not strides:
        strides = [1]

    indices = np.arange(count)
    angle_stack: list[np.ndarray] = []
    cross_stack: list[np.ndarray] = []
    for stride in strides:
        if closed:
            prev_idx = (indices - stride) % count
            next_idx = (indices + stride) % count
            valid = np.ones(count, dtype=bool)
        else:
            prev_idx = indices - stride
            next_idx = indices + stride
            valid = (prev_idx >= 0) & (next_idx < count)
            prev_idx = np.clip(prev_idx, 0, count - 1)
            next_idx = np.clip(next_idx, 0, count - 1)
        prev_pts = pts[prev_idx]
        next_pts = pts[next_idx]
        incoming = pts - prev_pts
        outgoing = next_pts - pts
        in_len = np.linalg.norm(incoming, axis=1)
        out_len = np.linalg.norm(outgoing, axis=1)
        valid = valid & (in_len > 1e-12) & (out_len > 1e-12)
        cos_angle = np.zeros(count, dtype=np.float64)
        cos_angle[valid] = np.clip(
            np.einsum("ij,ij->i", incoming[valid], outgoing[valid])
            / (in_len[valid] * out_len[valid]),
            -1.0,
            1.0,
        )
        angles_w = np.degrees(np.arccos(cos_angle))
        angles_w[~valid] = 0.0
        cross_w = (
            incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
        )
        cross_w[~valid] = 0.0
        angle_stack.append(angles_w)
        cross_stack.append(cross_w)

    angle_mat = np.vstack(angle_stack)
    cross_mat = np.vstack(cross_stack)
    robust_angles = np.median(angle_mat, axis=0)
    return angle_mat, cross_mat, robust_angles, strides


def _suppress_discrete_local_extrema(
    values: np.ndarray,
    mask: np.ndarray,
    half_width: int = 2,
    neighbor_ratio: float = 0.5,
) -> np.ndarray:
    """
    忽略过窄的离散极值尖峰。

    真折角在邻域内仍有一定抬升；孤立单点尖峰剔除。
    """
    count = len(values)
    if count == 0:
        return mask
    kept = np.asarray(mask, dtype=bool).copy()
    width = max(int(half_width), 1)
    for index in np.flatnonzero(kept).tolist():
        peak = float(values[index])
        if peak <= 1e-9:
            kept[index] = False
            continue
        neighbors = [
            float(values[(index + offset) % count])
            for offset in range(-width, width + 1)
            if offset != 0
        ]
        high_neighbors = sum(
            1 for value in neighbors if value >= peak * float(neighbor_ratio)
        )
        if high_neighbors == 0:
            kept[index] = False
    return kept


def detect_concave_fold_indices(
    points_2d: np.ndarray,
    fold_angle_deg: float = _CONCAVE_FOLD_ANGLE_DEG,
    sample_strides: Sequence[int] | None = None,
    min_separation_frac: float = 0.02,
    max_folds: int = _MAX_CONCAVE_FOLDS,
    closed: bool = True,
) -> list[int]:
    """
    检测折线上明显的凹面折角（逆时针：右转 / 叉积为负）。

    使用 2/4/8/16 多重采样。相对凸角检测更宽松：大步长只需多数过线，
    避免真实凹折因锯齿在单一大尺度掉阈值而被漏标。
    """
    pts = _as_float_array(points_2d)
    count = len(pts)
    if count < 3:
        return []

    angle_mat, cross_mat, robust_angles, strides = _multi_stride_turn_fields(
        pts,
        sample_strides=sample_strides,
        closed=closed,
    )
    if len(strides) == 0:
        return []

    threshold = float(max(fold_angle_deg, 1.0))
    votes = np.sum(angle_mat >= threshold, axis=0)
    min_votes = min(int(_CORNER_STRIDE_MIN_VOTES), len(strides))
    large_strides = [
        index for index, stride in enumerate(strides) if stride >= 8
    ]
    if large_strides:
        large_vals = angle_mat[np.asarray(large_strides, dtype=np.int32)]
        # 宽松：至少一个大步长过阈值，且大步长中位数不太弱
        large_ok = (
            np.any(large_vals >= threshold, axis=0)
            & (np.median(large_vals, axis=0) >= threshold * 0.65)
        )
    else:
        large_ok = np.ones(count, dtype=bool)

    # 凹折符号：在转角较大的尺度上统计叉积
    soft_mask = angle_mat >= (threshold * 0.6)
    cross_votes = np.sum(
        np.where(soft_mask, np.sign(cross_mat), 0.0),
        axis=0,
    )
    candidates = (
        (robust_angles >= threshold)
        & (votes >= min_votes)
        & large_ok
        & (cross_votes < 0.0)
    )
    # 仅剔除极窄孤立尖峰，保留真实凹折
    candidates = _suppress_discrete_local_extrema(
        robust_angles,
        candidates,
        half_width=2,
        neighbor_ratio=0.35,
    )

    # 局部极大（闭环或开链）
    for index in np.flatnonzero(candidates).tolist():
        if closed:
            left = float(robust_angles[(index - 1) % count])
            right = float(robust_angles[(index + 1) % count])
        else:
            if index <= 0 or index >= count - 1:
                candidates[index] = False
                continue
            left = float(robust_angles[index - 1])
            right = float(robust_angles[index + 1])
        peak = float(robust_angles[index])
        if peak + 1e-9 < left and peak + 1e-9 < right:
            candidates[index] = False

    if not closed:
        # 开链端点不当折角
        candidates[0] = False
        candidates[-1] = False

    candidate_idx = np.flatnonzero(candidates)
    if len(candidate_idx) == 0:
        return []

    order = candidate_idx[np.argsort(robust_angles[candidate_idx])[::-1]]
    min_sep = max(1, int(round(count * float(min_separation_frac))))
    kept: list[int] = []
    for index in order.tolist():
        conflict = False
        for existing in kept:
            distance = abs(index - existing)
            if closed:
                distance = min(distance, count - distance)
            if distance < min_sep:
                conflict = True
                break
        if not conflict:
            kept.append(int(index))
        if len(kept) >= int(max_folds):
            break
    return sorted(kept)


def point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    """点到折线的最短距离。"""
    pts = _as_float_array(polyline)
    query = _as_float_array(point).reshape(3)
    if len(pts) == 0:
        return float("inf")
    if len(pts) == 1:
        return float(np.linalg.norm(query - pts[0]))
    best = float("inf")
    for index in range(len(pts) - 1):
        a = pts[index]
        b = pts[index + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-18:
            dist = float(np.linalg.norm(query - a))
        else:
            t = float(np.clip(np.dot(query - a, ab) / denom, 0.0, 1.0))
            dist = float(np.linalg.norm(query - (a + t * ab)))
        best = min(best, dist)
    return best


def filter_handle_outliers(
    handles: Sequence[np.ndarray],
    polyline: np.ndarray,
    max_distance: float,
) -> list[np.ndarray]:
    """忽略偏离边线过远的离散控制手柄（面内漂浮噪声）。"""
    limit = float(max(max_distance, 0.0))
    kept: list[np.ndarray] = []
    for handle in handles:
        point = _as_float_array(handle).reshape(3)
        if point_to_polyline_distance(point, polyline) <= limit:
            kept.append(point)
    return kept



def side_interior_max_turn_deg(points: np.ndarray) -> float:
    """边链内部（不含端点）的最大转角，用于发现被误并的尖角。"""
    pts = _as_float_array(points)
    if len(pts) < 3:
        return 0.0
    max_turn = 0.0
    for index in range(1, len(pts) - 1):
        incoming = pts[index] - pts[index - 1]
        outgoing = pts[index + 1] - pts[index]
        in_len = float(np.linalg.norm(incoming))
        out_len = float(np.linalg.norm(outgoing))
        if in_len < 1e-12 or out_len < 1e-12:
            continue
        cos_angle = float(
            np.clip(
                np.dot(incoming, outgoing) / (in_len * out_len),
                -1.0,
                1.0,
            )
        )
        max_turn = max(max_turn, float(np.degrees(np.arccos(cos_angle))))
    return max_turn


def detect_side_fold_indices(
    points_2d: np.ndarray,
    fold_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
    sample_strides: Sequence[int] | None = None,
    min_separation_frac: float = 0.02,
    max_folds: int = _MAX_CONCAVE_FOLDS,
) -> list[int]:
    """
    开链边上检测明显折角（凸/凹均计）。

    与凹折检测同用多重采样，但不限制叉积符号，用于判定边应折线拟合。
    """
    pts = _as_float_array(points_2d)
    count = len(pts)
    if count < 3:
        return []

    angle_mat, _cross_mat, robust_angles, strides = _multi_stride_turn_fields(
        pts,
        sample_strides=sample_strides,
        closed=False,
    )
    if len(strides) == 0:
        return []

    threshold = float(max(fold_angle_deg, 1.0))
    votes = np.sum(angle_mat >= threshold, axis=0)
    min_votes = min(int(_CORNER_STRIDE_MIN_VOTES), len(strides))
    large_strides = [
        index for index, stride in enumerate(strides) if stride >= 8
    ]
    if large_strides:
        large_vals = angle_mat[np.asarray(large_strides, dtype=np.int32)]
        large_ok = (
            np.any(large_vals >= threshold, axis=0)
            & (np.median(large_vals, axis=0) >= threshold * 0.65)
        )
    else:
        large_ok = np.ones(count, dtype=bool)

    candidates = (
        (robust_angles >= threshold)
        & (votes >= min_votes)
        & large_ok
    )
    candidates = _suppress_discrete_local_extrema(
        robust_angles,
        candidates,
        half_width=2,
        neighbor_ratio=0.35,
    )

    for index in np.flatnonzero(candidates).tolist():
        if index <= 0 or index >= count - 1:
            candidates[index] = False
            continue
        left = float(robust_angles[index - 1])
        right = float(robust_angles[index + 1])
        peak = float(robust_angles[index])
        if peak + 1e-9 < left and peak + 1e-9 < right:
            candidates[index] = False

    candidates[0] = False
    candidates[-1] = False

    candidate_idx = np.flatnonzero(candidates)
    if len(candidate_idx) == 0:
        return []

    order = candidate_idx[np.argsort(robust_angles[candidate_idx])[::-1]]
    min_sep = max(1, int(round(count * float(min_separation_frac))))
    kept: list[int] = []
    for index in order.tolist():
        if any(abs(index - existing) < min_sep for existing in kept):
            continue
        kept.append(int(index))
        if len(kept) >= int(max_folds):
            break
    return sorted(kept)


def extract_polyline_keypoints(
    points: np.ndarray,
    fold_indices: Sequence[int],
) -> np.ndarray:
    """端点 + 折角点构成折线关键点。"""
    pts = _as_float_array(points)
    if len(pts) == 0:
        return pts
    if len(pts) == 1:
        return pts.copy()
    indices = {0, len(pts) - 1}
    for index in fold_indices:
        idx = int(index)
        if 0 < idx < len(pts) - 1:
            indices.add(idx)
    ordered = sorted(indices)
    return pts[np.asarray(ordered, dtype=np.int32)].copy()


def douglas_peucker_indices(
    points: np.ndarray,
    epsilon: float,
) -> list[int]:
    """Douglas-Peucker 简化，返回保留点下标（含端点，升序）。"""
    pts = _as_float_array(points)
    count = len(pts)
    if count <= 2:
        return list(range(count))
    eps = float(max(epsilon, 0.0))

    def _recurse(start: int, end: int, kept: set[int]) -> None:
        if end <= start + 1:
            return
        segment = pts[end] - pts[start]
        seg_len = float(np.linalg.norm(segment))
        max_dist = -1.0
        max_index = start
        for index in range(start + 1, end):
            if seg_len < 1e-12:
                dist = float(np.linalg.norm(pts[index] - pts[start]))
            else:
                dist = float(
                    np.linalg.norm(
                        np.cross(segment, pts[index] - pts[start])
                    )
                    / seg_len
                )
            if dist > max_dist:
                max_dist = dist
                max_index = index
        if max_dist > eps:
            kept.add(max_index)
            _recurse(start, max_index, kept)
            _recurse(max_index, end, kept)

    kept_indices: set[int] = {0, count - 1}
    _recurse(0, count - 1, kept_indices)
    return sorted(kept_indices)


def _polyline_interior_max_turn_deg(points: np.ndarray) -> float:
    """折线关键点序列内部最大转角。"""
    return side_interior_max_turn_deg(points)


def _max_points_to_polyline_distance(
    points: np.ndarray,
    polyline: np.ndarray,
) -> float:
    pts = _as_float_array(points)
    poly = _as_float_array(polyline)
    if len(pts) == 0 or len(poly) == 0:
        return 0.0
    return float(
        max(point_to_polyline_distance(point, poly) for point in pts)
    )


def _bezier_spans_to_polyline(
    spans: Sequence[np.ndarray],
    samples_per_span: int = 12,
) -> np.ndarray:
    """把多段三次贝塞尔采样成连续折线（段间共享端点只保留一次）。"""
    if not spans:
        return np.zeros((0, 3), dtype=np.float64)
    parts: list[np.ndarray] = []
    for index, controls in enumerate(spans):
        sampled = sample_cubic_bezier(
            controls,
            max(int(samples_per_span), 4),
        )
        parts.append(sampled if index == 0 else sampled[1:])
    return np.vstack(parts)


def classify_side_fit_mode(
    points: np.ndarray,
    points_2d: np.ndarray | None = None,
    fold_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
) -> tuple[str, np.ndarray, list[np.ndarray] | None]:
    """
    判定边为折线或平滑曲线拟合。

    用 Douglas-Peucker 压锯齿；若简化后仍有明显折角则折线拟合，
    否则近直线用两端点，缓弧用多段贝塞尔。
    返回 (fit_mode, polyline_or_side_points, bezier_spans|None)。
    """
    pts = _as_float_array(points)
    if len(pts) < 2:
        raise RegionFitError("边点数不足，无法分类拟合")
    if len(pts) == 2:
        return "POLYLINE", pts.copy(), None

    length = float(polyline_length(pts))
    # 略大于典型边界锯齿幅度，避免把噪声当成折点
    epsilon = max(length * 0.012, 1e-6)
    simplified_idx = douglas_peucker_indices(pts, epsilon)
    keypoints = pts[np.asarray(simplified_idx, dtype=np.int32)].copy()
    key_turn = _polyline_interior_max_turn_deg(keypoints)
    poly_err = _max_points_to_polyline_distance(pts, keypoints)

    # 明显折线：简化后仍有大折角
    if len(keypoints) >= 3 and key_turn >= float(fold_angle_deg):
        return "POLYLINE", keypoints, None

    # 近直线
    ends = pts[[0, -1]]
    if _max_points_to_polyline_distance(pts, ends) <= max(length * 0.02, 1e-6):
        return "POLYLINE", ends.copy(), None

    # 多重采样折角作补充：真折角但 DP 阈值偏宽时仍抓折线
    side_2d = (
        _as_float_array(points_2d)
        if points_2d is not None
        else pts[:, :2]
    )
    folds = detect_side_fold_indices(
        side_2d,
        fold_angle_deg=fold_angle_deg,
        min_separation_frac=0.06,
    )
    if folds:
        fold_keys = extract_polyline_keypoints(pts, folds)
        fold_turn = _polyline_interior_max_turn_deg(fold_keys)
        fold_err = _max_points_to_polyline_distance(pts, fold_keys)
        if (
            len(fold_keys) >= 3
            and fold_turn >= float(fold_angle_deg)
            and fold_err <= max(length * 0.04, poly_err * 1.25)
        ):
            return "POLYLINE", fold_keys, None

    spans = fit_bezier_polyline_spans(pts)
    return "CURVE", pts.copy(), spans


def _side_from_loop(
    loop_points: np.ndarray,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    count = len(loop_points)
    if count == 0:
        return np.empty((0, loop_points.shape[1]), dtype=np.float64)
    indices = [start_index]
    cursor = start_index
    guard = 0
    while cursor != end_index:
        cursor = (cursor + 1) % count
        indices.append(cursor)
        guard += 1
        if guard > count:
            raise RegionFitError("边链提取失败")
    return loop_points[np.asarray(indices, dtype=np.int32)]


def split_loop_into_sides(
    loop_points: np.ndarray,
    corner_indices: Sequence[int],
) -> list[np.ndarray]:
    """按折角点将闭环拆成边链（每条含两端角点）；不少于 2 个折角。"""
    if len(corner_indices) < 2:
        raise RegionFitError("折角不足，无法拆分边界段")
    ordered = sorted({int(i) % len(loop_points) for i in corner_indices})
    if len(ordered) < 2:
        raise RegionFitError("折角不足，无法拆分边界段")
    sides: list[np.ndarray] = []
    for index, start in enumerate(ordered):
        end = ordered[(index + 1) % len(ordered)]
        sides.append(_side_from_loop(loop_points, start, end))
    return sides


def split_polyline_at_significant_folds(
    points: np.ndarray,
    fold_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
) -> list[np.ndarray]:
    """
    仅在折线内部明显折角处断开；锯齿/小平折保留在同一段内。

    避免把 Douglas-Peucker 的每个短边都当成独立着色段。
    """
    pts = _as_float_array(points)
    if len(pts) < 3:
        return [pts.copy()] if len(pts) >= 2 else []
    threshold = float(max(fold_angle_deg, 1.0))
    break_indices = [0]
    for index in range(1, len(pts) - 1):
        incoming = pts[index] - pts[index - 1]
        outgoing = pts[index + 1] - pts[index]
        in_len = float(np.linalg.norm(incoming))
        out_len = float(np.linalg.norm(outgoing))
        if in_len < 1e-12 or out_len < 1e-12:
            continue
        cos_angle = float(
            np.clip(
                np.dot(incoming, outgoing) / (in_len * out_len),
                -1.0,
                1.0,
            )
        )
        turn = float(np.degrees(np.arccos(cos_angle)))
        if turn >= threshold:
            break_indices.append(index)
    break_indices.append(len(pts) - 1)
    segments: list[np.ndarray] = []
    for start, end in zip(break_indices[:-1], break_indices[1:]):
        if end <= start:
            continue
        segments.append(pts[start : end + 1].copy())
    return segments if segments else [pts.copy()]


def resplit_sides_at_interior_folds(
    sides: Sequence[np.ndarray],
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    fold_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
) -> list[np.ndarray]:
    """
    在已拆边的中段再检凸/凹折角并断开。

    解决「整段同色却含明显折弯」：首轮角点漏检时，中段折角仍应拆段。
    """
    result: list[np.ndarray] = []
    for side in sides:
        pts = _as_float_array(side)
        if len(pts) < 5:
            result.append(pts.copy())
            continue
        side_2d = project_points_to_plane(pts, origin, axis_u, axis_v)
        folds = detect_side_fold_indices(
            side_2d,
            fold_angle_deg=fold_angle_deg,
            min_separation_frac=0.08,
        )
        if not folds:
            result.append(pts.copy())
            continue
        cuts = sorted({0, len(pts) - 1, *[int(i) for i in folds]})
        for start, end in zip(cuts[:-1], cuts[1:]):
            if end > start:
                result.append(pts[start : end + 1].copy())
    return result


def resplit_sides_near_points(
    sides: Sequence[np.ndarray],
    points: Sequence[np.ndarray],
    max_dist: float,
    end_margin_frac: float = 0.05,
) -> list[np.ndarray]:
    """
    若给定点贴近某边中段，则在最近点切开。

    用于在共线合并之后强制恢复邻域 T 接缝切点。
    """
    if not points:
        return [np.asarray(side, dtype=np.float64).copy() for side in sides]
    limit = float(max(max_dist, 0.0))
    result: list[np.ndarray] = []
    for side in sides:
        pts = _as_float_array(side)
        if len(pts) < 2:
            result.append(pts.copy())
            continue
        params = polyline_parameters(pts)
        length = float(polyline_length(pts))
        margin = max(float(end_margin_frac), 0.0)
        cut_ts: list[float] = []
        for point in points:
            p = _as_float_array(point).reshape(3)
            best_t = None
            best_dist = limit + 1.0
            for index in range(len(pts) - 1):
                a = pts[index]
                b = pts[index + 1]
                ab = b - a
                ab_len2 = float(np.dot(ab, ab))
                if ab_len2 < 1e-18:
                    continue
                local_t = float(np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0))
                proj = a + local_t * ab
                dist = float(np.linalg.norm(p - proj))
                if dist < best_dist:
                    best_dist = dist
                    seg_param = float(params[index])
                    next_param = float(params[index + 1])
                    best_t = seg_param + local_t * (next_param - seg_param)
            if (
                best_t is not None
                and best_dist <= limit
                and margin < best_t < 1.0 - margin
            ):
                cut_ts.append(best_t)
        if not cut_ts:
            result.append(pts.copy())
            continue
        cut_ts = sorted(set(cut_ts))
        # 用弧长参数切开：在最近点插入并拆段
        anchors = [0.0] + cut_ts + [1.0]
        for start_t, end_t in zip(anchors[:-1], anchors[1:]):
            if end_t <= start_t + 1e-9:
                continue
            # 在 [start_t, end_t] 间取原子点
            mask = (params >= start_t - 1e-9) & (params <= end_t + 1e-9)
            chunk = pts[mask]
            if len(chunk) < 2:
                # 退化：两端插值
                start_pt = np.array(
                    [
                        np.interp(start_t, params, pts[:, axis])
                        for axis in range(3)
                    ],
                    dtype=np.float64,
                )
                end_pt = np.array(
                    [
                        np.interp(end_t, params, pts[:, axis])
                        for axis in range(3)
                    ],
                    dtype=np.float64,
                )
                chunk = np.vstack((start_pt, end_pt))
            else:
                chunk = chunk.copy()
                chunk[0] = np.array(
                    [
                        np.interp(start_t, params, pts[:, axis])
                        for axis in range(3)
                    ],
                    dtype=np.float64,
                )
                chunk[-1] = np.array(
                    [
                        np.interp(end_t, params, pts[:, axis])
                        for axis in range(3)
                    ],
                    dtype=np.float64,
                )
            if length > 0 and polyline_length(chunk) >= limit * 0.25:
                result.append(chunk)
            elif result:
                result[-1] = np.vstack((result[-1][:-1], chunk))
            else:
                result.append(chunk)
    return result if result else [np.asarray(sides[0], dtype=np.float64).copy()]


def merge_short_boundary_sides(
    sides: Sequence[np.ndarray],
    min_length: float,
    min_sides: int = 2,
) -> list[np.ndarray]:
    """
    把过短碎段并入邻边（优先并向转角更平的一侧）。

    抑制 T 接缝处网格锯齿造成的短色段。
    """
    result = [np.asarray(side, dtype=np.float64).copy() for side in sides]
    threshold = float(max(min_length, 0.0))
    guard = 0
    while len(result) > int(min_sides):
        lengths = [float(polyline_length(side)) for side in result]
        short_i = int(np.argmin(lengths))
        if lengths[short_i] >= threshold:
            break
        left = (short_i - 1) % len(result)
        right = (short_i + 1) % len(result)
        turn_left = _junction_turn_deg(result[left], result[short_i])
        turn_right = _junction_turn_deg(result[short_i], result[right])
        if turn_left <= turn_right:
            merge_a, merge_b = left, short_i
        else:
            merge_a, merge_b = short_i, right
        merged = np.vstack((result[merge_a][:-1], result[merge_b]))
        if merge_b > merge_a:
            result[merge_a] = merged
            del result[merge_b]
        else:
            result = [merged] + [
                result[index]
                for index in range(len(result))
                if index not in {merge_a, merge_b}
            ]
        guard += 1
        if guard > len(sides) + 8:
            break
    return result


def make_segment_color(seed: int) -> tuple[float, float, float, float]:
    """由种子生成饱和、互异感强的 RGBA（确定性随机）。"""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    hue = float(rng.random())
    sat = float(0.55 + 0.40 * rng.random())
    val = float(0.72 + 0.28 * rng.random())
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)
    i_mod = i % 6
    if i_mod == 0:
        r, g, b = val, t, p
    elif i_mod == 1:
        r, g, b = q, val, p
    elif i_mod == 2:
        r, g, b = p, val, t
    elif i_mod == 3:
        r, g, b = p, q, val
    elif i_mod == 4:
        r, g, b = t, p, val
    else:
        r, g, b = val, p, q
    return (float(r), float(g), float(b), 1.0)


def fit_cubic_bezier_controls(points: np.ndarray) -> np.ndarray:
    """
    将折线拟合成三次贝塞尔控制点 (P0,P1,P2,P3)，端点固定。

    中间两点用弧长参数化最小二乘求解；退化时退回弦长三等分。
    """
    pts = _as_float_array(points)
    if len(pts) < 2:
        raise RegionFitError("折线点数不足，无法拟合贝塞尔")
    p0 = pts[0].copy()
    p3 = pts[-1].copy()
    if len(pts) == 2 or polyline_length(pts) < 1e-12:
        delta = p3 - p0
        return np.vstack((p0, p0 + delta / 3.0, p0 + 2.0 * delta / 3.0, p3))

    params = polyline_parameters(pts)
    # 跳过端点：那里对 P1/P2 的系数为 0
    inner = (params > 1e-9) & (params < 1.0 - 1e-9)
    if not np.any(inner):
        delta = p3 - p0
        return np.vstack((p0, p0 + delta / 3.0, p0 + 2.0 * delta / 3.0, p3))

    t = params[inner]
    q = pts[inner]
    omt = 1.0 - t
    a1 = 3.0 * (omt ** 2) * t
    a2 = 3.0 * omt * (t ** 2)
    rhs = q - (omt ** 3)[:, None] * p0 - (t ** 3)[:, None] * p3
    design = np.column_stack((a1, a2))
    gram = design.T @ design
    if float(np.linalg.det(gram)) < 1e-14:
        delta = p3 - p0
        return np.vstack((p0, p0 + delta / 3.0, p0 + 2.0 * delta / 3.0, p3))

    p1 = np.empty(3, dtype=np.float64)
    p2 = np.empty(3, dtype=np.float64)
    for axis in range(3):
        solved = np.linalg.solve(gram, design.T @ rhs[:, axis])
        p1[axis] = float(solved[0])
        p2[axis] = float(solved[1])
    return np.vstack((p0, p1, p2, p3))


def fit_bezier_polyline_spans(
    points: np.ndarray,
    span_points: int = 12,
) -> list[np.ndarray]:
    """
    把长折线拆成多段三次贝塞尔，避免单段三次在长弧上弦切造成视觉断口。
    返回若干 (4,3) 控制点数组，首尾相接。
    """
    pts = _as_float_array(points)
    if len(pts) < 2:
        raise RegionFitError("折线点数不足，无法拟合贝塞尔")
    if len(pts) <= max(int(span_points), 4):
        return [fit_cubic_bezier_controls(pts)]

    step = max(int(span_points) - 1, 3)
    spans: list[np.ndarray] = []
    start = 0
    while start < len(pts) - 1:
        end = min(start + step, len(pts) - 1)
        # 最后一段太短则并入前一段
        if end < len(pts) - 1 and (len(pts) - 1 - end) < max(step // 2, 2):
            end = len(pts) - 1
        spans.append(fit_cubic_bezier_controls(pts[start : end + 1]))
        if end >= len(pts) - 1:
            break
        start = end
    return spans


def sample_cubic_bezier(controls: np.ndarray, count: int) -> np.ndarray:
    """按均匀参数 t 采样三次贝塞尔曲线。"""
    ctrl = _as_float_array(controls)
    if ctrl.shape != (4, 3):
        raise RegionFitError("贝塞尔控制点必须为 (4,3)")
    target = max(int(count), 2)
    t = np.linspace(0.0, 1.0, target, dtype=np.float64)
    omt = 1.0 - t
    return (
        (omt ** 3)[:, None] * ctrl[0]
        + (3.0 * (omt ** 2) * t)[:, None] * ctrl[1]
        + (3.0 * omt * (t ** 2))[:, None] * ctrl[2]
        + (t ** 3)[:, None] * ctrl[3]
    )


def share_closed_side_endpoints(sides: Sequence[np.ndarray]) -> list[np.ndarray]:
    """强制闭环相邻边共享端点。"""
    shared_sides = [np.asarray(side, dtype=np.float64).copy() for side in sides]
    if not shared_sides:
        return shared_sides
    for index in range(len(shared_sides)):
        nxt = (index + 1) % len(shared_sides)
        shared = 0.5 * (shared_sides[index][-1] + shared_sides[nxt][0])
        shared_sides[index][-1] = shared
        shared_sides[nxt][0] = shared
    return shared_sides


def _side_unit_direction(side: np.ndarray) -> np.ndarray | None:
    pts = _as_float_array(side)
    if len(pts) < 2:
        return None
    delta = pts[-1] - pts[0]
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
        return None
    return delta / length


def _symmetric_mean_polyline_distance(
    side_a: np.ndarray,
    side_b: np.ndarray,
    samples: int = 12,
) -> float:
    """两条折线的对称平均最近点距离。"""
    a = resample_polyline(_as_float_array(side_a), max(int(samples), 4))
    b = resample_polyline(_as_float_array(side_b), max(int(samples), 4))
    dist_ab = float(np.mean([point_to_polyline_distance(p, b) for p in a]))
    dist_ba = float(np.mean([point_to_polyline_distance(p, a) for p in b]))
    return 0.5 * (dist_ab + dist_ba)


def _segments_projective_overlap(
    side_a: np.ndarray,
    side_b: np.ndarray,
    min_overlap_frac: float = 0.25,
) -> bool:
    """沿 side_a 主方向投影，两段是否有足够重叠。"""
    a = _as_float_array(side_a)
    b = _as_float_array(side_b)
    direction = _side_unit_direction(a)
    if direction is None:
        return False
    origin = a[0]
    proj_a = np.dot(a - origin, direction)
    proj_b = np.dot(b - origin, direction)
    a0, a1 = float(np.min(proj_a)), float(np.max(proj_a))
    b0, b1 = float(np.min(proj_b)), float(np.max(proj_b))
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    span = max(a1 - a0, b1 - b0, 1e-12)
    return overlap / span >= float(min_overlap_frac)


def average_parallel_polylines(
    side_a: np.ndarray,
    side_b: np.ndarray,
    samples: int = 16,
) -> np.ndarray:
    """将近平行对边融并为中线（自动对齐方向）。"""
    return average_polyline_cluster([side_a, side_b], samples=samples)


def average_polyline_cluster(
    sides: Sequence[np.ndarray],
    samples: int = 16,
) -> np.ndarray:
    """多条近平行折线融并为一条中线。"""
    members = [_as_float_array(side) for side in sides if len(side) >= 2]
    if not members:
        raise RegionFitError("空边簇无法融并")
    if len(members) == 1:
        return members[0].copy()
    lengths = [float(polyline_length(poly)) for poly in members]
    ref = members[int(np.argmax(lengths))]
    dir_ref = _side_unit_direction(ref)
    count = max(int(samples), 4)
    stacked = [resample_polyline(ref, count)]
    for side in members:
        if side is ref:
            continue
        work = side
        dir_b = _side_unit_direction(work)
        if (
            dir_ref is not None
            and dir_b is not None
            and float(np.dot(dir_ref, dir_b)) < 0.0
        ):
            work = work[::-1].copy()
        stacked.append(resample_polyline(work, count))
    return np.mean(np.stack(stacked, axis=0), axis=0)


def collect_side_endpoints(sides: Sequence[np.ndarray]) -> list[np.ndarray]:
    """收集边端点作为邻近切分关键点。"""
    points: list[np.ndarray] = []
    for side in sides:
        pts = _as_float_array(side)
        if len(pts) == 0:
            continue
        points.append(pts[0].copy())
        if len(pts) > 1:
            points.append(pts[-1].copy())
    return points


def _sides_merge_compatible(
    side_a: np.ndarray,
    side_b: np.ndarray,
    proximity: float,
    parallel_dot: float = _MULTI_ISLAND_PARALLEL_DOT,
    overlap_frac: float = _MULTI_ISLAND_OVERLAP_FRAC,
) -> bool:
    dir_a = _side_unit_direction(side_a)
    dir_b = _side_unit_direction(side_b)
    if dir_a is None or dir_b is None:
        return False
    if abs(float(np.dot(dir_a, dir_b))) < float(parallel_dot):
        return False
    if not _segments_projective_overlap(
        side_a,
        side_b,
        min_overlap_frac=overlap_frac,
    ):
        return False
    return _symmetric_mean_polyline_distance(side_a, side_b) <= float(proximity)


def harmonize_multi_island_sides(
    island_sides: Sequence[Sequence[np.ndarray]],
    proximity: float,
) -> tuple[list[list[np.ndarray]], set[tuple[int, int]]]:
    """
    多岛边界协调：靠近的点切开线；近平行线聚类后每簇只保留一条代表边。

    返回 (各岛边列表, 抑制绘制的 (island_index, side_index) 集合)。
    次要岛上的重复对边不再画出来，避免多条平行色管叠在一起。
    """
    if len(island_sides) < 2:
        copied = [
            [np.asarray(side, dtype=np.float64).copy() for side in sides]
            for sides in island_sides
        ]
        return copied, set()

    limit = float(max(proximity, 0.0))
    keypoints = [collect_side_endpoints(sides) for sides in island_sides]

    # 1) 他岛端点落到本岛边上 → 切开
    cut_sides: list[list[np.ndarray]] = []
    for index, sides in enumerate(island_sides):
        foreign: list[np.ndarray] = []
        for other, points in enumerate(keypoints):
            if other != index:
                foreign.extend(points)
        cut = resplit_sides_near_points(
            [np.asarray(side, dtype=np.float64) for side in sides],
            foreign,
            max_dist=limit,
        )
        cut_sides.append(share_closed_side_endpoints(cut))

    # 2) 跨岛近平行边聚类（并查集）
    flat: list[tuple[int, int, np.ndarray]] = []
    for island_index, sides in enumerate(cut_sides):
        for side_index, side in enumerate(sides):
            flat.append((island_index, side_index, side))

    parent = list(range(len(flat)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(len(flat)):
        ia, _sa, side_a = flat[a]
        for b in range(a + 1, len(flat)):
            ib, _sb, side_b = flat[b]
            if ia == ib:
                continue
            if _sides_merge_compatible(side_a, side_b, limit):
                _union(a, b)

    clusters: dict[int, list[int]] = {}
    for index in range(len(flat)):
        root = _find(index)
        clusters.setdefault(root, []).append(index)

    result: list[list[np.ndarray]] = [
        [side.copy() for side in sides] for sides in cut_sides
    ]
    suppress: set[tuple[int, int]] = set()

    for members in clusters.values():
        if len(members) < 2:
            continue
        polylines = [flat[index][2] for index in members]
        lengths = [float(polyline_length(poly)) for poly in polylines]
        primary_local = int(np.argmax(lengths))
        primary_flat = members[primary_local]
        midline = average_polyline_cluster(polylines)
        pi, psi, _ = flat[primary_flat]
        result[pi][psi] = midline
        for local_index, flat_index in enumerate(members):
            if local_index == primary_local:
                continue
            ji, jsi, _ = flat[flat_index]
            # 保留拓扑端点，但不绘制，避免平行双线
            result[ji][jsi] = result[ji][jsi].copy()
            suppress.add((int(ji), int(jsi)))

    # 3) 仅对仍绘制的边，再用「绘制边」端点做一次近点切开
    draw_keypoints: list[list[np.ndarray]] = []
    for island_index, sides in enumerate(result):
        pts: list[np.ndarray] = []
        for side_index, side in enumerate(sides):
            if (island_index, side_index) in suppress:
                continue
            pts.extend(collect_side_endpoints([side]))
        draw_keypoints.append(pts)

    for island_index, sides in enumerate(result):
        foreign = []
        for other, points in enumerate(draw_keypoints):
            if other != island_index:
                foreign.extend(points)
        if not foreign:
            continue
        kept_indices = [
            side_index
            for side_index in range(len(sides))
            if (island_index, side_index) not in suppress
        ]
        if not kept_indices:
            continue
        kept_sides = [sides[side_index] for side_index in kept_indices]
        recut = resplit_sides_near_points(kept_sides, foreign, max_dist=limit)
        # recut 可能改变边数：写回时扩展/收缩该岛边列表较复杂；
        # 仅当边数不变时替换，否则跳过二次切开。
        if len(recut) == len(kept_sides):
            for local, side_index in enumerate(kept_indices):
                result[island_index][side_index] = recut[local]
        result[island_index] = share_closed_side_endpoints(result[island_index])

    return result, suppress


def build_fit_work_items(
    loops: Sequence[Sequence[int]],
    vertices: np.ndarray,
    stage: str,
    stitch_gap_frac: float = _MULTI_ISLAND_MERGE_GAP_FRAC,
    bridge_gap_frac: float = 0.25,
    bridge_enabled: bool = True,
    interior_points: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """
    按拟合阶段生成工作项（points / loop_ids / merged_from）。

    stage:
      ISLANDS — 每岛独立外围闭环
      STITCH  — 仅缝合邻近孤岛
      BRIDGE  — 在缝合基础上桥接更远的孤岛（条带外包络）
    """
    verts = _as_float_array(vertices)
    valid = [list(int(v) for v in loop) for loop in loops if len(loop) >= 3]
    meta: dict = {
        "stage": str(stage),
        "bridge_links": [],
        "stitch_gap": 0.0,
        "bridge_gap": 0.0,
        "extent": 0.0,
    }
    if not valid:
        return [], meta

    stacked = np.vstack(
        [verts[np.asarray(loop, dtype=np.int32)] for loop in valid]
    )
    extent = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    stitch_gap = max(extent * float(max(stitch_gap_frac, 0.0)), 1e-4)
    bridge_gap = max(
        extent * float(max(bridge_gap_frac, stitch_gap_frac, 0.0)),
        stitch_gap,
    )
    meta["extent"] = extent
    meta["stitch_gap"] = stitch_gap
    meta["bridge_gap"] = bridge_gap

    stage_key = str(stage).upper()
    if stage_key == "ISLANDS":
        items = [
            {
                "points": collapse_ultra_reflex_spike_vertices_closed_loop(
                    verts[np.asarray(loop, dtype=np.int32)]
                ),
                "loop_ids": loop,
                "merged_from": 1,
            }
            for loop in valid
        ]
        # 收束后点列可能不再对应原始顶点下标
        for item in items:
            collapsed = item["points"]
            ids = item.get("loop_ids")
            if ids is None:
                continue
            original = verts[np.asarray(ids, dtype=np.int32)]
            if len(collapsed) != len(original) or not np.allclose(
                collapsed, original, atol=1e-9
            ):
                item["loop_ids"] = None
        return items, meta

    if stage_key == "STITCH":
        items = merge_nearby_loops_to_outer_contours(
            valid, verts, max_gap=stitch_gap
        )
        seam_links: list[tuple[np.ndarray, np.ndarray]] = []
        for item in items:
            for seam in item.get("stitch_seams") or []:
                pts = _as_float_array(seam)
                for index in range(len(pts) - 1):
                    seam_links.append((pts[index].copy(), pts[index + 1].copy()))
        meta["bridge_links"] = seam_links
        return items, meta

    # BRIDGE：用更大间隙聚簇；簇内优先折角缝合，失败再条带外包络
    if not bridge_enabled:
        items = merge_nearby_loops_to_outer_contours(
            valid, verts, max_gap=stitch_gap
        )
        return items, meta

    clusters = cluster_nearby_boundary_loops(valid, verts, max_gap=bridge_gap)
    items: list[dict] = []
    bridge_links: list[tuple[np.ndarray, np.ndarray]] = []
    for cluster in clusters:
        cluster_loops = [valid[index] for index in cluster]
        if len(cluster_loops) == 1:
            ids = cluster_loops[0]
            items.append(
                {
                    "points": verts[np.asarray(ids, dtype=np.int32)].copy(),
                    "loop_ids": ids,
                    "merged_from": 1,
                }
            )
            continue
        centroids = []
        for loop in cluster_loops:
            pts = verts[np.asarray(loop, dtype=np.int32)]
            centroids.append(pts.mean(axis=0))
        for index in range(len(centroids) - 1):
            bridge_links.append(
                (centroids[index].copy(), centroids[index + 1].copy())
            )
        # 先折角缝合 / 消对边；失败再用桥接外包络
        envelope, seams = _outer_contour_for_nearby_cluster(
            cluster_loops,
            verts,
            max_gap=stitch_gap,
            allow_band_fallback=True,
        )
        for seam in seams:
            if len(seam) >= 2:
                bridge_links.append((seam[0].copy(), seam[-1].copy()))
        if envelope is None or len(envelope) < 3:
            try:
                envelope, _, _ = combine_boundary_islands(
                    cluster_loops,
                    verts,
                    interior_points=interior_points,
                    prefer_outer_envelope=True,
                )
            except RegionFitError:
                for ids in cluster_loops:
                    items.append(
                        {
                            "points": verts[
                                np.asarray(ids, dtype=np.int32)
                            ].copy(),
                            "loop_ids": ids,
                            "merged_from": 1,
                        }
                    )
                continue
        items.append(
            {
                "points": _as_float_array(envelope).copy(),
                "loop_ids": None,
                "merged_from": int(len(cluster_loops)),
            }
        )
    meta["bridge_links"] = bridge_links
    return items, meta


def extract_island_longest_sides(
    region_ids: np.ndarray,
    target_id,
    vertices: np.ndarray,
    loop_start: np.ndarray,
    loop_total: np.ndarray,
    loop_vertex_indices: np.ndarray,
    corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
    max_sides: int = 4,
    bezier_samples: int = 24,
    stage: str = "STITCH",
    stitch_gap_frac: float = _MULTI_ISLAND_MERGE_GAP_FRAC,
    bridge_gap_frac: float = 0.25,
    bridge_enabled: bool = True,
    min_perimeter_frac: float = _MIN_ISLAND_PERIMETER_FRAC,
    interior_points: np.ndarray | None = None,
) -> dict:
    """
    对领域内每个 island 边界环按折角拆成段并拟合。

    stage 控制多岛策略：ISLANDS / STITCH / BRIDGE。
    每条平滑折线段或曲线段分配独立随机颜色（segment_colors）。
    max_sides 保留兼容，调试拆分不再使用它压边数。
    """
    _ = max_sides  # 兼容旧调用签名；按折角拆分不压边数
    verts = _as_float_array(vertices)
    edge_faces = build_edge_face_adjacency(
        loop_start,
        loop_total,
        loop_vertex_indices,
    )
    loops = extract_region_boundary_loops(
        region_ids,
        target_id,
        loop_start,
        loop_total,
        loop_vertex_indices,
    )
    loops = filter_significant_boundary_loops(
        loops,
        verts,
        min_perimeter_frac=min_perimeter_frac,
    )
    if not loops:
        raise RegionFitError("过滤极小碎环后无可用边界")

    work_items, stage_meta = build_fit_work_items(
        loops,
        verts,
        stage=stage,
        stitch_gap_frac=stitch_gap_frac,
        bridge_gap_frac=bridge_gap_frac,
        bridge_enabled=bridge_enabled,
        interior_points=interior_points,
    )
    if not work_items:
        raise RegionFitError("过滤极小碎环后无可用边界")

    sample_n = max(int(bezier_samples), 4)
    island_drafts: list[dict] = []

    for island_index, item in enumerate(work_items):
        loop_pts = _as_float_array(item["points"])
        loop_ids = item.get("loop_ids")
        if len(loop_pts) < 3:
            continue
        origin, axis_u, axis_v = _loop_pca_basis(loop_pts)
        pts_2d_raw = project_points_to_plane(loop_pts, origin, axis_u, axis_v)
        if _signed_area_2d(pts_2d_raw) < 0.0:
            loop_pts = loop_pts[::-1]
            if loop_ids is not None:
                loop_ids = list(reversed(loop_ids))

        # 内角>355°的尖刺收束到上一级交点（所有拟合阶段）
        collapsed = collapse_ultra_reflex_spike_vertices_closed_loop(loop_pts)
        if len(collapsed) != len(loop_pts) or not np.allclose(
            collapsed, loop_pts, atol=1e-9
        ):
            loop_pts = collapsed
            loop_ids = None

        # 仅原始网格环可做邻域 T 接缝切点；融并外轮廓只靠几何折角
        if loop_ids is not None:
            neighbor_ids = boundary_loop_neighbor_ids(
                loop_ids,
                region_ids,
                edge_faces,
                target_id,
            )
            neighbor_vertices = neighbor_change_vertex_indices(neighbor_ids)
        else:
            neighbor_vertices = []

        sample_count = int(
            np.clip(
                len(loop_pts) * 2,
                _BOUNDARY_SAMPLES_MIN,
                _BOUNDARY_SAMPLES_MAX,
            )
        )
        loop_rs = resample_closed_polyline(loop_pts, sample_count)
        # 重采样后可能再次出现圆滑尖刺，再收束一次
        loop_rs = collapse_ultra_reflex_spike_vertices_closed_loop(loop_rs)
        if len(loop_rs) < 3:
            continue
        sample_count = len(loop_rs)
        pts_2d = project_points_to_plane(loop_rs, origin, axis_u, axis_v)
        normal_guess = _normalize(np.cross(axis_u, axis_v))
        angle_labels = collect_interior_angle_labels(
            loop_rs,
            min_deviation_deg=12.0,
            normal=normal_guess,
        )
        corners = detect_corner_indices(
            pts_2d,
            corner_angle_deg,
            max_corners=_MAX_SPLIT_CORNERS,
            convex_only=False,
            include_strong_concave=True,
        )
        fold_indices = detect_concave_fold_indices(
            pts_2d,
            fold_angle_deg=corner_angle_deg,
            max_folds=_MAX_CONCAVE_FOLDS,
        )
        neighbor_splits = (
            map_loop_indices_to_resampled(
                loop_pts,
                neighbor_vertices,
                sample_count,
            )
            if neighbor_vertices
            else []
        )
        split_indices = sorted(
            set(corners) | set(fold_indices) | set(neighbor_splits)
        )
        if len(split_indices) < 2:
            half = max(sample_count // 2, 1)
            split_indices = [0, half]

        island_folds = [loop_rs[index].copy() for index in fold_indices]
        ordered_splits = sorted(
            {int(i) % sample_count for i in split_indices}
        )
        sides_3d = split_loop_into_sides(loop_rs, ordered_splits)
        neighbor_split_set = {int(i) % sample_count for i in neighbor_splits}
        protected = [
            side_index
            for side_index, _start in enumerate(ordered_splits)
            if ordered_splits[(side_index + 1) % len(ordered_splits)]
            in neighbor_split_set
        ]
        sides_3d = merge_collinear_adjacent_sides(
            sides_3d,
            max_turn_deg=corner_angle_deg,
            min_sides=2,
            protected_junctions=protected,
        )
        sides_3d = share_closed_side_endpoints(sides_3d)
        sides_3d = resplit_sides_at_interior_folds(
            sides_3d,
            origin,
            axis_u,
            axis_v,
            fold_angle_deg=corner_angle_deg,
        )
        sides_3d = share_closed_side_endpoints(sides_3d)
        loop_len = float(polyline_length(np.vstack((loop_rs, loop_rs[:1]))))
        sides_3d = merge_short_boundary_sides(
            sides_3d,
            min_length=max(loop_len * _MIN_SEGMENT_LOOP_FRAC, 1e-4),
            min_sides=2,
        )
        sides_3d = share_closed_side_endpoints(sides_3d)
        if neighbor_vertices:
            neighbor_points = [
                loop_pts[int(index) % len(loop_pts)].copy()
                for index in neighbor_vertices
            ]
            sides_3d = resplit_sides_near_points(
                sides_3d,
                neighbor_points,
                max_dist=max(loop_len * 0.01, 1e-4),
            )
            sides_3d = share_closed_side_endpoints(sides_3d)

        for side in sides_3d:
            if len(side) < 5:
                continue
            side_2d = project_points_to_plane(side, origin, axis_u, axis_v)
            mid_folds = detect_side_fold_indices(
                side_2d,
                fold_angle_deg=corner_angle_deg,
                min_separation_frac=0.08,
            )
            for fold_index in mid_folds:
                island_folds.append(side[int(fold_index)].copy())

        if island_folds:
            deduped: list[np.ndarray] = []
            min_fold_dist = max(loop_len * 0.015, 1e-4)
            for point in island_folds:
                if all(
                    float(np.linalg.norm(point - existing)) >= min_fold_dist
                    for existing in deduped
                ):
                    deduped.append(point)
            island_folds = deduped

        island_drafts.append(
            {
                "island_index": int(island_index),
                "sides_3d": sides_3d,
                "origin": origin,
                "axis_u": axis_u,
                "axis_v": axis_v,
                "split_count": int(len(ordered_splits)),
                "island_folds": island_folds,
                "merged_from": int(item.get("merged_from", 1)),
                "angle_labels": angle_labels,
            }
        )

    if not island_drafts:
        raise RegionFitError("未提取到任何 island 边界边")

    islands: list[dict] = []
    wire_vertices: list[np.ndarray] = []
    wire_edges: list[tuple[int, int]] = []
    wire_color_ids: list[int] = []
    control_points: list[np.ndarray] = []
    control_color_ids: list[int] = []
    concave_fold_points: list[np.ndarray] = []
    interior_angle_labels: list[dict] = []
    segment_colors: list[tuple[float, float, float, float]] = []
    vertex_cursor = 0
    all_points: list[np.ndarray] = []

    for draft in island_drafts:
        island_index = int(draft["island_index"])
        sides_3d = draft["sides_3d"]
        origin = draft["origin"]
        axis_u = draft["axis_u"]
        axis_v = draft["axis_v"]
        island_folds = draft["island_folds"]
        concave_fold_points.extend(island_folds)
        interior_angle_labels.extend(draft.get("angle_labels") or [])

        beziers: list[dict] = []
        sides_sampled: list[np.ndarray] = []
        lengths: list[float] = []

        def _append_polyline_segment(
            polyline: np.ndarray,
            side_index: int,
            _island_index: int = island_index,
        ) -> None:
            nonlocal vertex_cursor
            poly = _as_float_array(polyline).copy()
            if len(poly) < 2:
                return
            color_id = len(segment_colors)
            seed = (
                0xA11E0000
                ^ (int(_island_index) << 16)
                ^ (int(side_index) << 8)
                ^ int(color_id)
            )
            segment_colors.append(make_segment_color(seed))
            sampled = resample_polyline(poly, max(sample_n, len(poly)))
            sampled = sampled.copy()
            sampled[0] = poly[0]
            sampled[-1] = poly[-1]
            length = float(polyline_length(poly))
            beziers.append(
                {
                    "fit_mode": "POLYLINE",
                    "polyline": poly,
                    "spans": None,
                    "color_id": color_id,
                    "length": length,
                    "side_index": int(side_index),
                }
            )
            sides_sampled.append(sampled)
            lengths.append(length)
            all_points.append(sampled)
            all_points.append(poly)
            wire_vertices.append(sampled)
            for offset in range(len(sampled) - 1):
                wire_edges.append(
                    (vertex_cursor + offset, vertex_cursor + offset + 1)
                )
                wire_color_ids.append(color_id)
            vertex_cursor += len(sampled)
            for point in poly:
                control_points.append(point)
                control_color_ids.append(color_id)

        def _append_curve_segment(
            side: np.ndarray,
            spans: list[np.ndarray],
            side_index: int,
            _island_index: int = island_index,
        ) -> None:
            nonlocal vertex_cursor
            color_id = len(segment_colors)
            seed = (
                0xA11E0000
                ^ (int(_island_index) << 16)
                ^ (int(side_index) << 8)
                ^ int(color_id)
            )
            segment_colors.append(make_segment_color(seed))
            sampled_parts = [
                sample_cubic_bezier(
                    controls,
                    max(sample_n // max(len(spans), 1), 6),
                )
                for controls in spans
            ]
            sampled_list = [sampled_parts[0]]
            for part in sampled_parts[1:]:
                sampled_list.append(part[1:])
            sampled = np.vstack(sampled_list).copy()
            sampled[0] = side[0]
            sampled[-1] = side[-1]
            length = float(polyline_length(sampled))
            beziers.append(
                {
                    "fit_mode": "CURVE",
                    "polyline": None,
                    "spans": spans,
                    "color_id": color_id,
                    "length": length,
                    "side_index": int(side_index),
                }
            )
            sides_sampled.append(sampled)
            lengths.append(length)
            all_points.append(sampled)
            for span_controls in spans:
                all_points.append(span_controls)
            wire_vertices.append(sampled)
            for offset in range(len(sampled) - 1):
                wire_edges.append(
                    (vertex_cursor + offset, vertex_cursor + offset + 1)
                )
                wire_color_ids.append(color_id)
            vertex_cursor += len(sampled)
            control_points.append(side[0])
            control_color_ids.append(color_id)
            handle_limit = float(
                max(polyline_length(side) * _HANDLE_OUTLIER_FRAC, 1e-4)
            )
            raw_handles = []
            for span_controls in spans:
                raw_handles.append(span_controls[1])
                raw_handles.append(span_controls[2])
            for handle in filter_handle_outliers(raw_handles, side, handle_limit):
                control_points.append(handle)
                control_color_ids.append(color_id)
            control_points.append(side[-1])
            control_color_ids.append(color_id)

        for side_index, side in enumerate(sides_3d):
            if side_index in draft.get("suppress_sides", set()):
                continue
            side_2d = project_points_to_plane(side, origin, axis_u, axis_v)
            fit_mode, key_or_side, spans = classify_side_fit_mode(
                side,
                points_2d=side_2d,
                fold_angle_deg=corner_angle_deg,
            )
            if fit_mode == "POLYLINE":
                polyline = _as_float_array(key_or_side).copy()
                polyline[0] = side[0]
                polyline[-1] = side[-1]
                for segment in split_polyline_at_significant_folds(
                    polyline,
                    fold_angle_deg=corner_angle_deg,
                ):
                    _append_polyline_segment(segment, side_index)
                continue

            assert spans is not None
            _append_curve_segment(side, spans, side_index)

        islands.append(
            {
                "island_index": int(island_index),
                "sides": sides_sampled,
                "lengths": lengths,
                "corner_count": int(draft["split_count"]),
                "beziers": beziers,
                "concave_fold_count": int(len(island_folds)),
                "concave_fold_points": island_folds,
                "merged_from": int(draft.get("merged_from", 1)),
            }
        )

    if not islands:
        raise RegionFitError("未提取到任何 island 边界边")
    if not wire_vertices:
        raise RegionFitError("island 边界边为空")

    stacked = np.vstack(all_points)
    extent = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    bevel_depth = float(max(extent * 0.004, 1e-4))
    control_radius = float(max(extent * 0.006, bevel_depth * 1.5))
    fold_radius = float(max(extent * 0.018, control_radius * 3.0))

    return {
        "islands": islands,
        "island_count": len(islands),
        "wire_vertices": np.vstack(wire_vertices),
        "wire_edges": wire_edges,
        "wire_color_ids": wire_color_ids,
        "control_points": np.asarray(control_points, dtype=np.float64),
        "control_color_ids": control_color_ids,
        "segment_colors": segment_colors,
        "concave_fold_points": (
            np.asarray(concave_fold_points, dtype=np.float64)
            if concave_fold_points
            else np.zeros((0, 3), dtype=np.float64)
        ),
        "interior_angle_labels": interior_angle_labels,
        "bevel_depth": bevel_depth,
        "control_radius": control_radius,
        "fold_radius": fold_radius,
        "stage": stage_meta.get("stage", stage),
        "stitch_gap": float(stage_meta.get("stitch_gap", 0.0)),
        "bridge_gap": float(stage_meta.get("bridge_gap", 0.0)),
        "bridge_links": stage_meta.get("bridge_links", []),
        "extent": float(stage_meta.get("extent", extent)),
    }



def _junction_turn_deg(side_a: np.ndarray, side_b: np.ndarray) -> float:
    """两条邻接边在共享角点处的转角（度）。"""
    a = _as_float_array(side_a)
    b = _as_float_array(side_b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    incoming = a[-1] - a[-2]
    outgoing = b[1] - b[0]
    in_len = float(np.linalg.norm(incoming))
    out_len = float(np.linalg.norm(outgoing))
    if in_len < 1e-12 or out_len < 1e-12:
        return 0.0
    cos_angle = float(
        np.clip(np.dot(incoming, outgoing) / (in_len * out_len), -1.0, 1.0)
    )
    return float(np.degrees(np.arccos(cos_angle)))


def merge_collinear_adjacent_sides(
    sides: Sequence[np.ndarray],
    max_turn_deg: float = DEFAULT_CORNER_ANGLE_DEG,
    min_sides: int = 3,
    protected_junctions: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """
    合并转角小于阈值的邻接边，把被误拆的连续长边接回去。

    只合并平缓接头，保留尖角；避免「最长四边」丢掉中间短段后出现断口。
    protected_junctions: 禁止合并的接头下标（边 i 与 i+1 之间），
    用于保留邻域 ID 变化造成的 T 接缝切点。
    """
    result = [np.asarray(side, dtype=np.float64).copy() for side in sides]
    if len(result) <= int(min_sides):
        return result
    threshold = float(max(max_turn_deg, 1.0))
    protected = {int(i) % max(len(result), 1) for i in (protected_junctions or [])}
    guard = 0
    while len(result) > int(min_sides):
        turns = [
            _junction_turn_deg(result[i], result[(i + 1) % len(result)])
            for i in range(len(result))
        ]
        candidates = [
            (float(turns[i]), i)
            for i in range(len(result))
            if i not in protected
        ]
        if not candidates:
            break
        _turn, merge_left = min(candidates, key=lambda item: item[0])
        if turns[merge_left] >= threshold:
            break
        merge_right = (merge_left + 1) % len(result)
        merged = np.vstack((result[merge_left][:-1], result[merge_right]))
        if merge_right > merge_left:
            result[merge_left] = merged
            del result[merge_right]
            new_protected: set[int] = set()
            for junction in protected:
                if junction == merge_left:
                    continue
                if junction > merge_left:
                    new_protected.add(junction - 1)
                else:
                    new_protected.add(junction)
            protected = {j % max(len(result), 1) for j in new_protected}
        else:
            result = [merged] + result[1:merge_left]
            protected = set()
        guard += 1
        if guard > len(sides) + 8:
            break
    return result


def merge_paired_collinear_adjacent_sides(
    sides_2d: Sequence[np.ndarray],
    sides_3d: Sequence[np.ndarray],
    max_turn_deg: float = DEFAULT_CORNER_ANGLE_DEG,
    min_sides: int = 3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """按 3D 转角同步合并近共线的 2D/3D 邻接边。"""
    result_2d = [np.asarray(side, dtype=np.float64).copy() for side in sides_2d]
    result_3d = [np.asarray(side, dtype=np.float64).copy() for side in sides_3d]
    if len(result_2d) != len(result_3d):
        raise RegionFitError("2D/3D 边数不一致，无法同步合并")
    if len(result_3d) <= int(min_sides):
        return result_2d, result_3d
    threshold = float(max(max_turn_deg, 1.0))
    guard = 0
    n0 = len(result_3d)
    while len(result_3d) > int(min_sides):
        turns = [
            _junction_turn_deg(result_3d[i], result_3d[(i + 1) % len(result_3d)])
            for i in range(len(result_3d))
        ]
        merge_left = int(np.argmin(turns))
        if turns[merge_left] >= threshold:
            break
        merge_right = (merge_left + 1) % len(result_3d)
        merged_2d = np.vstack(
            (result_2d[merge_left][:-1], result_2d[merge_right])
        )
        merged_3d = np.vstack(
            (result_3d[merge_left][:-1], result_3d[merge_right])
        )
        if merge_right > merge_left:
            result_2d[merge_left] = merged_2d
            result_3d[merge_left] = merged_3d
            del result_2d[merge_right]
            del result_3d[merge_right]
        else:
            result_2d = [merged_2d] + result_2d[1:merge_left]
            result_3d = [merged_3d] + result_3d[1:merge_left]
        guard += 1
        if guard > n0 + 8:
            break
    return result_2d, result_3d


def reduce_sides_to_count(
    sides: Sequence[np.ndarray],
    target_count: int,
    max_merge_turn_deg: float | None = None,
) -> list[np.ndarray]:
    """
    将边数降到 target_count，保持闭环首尾相接。

    优先合并转角最平缓的邻接边对。若提供 max_merge_turn_deg，
    则拒绝合并更锐的尖角，宁可保留多于 target 的边，也不制造 L 形折边。
    """
    result = [np.asarray(side, dtype=np.float64).copy() for side in sides]
    target = max(int(target_count), 3)
    merge_limit = (
        None
        if max_merge_turn_deg is None
        else float(max(max_merge_turn_deg, 1.0))
    )
    while len(result) > target:
        turns = [
            _junction_turn_deg(result[i], result[(i + 1) % len(result)])
            for i in range(len(result))
        ]
        merge_left = int(np.argmin(turns))
        if merge_limit is not None and turns[merge_left] >= merge_limit:
            break
        merge_right = (merge_left + 1) % len(result)
        merged = np.vstack((result[merge_left][:-1], result[merge_right]))
        if merge_right > merge_left:
            result[merge_left] = merged
            del result[merge_right]
        else:
            # 环尾与环首相接：merged 替换两侧后只留一条
            result = [merged] + result[1:merge_left]
    return result


def reduce_paired_sides_to_count(
    sides_2d: Sequence[np.ndarray],
    sides_3d: Sequence[np.ndarray],
    target_count: int,
    max_merge_turn_deg: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """按 3D 邻接转角同步合并 2D/3D 边，降到 target_count。"""
    result_2d = [np.asarray(side, dtype=np.float64).copy() for side in sides_2d]
    result_3d = [np.asarray(side, dtype=np.float64).copy() for side in sides_3d]
    if len(result_2d) != len(result_3d):
        raise RegionFitError("2D/3D 边数不一致，无法同步合并")
    target = max(int(target_count), 3)
    merge_limit = (
        None
        if max_merge_turn_deg is None
        else float(max(max_merge_turn_deg, 1.0))
    )
    while len(result_3d) > target:
        turns = [
            _junction_turn_deg(result_3d[i], result_3d[(i + 1) % len(result_3d)])
            for i in range(len(result_3d))
        ]
        merge_left = int(np.argmin(turns))
        if merge_limit is not None and turns[merge_left] >= merge_limit:
            break
        merge_right = (merge_left + 1) % len(result_3d)
        merged_2d = np.vstack(
            (result_2d[merge_left][:-1], result_2d[merge_right])
        )
        merged_3d = np.vstack(
            (result_3d[merge_left][:-1], result_3d[merge_right])
        )
        if merge_right > merge_left:
            result_2d[merge_left] = merged_2d
            result_3d[merge_left] = merged_3d
            del result_2d[merge_right]
            del result_3d[merge_right]
        else:
            result_2d = [merged_2d] + result_2d[1:merge_left]
            result_3d = [merged_3d] + result_3d[1:merge_left]
    return result_2d, result_3d


def classify_tri_or_quad(
    sides: Sequence[np.ndarray],
    triangle_ratio: float = DEFAULT_TRIANGLE_RATIO,
) -> tuple[str, list[np.ndarray]]:
    """
    保留最多四条主边；第四边（最短边）远短于第二短边时视为三边。

    以第二短边为基准而非最长边：扁长四边域（如弯曲条带）的两条
    短边长度接近，不会被误判成三边；真三角域的第四边是远小于
    其余三条边的碎边，仍会正确触发三边判定。
    """
    if len(sides) < 3:
        raise RegionFitError("边数不足，无法拟合")
    working = [np.asarray(side, dtype=np.float64) for side in sides]
    if len(working) > 4:
        working = reduce_sides_to_count(working, 4)
    lengths = [polyline_length(side) for side in working]
    if len(working) == 3:
        return "TRI", working
    order = sorted(range(len(working)), key=lambda i: lengths[i], reverse=True)
    second_shortest = lengths[order[2]]
    fourth = lengths[order[3]]
    ratio = max(float(triangle_ratio), 0.0)
    if second_shortest > 1e-12 and fourth < second_shortest * ratio:
        short_index = order[3]
        # 丢弃最短边后，把其两端并入相邻边，保持环序三边
        left = (short_index - 1) % 4
        right = (short_index + 1) % 4
        tip = 0.5 * (working[short_index][0] + working[short_index][-1])
        kept: list[np.ndarray] = []
        for index, side in enumerate(working):
            if index == short_index:
                continue
            side = side.copy()
            if index == left:
                side[-1] = tip
            if index == right:
                side[0] = tip
            kept.append(side)
        return "TRI", kept
    return "QUAD", working


def bridge_concave_notches(
    sides_2d: Sequence[np.ndarray],
    sides_3d: Sequence[np.ndarray] | None = None,
    angle_threshold_deg: float = DEFAULT_CORNER_ANGLE_DEG,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """
    对边链内部的凹口「沿拟合线」桥接。

    仅当边链内部存在明显折角、且该子链整体偏向领域内侧时，
    才用直线段桥接；平缓的整体弧形（如弯曲条带的内弧边）保持不变。
    要求闭环为逆时针方向：行进方向左侧为领域内侧。
    """
    out_2d: list[np.ndarray] = []
    out_3d: list[np.ndarray] | None = [] if sides_3d is not None else None
    threshold = float(max(angle_threshold_deg, 1.0))

    for index, raw_side in enumerate(sides_2d):
        side_2d = _as_float_array(raw_side).copy()
        side_3d = (
            _as_float_array(sides_3d[index]).copy()
            if sides_3d is not None
            else None
        )

        def _push() -> None:
            out_2d.append(side_2d)
            if out_3d is not None:
                out_3d.append(side_3d)

        count = len(side_2d)
        if count < 5:
            _push()
            continue

        w = max(1, count // 16)
        sharp: list[int] = []
        for i in range(1, count - 1):
            a = side_2d[max(i - w, 0)]
            b = side_2d[i]
            c = side_2d[min(i + w, count - 1)]
            vin = b - a
            vout = c - b
            in_len = float(np.linalg.norm(vin))
            out_len = float(np.linalg.norm(vout))
            if in_len < 1e-12 or out_len < 1e-12:
                continue
            cos_angle = float(
                np.clip(vin.dot(vout) / (in_len * out_len), -1.0, 1.0)
            )
            if float(np.degrees(np.arccos(cos_angle))) >= threshold:
                sharp.append(i)
        if not sharp:
            _push()
            continue

        i0 = max(sharp[0] - w, 1)
        i1 = min(sharp[-1] + w, count - 2)
        if i1 - i0 < 1:
            _push()
            continue

        anchor_a = side_2d[i0]
        anchor_b = side_2d[i1]
        chord = anchor_b - anchor_a
        chord_len = float(np.linalg.norm(chord))
        if chord_len < 1e-12:
            _push()
            continue
        # 逆时针环行进方向左侧为领域内侧
        left = np.array([-chord[1], chord[0]], dtype=np.float64) / chord_len
        segment = side_2d[i0 : i1 + 1]
        deviation = float(np.mean((segment - anchor_a) @ left))
        if deviation <= chord_len * 0.01:
            # 子链向外凸出（属于领域本体），保留原边界
            _push()
            continue

        t = np.linspace(0.0, 1.0, i1 - i0 + 1, dtype=np.float64)[:, None]
        side_2d[i0 : i1 + 1] = anchor_a + t * chord
        if side_3d is not None:
            a3 = side_3d[i0]
            b3 = side_3d[i1]
            side_3d[i0 : i1 + 1] = a3 + t * (b3 - a3)
        _push()

    return out_2d, out_3d


def extend_quad_corners_by_tangents(
    sides_2d: Sequence[np.ndarray],
    sides_3d: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    将相邻主边端部切线延长到交点，形成规整四边壁面。

    单 island 的凹多边形边界常沿扫描分割凹口收缩；这里只调整四个
    结构角点，保留每条边内部曲线，让长边按端部方向顺延。
    """
    if len(sides_2d) != 4 or len(sides_3d) != 4:
        return (
            [np.asarray(side, dtype=np.float64).copy() for side in sides_2d],
            [np.asarray(side, dtype=np.float64).copy() for side in sides_3d],
        )

    out2 = [np.asarray(side, dtype=np.float64).copy() for side in sides_2d]
    out3 = [np.asarray(side, dtype=np.float64).copy() for side in sides_3d]
    side_lengths = [max(polyline_length(side), 1e-9) for side in out2]
    extend_limit = float(np.median(side_lengths)) * 0.9

    def _endpoint_tangent(
        points: np.ndarray,
        at_end: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64)
        count = len(pts)
        if count < 2:
            return pts[-1 if at_end else 0], np.zeros(2, dtype=np.float64)
        window = min(max(count // 6, 2), 8, count)
        segment = pts[-window:] if at_end else pts[:window]
        center = segment.mean(axis=0)
        _, _, vh = np.linalg.svd(segment - center, full_matrices=False)
        direction = _normalize(np.array([vh[0, 0], vh[0, 1], 0.0]))[:2]
        reference = (pts[-1] - pts[-2]) if at_end else (pts[1] - pts[0])
        if float(direction.dot(reference)) < 0.0:
            direction = -direction
        return pts[-1 if at_end else 0], direction

    def _line_intersection(
        p: np.ndarray,
        d: np.ndarray,
        q: np.ndarray,
        e: np.ndarray,
    ) -> tuple[float, float, np.ndarray] | None:
        mat = np.column_stack((d, -e))
        det = float(np.linalg.det(mat))
        if abs(det) < 1e-8:
            return None
        t, u = np.linalg.solve(mat, q - p)
        point = p + float(t) * d
        return float(t), float(u), point

    def _endpoint_tangent_3d(
        points3: np.ndarray,
        points2: np.ndarray,
        at_end: bool,
    ) -> np.ndarray:
        if len(points3) < 2 or len(points2) < 2:
            return np.zeros(3, dtype=np.float64)
        if at_end:
            d2 = points2[-1] - points2[-2]
            d3 = points3[-1] - points3[-2]
        else:
            d2 = points2[1] - points2[0]
            d3 = points3[1] - points3[0]
        scale = max(float(np.linalg.norm(d2)), 1e-9)
        return d3 / scale

    for index in range(4):
        next_index = (index + 1) % 4
        p, d = _endpoint_tangent(out2[index], at_end=True)
        q, e = _endpoint_tangent(out2[next_index], at_end=False)
        if float(np.linalg.norm(d)) < 1e-9 or float(np.linalg.norm(e)) < 1e-9:
            continue
        hit = _line_intersection(p, d, q, e)
        if hit is None:
            continue
        t, u, point2 = hit
        current = 0.5 * (p + q)
        move = float(np.linalg.norm(point2 - current))
        if move < 1e-6 or move > extend_limit:
            continue
        if abs(t) > extend_limit or abs(u) > extend_limit:
            continue

        d3_prev = _endpoint_tangent_3d(out3[index], out2[index], at_end=True)
        d3_next = _endpoint_tangent_3d(
            out3[next_index],
            out2[next_index],
            at_end=False,
        )
        point3_prev = out3[index][-1] + t * d3_prev
        point3_next = out3[next_index][0] + u * d3_next
        point3 = 0.5 * (point3_prev + point3_next)
        out2[index][-1] = point2
        out2[next_index][0] = point2
        out3[index][-1] = point3
        out3[next_index][0] = point3

    return out2, out3


def _signed_area_2d(points: np.ndarray) -> float:
    pts = _as_float_array(points)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def coons_patch(
    bottom: np.ndarray,
    right: np.ndarray,
    top: np.ndarray,
    left: np.ndarray,
) -> np.ndarray:
    """
    双线性 Coons 插值。

    输入边需已统一采样：
      bottom/top: (nu+1, 3)
      left/right: (nv+1, 3)
    返回 vertices: (nv+1, nu+1, 3)
    """
    c0 = _as_float_array(bottom)
    c1 = _as_float_array(top)
    d0 = _as_float_array(left)
    d1 = _as_float_array(right)
    if len(c0) != len(c1) or len(d0) != len(d1):
        raise RegionFitError("Coons 对边采样数量必须一致")
    if len(c0) < 2 or len(d0) < 2:
        raise RegionFitError("Coons 边采样过少")

    # 角点对齐修正
    c0 = c0.copy()
    c1 = c1.copy()
    d0 = d0.copy()
    d1 = d1.copy()
    c0[0] = d0[0]
    c0[-1] = d1[0]
    c1[0] = d0[-1]
    c1[-1] = d1[-1]

    nu = len(c0) - 1
    nv = len(d0) - 1
    u = np.linspace(0.0, 1.0, nu + 1, dtype=np.float64)
    v = np.linspace(0.0, 1.0, nv + 1, dtype=np.float64)
    grid = np.empty((nv + 1, nu + 1, 3), dtype=np.float64)

    p00 = c0[0]
    p10 = c0[-1]
    p01 = c1[0]
    p11 = c1[-1]

    for j in range(nv + 1):
        for i in range(nu + 1):
            uu = u[i]
            vv = v[j]
            ruled_u = (1.0 - vv) * c0[i] + vv * c1[i]
            ruled_v = (1.0 - uu) * d0[j] + uu * d1[j]
            bilinear = (
                (1.0 - uu) * (1.0 - vv) * p00
                + uu * (1.0 - vv) * p10
                + (1.0 - uu) * vv * p01
                + uu * vv * p11
            )
            grid[j, i] = ruled_u + ruled_v - bilinear
    return grid


def patch_grid_faces(nu: int, nv: int) -> list[tuple[int, int, int, int]]:
    """规则四边形面索引（行优先）。"""
    faces: list[tuple[int, int, int, int]] = []
    for j in range(nv):
        for i in range(nu):
            v00 = j * (nu + 1) + i
            v10 = v00 + 1
            v01 = (j + 1) * (nu + 1) + i
            v11 = v01 + 1
            faces.append((v00, v10, v11, v01))
    return faces


def triangular_patch_faces(
    nu: int,
    nv: int,
) -> list[tuple[int, ...]]:
    """
    顶端退化四边网格的面：最上行合并为三角形，其余为四边形。

    顶点布局与 coons_patch 相同，但最后一行全部映射到同一 tip 索引前
    需要先在顶点层合并；本函数假设 tip 行已折叠，顶点按
    rows 0..nv-1 完整，外加 1 个 tip 顶点。
    """
    faces: list[tuple[int, ...]] = []
    width = nu + 1
    # 完整四边形行：0 .. nv-2
    for j in range(max(nv - 1, 0)):
        for i in range(nu):
            v00 = j * width + i
            v10 = v00 + 1
            v01 = (j + 1) * width + i
            v11 = v01 + 1
            faces.append((v00, v10, v11, v01))
    if nv >= 1:
        tip = nv * width  # 单一 tip 顶点放在折叠后数组末尾
        base_row = (nv - 1) * width
        for i in range(nu):
            faces.append((base_row + i, base_row + i + 1, tip))
    return faces


def build_triangular_patch(
    side_a: np.ndarray,
    side_b: np.ndarray,
    side_c: np.ndarray,
    segments_long: int,
    segments_base: int,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """
    三边拟合：两条长边同分段，底边单独分段。

    side_a/side_b 为两条长边（从底角指向 tip），side_c 为底边。
    """
    seg_v = max(int(segments_long), MIN_SEGMENTS)
    seg_u = max(int(segments_base), MIN_SEGMENTS)
    left = resample_polyline(side_a, seg_v + 1)
    right = resample_polyline(side_b, seg_v + 1)
    bottom = resample_polyline(side_c, seg_u + 1)
    tip = 0.5 * (left[-1] + right[-1])
    left[-1] = tip
    right[-1] = tip
    top = np.repeat(tip[None, :], seg_u + 1, axis=0)
    grid = coons_patch(bottom, right, top, left)

    # 折叠 tip 行
    body = grid[:-1].reshape(-1, 3)
    vertices = np.vstack((body, tip[None, :]))
    faces = triangular_patch_faces(seg_u, seg_v)
    return vertices, faces


def soft_snap_quad_grid_to_points(
    vertices: np.ndarray,
    segments_u: int,
    segments_v: int,
    samples: np.ndarray,
    *,
    strength: float = 0.95,
) -> np.ndarray:
    """
    将 Coons 规则网格吸附到表面采样点，减轻弦面抄近路导致的角部悬空。

    每个顶点取邻域样本中位数目标，按强度混合；边界顶点允许有限外扩。
    最后做一次四邻域平滑，抑制折角。
    """
    su = max(int(segments_u), MIN_SEGMENTS)
    sv = max(int(segments_v), MIN_SEGMENTS)
    grid = _as_float_array(vertices).reshape((sv + 1, su + 1, 3)).copy()
    pts = _as_float_array(samples)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    if len(pts) < 8:
        return grid.reshape(-1, 3)
    if len(pts) > 12000:
        pts = pts[:: int(np.ceil(len(pts) / 12000))]

    blend = float(np.clip(strength, 0.0, 1.0))
    # 允许位移上限：相对网格尺度
    scale = float(
        np.median(np.linalg.norm(np.diff(grid[0], axis=0), axis=1))
        + np.median(np.linalg.norm(grid[-1] - grid[0], axis=1))
    )
    max_move = max(scale * 1.25, 1e-6)
    k_neigh = 12

    snapped = grid.copy()
    flat = grid.reshape(-1, 3)
    # 批量：对每个网格点找 k 近邻（网格很小，可暴力）
    for idx in range(len(flat)):
        g = flat[idx]
        d2 = ((pts - g) ** 2).sum(axis=1)
        if len(d2) > k_neigh:
            nn = np.argpartition(d2, k_neigh)[:k_neigh]
        else:
            nn = np.arange(len(d2))
        target = np.median(pts[nn], axis=0)
        delta = target - g
        dist = float(np.linalg.norm(delta))
        if dist > max_move:
            delta *= max_move / dist
        j, i = divmod(idx, su + 1)
        on_border = j in (0, sv) or i in (0, su)
        w = blend * (0.7 if on_border else 1.0)
        snapped[j, i] = g + w * delta

    # 一次轻量平滑（含边界，保持规则网格）
    smooth = snapped.copy()
    for j in range(sv + 1):
        for i in range(su + 1):
            acc = snapped[j, i].copy()
            n = 1.0
            if i > 0:
                acc += snapped[j, i - 1]
                n += 1.0
            if i < su:
                acc += snapped[j, i + 1]
                n += 1.0
            if j > 0:
                acc += snapped[j - 1, i]
                n += 1.0
            if j < sv:
                acc += snapped[j + 1, i]
                n += 1.0
            smooth[j, i] = 0.65 * snapped[j, i] + 0.35 * (acc / n)

    return smooth.reshape(-1, 3)


def build_quad_patch(
    sides: Sequence[np.ndarray],
    segments_u: int,
    segments_v: int,
    *,
    shared_param: bool = False,
    surface_samples: np.ndarray | None = None,
    snap_strength: float = 0.85,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """
    四边 Coons：对边同分段。

    shared_param=True 时对边按下标重采样（条带内外弧共享 u 参数），
    否则按弧长重采样。若提供 surface_samples，则对网格做列向软吸附。
    """
    if len(sides) != 4:
        raise RegionFitError("四边拟合需要恰好四条边")
    seg_u = max(int(segments_u), MIN_SEGMENTS)
    seg_v = max(int(segments_v), MIN_SEGMENTS)
    resample = resample_polyline_by_index if shared_param else resample_polyline
    bottom = resample(sides[0], seg_u + 1)
    right = resample(sides[1], seg_v + 1)
    top = resample(sides[2][::-1], seg_u + 1)
    left = resample(sides[3][::-1], seg_v + 1)
    grid = coons_patch(bottom, right, top, left)
    vertices = grid.reshape(-1, 3)
    if surface_samples is not None and len(surface_samples) > 0:
        vertices = soft_snap_quad_grid_to_points(
            vertices,
            seg_u,
            seg_v,
            surface_samples,
            strength=snap_strength,
        )
    faces = patch_grid_faces(seg_u, seg_v)
    return vertices, faces


def _orient_faces(
    vertices: np.ndarray,
    faces: Sequence[tuple[int, ...]],
    reference_normal: np.ndarray,
) -> list[tuple[int, ...]]:
    """
    统一面朝向：网格由参数域构建，绕序本身一致，
    只按面积加权法线和与参考法线的点积决定整体是否翻转。
    逐面翻转会在强弯曲曲面上破坏绕序一致性，禁止使用。
    """
    ref = _normalize(_as_float_array(reference_normal))
    verts = _as_float_array(vertices)
    total = np.zeros(3, dtype=np.float64)
    cleaned = [tuple(face) for face in faces if len(face) >= 3]
    for face in cleaned:
        a = verts[face[0]]
        b = verts[face[1]]
        c = verts[face[2]]
        total += np.cross(b - a, c - a)
    if float(total.dot(ref)) < 0.0:
        return [tuple(reversed(face)) for face in cleaned]
    return cleaned


def _order_triangle_sides(
    sides: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (long0, long1, base)，long 从 base 角指向 tip。"""
    ordered = sorted(
        (np.asarray(side, dtype=np.float64) for side in sides),
        key=polyline_length,
        reverse=True,
    )
    long0, long1, base = ordered[0], ordered[1], ordered[2]
    tip_candidates = [
        long0[0],
        long0[-1],
        long1[0],
        long1[-1],
    ]
    # tip ≈ 两条长边的公共端点
    tip = None
    for p in tip_candidates:
        d0 = min(float(np.linalg.norm(p - long0[0])), float(np.linalg.norm(p - long0[-1])))
        d1 = min(float(np.linalg.norm(p - long1[0])), float(np.linalg.norm(p - long1[-1])))
        if d0 < 1e-8 and d1 < 1e-8:
            tip = p
            break
    if tip is None:
        tip = 0.5 * (long0[-1] + long1[-1])

    def _from_base_to_tip(side: np.ndarray) -> np.ndarray:
        if float(np.linalg.norm(side[-1] - tip)) <= float(
            np.linalg.norm(side[0] - tip)
        ):
            return side
        return side[::-1]

    long0 = _from_base_to_tip(long0)
    long1 = _from_base_to_tip(long1)
    # base 方向：从 long0 底到 long1 底
    if float(np.linalg.norm(base[0] - long0[0])) > float(
        np.linalg.norm(base[-1] - long0[0])
    ):
        base = base[::-1]
    return long0, long1, base


def _loop_pca_basis(
    loop_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    边界环点集的 PCA 基底：返回 (centroid, axis_u, axis_v)。

    对跨越大弧度的弯曲条带，面法线均值方向会与边界主平面共面，
    投影会折叠自交；改用边界点最佳拟合平面保证投影展开。
    """
    pts = _as_float_array(loop_points)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = _normalize(vh[0])
    axis_v = _normalize(vh[1])
    return centroid, axis_u, axis_v


def analyze_region_fit_topology(
    region_ids: np.ndarray,
    target_id,
    vertices: np.ndarray,
    loop_start: np.ndarray,
    loop_total: np.ndarray,
    loop_vertex_indices: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    face_centers: np.ndarray,
    triangle_ratio: float = DEFAULT_TRIANGLE_RATIO,
    corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
    adjacency_offsets: np.ndarray | None = None,
    adjacency_indices: np.ndarray | None = None,
    stitch_gap_frac: float = _MULTI_ISLAND_MERGE_GAP_FRAC,
    bridge_gap_frac: float = 0.25,
    bridge_enabled: bool = True,
    min_perimeter_frac: float = _MIN_ISLAND_PERIMETER_FRAC,
) -> dict:
    """分析领域边界并返回拟合拓扑描述；target_id 可为编号集合。"""
    work_ids = _as_int_array(region_ids)

    loops = extract_region_boundary_loops(
        work_ids,
        target_id,
        loop_start,
        loop_total,
        loop_vertex_indices,
    )
    loops = filter_significant_boundary_loops(
        loops,
        vertices,
        min_perimeter_frac=min_perimeter_frac,
    )
    if not loops:
        raise RegionFitError("过滤极小碎环后无可用边界")
    # 目标面心 + 邻接/-走廊 -1 碎面，供跨岛包络延长相接
    interior = collect_island_bridge_interiors(
        work_ids,
        target_id,
        face_centers,
        adjacency_offsets=adjacency_offsets,
        adjacency_indices=adjacency_indices,
    )
    # 成面默认走桥接阶段：邻近缝合 + 远岛条带包络（可由参数关闭）
    work_items, stage_meta = build_fit_work_items(
        loops,
        vertices,
        stage="BRIDGE" if bridge_enabled else "STITCH",
        stitch_gap_frac=stitch_gap_frac,
        bridge_gap_frac=bridge_gap_frac,
        bridge_enabled=bridge_enabled,
        interior_points=interior,
    )
    if not work_items:
        raise RegionFitError("过滤极小碎环后无可用边界")
    primary = max(
        work_items,
        key=lambda item: float(
            polyline_length(
                np.vstack(
                    (
                        _as_float_array(item["points"]),
                        _as_float_array(item["points"])[:1],
                    )
                )
            )
        ),
    )
    verts = _as_float_array(vertices)
    gap = (
        float(stage_meta["bridge_gap"])
        if bridge_enabled
        else float(stage_meta["stitch_gap"])
    )
    clusters = cluster_nearby_boundary_loops(loops, verts, max_gap=gap)
    if not clusters:
        raise RegionFitError("过滤极小碎环后无可用边界")

    def _cluster_perimeter(members: Sequence[int]) -> float:
        total = 0.0
        for index in members:
            pts = verts[np.asarray(loops[index], dtype=np.int32)]
            total += float(polyline_length(np.vstack((pts, pts[:1]))))
        return total

    primary_cluster = max(clusters, key=_cluster_perimeter)
    fit_loops = [list(loops[index]) for index in primary_cluster]
    try:
        loop_pts, band_sides, _band_samples = combine_boundary_islands(
            fit_loops,
            vertices,
            interior_points=interior,
            prefer_outer_envelope=len(fit_loops) > 1,
        )
    except RegionFitError:
        loop_pts = _as_float_array(primary["points"])
        band_sides = None
    _, normal, _, _ = compute_region_frame(
        face_normals,
        face_areas,
        work_ids,
        target_id,
        face_centers,
    )
    # 2D 分析基底用边界点 PCA 平面，避免弯曲条带投影折叠
    origin, axis_u, axis_v = _loop_pca_basis(loop_pts)
    shared_param = False

    if band_sides is not None and len(band_sides) == 4:
        # 多 island 条带：直接使用共享 u 分箱的四边，跳过转角检测与
        # 整环弧长重采样（二者都会破坏内外弧的参数对应，导致扭曲）。
        # 结构上恒为四边（两长弧+两端帽），即使一端很短也不坍成三边。
        sides_3d = [np.asarray(side, dtype=np.float64) for side in band_sides]
        pts_2d_raw = project_points_to_plane(loop_pts, origin, axis_u, axis_v)
        if _signed_area_2d(pts_2d_raw) < 0.0:
            sides_3d = [side[::-1] for side in reversed(sides_3d)]
        # 旋转使 sides[0]/sides[2] 为两条长边（U 向沿条带）
        lengths = [polyline_length(side) for side in sides_3d]
        if lengths[0] + lengths[2] < lengths[1] + lengths[3]:
            sides_3d = [sides_3d[1], sides_3d[2], sides_3d[3], sides_3d[0]]
        topology = "QUAD"
        shared_param = True
        corner_count = 4
    else:
        pts_2d_raw = project_points_to_plane(loop_pts, origin, axis_u, axis_v)
        # 统一为逆时针（角点检测与凹口桥接依赖此方向约定）
        if _signed_area_2d(pts_2d_raw) < 0.0:
            loop_pts = loop_pts[::-1]

        # 闭环按弧长均匀重采样：3D 边链直接来自采样点，保留扫描起伏
        sample_count = int(
            np.clip(
                len(loop_pts) * 2,
                _BOUNDARY_SAMPLES_MIN,
                _BOUNDARY_SAMPLES_MAX,
            )
        )
        loop_rs = resample_closed_polyline(loop_pts, sample_count)
        pts_2d = project_points_to_plane(loop_rs, origin, axis_u, axis_v)

        corners = detect_corner_indices(
            pts_2d,
            corner_angle_deg,
            max_corners=_MAX_CORNER_CANDIDATES,
            include_strong_concave=True,
        )
        if len(corners) < 3:
            # 无明显角点（近圆/椭圆域）：按弧长均匀取 4 个角点
            quarter = max(sample_count // 4, 1)
            corners = [0, quarter, 2 * quarter, 3 * quarter]

        sides_3d = split_loop_into_sides(loop_rs, corners)
        sides_2d = split_loop_into_sides(pts_2d, corners)
        sides_2d, sides_3d = merge_paired_collinear_adjacent_sides(
            sides_2d,
            sides_3d,
            max_turn_deg=corner_angle_deg,
            min_sides=4,
        )
        sides_2d, sides_3d = bridge_concave_notches(
            sides_2d,
            sides_3d,
            corner_angle_deg,
        )
        if len(sides_3d) > 4:
            sides_2d, sides_3d = reduce_paired_sides_to_count(
                sides_2d,
                sides_3d,
                4,
                max_merge_turn_deg=max(
                    corner_angle_deg,
                    _STRONG_CONCAVE_ANGLE_DEG,
                ),
            )
        # 拟合面仍需要三/四边；若尖角过多则再强制收到 4（成面阶段）
        if len(sides_3d) > 4:
            sides_2d, sides_3d = reduce_paired_sides_to_count(
                sides_2d,
                sides_3d,
                4,
            )
        if len(sides_3d) == 4:
            sides_2d, sides_3d = extend_quad_corners_by_tangents(
                sides_2d,
                sides_3d,
            )
        topology, sides_3d = classify_tri_or_quad(sides_3d, triangle_ratio)
        corner_count = len(corners)

    lengths = tuple(polyline_length(side) for side in sides_3d)
    return {
        "topology": topology,
        "sides": sides_3d,
        "side_lengths": lengths,
        "normal": normal,
        "origin": origin,
        "loop_count": len(fit_loops),
        "corner_count": corner_count,
        "shared_param": shared_param,
        "surface_samples": interior,
        "cluster_count": len(clusters),
        "skipped_far_loops": max(len(loops) - len(fit_loops), 0),
        "stitch_gap": float(stage_meta.get("stitch_gap", 0.0)),
        "bridge_gap": float(stage_meta.get("bridge_gap", 0.0)),
    }


def fit_region_surface(
    region_ids: np.ndarray,
    target_id,
    vertices: np.ndarray,
    loop_start: np.ndarray,
    loop_total: np.ndarray,
    loop_vertex_indices: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    face_centers: np.ndarray,
    segments_u: int = DEFAULT_SEG_U,
    segments_v: int = DEFAULT_SEG_V,
    triangle_ratio: float = DEFAULT_TRIANGLE_RATIO,
    corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
    adjacency_offsets: np.ndarray | None = None,
    adjacency_indices: np.ndarray | None = None,
    stitch_gap_frac: float = _MULTI_ISLAND_MERGE_GAP_FRAC,
    bridge_gap_frac: float = 0.25,
    bridge_enabled: bool = True,
    min_perimeter_frac: float = _MIN_ISLAND_PERIMETER_FRAC,
) -> RegionFitResult:
    """拟合指定领域（或多个领域并集）为三边或四边规则网格。"""
    analysis = analyze_region_fit_topology(
        region_ids=region_ids,
        target_id=target_id,
        vertices=vertices,
        loop_start=loop_start,
        loop_total=loop_total,
        loop_vertex_indices=loop_vertex_indices,
        face_normals=face_normals,
        face_areas=face_areas,
        face_centers=face_centers,
        triangle_ratio=triangle_ratio,
        corner_angle_deg=corner_angle_deg,
        adjacency_offsets=adjacency_offsets,
        adjacency_indices=adjacency_indices,
        stitch_gap_frac=stitch_gap_frac,
        bridge_gap_frac=bridge_gap_frac,
        bridge_enabled=bridge_enabled,
        min_perimeter_frac=min_perimeter_frac,
    )
    topology = str(analysis["topology"])
    sides = analysis["sides"]
    normal = analysis["normal"]
    warnings: list[str] = []
    loops_n = int(analysis["loop_count"])
    if loops_n > 1:
        warnings.append(
            f"检测到 {loops_n} 个 island，已按边沿曲率延长相接"
        )

    seg_u = int(np.clip(segments_u, MIN_SEGMENTS, MAX_SEGMENTS))
    seg_v = int(np.clip(segments_v, MIN_SEGMENTS, MAX_SEGMENTS))

    if topology == "TRI":
        long0, long1, base = _order_triangle_sides(sides)
        # segments_v → 两条长边；segments_u → 底边
        mesh_verts, faces = build_triangular_patch(
            long0,
            long1,
            base,
            segments_long=seg_v,
            segments_base=seg_u,
        )
    else:
        shared = bool(analysis.get("shared_param", False))
        # 多 island：用走廊采样把 Coons 弦段吸回表面，使孤岛延长段相接
        samples = analysis.get("surface_samples") if shared else None
        mesh_verts, faces = build_quad_patch(
            sides,
            seg_u,
            seg_v,
            shared_param=shared,
            surface_samples=samples,
            snap_strength=0.9 if shared else 0.0,
        )

    faces = _orient_faces(mesh_verts, faces, normal)
    return RegionFitResult(
        vertices=np.asarray(mesh_verts, dtype=np.float64),
        faces=list(faces),
        topology=topology,
        segments_u=seg_u,
        segments_v=seg_v,
        side_lengths=tuple(float(v) for v in analysis["side_lengths"]),
        warnings=warnings,
    )


__all__ = (
    "DEFAULT_TRIANGLE_RATIO",
    "DEFAULT_SEG_U",
    "DEFAULT_SEG_V",
    "MIN_SEGMENTS",
    "MAX_SEGMENTS",
    "RegionFitError",
    "RegionFitResult",
    "analyze_region_fit_topology",
    "bridge_concave_notches",
    "bridge_reentrant_corners_closed_loop",
    "collapse_ultra_reflex_spike_vertices_closed_loop",
    "stitch_two_loops_by_corner_seams",
    "build_fit_work_items",
    "build_quad_patch",
    "build_triangular_patch",
    "classify_tri_or_quad",
    "collect_island_bridge_interiors",
    "collect_interior_angle_labels",
    "combine_boundary_islands",
    "coons_patch",
    "detect_corner_indices",
    "detect_concave_fold_indices",
    "extend_quad_corners_by_tangents",
    "extract_island_longest_sides",
    "extract_region_boundary_loops",
    "build_edge_face_adjacency",
    "boundary_loop_neighbor_ids",
    "neighbor_change_vertex_indices",
    "map_loop_indices_to_resampled",
    "resplit_sides_near_points",
    "filter_handle_outliers",
    "filter_significant_boundary_loops",
    "fit_bezier_polyline_spans",
    "fit_cubic_bezier_controls",
    "fit_region_surface",
    "merge_collinear_adjacent_sides",
    "merge_nearby_loops_to_outer_contours",
    "cluster_nearby_boundary_loops",
    "loop_has_internal_corridor",
    "outer_contour_from_boundary_loops",
    "outer_contour_by_dissolving_facing_edges",
    "point_to_polyline_distance",
    "sample_cubic_bezier",
    "polyline_length",
    "polyline_parameters",
    "resample_closed_polyline",
    "resample_polyline",
    "select_primary_boundary_loop",
    "side_interior_max_turn_deg",
    "soft_snap_quad_grid_to_points",
)
