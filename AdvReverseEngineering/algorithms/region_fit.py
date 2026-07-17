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


def extract_region_boundary_loops(
    region_ids: np.ndarray,
    target_id: int,
    loop_start: np.ndarray,
    loop_total: np.ndarray,
    loop_vertex_indices: np.ndarray,
) -> list[list[int]]:
    """
    从 polygon loops 提取目标领域的有序边界闭环。

    通过取消领域内部成对有向半边保留外轮廓与跨领域边。
    """
    ids = _as_int_array(region_ids)
    starts = _as_int_array(loop_start)
    totals = _as_int_array(loop_total)
    loops = _as_int_array(loop_vertex_indices)
    if len(ids) == 0:
        return []
    if target_id < 0 or not np.any(ids == int(target_id)):
        raise RegionFitError(f"领域 {target_id} 不存在")

    halfedge_count: dict[tuple[int, int], int] = defaultdict(int)
    face_indices = np.flatnonzero(ids == int(target_id))
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
        raise RegionFitError(f"领域 {target_id} 没有可提取的外边界")

    outgoing: dict[int, list[int]] = defaultdict(list)
    for v0, v1 in boundary_edges:
        outgoing[v0].append(v1)

    for vertex, neighbors in outgoing.items():
        if len(neighbors) != 1:
            # 分支/非流形：暂时仍尝试追踪，但若失败会报错
            if len(neighbors) == 0:
                raise RegionFitError(
                    f"领域 {target_id} 边界在顶点 {vertex} 处中断"
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
                    f"领域 {target_id} 边界不闭合（顶点 {current}）"
                )
            # 多分支时选第一条未使用出边
            nxt = candidates[0]
            unused[(current, nxt)] -= 1
            current = nxt
            guard += 1
            if guard > max_steps:
                raise RegionFitError(f"领域 {target_id} 边界追踪溢出")
        if len(loop) >= 3:
            closed_loops.append(loop)

    if not closed_loops:
        raise RegionFitError(f"领域 {target_id} 未形成有效闭环边界")
    return closed_loops


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


def compute_region_frame(
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    region_ids: np.ndarray,
    target_id: int,
    face_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (origin, normal, axis_u, axis_v)。"""
    ids = _as_int_array(region_ids)
    mask = ids == int(target_id)
    if not np.any(mask):
        raise RegionFitError(f"领域 {target_id} 不存在")
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
) -> list[int]:
    """
    在均匀采样的闭环折线上按窗口化转角检测角点。

    使用 ±window 采样点的弦向量计算转角，抑制扫描噪声；
    按转角强度做非极大值抑制，最多保留 max_corners 个凸角点。
    要求闭环为逆时针方向（convex_only 依赖叉积符号）。
    """
    pts = _as_float_array(points_2d)
    count = len(pts)
    if count < 3:
        return []
    w = int(window) if window else max(1, count // 32)
    w = min(w, max((count - 1) // 2, 1))

    indices = np.arange(count)
    prev_pts = pts[(indices - w) % count]
    next_pts = pts[(indices + w) % count]
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
    angles = np.degrees(np.arccos(cos_angle))
    angles[~valid] = 0.0
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]

    candidates = angles >= float(max(angle_threshold_deg, 1.0))
    if convex_only:
        candidates &= cross > 0.0
    candidate_idx = np.flatnonzero(candidates)
    if len(candidate_idx) == 0:
        return []

    order = candidate_idx[np.argsort(angles[candidate_idx])[::-1]]
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
    """按角点将闭环拆成边链（每条含两端角点）。"""
    if len(corner_indices) < 3:
        raise RegionFitError("角点不足，无法形成三边/四边拓扑")
    ordered = sorted({int(i) % len(loop_points) for i in corner_indices})
    sides: list[np.ndarray] = []
    for index, start in enumerate(ordered):
        end = ordered[(index + 1) % len(ordered)]
        sides.append(_side_from_loop(loop_points, start, end))
    return sides


def reduce_sides_to_count(
    sides: Sequence[np.ndarray],
    target_count: int,
) -> list[np.ndarray]:
    """通过合并最短边，将边数降到 target_count。"""
    result = [np.asarray(side, dtype=np.float64).copy() for side in sides]
    target = max(int(target_count), 3)
    while len(result) > target:
        lengths = [polyline_length(side) for side in result]
        short_index = int(np.argmin(lengths))
        left = (short_index - 1) % len(result)
        right = (short_index + 1) % len(result)
        # 把最短边并入较长的邻边
        left_len = lengths[left]
        right_len = lengths[right]
        if left_len >= right_len:
            merged = np.vstack((result[left][:-1], result[short_index]))
            result[left] = merged
            del result[short_index]
        else:
            merged = np.vstack((result[short_index][:-1], result[right]))
            result[short_index] = merged
            del result[right]
    return result


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


def build_quad_patch(
    sides: Sequence[np.ndarray],
    segments_u: int,
    segments_v: int,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """四边 Coons：对边同分段。"""
    if len(sides) != 4:
        raise RegionFitError("四边拟合需要恰好四条边")
    seg_u = max(int(segments_u), MIN_SEGMENTS)
    seg_v = max(int(segments_v), MIN_SEGMENTS)
    bottom = resample_polyline(sides[0], seg_u + 1)
    right = resample_polyline(sides[1], seg_v + 1)
    top = resample_polyline(sides[2][::-1], seg_u + 1)
    left = resample_polyline(sides[3][::-1], seg_v + 1)
    grid = coons_patch(bottom, right, top, left)
    vertices = grid.reshape(-1, 3)
    faces = patch_grid_faces(seg_u, seg_v)
    return vertices, faces


def _orient_faces(
    vertices: np.ndarray,
    faces: Sequence[tuple[int, ...]],
    reference_normal: np.ndarray,
) -> list[tuple[int, ...]]:
    ref = _normalize(_as_float_array(reference_normal))
    oriented: list[tuple[int, ...]] = []
    verts = _as_float_array(vertices)
    for face in faces:
        if len(face) < 3:
            continue
        a = verts[face[0]]
        b = verts[face[1]]
        c = verts[face[2]]
        normal = np.cross(b - a, c - a)
        if float(normal.dot(ref)) < 0.0:
            oriented.append(tuple(reversed(face)))
        else:
            oriented.append(tuple(face))
    return oriented


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


def analyze_region_fit_topology(
    region_ids: np.ndarray,
    target_id: int,
    vertices: np.ndarray,
    loop_start: np.ndarray,
    loop_total: np.ndarray,
    loop_vertex_indices: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    face_centers: np.ndarray,
    triangle_ratio: float = DEFAULT_TRIANGLE_RATIO,
    corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
) -> dict:
    """分析领域边界并返回拟合拓扑描述（不含网格）。"""
    loops = extract_region_boundary_loops(
        region_ids,
        target_id,
        loop_start,
        loop_total,
        loop_vertex_indices,
    )
    primary = select_primary_boundary_loop(loops, vertices)
    origin, normal, axis_u, axis_v = compute_region_frame(
        face_normals,
        face_areas,
        region_ids,
        target_id,
        face_centers,
    )
    loop_pts = _as_float_array(vertices)[np.asarray(primary, dtype=np.int32)]
    pts_2d_raw = project_points_to_plane(loop_pts, origin, axis_u, axis_v)
    # 统一为逆时针（角点检测与凹口桥接依赖此方向约定）
    if _signed_area_2d(pts_2d_raw) < 0.0:
        loop_pts = loop_pts[::-1]

    # 闭环按弧长均匀重采样：3D 边链直接来自采样点，保留扫描起伏
    sample_count = int(
        np.clip(len(loop_pts) * 2, _BOUNDARY_SAMPLES_MIN, _BOUNDARY_SAMPLES_MAX)
    )
    loop_rs = resample_closed_polyline(loop_pts, sample_count)
    pts_2d = project_points_to_plane(loop_rs, origin, axis_u, axis_v)

    corners = detect_corner_indices(
        pts_2d,
        corner_angle_deg,
        max_corners=4,
    )
    if len(corners) < 3:
        # 无明显角点（近圆/椭圆域）：按弧长均匀取 4 个角点
        quarter = max(sample_count // 4, 1)
        corners = [0, quarter, 2 * quarter, 3 * quarter]

    sides_3d = split_loop_into_sides(loop_rs, corners)
    sides_2d = split_loop_into_sides(pts_2d, corners)
    sides_2d, sides_3d = bridge_concave_notches(
        sides_2d,
        sides_3d,
        corner_angle_deg,
    )
    topology, sides_3d = classify_tri_or_quad(sides_3d, triangle_ratio)

    lengths = tuple(polyline_length(side) for side in sides_3d)
    return {
        "topology": topology,
        "sides": sides_3d,
        "side_lengths": lengths,
        "normal": normal,
        "origin": origin,
        "loop_count": len(loops),
        "corner_count": len(corners),
    }


def fit_region_surface(
    region_ids: np.ndarray,
    target_id: int,
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
) -> RegionFitResult:
    """拟合指定领域为三边或四边规则网格。"""
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
    )
    topology = str(analysis["topology"])
    sides = analysis["sides"]
    normal = analysis["normal"]
    warnings: list[str] = []
    if int(analysis["loop_count"]) > 1:
        warnings.append("检测到多个边界环，已使用最长外环")

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
        mesh_verts, faces = build_quad_patch(sides, seg_u, seg_v)

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
    "build_quad_patch",
    "build_triangular_patch",
    "classify_tri_or_quad",
    "coons_patch",
    "detect_corner_indices",
    "extract_region_boundary_loops",
    "fit_region_surface",
    "polyline_length",
    "polyline_parameters",
    "resample_closed_polyline",
    "resample_polyline",
    "select_primary_boundary_loop",
)
