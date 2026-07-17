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
_EXTEND_LENGTH_FACTOR = 8.0


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
) -> list[int]:
    """按转角检测闭环折线角点索引。"""
    pts = _as_float_array(points_2d)
    count = len(pts)
    if count < 3:
        return []
    cos_limit = float(np.cos(np.radians(max(angle_threshold_deg, 1.0))))
    corners: list[int] = []
    for index in range(count):
        prev_pt = pts[(index - 1) % count]
        curr_pt = pts[index]
        next_pt = pts[(index + 1) % count]
        incoming = curr_pt - prev_pt
        outgoing = next_pt - curr_pt
        in_len = float(np.linalg.norm(incoming))
        out_len = float(np.linalg.norm(outgoing))
        if in_len < 1e-12 or out_len < 1e-12:
            continue
        incoming /= in_len
        outgoing /= out_len
        # 转角越大，点积越小
        if float(incoming.dot(outgoing)) <= cos_limit:
            corners.append(index)
    return corners


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
    保留最多四条主边；第四边短于最长边 * ratio 时视为三边。
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
    longest = lengths[order[0]]
    fourth = lengths[order[3]]
    ratio = max(float(triangle_ratio), 0.0)
    if longest > 1e-12 and fourth < longest * ratio:
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


def _line_intersection_2d(
    p0: np.ndarray,
    d0: np.ndarray,
    p1: np.ndarray,
    d1: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """二维射线求交，返回 (点, t0, t1)；平行则 None。"""
    matrix = np.array(
        [[d0[0], -d1[0]], [d0[1], -d1[1]]],
        dtype=np.float64,
    )
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-10:
        return None
    delta = p1 - p0
    params = np.linalg.solve(matrix, delta)
    t0 = float(params[0])
    t1 = float(params[1])
    return p0 + d0 * t0, t0, t1


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """单调链凸包，返回逆时针顶点（不含重复终点）。"""
    pts = _as_float_array(points)
    if len(pts) <= 1:
        return pts.copy()
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    ordered = pts[order]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if not hull:
        return pts[:1].copy()
    return np.asarray(hull, dtype=np.float64)


def extend_concave_corners(
    sides: Sequence[np.ndarray],
    reference_normal_2d_sign: float = 1.0,
) -> list[np.ndarray]:
    """
    对凹轮廓避免沿凹口收缩：

    1. 若角点集非凸，改用凸包角点并重建直边（拟合域外扩）；
    2. 对仍存在的局部凹折线，用相邻主边端部切线外延求交替换凹点。
    """
    if len(sides) < 3:
        return [np.asarray(side, dtype=np.float64).copy() for side in sides]

    result = [np.asarray(side, dtype=np.float64).copy() for side in sides]
    corners = np.asarray([side[0] for side in result], dtype=np.float64)
    hull = _convex_hull_2d(corners)
    if len(hull) >= 3 and len(hull) < len(corners):
        # 凹多边形：用凸包外扩角点，避免拟合线陷入凹口
        rebuilt: list[np.ndarray] = []
        for index in range(len(hull)):
            rebuilt.append(
                np.asarray(
                    [hull[index], hull[(index + 1) % len(hull)]],
                    dtype=np.float64,
                )
            )
        return rebuilt

    median_len = float(
        np.median([max(polyline_length(side), 1e-12) for side in result])
    )
    max_extend = median_len * _EXTEND_LENGTH_FACTOR

    for index in range(len(result)):
        prev_side = result[index]
        next_index = (index + 1) % len(result)
        next_side = result[next_index]
        if len(prev_side) < 2 or len(next_side) < 2:
            continue
        corner = prev_side[-1]
        d_prev = corner - prev_side[-2]
        d_next = next_side[1] - corner
        cross = float(d_prev[0] * d_next[1] - d_prev[1] * d_next[0])
        is_concave = cross * float(reference_normal_2d_sign) < 0.0
        if not is_concave:
            continue

        # 取两侧「主边」：跳过紧贴凹点的短折，用更靠外的点定义切线
        prev_anchor = prev_side[0] if len(prev_side) >= 2 else prev_side[-2]
        next_anchor = next_side[-1] if len(next_side) >= 2 else next_side[1]
        prev_dir = _normalize(
            np.array(
                [
                    corner[0] - prev_anchor[0],
                    corner[1] - prev_anchor[1],
                    0.0,
                ],
                dtype=np.float64,
            )
        )[:2]
        next_dir = _normalize(
            np.array(
                [
                    next_anchor[0] - corner[0],
                    next_anchor[1] - corner[1],
                    0.0,
                ],
                dtype=np.float64,
            )
        )[:2]
        if float(np.linalg.norm(prev_dir)) < 1e-8:
            continue
        if float(np.linalg.norm(next_dir)) < 1e-8:
            continue

        # 从两侧锚点所在直线外延求交
        hit = _line_intersection_2d(
            prev_anchor,
            prev_dir,
            next_anchor,
            -next_dir,
        )
        if hit is None:
            continue
        point, t0, t1 = hit
        if t0 < -1e-6 or t1 < -1e-6:
            continue
        dist = float(np.linalg.norm(point - corner))
        if dist > max_extend:
            continue
        # 仅当新交点相对凹点更靠外（远离多边形质心）时采用
        centroid = corners.mean(axis=0)
        if float(np.linalg.norm(point - centroid)) <= float(
            np.linalg.norm(corner - centroid)
        ) + 1e-9:
            continue
        prev_side = prev_side.copy()
        next_side = next_side.copy()
        prev_side[-1] = point
        next_side[0] = point
        result[index] = prev_side
        result[next_index] = next_side
    return result


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
    pts_2d = project_points_to_plane(loop_pts, origin, axis_u, axis_v)
    # 统一为逆时针
    if _signed_area_2d(pts_2d) < 0.0:
        pts_2d = pts_2d[::-1]
        loop_pts = loop_pts[::-1]

    corners = detect_corner_indices(pts_2d, corner_angle_deg)
    if len(corners) < 3:
        # 回退：按累计弧长均匀取 4 个角点
        params = polyline_parameters(
            np.vstack((pts_2d, pts_2d[:1]))
        )[:-1]
        corners = [
            int(np.argmin(np.abs(params - target)))
            for target in (0.0, 0.25, 0.5, 0.75)
        ]
        corners = sorted(set(corners))
        if len(corners) < 3:
            raise RegionFitError("无法检测足够角点")

    sides_2d = split_loop_into_sides(pts_2d, corners)
    sides_2d = extend_concave_corners(sides_2d, reference_normal_2d_sign=1.0)
    topology, sides_2d = classify_tri_or_quad(sides_2d, triangle_ratio)

    sides_3d = [
        unproject_points_from_plane(side, origin, axis_u, axis_v)
        for side in sides_2d
    ]
    # 把高度抬回：用原边界最近点替换平面点，保留 3D 起伏
    sides_3d = [
        _lift_side_to_original(side, loop_pts) for side in sides_3d
    ]
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


def _lift_side_to_original(
    side_planar: np.ndarray,
    original_loop: np.ndarray,
) -> np.ndarray:
    """将平面边采样点吸附到最近的原边界点，保留扫描起伏。"""
    side = _as_float_array(side_planar).copy()
    loop = _as_float_array(original_loop)
    if len(side) == 0 or len(loop) == 0:
        return side
    for index, point in enumerate(side):
        distances = np.linalg.norm(loop - point, axis=1)
        nearest = int(np.argmin(distances))
        # 仅对非外延角点做吸附；外延点距环较远则保留
        if float(distances[nearest]) <= max(
            polyline_length(loop) * 0.02,
            1e-4,
        ):
            side[index] = loop[nearest]
    return side


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
    "build_quad_patch",
    "build_triangular_patch",
    "classify_tri_or_quad",
    "coons_patch",
    "detect_corner_indices",
    "extend_concave_corners",
    "extract_region_boundary_loops",
    "fit_region_surface",
    "polyline_length",
    "polyline_parameters",
    "resample_polyline",
    "select_primary_boundary_loop",
)
