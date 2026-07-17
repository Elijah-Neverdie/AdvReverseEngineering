# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""基于编辑模式选区的工业摆正算法。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.math import normalize
from ..utils.mesh import MeshData


@dataclass(frozen=True)
class SelectedPlaneResult:
    """编辑模式选区摆正结果。"""

    rotation: np.ndarray
    pivot: np.ndarray
    method_label: str
    plane_point: np.ndarray
    plane_normal: np.ndarray


def fit_selected_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """使用 SVD 对选中点进行最小二乘平面拟合。"""
    if len(points) < 3:
        raise ValueError("至少需要选择 3 个不共线顶点")

    center = points.mean(axis=0)
    centered = points - center
    _, singular_values, vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    if len(singular_values) < 2 or singular_values[1] < 1e-10:
        raise ValueError("选中元素近似共线，无法确定底面")
    return center, normalize(vectors[-1])


def _orient_normal_towards_object(
    normal: np.ndarray,
    plane_point: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """让平面法线指向物体主体，使摆正后物体尽量位于 Z 正方向。"""
    signed = (vertices - plane_point) @ normal
    non_near = signed[np.abs(signed) > 1e-9]
    if len(non_near) and float(np.median(non_near)) < 0.0:
        return -normal
    return normal


def collect_plane_points(
    vertices: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    threshold_ratio: float = 0.02,
) -> tuple[np.ndarray, float]:
    """收集距选定平面不超过包围盒最大尺寸 2% 的物体顶点。"""
    dimensions = vertices.max(axis=0) - vertices.min(axis=0)
    threshold = max(float(dimensions.max()) * threshold_ratio, 1e-6)
    distances = np.abs((vertices - plane_point) @ plane_normal)
    return vertices[distances <= threshold], threshold


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """建立平面内稳定的正交基。"""
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis_u = normalize(np.cross(reference, normal))
    axis_v = normalize(np.cross(normal, axis_u))
    return axis_u, axis_v


def _project_to_plane(
    points: np.ndarray,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    """将三维点批量投影到平面二维坐标。"""
    centered = points - origin
    return np.column_stack((centered @ axis_u, centered @ axis_v))


def _pca_axes_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算二维点集主轴与中心。"""
    center = np.median(points, axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return vectors[:, order], center


def _symmetry_score(points: np.ndarray, axis: np.ndarray) -> float:
    """通过二维占据直方图与镜像图比较，估计指定轴的镜像对称度。"""
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    perpendicular = np.array([-axis[1], axis[0]], dtype=np.float64)
    center = np.median(points, axis=0)
    coordinates = np.column_stack(
        ((points - center) @ perpendicular, (points - center) @ axis)
    )

    span = np.ptp(coordinates, axis=0)
    if float(span.min()) < 1e-10:
        return 0.0

    bins = int(np.clip(np.sqrt(len(points)), 24, 72))
    max_x = max(float(np.abs(coordinates[:, 0]).max()), 1e-9)
    min_y = float(coordinates[:, 1].min())
    max_y = float(coordinates[:, 1].max())
    histogram, _, _ = np.histogram2d(
        coordinates[:, 0],
        coordinates[:, 1],
        bins=bins,
        range=((-max_x, max_x), (min_y, max_y)),
    )
    mirrored = histogram[::-1, :]
    denominator = float(histogram.sum() + mirrored.sum())
    if denominator < 1e-12:
        return 0.0
    difference = float(np.abs(histogram - mirrored).sum())
    return 1.0 - difference / denominator


def _detect_symmetry_axis(
    projected: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """在二维 PCA 两条主轴中检测最可信的镜像轴。"""
    if len(projected) < 12:
        return None, 0.0

    axes, _ = _pca_axes_2d(projected)
    candidates = (axes[:, 0], axes[:, 1])
    scores = [_symmetry_score(projected, axis) for axis in candidates]
    best = int(np.argmax(scores))
    if scores[best] < 0.76:
        return None, scores[best]
    return candidates[best], scores[best]


def _convex_hull_2d(points: np.ndarray, max_points: int = 12000) -> np.ndarray:
    """单调链算法计算二维凸包；先量化去重以控制大网格开销。"""
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]

    unique = np.unique(np.round(points, decimals=8), axis=0)
    if len(unique) < 3:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (
            a[1] - o[1]
        ) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _detect_right_angle_axis(
    projected: np.ndarray,
    angle_tolerance_deg: float = 12.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """在平面投影凸包中寻找接近 90° 且边长显著的角。"""
    hull = _convex_hull_2d(projected)
    if len(hull) < 3:
        return None, None

    previous = np.roll(hull, 1, axis=0) - hull
    following = np.roll(hull, -1, axis=0) - hull
    previous_length = np.linalg.norm(previous, axis=1)
    following_length = np.linalg.norm(following, axis=1)
    valid = (previous_length > 1e-9) & (following_length > 1e-9)
    normalized_previous = previous / np.maximum(
        previous_length[:, None],
        1e-12,
    )
    normalized_following = following / np.maximum(
        following_length[:, None],
        1e-12,
    )
    cosines = np.abs(np.sum(
        normalized_previous * normalized_following,
        axis=1,
    ))
    tolerance = float(np.sin(np.radians(angle_tolerance_deg)))
    candidates = np.flatnonzero(valid & (cosines <= tolerance))
    if len(candidates) == 0:
        return None, None

    weights = previous_length[candidates] * following_length[candidates]
    index = int(candidates[int(np.argmax(weights))])
    first = normalized_previous[index]
    second = normalized_following[index]
    y_axis = first if previous_length[index] >= following_length[index] else second
    return y_axis, hull[index]


def _side_axis(
    mesh_data: MeshData,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    从侧面寻找 Y 方向。

    优先选取面积近似、法线相反的平行面；否则使用最大侧面法线。
    """
    normals = mesh_data["normals"]
    areas = mesh_data["areas"]
    dot_up = np.abs(normals @ plane_normal)
    side_indices = np.flatnonzero(dot_up <= 0.35)
    if len(side_indices) == 0:
        axis_u, _ = _plane_basis(plane_normal)
        return axis_u, "底面中心 + 默认侧向"

    side_normals = normals[side_indices]
    side_areas = areas[side_indices]
    order = np.argsort(side_areas)[::-1][: min(256, len(side_indices))]
    best_pair: tuple[int, int] | None = None
    best_weight = -1.0

    for position, local_i in enumerate(order):
        normal_i = side_normals[local_i]
        area_i = float(side_areas[local_i])
        remaining = order[position + 1:]
        if len(remaining) == 0:
            continue
        dots = side_normals[remaining] @ normal_i
        area_j = side_areas[remaining]
        ratio = np.minimum(area_i, area_j) / np.maximum(area_i, area_j)
        valid = (dots <= -0.94) & (ratio >= 0.72)
        if not np.any(valid):
            continue
        candidates = remaining[valid]
        candidate_weights = area_i + side_areas[candidates]
        candidate = int(candidates[int(np.argmax(candidate_weights))])
        weight = area_i + float(side_areas[candidate])
        if weight > best_weight:
            best_pair = (int(local_i), candidate)
            best_weight = weight

    if best_pair is not None:
        axis = side_normals[best_pair[0]]
        axis -= np.dot(axis, plane_normal) * plane_normal
        return normalize(axis), "底面中心 + 平行等面积侧面"

    largest = int(np.argmax(side_areas))
    axis = side_normals[largest]
    axis -= np.dot(axis, plane_normal) * plane_normal
    return normalize(axis), "底面中心 + 最大侧面"


def _frame_rotation(y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """构造将源 Y/Z 轴分别映射到世界 Y/Z 的旋转矩阵。"""
    z_axis = normalize(z_axis)
    y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
    y_axis = normalize(y_axis)
    x_axis = normalize(np.cross(y_axis, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.vstack((x_axis, y_axis, z_axis))


def _axis_2d_to_3d(
    axis: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    """二维平面方向还原为三维世界方向。"""
    return normalize(axis[0] * axis_u + axis[1] * axis_v)


def _symmetry_pivot(
    plane_points: np.ndarray,
    plane_point: np.ndarray,
    symmetry_axis: np.ndarray,
    plane_normal: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """取镜像轴附近最远两点的中心作为原点。"""
    relative = plane_points - plane_point
    perpendicular = normalize(np.cross(plane_normal, symmetry_axis))
    distance_to_axis = np.abs(relative @ perpendicular)
    near_axis = plane_points[
        distance_to_axis <= max(threshold, 1e-6)
    ]
    if len(near_axis) < 2:
        near_axis = plane_points
    coordinates = (near_axis - plane_point) @ symmetry_axis
    first = near_axis[int(np.argmin(coordinates))]
    second = near_axis[int(np.argmax(coordinates))]
    return (first + second) * 0.5


def orient_from_selected_plane(
    mesh_data: MeshData,
    selected_points: np.ndarray,
) -> SelectedPlaneResult:
    """执行选区平面 → 对称/直角/侧面回退的完整摆正流程。"""
    plane_point, plane_normal = fit_selected_plane(selected_points)
    plane_normal = _orient_normal_towards_object(
        plane_normal,
        plane_point,
        mesh_data["vertices"],
    )
    plane_points, threshold = collect_plane_points(
        mesh_data["vertices"],
        plane_point,
        plane_normal,
    )
    if len(plane_points) < 3:
        plane_points = selected_points

    axis_u, axis_v = _plane_basis(plane_normal)
    projected = _project_to_plane(
        plane_points,
        plane_point,
        axis_u,
        axis_v,
    )

    symmetry_axis_2d, _ = _detect_symmetry_axis(projected)
    if symmetry_axis_2d is not None:
        y_axis = _axis_2d_to_3d(symmetry_axis_2d, axis_u, axis_v)
        pivot = _symmetry_pivot(
            plane_points,
            plane_point,
            y_axis,
            plane_normal,
            threshold,
        )
        label = "选区平面 + 镜像轴"
    else:
        right_axis_2d, corner_2d = _detect_right_angle_axis(projected)
        if right_axis_2d is not None and corner_2d is not None:
            y_axis = _axis_2d_to_3d(right_axis_2d, axis_u, axis_v)
            pivot = (
                plane_point
                + corner_2d[0] * axis_u
                + corner_2d[1] * axis_v
            )
            label = "选区平面 + 直角"
        else:
            y_axis, fallback_label = _side_axis(mesh_data, plane_normal)
            minimum = projected.min(axis=0)
            maximum = projected.max(axis=0)
            center_2d = (minimum + maximum) * 0.5
            pivot = (
                plane_point
                + center_2d[0] * axis_u
                + center_2d[1] * axis_v
            )
            label = fallback_label

    rotation = _frame_rotation(y_axis, plane_normal)
    return SelectedPlaneResult(
        rotation=rotation,
        pivot=pivot,
        method_label=label,
        plane_point=plane_point,
        plane_normal=plane_normal,
    )
