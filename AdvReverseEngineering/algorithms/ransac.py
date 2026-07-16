# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""RANSAC 最大平面检测算法。"""

from __future__ import annotations

import numpy as np

from ..utils.math import build_full_orientation, normalize
from ..utils.mesh import subsample_array


def _fit_plane_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由三点拟合平面，返回 (平面上一点, 单位法线)。"""
    p0, p1, p2 = points
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return p0, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return p0, normal / norm


def _point_plane_distances(
    points: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """批量计算点到平面距离。"""
    vectors = points - plane_point
    return np.abs(vectors @ plane_normal)


def ransac_largest_plane(
    vertices: np.ndarray,
    distance_threshold: float = 0.005,
    max_iterations: int = 400,
    sample_count: int = 80000,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """
    RANSAC 检测最大支持平面。

    参数:
        vertices: 世界空间顶点 (N, 3)
        distance_threshold: 内点距离阈值
        max_iterations: 最大迭代次数
        sample_count: 参与计算的下采样顶点数

    返回:
        plane_normal: 平面法线（朝下半球）
        inlier_count: 内点数量
    """
    points = subsample_array(vertices, sample_count, seed=seed)
    if len(points) < 3:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64), 0

    rng = np.random.default_rng(seed)
    best_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    best_inliers = 0

    for _ in range(max_iterations):
        indices = rng.choice(len(points), 3, replace=False)
        plane_point, plane_normal = _fit_plane_from_points(points[indices])

        distances = _point_plane_distances(points, plane_point, plane_normal)
        inlier_mask = distances < distance_threshold
        inlier_count = int(inlier_mask.sum())

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_normal = plane_normal.copy()
            if inlier_count > len(points) * 0.85:
                break

    if best_normal[2] > 0.0:
        best_normal = -best_normal
    return normalize(best_normal), best_inliers


def orientation_matrix_ransac(
    vertices: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """RANSAC 底面检测完整摆正旋转矩阵。"""
    bbox = vertices.max(axis=0) - vertices.min(axis=0)
    threshold = max(float(bbox.max()) * 0.002, 1e-4)
    normal, _ = ransac_largest_plane(vertices, distance_threshold=threshold)
    return build_full_orientation(normal, vertices, centroid)
