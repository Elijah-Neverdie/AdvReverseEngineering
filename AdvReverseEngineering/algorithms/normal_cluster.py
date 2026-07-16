# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""法线聚类算法。"""

from __future__ import annotations

import numpy as np

from ..utils.math import build_full_orientation, normalize


def _canonical_axis_candidates() -> np.ndarray:
    """六个主轴方向候选（用于快速法线投票）。"""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )


def cluster_dominant_normal(
    normals: np.ndarray,
    areas: np.ndarray,
    angle_threshold_deg: float = 22.5,
) -> np.ndarray:
    """
    按角度聚类面法线，返回面积加权最大的主方向。

    先将法线映射到最近的主轴桶，再对相反方向合并统计，
    适合扫描模型的大平面检测。
    """
    if len(normals) == 0:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)

    cos_limit = float(np.cos(np.radians(angle_threshold_deg)))
    candidates = _canonical_axis_candidates()
    scores = np.zeros(len(candidates), dtype=np.float64)

    # 法线点积矩阵 (F, 6)，批量计算
    dots = np.abs(normals @ candidates.T)
    nearest = np.argmax(dots, axis=1)
    nearest_dots = dots[np.arange(len(normals)), nearest]
    valid = nearest_dots >= cos_limit

    weighted_areas = areas.copy()
    weighted_areas[~valid] = 0.0
    for axis_index in range(len(candidates)):
        mask = nearest == axis_index
        scores[axis_index] = float(weighted_areas[mask].sum())

    # 合并相反方向票数
    merged = scores.copy()
    for i in range(3):
        merged[i] = scores[i] + scores[i + 3]
        merged[i + 3] = merged[i]

    best_index = int(np.argmax(merged))
    dominant = candidates[best_index]
    if dominant[2] > 0.0:
        dominant = -dominant
    return normalize(dominant)


def orientation_matrix_normal_cluster(
    vertices: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """法线聚类完整摆正旋转矩阵。"""
    dominant = cluster_dominant_normal(normals, areas)
    return build_full_orientation(dominant, vertices, centroid)
