# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""法线聚类算法。"""

from __future__ import annotations

import numpy as np

from ..utils.math import build_full_orientation, normalize


def cluster_dominant_normal(
    normals: np.ndarray,
    areas: np.ndarray,
    angle_threshold_deg: float = 22.5,
) -> np.ndarray:
    """
    按角度聚类面法线，返回面积加权最大的主方向。

    通过面积加权方向张量合并 n 与 -n，因此无需依赖当前世界六轴，
    也不会把错误的正负轴配成一组。随后对主方向角度范围内的法线
    做一次加权均值精修。
    """
    if len(normals) == 0:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)

    valid_areas = np.maximum(
        np.asarray(areas, dtype=np.float64),
        0.0,
    )
    if float(valid_areas.sum()) < 1e-12:
        valid_areas = np.ones(len(normals), dtype=np.float64)

    # Σ area * (n nᵀ) 天然把正反法线视为同一无向轴。
    direction_tensor = normals.T @ (normals * valid_areas[:, None])
    eigenvalues, eigenvectors = np.linalg.eigh(direction_tensor)
    dominant = normalize(eigenvectors[:, int(np.argmax(eigenvalues))])

    cos_limit = float(np.cos(np.radians(angle_threshold_deg)))
    signed_dots = normals @ dominant
    cluster_mask = np.abs(signed_dots) >= cos_limit
    if np.any(cluster_mask):
        cluster_normals = normals[cluster_mask].copy()
        cluster_weights = valid_areas[cluster_mask]
        signs = np.where(signed_dots[cluster_mask] < 0.0, -1.0, 1.0)
        cluster_normals *= signs[:, None]
        refined = np.sum(
            cluster_normals * cluster_weights[:, None],
            axis=0,
        )
        if float(np.linalg.norm(refined)) > 1e-12:
            dominant = normalize(refined)

    # 对单独使用的法线策略，优先让结果落在世界下半球。
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
