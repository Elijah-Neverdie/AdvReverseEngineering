# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""有向包围盒（OBB）精修算法。"""

from __future__ import annotations

import numpy as np

from ..utils.math import (
    aabb_volume,
    euler_xyz_to_matrix,
)
from ..utils.mesh import subsample_array


def obb_refine_rotation(
    vertices: np.ndarray,
    initial_rotation: np.ndarray,
    search_deg: float = 15.0,
    step_deg: float = 3.0,
    sample_count: int = 60000,
    seed: int = 42,
) -> np.ndarray:
    """
    在初始旋转附近 ±search_deg 范围内搜索最小 AABB 体积。

    不全局搜索，仅在给定初值附近做网格扰动，符合工业扫描摆正流程。
    """
    points = subsample_array(vertices, sample_count, seed=seed)
    if len(points) == 0:
        return initial_rotation.copy()

    best_rotation = initial_rotation.copy()
    best_volume = aabb_volume(points @ best_rotation.T)

    angles = np.arange(-search_deg, search_deg + 0.1, step_deg)
    angles_rad = np.radians(angles)

    for rx in angles_rad:
        for ry in angles_rad:
            for rz in angles_rad:
                delta = euler_xyz_to_matrix(rx, ry, rz)
                candidate = delta @ initial_rotation
                volume = aabb_volume(points @ candidate.T)
                if volume < best_volume:
                    best_volume = volume
                    best_rotation = candidate

    return best_rotation


def orientation_matrix_obb(
    vertices: np.ndarray,
    initial_rotation: np.ndarray,
) -> np.ndarray:
    """OBB 精修旋转矩阵。"""
    return obb_refine_rotation(vertices, initial_rotation)
