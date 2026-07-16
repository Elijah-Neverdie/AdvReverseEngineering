# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""
自动摆正策略调度。

每次点击「自动摆正」切换一种策略，便于快速对比效果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .normal_cluster import orientation_matrix_normal_cluster
from .obb import orientation_matrix_obb
from .pca import estimate_ground_normal_pca, orientation_matrix_pca
from .ransac import orientation_matrix_ransac
from ..utils.mesh import MeshData


@dataclass(frozen=True)
class OrientationStrategy:
    """摆正策略定义。"""

    identifier: str
    label: str
    compute: Callable[[MeshData, dict], np.ndarray]


def _strategy_pca(mesh_data: MeshData, _settings: dict) -> np.ndarray:
    return orientation_matrix_pca(mesh_data["vertices"])


def _strategy_ransac(mesh_data: MeshData, _settings: dict) -> np.ndarray:
    return orientation_matrix_ransac(
        mesh_data["vertices"],
        mesh_data["centroid"],
    )


def _strategy_normal(mesh_data: MeshData, _settings: dict) -> np.ndarray:
    return orientation_matrix_normal_cluster(
        mesh_data["vertices"],
        mesh_data["normals"],
        mesh_data["areas"],
        mesh_data["centroid"],
    )


def _strategy_obb(mesh_data: MeshData, settings: dict) -> np.ndarray:
    # OBB 以 PCA 结果为初值
    initial = orientation_matrix_pca(mesh_data["vertices"])
    if not settings.get("obb_refinement", True):
        return initial
    return orientation_matrix_obb(mesh_data["vertices"], initial)


def _strategy_combined(mesh_data: MeshData, settings: dict) -> np.ndarray:
    """
    组合流程: PCA → RANSAC → 法线聚类 → OBB。

    各步骤根据 UI 开关启用，逐步修正旋转。
    """
    vertices = mesh_data["vertices"]
    centroid = mesh_data["centroid"]

    if settings.get("use_pca", True):
        rotation = orientation_matrix_pca(vertices)
    else:
        rotation = np.eye(3, dtype=np.float64)

    if settings.get("detect_largest_plane", True):
        ransac_rot = orientation_matrix_ransac(vertices, centroid)
        rotation = ransac_rot

    if settings.get("normal_clustering", True):
        normal_rot = orientation_matrix_normal_cluster(
            vertices,
            mesh_data["normals"],
            mesh_data["areas"],
            centroid,
        )
        rotation = normal_rot

    if settings.get("obb_refinement", True):
        rotation = orientation_matrix_obb(vertices, rotation)

    return rotation


ORIENTATION_STRATEGIES: tuple[OrientationStrategy, ...] = (
    OrientationStrategy("PCA", "PCA 主方向", _strategy_pca),
    OrientationStrategy("RANSAC", "RANSAC 最大平面", _strategy_ransac),
    OrientationStrategy("NORMAL", "法线聚类", _strategy_normal),
    OrientationStrategy("OBB", "OBB 包围盒精修", _strategy_obb),
    OrientationStrategy("COMBINED", "组合流程", _strategy_combined),
)


def get_strategy(index: int) -> OrientationStrategy:
    """按索引获取策略（循环取模）。"""
    return ORIENTATION_STRATEGIES[index % len(ORIENTATION_STRATEGIES)]


def compute_up_axis(rotation: np.ndarray) -> np.ndarray:
    """从旋转矩阵提取当前估计上方向（世界 Z）。"""
    return rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)


def estimate_up_from_mesh(mesh_data: MeshData) -> np.ndarray:
    """估计网格上方向，用于分析面板显示。"""
    normal = estimate_ground_normal_pca(mesh_data["vertices"])
    return -normal
