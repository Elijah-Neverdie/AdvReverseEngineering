# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""主成分分析（PCA）算法。"""

from __future__ import annotations

import numpy as np

from ..utils.math import build_full_orientation, ensure_right_handed


def compute_pca(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算顶点 PCA。

    返回:
        centroid: 质心 (3,)
        eigenvalues: 特征值升序 (3,)
        eigenvectors: 特征向量列向量 (3, 3)
    """
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvectors = ensure_right_handed(eigenvectors)
    return centroid, eigenvalues, eigenvectors


def estimate_ground_normal_pca(vertices: np.ndarray) -> np.ndarray:
    """
    用 PCA 估计底面法线。

    最小方差轴通常对应物体最薄方向，即底面法线方向。
    """
    _, _, eigenvectors = compute_pca(vertices)
    normal = eigenvectors[:, 0]
    # 使法线朝下（世界 -Z 半球）
    if normal[2] > 0.0:
        normal = -normal
    return normal


def orientation_matrix_pca(vertices: np.ndarray) -> np.ndarray:
    """PCA 完整摆正旋转矩阵。"""
    centroid, _, eigenvectors = compute_pca(vertices)
    normal = eigenvectors[:, 0]
    if normal[2] > 0.0:
        normal = -normal
    return build_full_orientation(normal, vertices, centroid)
