# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""底面检测算法。"""

from __future__ import annotations

import numpy as np

from ..utils.mesh import MeshData


def detect_bottom_face_indices(
    mesh_data: MeshData,
    z_tolerance_ratio: float = 0.02,
    normal_threshold: float = 0.85,
) -> list[int]:
    """
    检测当前作为底面的面索引。

    摆正后底面应近似水平（法线接近 ±Z），且位于最低 Z 区域。
    不强制法线朝下，兼容对象模式与编辑模式两种轴向约定。
    """
    vertices = mesh_data["vertices"]
    normals = mesh_data["normals"]
    centers = mesh_data["face_centers"]

    if len(vertices) == 0 or len(centers) == 0:
        return []

    min_z = float(vertices[:, 2].min())
    z_range = float(vertices[:, 2].max() - min_z)
    z_tolerance = max(z_range * z_tolerance_ratio, 1e-5)

    center_z = centers[:, 2]
    horizontal = np.abs(normals[:, 2]) >= normal_threshold

    # 优先：最低高度带内的近水平面
    lowest_horizontal = (
        (center_z <= min_z + z_tolerance)
        & horizontal
    )
    indices = np.flatnonzero(lowest_horizontal)
    if len(indices) > 0:
        return indices.tolist()

    # 回退：略放宽高度，但仍要求近似水平
    relaxed = (
        (center_z <= min_z + z_tolerance * 3.0)
        & horizontal
    )
    indices = np.flatnonzero(relaxed)
    if len(indices) > 0:
        return indices.tolist()

    # 最终回退：最低高度带内的所有面
    return np.flatnonzero(center_z <= min_z + z_tolerance).tolist()
