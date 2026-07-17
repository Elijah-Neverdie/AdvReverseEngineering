# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""底面检测算法。"""

from __future__ import annotations

import numpy as np

from ..utils.mesh import MeshData


def detect_bottom_face_indices(
    mesh_data: MeshData,
    z_tolerance_ratio: float = 0.02,
    normal_threshold: float = 0.75,
) -> list[int]:
    """
    检测当前作为底面的三角面索引。

    摆正后底面应朝下（法线接近 -Z），且位于最低 Z 区域。
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
    down_alignment = -normals[:, 2]

    # 批量筛选：最低高度带内的近水平面，或略高但法线明确朝下的面。
    lowest_band = (
        (center_z <= min_z + z_tolerance)
        & (down_alignment >= 0.25)
    )
    downward_band = (
        (center_z <= min_z + z_tolerance * 3.0)
        & (down_alignment >= normal_threshold)
    )
    return np.flatnonzero(lowest_band | downward_band).tolist()
