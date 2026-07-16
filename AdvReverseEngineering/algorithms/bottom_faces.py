# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""底面检测算法。"""

from __future__ import annotations

import numpy as np

from ..utils.mesh import MeshData


def detect_bottom_face_indices(
    mesh_data: MeshData,
    mesh_polygons,
    z_tolerance_ratio: float = 0.02,
    normal_threshold: float = 0.75,
) -> list[int]:
    """
    检测当前作为底面的三角面索引。

    摆正后底面应朝下（法线接近 -Z），且位于最低 Z 区域。
    """
    vertices = mesh_data["vertices"]
    normals = mesh_data["normals"]

    if len(vertices) == 0 or len(mesh_polygons) == 0:
        return []

    min_z = float(vertices[:, 2].min())
    z_range = float(vertices[:, 2].max() - min_z)
    z_tolerance = max(z_range * z_tolerance_ratio, 1e-5)

    bottom_faces: list[int] = []

    for face_index, polygon in enumerate(mesh_polygons):
        vert_indices = polygon.vertices
        face_vertices = vertices[vert_indices]
        face_z_max = float(face_vertices[:, 2].max())
        face_z_mean = float(face_vertices[:, 2].mean())
        # 法线朝下程度（越接近 -Z 越大）
        normal_alignment = float(-normals[face_index][2])

        # 位于最低高度带内的面
        if face_z_max <= min_z + z_tolerance:
            bottom_faces.append(face_index)
            continue

        # 法线朝下且接近底面
        if normal_alignment >= normal_threshold:
            if face_z_mean <= min_z + z_tolerance * 3.0:
                bottom_faces.append(face_index)

    return bottom_faces
