# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""底面精确贴合 XOY 平面。"""

from __future__ import annotations

import numpy as np

from ..utils.math import apply_rotation_to_object, normalize, rotation_align_vector_to_axis
from ..utils.mesh import MeshData, extract_mesh_data


def collect_face_world_vertices(
    mesh_data: MeshData,
    face_indices: list[int],
    mesh_polygons,
) -> np.ndarray:
    """收集指定面的唯一世界空间顶点。"""
    if not face_indices:
        return np.empty((0, 3), dtype=np.float64)

    vertices = mesh_data["vertices"]
    index_set: set[int] = set()
    for face_index in face_indices:
        if 0 <= face_index < len(mesh_polygons):
            index_set.update(mesh_polygons[face_index].vertices)

    if not index_set:
        return np.empty((0, 3), dtype=np.float64)

    ordered = np.fromiter(sorted(index_set), dtype=np.int64)
    return vertices[ordered]


def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SVD 拟合平面，返回平面点与单位法线。"""
    if len(points) < 3:
        raise ValueError("底面点数不足，无法精对齐")

    center = points.mean(axis=0)
    centered = points - center
    _, singular_values, vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    if len(singular_values) < 2 or singular_values[1] < 1e-12:
        raise ValueError("底面点近似共线，无法精对齐")
    return center, normalize(vectors[-1])


def _lowest_band_points(vertices: np.ndarray) -> np.ndarray:
    """取最低 2% 高度带内的顶点作为底面候选点。"""
    min_z = float(vertices[:, 2].min())
    z_range = float(vertices[:, 2].max() - min_z)
    threshold = max(z_range * 0.02, 1e-6)
    return vertices[vertices[:, 2] <= min_z + threshold]


def snap_bottom_to_xy_plane(
    obj: "bpy.types.Object",
    mesh_data: MeshData,
    face_indices: list[int],
) -> None:
    """
    将检测到的底面精确贴合世界 XOY 平面。

    步骤:
        1. 拟合底面平面
        2. 使法线对齐 +Z，并保证物体主体位于 Z 正方向
        3. 平移使底面平均高度为 Z=0
    """
    from mathutils import Matrix, Vector

    points = collect_face_world_vertices(
        mesh_data,
        face_indices,
        obj.data.polygons,
    )
    if len(points) < 3:
        points = _lowest_band_points(mesh_data["vertices"])
    if len(points) < 3:
        raise ValueError("无法找到可用于精对齐的底面点")

    plane_point, plane_normal = fit_plane_svd(points)

    # 物体主体应位于平面法线正侧，对齐到 +Z 后物体朝上。
    centroid = mesh_data["centroid"]
    if float(np.dot(centroid - plane_point, plane_normal)) < 0.0:
        plane_normal = -plane_normal

    correction = rotation_align_vector_to_axis(
        plane_normal,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    apply_rotation_to_object(obj, correction, plane_point)

    # 旋转后重新读取世界坐标，把底面精确落到 Z=0。
    refreshed = extract_mesh_data(obj)
    grounded = collect_face_world_vertices(
        refreshed,
        face_indices,
        obj.data.polygons,
    )
    if len(grounded) < 3:
        grounded = _lowest_band_points(refreshed["vertices"])

    ground_z = float(grounded[:, 2].mean())
    obj.matrix_world = (
        Matrix.Translation(Vector((0.0, 0.0, -ground_z)))
        @ obj.matrix_world
    )
