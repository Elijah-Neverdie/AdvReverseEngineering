# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""网格数据批量读取工具。"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class MeshData(TypedDict):
    """一次读取的网格缓存数据（世界空间）。"""

    vertices: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    face_centers: np.ndarray
    centroid: np.ndarray


def subsample_array(
    array: np.ndarray,
    max_count: int,
    seed: int = 42,
) -> np.ndarray:
    """
    对数组随机下采样，控制计算量。

    参数:
        array: 输入数组 (N, ...)
        max_count: 最大保留数量
        seed: 随机种子，保证可复现
    """
    count = len(array)
    if count <= max_count:
        return array

    rng = np.random.default_rng(seed)
    indices = rng.choice(count, max_count, replace=False)
    return array[indices]


def extract_mesh_data(obj: "bpy.types.Object") -> MeshData:
    """
    一次性批量读取网格顶点和面法线（世界空间）。

    使用 foreach_get 避免逐元素 Python API 访问，适合大网格。
    """
    import bpy

    mesh = obj.data
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError("网格没有顶点数据")

    matrix = np.array(obj.matrix_world, dtype=np.float64)
    rotation = matrix[:3, :3]
    try:
        normal_matrix = np.linalg.inv(rotation).T
    except np.linalg.LinAlgError:
        normal_matrix = np.linalg.pinv(rotation).T

    vert_count = len(mesh.vertices)
    coords = np.empty(vert_count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(vert_count, 3)

    ones = np.ones((vert_count, 1), dtype=np.float64)
    homogeneous = np.hstack((coords, ones))
    vertices = (matrix @ homogeneous.T).T[:, :3]

    face_count = len(mesh.polygons)
    normals = np.empty(face_count * 3, dtype=np.float64)
    areas = np.empty(face_count, dtype=np.float64)
    face_centers = np.empty(face_count * 3, dtype=np.float64)
    mesh.polygons.foreach_get("normal", normals)
    mesh.polygons.foreach_get("area", areas)
    mesh.polygons.foreach_get("center", face_centers)
    normals = normals.reshape(face_count, 3)
    face_centers = face_centers.reshape(face_count, 3)
    normals = (normal_matrix @ normals.T).T
    center_ones = np.ones((face_count, 1), dtype=np.float64)
    homogeneous_centers = np.hstack((face_centers, center_ones))
    face_centers = (matrix @ homogeneous_centers.T).T[:, :3]

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    normals /= lengths

    return MeshData(
        vertices=vertices,
        normals=normals,
        areas=areas,
        face_centers=face_centers,
        centroid=vertices.mean(axis=0),
    )


def extract_selected_world_points(
    obj: "bpy.types.Object",
) -> np.ndarray:
    """
    读取编辑模式下选中的点、边或面所包含的唯一顶点。

    直接访问当前 EditMesh，确保尚未提交到 Mesh datablock 的选择状态
    也能被正确读取。
    """
    import bmesh

    if obj.mode != "EDIT":
        return np.empty((0, 3), dtype=np.float64)

    mesh = obj.data
    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
    bm = bmesh.from_edit_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    selected_indices: set[int] = {
        vertex.index for vertex in bm.verts if vertex.select
    }
    for edge in bm.edges:
        if edge.select:
            selected_indices.update(vertex.index for vertex in edge.verts)
    for face in bm.faces:
        if face.select:
            selected_indices.update(vertex.index for vertex in face.verts)

    if not selected_indices:
        return np.empty((0, 3), dtype=np.float64)

    ordered_indices = np.fromiter(
        sorted(selected_indices),
        dtype=np.int64,
    )
    local = np.empty((len(ordered_indices), 3), dtype=np.float64)
    for row, vertex_index in enumerate(ordered_indices):
        local[row] = bm.verts[int(vertex_index)].co

    matrix = np.array(obj.matrix_world, dtype=np.float64)
    ones = np.ones((len(local), 1), dtype=np.float64)
    homogeneous = np.hstack((local, ones))
    return (matrix @ homogeneous.T).T[:, :3]
