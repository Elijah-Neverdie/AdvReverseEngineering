# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""数学与几何变换工具。"""

from __future__ import annotations

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """单位化向量，零向量时返回世界 Z 轴。"""
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / length


def rotation_align_vector_to_axis(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    计算 3x3 旋转矩阵，将 source 方向旋转至 target 方向。

    使用 Rodrigues 公式，处理平行与反平行特殊情况。
    """
    src = normalize(source)
    tgt = normalize(target)
    dot = float(np.clip(np.dot(src, tgt), -1.0, 1.0))

    if dot > 0.999999:
        return np.eye(3, dtype=np.float64)

    if dot < -0.999999:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = normalize(axis)
        # 180° 旋转
        cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * cross @ cross

    axis = normalize(np.cross(src, tgt))
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + cross + cross @ cross * (1.0 / (1.0 + dot))


def rotation_align_to_negative_z(direction: np.ndarray) -> np.ndarray:
    """将给定方向对齐到世界 -Z（底面朝下）。"""
    target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return rotation_align_vector_to_axis(direction, target)


def rotation_z(angle_rad: float) -> np.ndarray:
    """绕世界 Z 轴旋转矩阵。"""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def aabb_volume(vertices: np.ndarray) -> float:
    """计算轴对齐包围盒体积。"""
    if len(vertices) == 0:
        return 0.0
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    size = maxs - mins
    return float(size[0] * size[1] * size[2])


def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """欧拉角 XYZ（弧度）转 3x3 旋转矩阵。"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def ensure_right_handed(rotation: np.ndarray) -> np.ndarray:
    """确保旋转矩阵为右手系。"""
    if np.linalg.det(rotation) < 0.0:
        fixed = rotation.copy()
        fixed[:, 0] *= -1.0
        return fixed
    return rotation


def build_full_orientation(
    ground_normal: np.ndarray,
    vertices: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """
    由底面法线构建完整摆正矩阵。

    步骤:
        1. 底面法线对齐 -Z
        2. 在 XY 平面内将最长轴对齐 X
    """
    rot_align = rotation_align_to_negative_z(ground_normal)
    rotated = (vertices - centroid) @ rot_align.T

    xy = rotated[:, :2]
    if len(xy) < 2:
        return rot_align

    cov = np.cov(xy.T)
    if cov.shape != (2, 2):
        return rot_align

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.arctan2(major[1], major[0]))
    rot_spin = rotation_z(-angle)
    return rot_spin @ rot_align


def rotation_to_matrix_4x4(rotation: np.ndarray):
    """将 3x3 NumPy 旋转矩阵转为 Blender 4x4 Matrix。"""
    from mathutils import Matrix

    mat = Matrix.Identity(4)
    for row in range(3):
        for col in range(3):
            mat[row][col] = float(rotation[row, col])
    return mat


def apply_rotation_to_object(
    obj: "bpy.types.Object",
    rotation: np.ndarray,
    pivot: np.ndarray,
) -> None:
    """绕世界空间 pivot 点施加旋转到对象。"""
    from mathutils import Matrix, Vector

    rot_mat = rotation_to_matrix_4x4(rotation)
    pivot_vec = Vector(pivot.tolist())
    translation = Matrix.Translation(pivot_vec)
    translation_inv = Matrix.Translation(-pivot_vec)
    obj.matrix_world = translation @ rot_mat @ translation_inv @ obj.matrix_world
