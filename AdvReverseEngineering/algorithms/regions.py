# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""
领域（Region）自动分割算法。

参考 Geomagic Design X 的 Auto Segment 思路：
    1. 边保护法线平滑，抑制扫描噪声但不软化硬边
    2. 种子区域生长：候选面须同时接近相邻面法线与领域平均法线，
       杜绝链式合并越过圆角/硬边的“泄漏”
    3. 按网格总面积占比过滤离散小领域
    4. 为有效领域生成稳定可复现的随机色
"""

from __future__ import annotations

from collections import deque
from math import sqrt
from typing import TypedDict

import numpy as np

from ..utils.mesh import FaceTopology


REGION_IGNORED_ID = -1
COLOR_SEED = 20260717


class RegionSegmentationResult(TypedDict):
    """领域分割结果。"""

    region_ids: np.ndarray
    region_count: int
    ignored_face_count: int
    ignored_region_count: int
    colors: np.ndarray
    total_area: float


def smooth_face_normals(
    normals: np.ndarray,
    areas: np.ndarray,
    topology: FaceTopology,
    iterations: int,
    edge_angle_limit_deg: float = 30.0,
) -> np.ndarray:
    """
    边保护的面法线平滑，抑制扫描噪声。

    仅对夹角不超过 edge_angle_limit_deg 的邻接面互相平均，
    硬边（如 90° 棱）两侧法线不会互相污染。全程 NumPy 向量化。
    """
    face_count = len(normals)
    if iterations <= 0 or face_count == 0:
        return np.asarray(normals, dtype=np.float64)

    face_a = topology["edge_face_a"]
    face_b = topology["edge_face_b"]
    if len(face_a) == 0:
        return np.asarray(normals, dtype=np.float64)

    result = np.asarray(normals, dtype=np.float64).copy()
    weights = np.maximum(np.asarray(areas, dtype=np.float64), 1e-12)
    cos_limit = float(np.cos(np.radians(max(edge_angle_limit_deg, 0.0))))

    for _ in range(int(iterations)):
        dots = np.sum(result[face_a] * result[face_b], axis=1)
        mask = dots >= cos_limit
        keep_a = face_a[mask]
        keep_b = face_b[mask]

        accumulated = result * weights[:, None]
        if len(keep_a):
            for axis in range(3):
                accumulated[:, axis] += np.bincount(
                    keep_a,
                    weights=result[keep_b, axis] * weights[keep_b],
                    minlength=face_count,
                )
                accumulated[:, axis] += np.bincount(
                    keep_b,
                    weights=result[keep_a, axis] * weights[keep_a],
                    minlength=face_count,
                )

        lengths = np.linalg.norm(accumulated, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-12)
        result = accumulated / lengths

    return result


def _seed_order_by_flatness(
    face_count: int,
    normals: np.ndarray,
    topology: FaceTopology,
) -> np.ndarray:
    """
    种子顺序：局部法线变化最小（最平坦）的面优先。

    先从平坦处生长可以让平面/圆柱等特征先占据领域，
    圆角过渡带留到最后自成小领域，便于按面积过滤。
    """
    face_a = topology["edge_face_a"]
    face_b = topology["edge_face_b"]
    min_dot = np.ones(face_count, dtype=np.float64)
    if len(face_a):
        dots = np.sum(normals[face_a] * normals[face_b], axis=1)
        np.minimum.at(min_dot, face_a, dots)
        np.minimum.at(min_dot, face_b, dots)
    return np.argsort(-min_dot, kind="stable")


def _grow_regions(
    normals: np.ndarray,
    areas: np.ndarray,
    topology: FaceTopology,
    cos_limit: float,
) -> np.ndarray:
    """
    种子区域生长，输出稠密领域标签（0..N-1）。

    双重判据：
        1. 候选面与其接壤面法线夹角 <= 阈值（局部平滑性）
        2. 候选面与领域面积加权平均法线夹角 <= 阈值（全局一致性）
    判据 2 阻止链式合并沿圆角逐面漂移、越过硬边。

    内层循环使用 Python list 索引与标量运算，
    实测比逐元素访问 NumPy 标量快数倍，可支撑百万面网格。
    """
    face_count = len(normals)
    offsets = topology["adjacency_offsets"].tolist()
    adjacency = topology["adjacency_indices"].tolist()

    nx = normals[:, 0].tolist()
    ny = normals[:, 1].tolist()
    nz = normals[:, 2].tolist()
    weight = np.maximum(
        np.asarray(areas, dtype=np.float64),
        1e-12,
    ).tolist()

    seed_order = _seed_order_by_flatness(face_count, normals, topology)

    labels = [-1] * face_count
    region_id = 0

    for seed in seed_order.tolist():
        if labels[seed] != -1:
            continue
        labels[seed] = region_id
        seed_weight = weight[seed]
        sum_x = nx[seed] * seed_weight
        sum_y = ny[seed] * seed_weight
        sum_z = nz[seed] * seed_weight

        queue = deque((seed,))
        while queue:
            face = queue.popleft()
            length = sqrt(
                sum_x * sum_x + sum_y * sum_y + sum_z * sum_z
            )
            if length < 1e-12:
                mean_x, mean_y, mean_z = nx[face], ny[face], nz[face]
            else:
                mean_x = sum_x / length
                mean_y = sum_y / length
                mean_z = sum_z / length

            face_x = nx[face]
            face_y = ny[face]
            face_z = nz[face]

            for slot in range(offsets[face], offsets[face + 1]):
                neighbor = adjacency[slot]
                if labels[neighbor] != -1:
                    continue
                cand_x = nx[neighbor]
                cand_y = ny[neighbor]
                cand_z = nz[neighbor]
                # 判据 1: 与接壤面法线接近
                if (
                    cand_x * face_x
                    + cand_y * face_y
                    + cand_z * face_z
                ) < cos_limit:
                    continue
                # 判据 2: 与领域平均法线接近（防泄漏关键）
                if (
                    cand_x * mean_x
                    + cand_y * mean_y
                    + cand_z * mean_z
                ) < cos_limit:
                    continue
                labels[neighbor] = region_id
                cand_weight = weight[neighbor]
                sum_x += cand_x * cand_weight
                sum_y += cand_y * cand_weight
                sum_z += cand_z * cand_weight
                queue.append(neighbor)

        region_id += 1

    return np.asarray(labels, dtype=np.int32)


def generate_region_colors(
    region_count: int,
    seed: int = COLOR_SEED,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    生成稳定、可区分的半透明随机色。

    使用黄金分割色相偏移，避免相邻编号颜色过于接近。
    """
    if region_count <= 0:
        return np.empty((0, 4), dtype=np.float32)

    rng = np.random.default_rng(seed)
    # 固定偏移后按黄金角排布色相，再轻微扰动饱和度/明度。
    base = float(rng.random())
    hues = (base + np.arange(region_count) * 0.61803398875) % 1.0
    saturations = 0.55 + 0.35 * rng.random(region_count)
    values = 0.70 + 0.25 * rng.random(region_count)

    colors = np.empty((region_count, 4), dtype=np.float32)
    for index in range(region_count):
        hue = float(hues[index])
        sat = float(saturations[index])
        val = float(values[index])
        sector = int(hue * 6.0)
        frac = hue * 6.0 - sector
        p = val * (1.0 - sat)
        q = val * (1.0 - sat * frac)
        t = val * (1.0 - sat * (1.0 - frac))
        sector %= 6
        if sector == 0:
            r, g, b = val, t, p
        elif sector == 1:
            r, g, b = q, val, p
        elif sector == 2:
            r, g, b = p, val, t
        elif sector == 3:
            r, g, b = p, q, val
        elif sector == 4:
            r, g, b = t, p, val
        else:
            r, g, b = val, p, q
        colors[index] = (r, g, b, alpha)
    return colors


def segment_regions_by_normal(
    normals: np.ndarray,
    areas: np.ndarray,
    topology: FaceTopology,
    angle_threshold_deg: float = 15.0,
    ignore_discrete: bool = True,
    min_area_ratio: float = 0.001,
    smooth_iterations: int = 0,
) -> RegionSegmentationResult:
    """
    按法线阈值进行领域分割（种子区域生长，防泄漏）。

    参数:
        normals: (F, 3) 单位法线
        areas: (F,) 面面积（任意一致尺度）
        topology: 共享边邻接拓扑
        angle_threshold_deg: 法线最大夹角（度），同时约束
            相邻面与领域平均法线两个判据
        ignore_discrete: 是否按面积占比忽略小领域
        min_area_ratio: 相对网格总面积的最小占比阈值
        smooth_iterations: 边保护法线平滑迭代次数，
            细碎扫描网格建议 1~3，规则网格可为 0
    """
    face_count = len(normals)
    if face_count == 0:
        return RegionSegmentationResult(
            region_ids=np.empty(0, dtype=np.int32),
            region_count=0,
            ignored_face_count=0,
            ignored_region_count=0,
            colors=np.empty((0, 4), dtype=np.float32),
            total_area=0.0,
        )

    normals = np.asarray(normals, dtype=np.float64)
    areas = np.maximum(np.asarray(areas, dtype=np.float64), 0.0)
    total_area = float(areas.sum())
    if total_area < 1e-18:
        areas = np.ones(face_count, dtype=np.float64)
        total_area = float(face_count)

    cos_limit = float(
        np.cos(np.radians(max(angle_threshold_deg, 0.0)))
    )

    # 平滑保护角取阈值两倍（至少 30°）：既能平均噪声，
    # 又不会让真正的硬边两侧互相污染。
    if smooth_iterations > 0:
        smooth_limit = max(float(angle_threshold_deg) * 2.0, 30.0)
        normals = smooth_face_normals(
            normals,
            areas,
            topology,
            iterations=smooth_iterations,
            edge_angle_limit_deg=smooth_limit,
        )

    if len(topology["edge_face_a"]) == 0:
        temp_ids = np.arange(face_count, dtype=np.int32)
    else:
        temp_ids = _grow_regions(normals, areas, topology, cos_limit)

    region_total = int(temp_ids.max()) + 1 if face_count else 0
    region_areas = np.bincount(
        temp_ids,
        weights=areas,
        minlength=region_total,
    )

    keep_mask = np.ones(region_total, dtype=bool)
    ignored_region_count = 0
    if ignore_discrete and min_area_ratio > 0.0:
        area_limit = total_area * float(min_area_ratio)
        keep_mask = region_areas >= area_limit
        ignored_region_count = int((~keep_mask).sum())

    remap = np.full(region_total, REGION_IGNORED_ID, dtype=np.int32)
    kept_indices = np.flatnonzero(keep_mask)
    remap[kept_indices] = np.arange(len(kept_indices), dtype=np.int32)
    region_ids = remap[temp_ids]

    ignored_face_count = int(np.count_nonzero(region_ids < 0))
    region_count = int(len(kept_indices))
    colors = generate_region_colors(region_count)

    return RegionSegmentationResult(
        region_ids=region_ids,
        region_count=region_count,
        ignored_face_count=ignored_face_count,
        ignored_region_count=ignored_region_count,
        colors=colors,
        total_area=total_area,
    )
