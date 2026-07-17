# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""
领域（Region）自动分割算法。

参考 Geomagic Design X 的 Auto Segment 思路：
    1. 以共享边为邻接
    2. 相邻面法线夹角不超过阈值则合并
    3. 按网格总面积占比过滤离散小领域
    4. 为有效领域生成稳定可复现的随机色
"""

from __future__ import annotations

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


def _union_find_labels(
    face_count: int,
    face_a: np.ndarray,
    face_b: np.ndarray,
    merge_mask: np.ndarray,
) -> np.ndarray:
    """对可合并的邻接边执行并查集，返回每面根标签。"""
    parent = np.arange(face_count, dtype=np.int32)

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        # 路径压缩
        while parent[index] != index:
            nxt = int(parent[index])
            parent[index] = root
            index = nxt
        return root

    for left, right, should_merge in zip(
        face_a.tolist(),
        face_b.tolist(),
        merge_mask.tolist(),
    ):
        if not should_merge:
            continue
        root_a = find(int(left))
        root_b = find(int(right))
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    labels = np.empty(face_count, dtype=np.int32)
    for index in range(face_count):
        labels[index] = find(index)
    return labels


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
) -> RegionSegmentationResult:
    """
    按法线阈值进行领域分割。

    参数:
        normals: (F, 3) 单位法线
        areas: (F,) 面面积（任意一致尺度）
        topology: 共享边邻接拓扑
        angle_threshold_deg: 相邻面法线最大夹角（度）
        ignore_discrete: 是否按面积占比忽略小领域
        min_area_ratio: 相对网格总面积的最小占比阈值
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

    face_a = topology["edge_face_a"]
    face_b = topology["edge_face_b"]
    if len(face_a) == 0:
        labels = np.arange(face_count, dtype=np.int32)
    else:
        dots = np.sum(
            normals[face_a] * normals[face_b],
            axis=1,
        )
        cos_limit = float(
            np.cos(np.radians(max(angle_threshold_deg, 0.0)))
        )
        merge_mask = dots >= cos_limit
        labels = _union_find_labels(
            face_count,
            face_a,
            face_b,
            merge_mask,
        )

    # 将并查集根标签压缩为稠密临时编号，便于面积聚合。
    unique_roots, inverse = np.unique(labels, return_inverse=True)
    temp_ids = inverse.astype(np.int32, copy=False)
    region_areas = np.bincount(
        temp_ids,
        weights=areas,
        minlength=len(unique_roots),
    )

    keep_mask = np.ones(len(unique_roots), dtype=bool)
    ignored_region_count = 0
    if ignore_discrete and min_area_ratio > 0.0:
        area_limit = total_area * float(min_area_ratio)
        keep_mask = region_areas >= area_limit
        ignored_region_count = int((~keep_mask).sum())

    remap = np.full(len(unique_roots), REGION_IGNORED_ID, dtype=np.int32)
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
