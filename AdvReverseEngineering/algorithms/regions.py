# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""
领域（Region）自动分割算法。

参考 Geomagic Design X 的 Auto Segment，并以 Blender 视图叠加层
「线框」阈值作为默认硬边判据：
    1. 边保护法线平滑，抑制扫描噪声但不软化硬边
    2. 按 Blender edge_fac 公式把相邻面法线夹角映射为线框可见性
    3. 种子区域生长：只越过「线框不可见」的平坦边；双重法线判据防泄漏
    4. 按网格总面积占比过滤离散小领域
    5. 为有效领域生成稳定可复现的随机色
"""

from __future__ import annotations

from collections import deque
from math import sqrt
from typing import TypedDict

import numpy as np

from ..utils.mesh import FaceTopology


REGION_IGNORED_ID = -1
COLOR_SEED = 20260717
# Blender extract_mesh_vbo_edge_fac.cc::edge_factor_calc 中的缩放常数：
# fac = clamp(200*(dot-1)+1, 0, 1)；约 acos(0.995)≈5.73° 以上恒为硬边。
_BLENDER_WIRE_FAC_SCALE = 200.0
_BLENDER_WIRE_FAC_HIDE = 254.0 / 255.0


class RegionSegmentationResult(TypedDict):
    """领域分割结果。"""

    region_ids: np.ndarray
    region_count: int
    ignored_face_count: int
    ignored_region_count: int
    colors: np.ndarray
    total_area: float


def blender_wire_edge_fac(cosine: np.ndarray | float) -> np.ndarray | float:
    """
    复刻 Blender 线框边因子 edge_factor_calc。

    cosine 为相邻面单位法线点积；返回 [0, 254/255]：
    越接近 0 表示越硬（线框更易显示），接近 1 表示越平坦。
    """
    cosine_arr = np.asarray(cosine, dtype=np.float64)
    fac = (_BLENDER_WIRE_FAC_SCALE * (cosine_arr - 1.0)) + 1.0
    fac = np.clip(fac, 0.0, 1.0) * _BLENDER_WIRE_FAC_HIDE
    if np.ndim(cosine) == 0:
        return float(fac)
    return fac.astype(np.float64, copy=False)


def blender_wire_step_param(wireframe_threshold: float) -> float:
    """
    复刻 overlay_wireframe.hh::wire_discard_threshold_get。

    着色器中边可见条件为：edge_fac <= wire_step_param。
    """
    threshold = float(np.sqrt(abs(float(wireframe_threshold))))
    return float(threshold * _BLENDER_WIRE_FAC_HIDE)


def wireframe_threshold_to_cos_limit(wireframe_threshold: float) -> float:
    """
    把视图叠加层线框阈值 T 换成「可合并」的最小法线点积。

    边在线框中可见（应作为领域边界）当 fac <= sqrt(T)*254/255，
    等价于（未触底时）cosine <= 1 + (sqrt(T)-1)/200。
    区域生长仅允许越过更平坦的边，即 cosine > 该值。
    """
    step = float(np.sqrt(abs(float(wireframe_threshold))))
    # fac = clamp(200*(c-1)+1,0,1)；fac > step 才能合并。
    # 反解未钳制段：c > 1 + (step - 1)/200。
    return float(1.0 + (step - 1.0) / _BLENDER_WIRE_FAC_SCALE)


def angle_threshold_to_cos_limit(angle_threshold_deg: float) -> float:
    """兼容旧的角度阈值：夹角小于等于该值才允许合并。"""
    return float(np.cos(np.radians(max(float(angle_threshold_deg), 0.0))))


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
                # 判据 1: 与接壤面法线接近（对齐 Blender 线框：<= 视为硬边）
                local_dot = (
                    cand_x * face_x
                    + cand_y * face_y
                    + cand_z * face_z
                )
                if local_dot <= cos_limit:
                    continue
                # 判据 2: 与领域平均法线接近（防泄漏关键）
                if (
                    cand_x * mean_x
                    + cand_y * mean_y
                    + cand_z * mean_z
                ) <= cos_limit:
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
    angle_threshold_deg: float | None = None,
    ignore_discrete: bool = True,
    min_area_ratio: float = 0.001,
    smooth_iterations: int = 0,
    wireframe_threshold: float | None = 0.1,
) -> RegionSegmentationResult:
    """
    按 Blender 线框硬边（或兼容角度阈值）进行领域分割。

    参数:
        normals: (F, 3) 单位法线
        areas: (F,) 面面积（任意一致尺度）
        topology: 共享边邻接拓扑
        angle_threshold_deg: 旧版角度阈值（度）；仅当
            wireframe_threshold 为 None 时生效
        ignore_discrete: 是否按面积占比忽略小领域
        min_area_ratio: 相对网格总面积的最小占比阈值
        smooth_iterations: 边保护法线平滑迭代次数，
            细碎扫描网格建议 1~3，规则网格可为 0
        wireframe_threshold: 对齐视图叠加层线框滑条（0~1），
            默认 0.1；越低领域边界越接近更硬的橘色线框边
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

    if wireframe_threshold is not None:
        cos_limit = wireframe_threshold_to_cos_limit(wireframe_threshold)
        # 平滑保护角固定约 30°：可平均 <30° 的扫描噪声，
        # 又不会让 Blender 线框硬边（fac=0，约 ≥5.7°）两侧互相污染。
        smooth_limit = 30.0
    else:
        angle = 15.0 if angle_threshold_deg is None else float(angle_threshold_deg)
        cos_limit = angle_threshold_to_cos_limit(angle)
        smooth_limit = max(angle * 2.0, 30.0)

    if smooth_iterations > 0:
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


def compute_region_centroids(
    region_ids: np.ndarray,
    face_centers: np.ndarray,
    areas: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    计算每个有效领域的面积加权中心。

    返回:
        {region_id: world_xyz ndarray(3,)}
    """
    ids = np.asarray(region_ids, dtype=np.int32)
    centers = np.asarray(face_centers, dtype=np.float64)
    weights = np.maximum(np.asarray(areas, dtype=np.float64), 0.0)
    result: dict[int, np.ndarray] = {}
    if len(ids) == 0:
        return result

    valid = ids >= 0
    if not np.any(valid):
        return result

    max_id = int(ids[valid].max())
    sum_w = np.bincount(
        ids[valid],
        weights=weights[valid],
        minlength=max_id + 1,
    )
    for axis in range(3):
        axis_sum = np.bincount(
            ids[valid],
            weights=centers[valid, axis] * weights[valid],
            minlength=max_id + 1,
        )
        for region_id in range(max_id + 1):
            total = float(sum_w[region_id])
            if total < 1e-18:
                continue
            if region_id not in result:
                result[region_id] = np.zeros(3, dtype=np.float64)
            result[region_id][axis] = float(axis_sum[region_id] / total)
    return result


def compute_region_label_anchors(
    region_ids: np.ndarray,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    areas: np.ndarray,
    offset_ratio: float = 0.015,
    min_offset: float = 1e-4,
) -> dict[int, dict]:
    """
    为每个领域选择最靠近面积加权中心的面，并把编号锚点放到其法线正方向。

    返回:
        {
            region_id: {
                "world_co": 面中心沿法线偏移后的点,
                "face_center": 面中心,
                "normal": 单位法线,
                "face_index": 选中的面索引,
            }
        }
    """
    centroids = compute_region_centroids(region_ids, face_centers, areas)
    ids = np.asarray(region_ids, dtype=np.int32)
    centers = np.asarray(face_centers, dtype=np.float64)
    normals = np.asarray(face_normals, dtype=np.float64)
    if len(centroids) == 0 or len(ids) == 0:
        return {}

    # 偏移量随模型尺度变化，保证数字略离开表面。
    extent = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    offset = max(extent * float(offset_ratio), float(min_offset))

    anchors: dict[int, dict] = {}
    for region_id, centroid in centroids.items():
        mask = ids == int(region_id)
        if not np.any(mask):
            continue
        face_indices = np.flatnonzero(mask)
        region_centers = centers[face_indices]
        distances = np.linalg.norm(region_centers - centroid, axis=1)
        best_local = int(np.argmin(distances))
        face_index = int(face_indices[best_local])
        normal = normals[face_index]
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            normal = normal / length
        face_center = centers[face_index]
        anchors[int(region_id)] = {
            "world_co": face_center + normal * offset,
            "face_center": face_center.copy(),
            "normal": normal.copy(),
            "face_index": face_index,
        }
    return anchors


def compact_region_ids(
    region_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    将有效领域 ID 压缩为连续的 0..K-1。

    返回:
        (新标签, 旧ID->新ID映射表, 新领域数)
        映射表长度 = max(old_id)+1，忽略槽为 REGION_IGNORED_ID。
    """
    ids = np.asarray(region_ids, dtype=np.int32).copy()
    if len(ids) == 0:
        return ids, np.empty(0, dtype=np.int32), 0

    valid = ids >= 0
    if not np.any(valid):
        return ids, np.empty(0, dtype=np.int32), 0

    unique = np.unique(ids[valid])
    remap = np.full(int(unique.max()) + 1, REGION_IGNORED_ID, dtype=np.int32)
    remap[unique] = np.arange(len(unique), dtype=np.int32)
    ids[valid] = remap[ids[valid]]
    return ids, remap, int(len(unique))


def remap_region_colors(
    colors: np.ndarray,
    id_remap: np.ndarray,
    new_count: int,
) -> np.ndarray:
    """
    按 compact 映射重排颜色表，保留锚点原色。

    id_remap[old_id] = new_id；未使用的旧色丢弃。
    """
    palette = np.asarray(colors, dtype=np.float32)
    if new_count <= 0:
        return np.empty((0, 4), dtype=np.float32)

    remapped = np.zeros((new_count, 4), dtype=np.float32)
    if len(palette) == 0 or len(id_remap) == 0:
        return generate_region_colors(new_count)

    for old_id, new_id in enumerate(id_remap.tolist()):
        if new_id < 0 or new_id >= new_count:
            continue
        if old_id < len(palette):
            remapped[new_id] = palette[old_id]
        else:
            remapped[new_id] = generate_region_colors(1)[0]
    return remapped


def merge_region_ids(
    region_ids: np.ndarray,
    colors: np.ndarray,
    anchor_id: int,
    source_id: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    将 source 领域合入 anchor，并压缩编号。

    返回:
        (新标签, 新颜色表, 新领域数, 压缩后锚点ID)
    """
    ids = np.asarray(region_ids, dtype=np.int32).copy()
    palette = np.asarray(colors, dtype=np.float32)
    if anchor_id < 0 or source_id < 0:
        raise ValueError("合并领域编号无效")
    if anchor_id == source_id:
        compacted, remap, count = compact_region_ids(ids)
        new_colors = remap_region_colors(palette, remap, count)
        new_anchor = int(remap[anchor_id]) if anchor_id < len(remap) else -1
        return compacted, new_colors, count, new_anchor

    ids[ids == source_id] = anchor_id
    compacted, remap, count = compact_region_ids(ids)
    new_colors = remap_region_colors(palette, remap, count)
    new_anchor = int(remap[anchor_id]) if anchor_id < len(remap) else -1
    return compacted, new_colors, count, new_anchor
