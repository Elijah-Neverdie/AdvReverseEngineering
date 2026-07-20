# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""画笔种子边、硬边代价补全与领域连通分量拆分。"""

from __future__ import annotations

import colorsys
import heapq
from typing import Iterable

import numpy as np

from .regions import (
    REGION_IGNORED_ID,
    blender_wire_edge_fac,
    blender_wire_step_param,
    generate_region_colors,
)


def _contrast_color(base_rgba: np.ndarray, offset_index: int = 0) -> np.ndarray:
    """
    生成与原领域强对比的颜色（互补色相 + 高饱和度）。

    拆分预览用它把新分出的领域与原领域清晰区分开。
    """
    base = np.asarray(base_rgba, dtype=np.float32).ravel()
    r, g, b = float(base[0]), float(base[1]), float(base[2])
    alpha = float(base[3]) if len(base) >= 4 else 0.55
    h, s, v = colorsys.rgb_to_hsv(
        min(max(r, 0.0), 1.0),
        min(max(g, 0.0), 1.0),
        min(max(b, 0.0), 1.0),
    )
    # 互补色相，多分量再各自错开，避免撞色。
    h = (h + 0.5 + 0.137 * offset_index) % 1.0
    s = min(1.0, max(0.75, s * 1.25 + 0.2))
    v = min(1.0, max(0.85, v + 0.15))
    nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
    return np.array([nr, ng, nb, alpha], dtype=np.float32)


def prepare_edge_costs(
    topology: dict,
    normals: np.ndarray,
    face_centers: np.ndarray,
    region_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    为每条共享边计算硬边偏好代价与中点。

    代价对齐 Blender 线框 edge_fac：越硬（fac→0）代价越低；
    跨领域边抬高代价，供 Dijkstra 规避，但仍可作终止。
    """
    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    edge_count = len(face_a)
    if edge_count == 0:
        empty_cost = np.empty(0, dtype=np.float64)
        empty_mid = np.empty((0, 3), dtype=np.float64)
        return empty_cost, empty_mid

    n0 = normals[face_a]
    n1 = normals[face_b]
    dots = np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0)
    # 0=硬边，接近 1=平坦（与视图线框一致）
    fac = np.asarray(blender_wire_edge_fac(dots), dtype=np.float64)

    centers = np.asarray(face_centers, dtype=np.float64)
    mids = 0.5 * (centers[face_a] + centers[face_b])

    # 基础代价：硬边接近 0.05，平坦接近 1.0
    costs = 0.05 + 0.95 * fac

    rid = np.asarray(region_ids, dtype=np.int32)
    cross = rid[face_a] != rid[face_b]
    costs = np.where(cross, costs + 50.0, costs)
    return costs.astype(np.float64, copy=False), mids.astype(np.float64, copy=False)


def candidate_hard_edges(
    topology: dict,
    normals: np.ndarray,
    region_ids: np.ndarray,
    target_rid: int,
    wireframe_threshold: float,
) -> np.ndarray:
    """
    目标领域内部的候选硬边（与「识别领域」同一套 Blender 线框判据）。

    边在线框中可见（edge_fac <= wire_step(T)）则入选。
    Ctrl+滚轮增大 T 会纳入更缓的折棱（例如曲面上的棱线）。
    """
    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    if len(face_a) == 0:
        return np.empty(0, dtype=np.int32)

    rid = np.asarray(region_ids, dtype=np.int32)
    target = int(target_rid)
    internal = (rid[face_a] == target) & (rid[face_b] == target)
    if not np.any(internal):
        return np.empty(0, dtype=np.int32)

    normals_arr = np.asarray(normals, dtype=np.float64)
    n0 = normals_arr[face_a]
    n1 = normals_arr[face_b]
    dots = np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0)
    fac = np.asarray(blender_wire_edge_fac(dots), dtype=np.float64)
    step = float(blender_wire_step_param(wireframe_threshold))
    mask = internal & (fac <= step)
    if not np.any(mask):
        return np.empty(0, dtype=np.int32)
    return np.flatnonzero(mask).astype(np.int32)


def group_candidate_edge_chains(
    candidate_edges: np.ndarray,
    edge_vert_a: np.ndarray,
    edge_vert_b: np.ndarray,
) -> list[np.ndarray]:
    """
    将候选硬边按共享顶点连通性合并为连续棱线链。

    同一折棱上的相邻小边会归为一条链，供点选时整链选中。
    """
    cand = np.asarray(candidate_edges, dtype=np.int32)
    if len(cand) == 0:
        return []

    cand_set = {int(e) for e in cand.tolist()}
    vert_to_edges: dict[int, list[int]] = {}
    edge_vert_a = np.asarray(edge_vert_a, dtype=np.int32)
    edge_vert_b = np.asarray(edge_vert_b, dtype=np.int32)
    for edge_index in cand_set:
        va = int(edge_vert_a[edge_index])
        vb = int(edge_vert_b[edge_index])
        vert_to_edges.setdefault(va, []).append(edge_index)
        vert_to_edges.setdefault(vb, []).append(edge_index)

    visited: set[int] = set()
    chains: list[np.ndarray] = []
    for start in cand.tolist():
        start = int(start)
        if start in visited:
            continue
        component: list[int] = []
        queue = [start]
        visited.add(start)
        while queue:
            edge_index = queue.pop()
            component.append(edge_index)
            for vert in (
                int(edge_vert_a[edge_index]),
                int(edge_vert_b[edge_index]),
            ):
                for neighbor in vert_to_edges.get(vert, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        chains.append(np.asarray(sorted(component), dtype=np.int32))
    return chains


def chain_splits_region(
    chain_edges: np.ndarray,
    region_ids: np.ndarray,
    topology: dict,
    target_rid: int,
) -> bool:
    """链作为切线能否把目标领域恰好一分为二。"""
    chain = np.asarray(chain_edges, dtype=np.int32)
    if len(chain) == 0:
        return False

    ids = np.asarray(region_ids, dtype=np.int32)
    target = int(target_rid)
    members = np.flatnonzero(ids == target)
    if len(members) == 0:
        return False

    cut_set = {int(e) for e in chain.tolist()}
    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    offsets = np.asarray(topology["adjacency_offsets"], dtype=np.int32)
    adj = np.asarray(topology["adjacency_indices"], dtype=np.int32)
    face_count = len(ids)

    barrier_pairs: set[tuple[int, int]] = set()
    for edge_index in cut_set:
        if edge_index < 0 or edge_index >= len(face_a):
            continue
        fa = int(face_a[edge_index])
        fb = int(face_b[edge_index])
        barrier_pairs.add((min(fa, fb), max(fa, fb)))

    visited = np.zeros(face_count, dtype=bool)
    components = 0
    for start in members.tolist():
        if visited[start]:
            continue
        components += 1
        stack = [int(start)]
        visited[start] = True
        while stack:
            face = stack.pop()
            begin = int(offsets[face])
            end = int(offsets[face + 1])
            for neighbor in adj[begin:end].tolist():
                neighbor = int(neighbor)
                if visited[neighbor] or int(ids[neighbor]) != target:
                    continue
                pair = (min(face, neighbor), max(face, neighbor))
                if pair in barrier_pairs:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)
    return components == 2


def filter_bisecting_candidate_chains(
    candidate_edges: np.ndarray,
    region_ids: np.ndarray,
    topology: dict,
    target_rid: int,
    edge_vert_a: np.ndarray,
    edge_vert_b: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    仅保留能把目标领域一分为二的候选棱线链。

    返回 (扁平边索引, 链列表)。
    """
    chains = group_candidate_edge_chains(
        candidate_edges,
        edge_vert_a,
        edge_vert_b,
    )
    good = [
        chain
        for chain in chains
        if chain_splits_region(chain, region_ids, topology, target_rid)
    ]
    if not good:
        return np.empty(0, dtype=np.int32), []
    flat = np.concatenate(good).astype(np.int32, copy=False)
    return flat, good


def stroke_hits_to_seed_edges(
    faces: Iterable[int],
    worlds: np.ndarray,
    topology: dict,
    face_centers: np.ndarray,
    region_ids: np.ndarray,
    target_rid: int,
) -> np.ndarray:
    """
    将笔迹命中面/点转换为目标领域内的种子边索引。

    优先取相邻命中面共享边，其次取命中面到笔迹点最近的同领域内部边。
    """
    face_list = [int(f) for f in faces]
    if len(face_list) == 0:
        return np.empty(0, dtype=np.int32)

    worlds = np.asarray(worlds, dtype=np.float64)
    if len(worlds) != len(face_list):
        raise ValueError("笔迹命中面与世界坐标数量不一致")

    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    rid = np.asarray(region_ids, dtype=np.int32)
    centers = np.asarray(face_centers, dtype=np.float64)

    # 面 → 边列表
    face_to_edges: dict[int, list[int]] = {}
    for edge_index, (fa, fb) in enumerate(zip(face_a.tolist(), face_b.tolist())):
        face_to_edges.setdefault(fa, []).append(edge_index)
        face_to_edges.setdefault(fb, []).append(edge_index)

    seeds: set[int] = set()

    # 相邻命中若共享边且属于目标领域，则收为种子。
    for index in range(len(face_list) - 1):
        f0 = face_list[index]
        f1 = face_list[index + 1]
        if rid[f0] != target_rid and rid[f1] != target_rid:
            continue
        edges0 = set(face_to_edges.get(f0, ()))
        shared = edges0.intersection(face_to_edges.get(f1, ()))
        for edge_index in shared:
            fa = int(face_a[edge_index])
            fb = int(face_b[edge_index])
            if rid[fa] == target_rid or rid[fb] == target_rid:
                seeds.add(int(edge_index))

    # 每个命中点：在该面的候选边中选最近中点。
    mids = 0.5 * (centers[face_a] + centers[face_b])
    for hit_face, world in zip(face_list, worlds):
        if int(rid[hit_face]) != int(target_rid):
            continue
        candidates = face_to_edges.get(int(hit_face), [])
        if not candidates:
            continue
        best_edge = None
        best_dist = float("inf")
        for edge_index in candidates:
            fa = int(face_a[edge_index])
            fb = int(face_b[edge_index])
            # 至少一侧属于目标领域
            if rid[fa] != target_rid and rid[fb] != target_rid:
                continue
            delta = mids[edge_index] - world
            dist = float(np.dot(delta, delta))
            if dist < best_dist:
                best_dist = dist
                best_edge = int(edge_index)
        if best_edge is not None:
            seeds.add(best_edge)

    if not seeds:
        return np.empty(0, dtype=np.int32)
    return np.asarray(sorted(seeds), dtype=np.int32)


def _edge_endpoint_vertices(
    seed_edges: np.ndarray,
    edge_vert_a: np.ndarray,
    edge_vert_b: np.ndarray,
) -> list[int]:
    """种子边中出现奇数次的顶点视为路径端点。"""
    counts: dict[int, int] = {}
    for edge_index in seed_edges.tolist():
        va = int(edge_vert_a[edge_index])
        vb = int(edge_vert_b[edge_index])
        counts[va] = counts.get(va, 0) + 1
        counts[vb] = counts.get(vb, 0) + 1
    ends = [v for v, c in counts.items() if c % 2 == 1]
    if ends:
        return ends
    # 闭环：任取一个顶点作延伸起点
    if counts:
        return [next(iter(counts))]
    return []


def _is_region_boundary_edge(
    edge_index: int,
    face_a: np.ndarray,
    face_b: np.ndarray,
    region_ids: np.ndarray,
    target_rid: int,
) -> bool:
    fa = int(face_a[edge_index])
    fb = int(face_b[edge_index])
    ra = int(region_ids[fa])
    rb = int(region_ids[fb])
    return (ra == target_rid) != (rb == target_rid)


def _is_internal_region_edge(
    edge_index: int,
    face_a: np.ndarray,
    face_b: np.ndarray,
    region_ids: np.ndarray,
    target_rid: int,
) -> bool:
    fa = int(face_a[edge_index])
    fb = int(face_b[edge_index])
    return (
        int(region_ids[fa]) == target_rid
        and int(region_ids[fb]) == target_rid
    )


def filter_internal_cut_edges(
    cut_edges: np.ndarray,
    topology: dict,
    region_ids: np.ndarray,
    target_rid: int,
) -> np.ndarray:
    """去掉跨领域边界边，只保留目标领域内部切边。"""
    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    rid = np.asarray(region_ids, dtype=np.int32)
    target = int(target_rid)
    kept: list[int] = []
    for edge_index in np.asarray(cut_edges, dtype=np.int32).tolist():
        edge_index = int(edge_index)
        if edge_index < 0 or edge_index >= len(face_a):
            continue
        if (
            int(rid[int(face_a[edge_index])]) == target
            and int(rid[int(face_b[edge_index])]) == target
        ):
            kept.append(edge_index)
    if not kept:
        return np.empty(0, dtype=np.int32)
    return np.asarray(kept, dtype=np.int32)


def _edge_touches_foreign_region(
    edge_index: int,
    face_a: np.ndarray,
    face_b: np.ndarray,
    offsets: np.ndarray,
    adj: np.ndarray,
    region_ids: np.ndarray,
    target_rid: int,
) -> bool:
    """内部边是否贴着外领域（一侧面子有跨领域邻接）——周界贴边。"""
    target = int(target_rid)
    for face in (int(face_a[edge_index]), int(face_b[edge_index])):
        begin = int(offsets[face])
        end = int(offsets[face + 1])
        for neighbor in adj[begin:end].tolist():
            if int(region_ids[int(neighbor)]) != target:
                return True
    return False


def grow_ridge_cut_to_boundary(
    seed_edge: int,
    topology: dict,
    region_ids: np.ndarray,
    target_rid: int,
    edge_costs: np.ndarray,
    edge_mids: np.ndarray,
    vert_edge_offsets: np.ndarray,
    vert_edge_indices: np.ndarray,
    edge_vert_a: np.ndarray,
    edge_vert_b: np.ndarray,
    vertices: np.ndarray | None = None,
) -> np.ndarray:
    """
    从种子短边两端沿棱线方向贪心延伸，形成完整分割硬棱。

    关键：不再用「到任意领域边界的最短路」——那会横向抄近道，
    只留下极短切线。改为每步在当前顶点选取「方向最连续 × 最硬」
    的下一条边，直到碰到领域边界或无合格续边。
    """
    del edge_mids  # 保留参数签名兼容调用方
    seed = int(seed_edge)
    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    edge_vert_a = np.asarray(edge_vert_a, dtype=np.int32)
    edge_vert_b = np.asarray(edge_vert_b, dtype=np.int32)
    edge_costs = np.asarray(edge_costs, dtype=np.float64)
    if seed < 0 or seed >= len(edge_vert_a):
        return np.empty(0, dtype=np.int32)

    # 种子必须是目标领域内部边，不能是已有领域分界
    if not _is_internal_region_edge(
        seed, face_a, face_b, region_ids, target_rid
    ):
        return np.empty(0, dtype=np.int32)

    offsets = np.asarray(topology["adjacency_offsets"], dtype=np.int32)
    adj = np.asarray(topology["adjacency_indices"], dtype=np.int32)
    rid = np.asarray(region_ids, dtype=np.int32)

    verts = (
        np.asarray(vertices, dtype=np.float64)
        if vertices is not None and len(vertices)
        else None
    )
    seed_hardness = max(0.0, 1.0 - float(edge_costs[seed]))
    # 续边硬度下限：允许比种子略软（缓棱中段）
    min_hardness = max(0.04, seed_hardness * 0.22)
    soft_hardness = max(0.02, seed_hardness * 0.12)
    # 主方向约 <55° 可续；横拐（周界溢出）仍拒绝
    min_principal_align = 0.55
    min_local_align = 0.35
    # 贴周界但朝主方向前进：允许；贴周界且偏横：拒绝
    min_perimeter_forward_align = 0.75
    max_steps = 200000

    def other_vert(edge_index: int, from_vert: int) -> int:
        va = int(edge_vert_a[edge_index])
        vb = int(edge_vert_b[edge_index])
        return vb if va == from_vert else va

    def edge_dir(edge_index: int, from_vert: int) -> np.ndarray | None:
        if verts is None:
            return None
        other = other_vert(edge_index, from_vert)
        if other < 0 or other >= len(verts) or from_vert >= len(verts):
            return None
        delta = verts[other] - verts[from_vert]
        length = float(np.linalg.norm(delta))
        if length < 1e-12:
            return None
        return delta / length

    def hardness(edge_index: int) -> float:
        return max(0.0, 1.0 - float(edge_costs[edge_index]))

    def greedy_extend(start_vert: int, prev_vert: int) -> list[int]:
        """
        从 start 沿种子主方向延伸到领域边界。
        - 与主方向同向的内部边可续（含接近边界的边）
        - 贴周界且横拐的边拒绝（防端点溢出）
        - 真领域边界边只作终止信号，不加入切线
        """
        path: list[int] = []
        current = int(start_vert)
        previous = int(prev_vert)
        used: set[int] = {seed}

        principal: np.ndarray | None = None
        if verts is not None and 0 <= previous < len(verts) and 0 <= current < len(verts):
            delta = verts[current] - verts[previous]
            length = float(np.linalg.norm(delta))
            if length > 1e-12:
                principal = delta / length

        for _ in range(max_steps):
            travel = None
            if verts is not None and 0 <= previous < len(verts) and 0 <= current < len(verts):
                delta = verts[current] - verts[previous]
                length = float(np.linalg.norm(delta))
                if length > 1e-12:
                    travel = delta / length
            if principal is None and travel is not None:
                principal = travel

            begin = int(vert_edge_offsets[current])
            end = int(vert_edge_offsets[current + 1])
            best_internal = None
            best_internal_score = -1.0
            best_internal_other = -1
            best_soft = None
            best_soft_score = -1.0
            best_soft_other = -1

            for edge_index in vert_edge_indices[begin:end].tolist():
                edge_index = int(edge_index)
                if edge_index in used:
                    continue
                v_other = other_vert(edge_index, current)
                if v_other == previous:
                    continue

                ed = edge_dir(edge_index, current) if verts is not None else None
                align_p = 1.0
                align_t = 1.0
                if principal is not None and ed is not None:
                    align_p = float(np.dot(principal, ed))
                if travel is not None and ed is not None:
                    align_t = float(np.dot(travel, ed))

                # 真·领域边界：朝向主方向则视为走到尽头（不加入切线）
                if _is_region_boundary_edge(
                    edge_index, face_a, face_b, region_ids, target_rid
                ):
                    continue

                if not _is_internal_region_edge(
                    edge_index, face_a, face_b, region_ids, target_rid
                ):
                    continue

                on_perimeter = _edge_touches_foreign_region(
                    edge_index, face_a, face_b, offsets, adj, rid, target_rid
                )
                # 周界横拐：拒绝；周界但朝主方向前进：允许（否则红线会中途断开）
                if on_perimeter and align_p < min_perimeter_forward_align:
                    continue

                if verts is not None:
                    if align_p < min_principal_align:
                        continue
                    if align_t < min_local_align:
                        continue

                hard = hardness(edge_index)
                score = hard * (0.15 + 0.55 * align_p + 0.30 * align_t)
                if on_perimeter:
                    # 略降权，但仍可选，以便走到边界
                    score *= 0.85

                if hard >= min_hardness and score > best_internal_score:
                    best_internal_score = score
                    best_internal = edge_index
                    best_internal_other = v_other
                elif (
                    hard >= soft_hardness
                    and align_p >= min_perimeter_forward_align
                    and score > best_soft_score
                ):
                    # 缓棱中段：方向够正时允许更软的续边
                    best_soft_score = score
                    best_soft = edge_index
                    best_soft_other = v_other

            # 无坐标：按硬度，周界横边无法判断时仍禁止「任意」周界
            if best_internal is None and verts is None:
                for edge_index in vert_edge_indices[begin:end].tolist():
                    edge_index = int(edge_index)
                    if edge_index in used:
                        continue
                    v_other = other_vert(edge_index, current)
                    if v_other == previous:
                        continue
                    if _is_region_boundary_edge(
                        edge_index, face_a, face_b, region_ids, target_rid
                    ):
                        continue
                    if not _is_internal_region_edge(
                        edge_index, face_a, face_b, region_ids, target_rid
                    ):
                        continue
                    hard = hardness(edge_index)
                    if hard < min_hardness:
                        continue
                    if hard > best_internal_score:
                        best_internal_score = hard
                        best_internal = edge_index
                        best_internal_other = v_other

            chosen = best_internal
            chosen_other = best_internal_other
            if chosen is None and best_soft is not None:
                chosen = best_soft
                chosen_other = best_soft_other

            if chosen is not None:
                path.append(int(chosen))
                used.add(int(chosen))
                previous = current
                current = int(chosen_other)
                if travel is not None and principal is not None:
                    blended = principal * 0.75 + travel * 0.25
                    norm = float(np.linalg.norm(blended))
                    if norm > 1e-12:
                        principal = blended / norm
                continue

            break

        return path

    completed: set[int] = {seed}
    va = int(edge_vert_a[seed])
    vb = int(edge_vert_b[seed])
    for edge_index in greedy_extend(va, vb):
        completed.add(int(edge_index))
    for edge_index in greedy_extend(vb, va):
        completed.add(int(edge_index))
    return filter_internal_cut_edges(
        np.asarray(sorted(completed), dtype=np.int32),
        topology,
        region_ids,
        target_rid,
    )


def _cut_edge_components(
    cut_edges: np.ndarray,
    edge_vert_a: np.ndarray,
    edge_vert_b: np.ndarray,
) -> list[list[int]]:
    """按共享顶点把切边分成连通分量。"""
    edges = [int(e) for e in np.asarray(cut_edges, dtype=np.int32).tolist()]
    if not edges:
        return []
    vert_to_edges: dict[int, list[int]] = {}
    for edge_index in edges:
        for vert in (
            int(edge_vert_a[edge_index]),
            int(edge_vert_b[edge_index]),
        ):
            vert_to_edges.setdefault(vert, []).append(edge_index)
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in edges:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            edge_index = stack.pop()
            component.append(edge_index)
            for vert in (
                int(edge_vert_a[edge_index]),
                int(edge_vert_b[edge_index]),
            ):
                for neighbor in vert_to_edges.get(vert, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        components.append(component)
    return components


def unify_cut_edges_as_line(
    cut_edges: np.ndarray,
    topology: dict,
    region_ids: np.ndarray,
    target_rid: int,
    edge_costs: np.ndarray,
    vert_edge_offsets: np.ndarray,
    vert_edge_indices: np.ndarray,
    edge_vert_a: np.ndarray,
    edge_vert_b: np.ndarray,
    max_bridges: int = 8,
) -> np.ndarray:
    """
    把多段点选切边合并成一条尽量连通的切线。

    先去掉跨领域边界边，再在目标领域内部用硬边代价最短路桥接
    各连通分量之间的缺口（多点选时常见）。
    """
    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    edge_vert_a = np.asarray(edge_vert_a, dtype=np.int32)
    edge_vert_b = np.asarray(edge_vert_b, dtype=np.int32)
    edge_costs = np.asarray(edge_costs, dtype=np.float64)
    offsets = np.asarray(vert_edge_offsets, dtype=np.int32)
    adj_edges = np.asarray(vert_edge_indices, dtype=np.int32)
    target = int(target_rid)

    completed = {
        int(e)
        for e in filter_internal_cut_edges(
            cut_edges, topology, region_ids, target
        ).tolist()
    }
    if not completed:
        return np.empty(0, dtype=np.int32)

    def endpoint_verts(edge_set: set[int]) -> set[int]:
        degree: dict[int, int] = {}
        for edge_index in edge_set:
            for vert in (
                int(edge_vert_a[edge_index]),
                int(edge_vert_b[edge_index]),
            ):
                degree[vert] = degree.get(vert, 0) + 1
        return {v for v, d in degree.items() if d == 1} or set(degree)

    def shortest_bridge(
        starts: set[int], goals: set[int]
    ) -> list[int] | None:
        if not starts or not goals:
            return None
        goal_set = set(goals)
        dist: dict[int, float] = {v: 0.0 for v in starts}
        prev_edge: dict[int, int | None] = {v: None for v in starts}
        prev_vert: dict[int, int | None] = {v: None for v in starts}
        heap: list[tuple[float, int]] = [(0.0, v) for v in starts]
        heapq.heapify(heap)
        reached: int | None = None
        while heap:
            cost_u, u = heapq.heappop(heap)
            if cost_u > dist.get(u, float("inf")) + 1e-12:
                continue
            if u in goal_set and u not in starts:
                reached = u
                break
            begin = int(offsets[u])
            end = int(offsets[u + 1])
            for edge_index in adj_edges[begin:end].tolist():
                edge_index = int(edge_index)
                if edge_index in completed:
                    continue
                if not _is_internal_region_edge(
                    edge_index, face_a, face_b, region_ids, target
                ):
                    continue
                va = int(edge_vert_a[edge_index])
                vb = int(edge_vert_b[edge_index])
                v_other = vb if va == u else va
                step = float(edge_costs[edge_index])
                if not np.isfinite(step) or step < 0.0:
                    step = 1.0
                new_cost = cost_u + step
                if new_cost + 1e-12 >= dist.get(v_other, float("inf")):
                    continue
                dist[v_other] = new_cost
                prev_edge[v_other] = edge_index
                prev_vert[v_other] = u
                heapq.heappush(heap, (new_cost, v_other))
        if reached is None:
            # 起点本身已在目标集合（分量已邻接）
            for v in starts:
                if v in goal_set:
                    return []
            return None
        path: list[int] = []
        cursor = reached
        while cursor is not None and prev_edge.get(cursor) is not None:
            edge_index = int(prev_edge[cursor])
            path.append(edge_index)
            cursor = prev_vert.get(cursor)
        return path

    for _ in range(int(max_bridges)):
        components = _cut_edge_components(
            np.asarray(sorted(completed), dtype=np.int32),
            edge_vert_a,
            edge_vert_b,
        )
        if len(components) <= 1:
            break
        # 桥接最近的一对分量端点
        best_path: list[int] | None = None
        best_cost = float("inf")
        for i in range(len(components)):
            set_i = set(components[i])
            ends_i = endpoint_verts(set_i)
            for j in range(i + 1, len(components)):
                set_j = set(components[j])
                ends_j = endpoint_verts(set_j)
                path = shortest_bridge(ends_i, ends_j)
                if path is None:
                    continue
                cost = 0.0
                for edge_index in path:
                    step = float(edge_costs[edge_index])
                    cost += step if np.isfinite(step) and step >= 0.0 else 1.0
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
        if best_path is None:
            break
        for edge_index in best_path:
            completed.add(int(edge_index))

    sealed = seal_cut_to_region_boundary(
        np.asarray(sorted(completed), dtype=np.int32),
        topology,
        region_ids,
        target,
        edge_costs,
        offsets,
        adj_edges,
        edge_vert_a,
        edge_vert_b,
    )
    if len(sealed):
        return sealed
    return np.asarray(sorted(completed), dtype=np.int32)


def complete_cut_edges_dijkstra(
    topology: dict,
    normals: np.ndarray,
    face_centers: np.ndarray,
    region_ids: np.ndarray,
    target_rid: int,
    seed_edges: np.ndarray,
    stroke_worlds: np.ndarray,
    stroke_screens: np.ndarray,
    edge_costs: np.ndarray,
    edge_mids: np.ndarray,
    vert_edge_offsets: np.ndarray,
    vert_edge_indices: np.ndarray,
    edge_vert_a: np.ndarray,
    edge_vert_b: np.ndarray,
    max_radius: float = 1.0,
) -> np.ndarray:
    """
    从种子边两端在目标领域内沿硬边代价图延伸补全。

    终止条件：抵达领域边界、另一端点、超出搜索半径或硬棱尽头。
    """
    del normals, face_centers, stroke_screens  # 预留扩展位

    seed_edges = np.asarray(seed_edges, dtype=np.int32)
    if len(seed_edges) == 0:
        return np.empty(0, dtype=np.int32)

    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    edge_vert_a = np.asarray(edge_vert_a, dtype=np.int32)
    edge_vert_b = np.asarray(edge_vert_b, dtype=np.int32)
    edge_costs = np.asarray(edge_costs, dtype=np.float64)
    edge_mids = np.asarray(edge_mids, dtype=np.float64)
    stroke_worlds = np.asarray(stroke_worlds, dtype=np.float64)

    completed: set[int] = set(int(e) for e in seed_edges.tolist())
    endpoints = _edge_endpoint_vertices(seed_edges, edge_vert_a, edge_vert_b)
    if not endpoints:
        return np.asarray(sorted(completed), dtype=np.int32)

    # 笔迹附近搜索盒：种子边中点 + 笔迹点
    focus_points = [edge_mids[e] for e in seed_edges.tolist()]
    if len(stroke_worlds):
        focus_points.extend(list(stroke_worlds))
    focus = np.asarray(focus_points, dtype=np.float64)
    focus_center = focus.mean(axis=0)
    radius_sq = float(max_radius) ** 2

    seed_verts = set(endpoints)
    for edge_index in seed_edges.tolist():
        seed_verts.add(int(edge_vert_a[edge_index]))
        seed_verts.add(int(edge_vert_b[edge_index]))

    def nearby(edge_index: int) -> bool:
        mid = edge_mids[edge_index]
        delta = mid - focus_center
        return float(np.dot(delta, delta)) <= radius_sq * 4.0

    def stroke_bias(edge_index: int) -> float:
        if len(stroke_worlds) == 0:
            return 0.0
        mid = edge_mids[edge_index]
        dists = np.sum((stroke_worlds - mid) ** 2, axis=1)
        nearest = float(np.sqrt(dists.min()))
        # 远离笔迹略增代价，引导补全仍贴近用户意图
        return 0.15 * nearest / max(float(max_radius), 1e-6)

    def dijkstra_from(start_vert: int) -> list[int]:
        """返回从起点延伸得到的边序列（不含已有种子边）。"""
        dist: dict[int, float] = {start_vert: 0.0}
        prev_edge: dict[int, int | None] = {start_vert: None}
        prev_vert: dict[int, int | None] = {start_vert: None}
        heap: list[tuple[float, int]] = [(0.0, start_vert)]
        goal_vert: int | None = None
        best_hard_vert: int | None = start_vert
        best_hard_score = -1.0

        while heap:
            cost_u, u = heapq.heappop(heap)
            if cost_u > dist.get(u, float("inf")) + 1e-12:
                continue

            start = int(vert_edge_offsets[u])
            end = int(vert_edge_offsets[u + 1])
            for edge_index in vert_edge_indices[start:end].tolist():
                edge_index = int(edge_index)
                if not nearby(edge_index):
                    continue

                va = int(edge_vert_a[edge_index])
                vb = int(edge_vert_b[edge_index])
                v_other = vb if va == u else va

                # 领域边界边：纳入路径并终止
                if _is_region_boundary_edge(
                    edge_index, face_a, face_b, region_ids, target_rid
                ):
                    new_cost = cost_u + 0.02 + stroke_bias(edge_index)
                    if new_cost < dist.get(v_other, float("inf")):
                        dist[v_other] = new_cost
                        prev_edge[v_other] = edge_index
                        prev_vert[v_other] = u
                        goal_vert = v_other
                        heap.clear()
                    break

                if not _is_internal_region_edge(
                    edge_index, face_a, face_b, region_ids, target_rid
                ):
                    continue

                # 已是种子/已补全边：可免费穿过，但不作为延伸目标
                if edge_index in completed:
                    step = 1e-6
                    new_cost = cost_u + step
                    if new_cost < dist.get(v_other, float("inf")):
                        dist[v_other] = new_cost
                        prev_edge[v_other] = edge_index
                        prev_vert[v_other] = u
                        heapq.heappush(heap, (new_cost, v_other))
                    continue

                step = float(edge_costs[edge_index]) + stroke_bias(edge_index)
                new_cost = cost_u + step
                if new_cost >= dist.get(v_other, float("inf")):
                    continue
                dist[v_other] = new_cost
                prev_edge[v_other] = edge_index
                prev_vert[v_other] = u

                # 记录沿硬边走出的最远点（代价越低越好，距离越远越好）
                hardness = max(0.0, 1.0 - float(edge_costs[edge_index]))
                score = hardness * (1.0 + new_cost)
                if score > best_hard_score and hardness > 0.35:
                    best_hard_score = score
                    best_hard_vert = v_other

                # 经非种子边接到其他种子顶点：闭合缺口
                if v_other in seed_verts and v_other != start_vert:
                    goal_vert = v_other
                    heap.clear()
                    break

                heapq.heappush(heap, (new_cost, v_other))

            if goal_vert is not None:
                break

        if goal_vert is None:
            goal_vert = best_hard_vert if best_hard_vert is not None else start_vert

        path_edges: list[int] = []
        cursor = goal_vert
        while cursor is not None and prev_edge.get(cursor) is not None:
            edge_index = int(prev_edge[cursor])
            if edge_index not in completed:
                path_edges.append(edge_index)
            cursor = prev_vert.get(cursor)
        return path_edges

    for endpoint in endpoints:
        for edge_index in dijkstra_from(int(endpoint)):
            completed.add(int(edge_index))

    return np.asarray(sorted(completed), dtype=np.int32)


def split_region_by_cut_edges(
    region_ids: np.ndarray,
    topology: dict,
    cut_edges: np.ndarray,
    colors: np.ndarray,
    target_rid: int | None = None,
    min_component_faces: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    将补全边作为邻接屏障，在受影响领域内做连通分量拆分。

    最大分量保留原 ID/颜色；其余分量分配新 ID/新色。忽略面不变。
    target_rid 非空时只拆分该领域，且只使用两侧都属于该领域的内部切边。
    min_component_faces: 新分量最少面数；None 时大领域默认约 1%，
    手动点选切分请传 1，避免「看起来切开却无预览」。
    """
    ids = np.asarray(region_ids, dtype=np.int32).copy()
    colors = np.asarray(colors, dtype=np.float32)
    face_count = len(ids)
    if face_count == 0:
        return ids, colors.copy(), 0

    cut_set = set(int(e) for e in np.asarray(cut_edges, dtype=np.int32).tolist())
    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    offsets = np.asarray(topology["adjacency_offsets"], dtype=np.int32)
    adj = np.asarray(topology["adjacency_indices"], dtype=np.int32)

    # 仅内部边（两侧同属目标/同一领域）才构成切开屏障
    barrier_pairs: set[tuple[int, int]] = set()
    affected: set[int] = set()
    for edge_index in cut_set:
        if edge_index < 0 or edge_index >= len(face_a):
            continue
        fa = int(face_a[edge_index])
        fb = int(face_b[edge_index])
        ra = int(ids[fa])
        rb = int(ids[fb])
        if ra < 0 or rb < 0 or ra != rb:
            continue
        if target_rid is not None and ra != int(target_rid):
            continue
        barrier_pairs.add((min(fa, fb), max(fa, fb)))
        affected.add(ra)

    if target_rid is not None:
        affected = {int(target_rid)} if barrier_pairs else set()

    if not affected or not barrier_pairs:
        region_count = int(ids.max()) + 1 if np.any(ids >= 0) else 0
        if len(colors) < region_count:
            extra = generate_region_colors(region_count - len(colors))
            colors = np.vstack((colors, extra)) if len(colors) else extra
        return ids, colors[:region_count].copy(), region_count

    next_id = int(ids.max()) + 1 if np.any(ids >= 0) else 0
    new_color_rows: list[np.ndarray] = []

    for rid in sorted(affected):
        members = np.flatnonzero(ids == rid)
        if len(members) == 0:
            continue

        visited = np.zeros(face_count, dtype=bool)
        components: list[list[int]] = []

        for start in members.tolist():
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            component: list[int] = []
            while stack:
                face = stack.pop()
                component.append(face)
                begin = int(offsets[face])
                end = int(offsets[face + 1])
                for neighbor in adj[begin:end].tolist():
                    neighbor = int(neighbor)
                    if visited[neighbor]:
                        continue
                    if int(ids[neighbor]) != rid:
                        continue
                    pair = (min(face, neighbor), max(face, neighbor))
                    if pair in barrier_pairs:
                        continue
                    visited[neighbor] = True
                    stack.append(neighbor)
            components.append(component)

        if len(components) <= 1:
            continue

        components.sort(key=len, reverse=True)
        total_faces = sum(len(c) for c in components)
        # 手动点选（指定 min_component_faces）或指定目标领域：保留所有非空新分量
        if min_component_faces is not None:
            min_faces = max(1, int(min_component_faces))
        elif target_rid is not None:
            min_faces = 1
        elif total_faces >= 100:
            min_faces = max(3, int(total_faces * 0.01))
        else:
            min_faces = 1
        new_parts = [c for c in components[1:] if len(c) >= min_faces]
        if not new_parts:
            continue

        base = colors[rid] if rid < len(colors) else np.array(
            [0.5, 0.5, 0.8, 0.55],
            dtype=np.float32,
        )
        for offset_index, component in enumerate(new_parts):
            new_rid = next_id
            next_id += 1
            for face in component:
                ids[face] = new_rid
            new_color_rows.append(_contrast_color(base, offset_index))

    region_count = int(ids.max()) + 1 if np.any(ids >= 0) else 0
    if new_color_rows:
        colors = np.vstack(
            (colors, np.vstack(new_color_rows))
        ).astype(np.float32)
    if len(colors) < region_count:
        extra = generate_region_colors(region_count - len(colors))
        colors = np.vstack((colors, extra)) if len(colors) else extra
    return ids, colors[:region_count].copy(), region_count


def count_components_after_cut(
    region_ids: np.ndarray,
    topology: dict,
    cut_edges: np.ndarray,
    target_rid: int,
) -> int:
    """切线作为屏障后，目标领域的面连通分量数。"""
    ids = np.asarray(region_ids, dtype=np.int32)
    target = int(target_rid)
    members = np.flatnonzero(ids == target)
    if len(members) == 0:
        return 0
    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    offsets = np.asarray(topology["adjacency_offsets"], dtype=np.int32)
    adj = np.asarray(topology["adjacency_indices"], dtype=np.int32)
    barrier_pairs: set[tuple[int, int]] = set()
    for edge_index in np.asarray(cut_edges, dtype=np.int32).tolist():
        edge_index = int(edge_index)
        if edge_index < 0 or edge_index >= len(face_a):
            continue
        fa = int(face_a[edge_index])
        fb = int(face_b[edge_index])
        if int(ids[fa]) != target or int(ids[fb]) != target:
            continue
        barrier_pairs.add((min(fa, fb), max(fa, fb)))
    if not barrier_pairs:
        return 1
    visited = np.zeros(len(ids), dtype=bool)
    components = 0
    for start in members.tolist():
        if visited[start]:
            continue
        components += 1
        stack = [int(start)]
        visited[start] = True
        while stack:
            face = stack.pop()
            begin = int(offsets[face])
            end = int(offsets[face + 1])
            for neighbor in adj[begin:end].tolist():
                neighbor = int(neighbor)
                if visited[neighbor] or int(ids[neighbor]) != target:
                    continue
                pair = (min(face, neighbor), max(face, neighbor))
                if pair in barrier_pairs:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)
    return components


def seal_cut_to_region_boundary(
    cut_edges: np.ndarray,
    topology: dict,
    region_ids: np.ndarray,
    target_rid: int,
    edge_costs: np.ndarray,
    vert_edge_offsets: np.ndarray,
    vert_edge_indices: np.ndarray,
    edge_vert_a: np.ndarray,
    edge_vert_b: np.ndarray,
) -> np.ndarray:
    """
    把切线悬空端点封到领域周界，堵住「看起来贯通、面仍可绕行」的缺口。
    """
    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    edge_vert_a = np.asarray(edge_vert_a, dtype=np.int32)
    edge_vert_b = np.asarray(edge_vert_b, dtype=np.int32)
    edge_costs = np.asarray(edge_costs, dtype=np.float64)
    offsets = np.asarray(vert_edge_offsets, dtype=np.int32)
    adj_edges = np.asarray(vert_edge_indices, dtype=np.int32)
    target = int(target_rid)

    completed = {
        int(e)
        for e in filter_internal_cut_edges(
            cut_edges, topology, region_ids, target
        ).tolist()
    }
    if not completed:
        return np.empty(0, dtype=np.int32)

    def vert_on_boundary(vert: int) -> bool:
        begin = int(offsets[vert])
        end = int(offsets[vert + 1])
        for edge_index in adj_edges[begin:end].tolist():
            if _is_region_boundary_edge(
                int(edge_index), face_a, face_b, region_ids, target
            ):
                return True
        return False

    degree: dict[int, int] = {}
    for edge_index in completed:
        for vert in (
            int(edge_vert_a[edge_index]),
            int(edge_vert_b[edge_index]),
        ):
            degree[vert] = degree.get(vert, 0) + 1
    dangling = [v for v, d in degree.items() if d == 1]

    def path_to_boundary(start: int) -> list[int]:
        if vert_on_boundary(start):
            return []
        dist: dict[int, float] = {start: 0.0}
        prev_edge: dict[int, int | None] = {start: None}
        prev_vert: dict[int, int | None] = {start: None}
        heap: list[tuple[float, int]] = [(0.0, start)]
        reached: int | None = None
        while heap:
            cost_u, u = heapq.heappop(heap)
            if cost_u > dist.get(u, float("inf")) + 1e-12:
                continue
            if u != start and vert_on_boundary(u):
                reached = u
                break
            begin = int(offsets[u])
            end = int(offsets[u + 1])
            for edge_index in adj_edges[begin:end].tolist():
                edge_index = int(edge_index)
                if edge_index in completed:
                    # 允许沿已有切线走到另一端，但不作为新封边
                    pass
                if not _is_internal_region_edge(
                    edge_index, face_a, face_b, region_ids, target
                ):
                    continue
                va = int(edge_vert_a[edge_index])
                vb = int(edge_vert_b[edge_index])
                v_other = vb if va == u else va
                step = float(edge_costs[edge_index])
                if not np.isfinite(step) or step < 0.0:
                    step = 1.0
                # 已在切线上的边代价极低，便于穿过
                if edge_index in completed:
                    step *= 0.01
                new_cost = cost_u + step
                if new_cost + 1e-12 >= dist.get(v_other, float("inf")):
                    continue
                dist[v_other] = new_cost
                prev_edge[v_other] = edge_index
                prev_vert[v_other] = u
                heapq.heappush(heap, (new_cost, v_other))
        if reached is None:
            return []
        path: list[int] = []
        cursor = reached
        while cursor is not None and prev_edge.get(cursor) is not None:
            edge_index = int(prev_edge[cursor])
            if edge_index not in completed:
                path.append(edge_index)
            cursor = prev_vert.get(cursor)
        return path

    for vert in dangling:
        for edge_index in path_to_boundary(int(vert)):
            completed.add(int(edge_index))

    return np.asarray(sorted(completed), dtype=np.int32)


def cut_edges_from_paint_corridor(
    painted_faces: np.ndarray,
    topology: dict,
    normals: np.ndarray,
    face_centers: np.ndarray,
    region_ids: np.ndarray,
    target_rid: int,
    stroke_worlds: np.ndarray,
    edge_costs: np.ndarray,
    edge_mids: np.ndarray,
    vert_edge_offsets: np.ndarray,
    vert_edge_indices: np.ndarray,
    edge_vert_a: np.ndarray,
    edge_vert_b: np.ndarray,
    max_radius: float = 1.0,
) -> tuple[np.ndarray, str]:
    """
    从涂红面走廊提取优先硬边的切线并补全。

    返回 (completed_edge_indices, status_message)。
    status 为空字符串表示成功；否则为失败原因。
    """
    painted = np.unique(np.asarray(painted_faces, dtype=np.int32))
    if len(painted) == 0:
        return np.empty(0, dtype=np.int32), "未涂绘任何面"

    rid = np.asarray(region_ids, dtype=np.int32)
    painted = painted[(painted >= 0) & (painted < len(rid))]
    painted = painted[rid[painted] == int(target_rid)]
    if len(painted) < 2:
        return np.empty(0, dtype=np.int32), "涂绘面过少，请加粗笔刷或继续涂绘"

    face_a = np.asarray(topology["edge_face_a"], dtype=np.int32)
    face_b = np.asarray(topology["edge_face_b"], dtype=np.int32)
    edge_costs = np.asarray(edge_costs, dtype=np.float64)
    edge_mids = np.asarray(edge_mids, dtype=np.float64)
    centers = np.asarray(face_centers, dtype=np.float64)
    stroke_worlds = np.asarray(stroke_worlds, dtype=np.float64)
    if len(stroke_worlds) == 0:
        stroke_worlds = centers[painted]

    painted_set = set(int(f) for f in painted.tolist())

    # 走廊内边：两侧同属目标领域，且至少一侧被涂红。
    corridor_edges: list[int] = []
    for edge_index in range(len(face_a)):
        fa = int(face_a[edge_index])
        fb = int(face_b[edge_index])
        if rid[fa] != target_rid or rid[fb] != target_rid:
            continue
        if fa not in painted_set and fb not in painted_set:
            continue
        corridor_edges.append(edge_index)

    if not corridor_edges:
        return np.empty(0, dtype=np.int32), "涂绘区域未形成可切走廊"

    corridor_arr = np.asarray(corridor_edges, dtype=np.int32)
    edge_vert_a = np.asarray(edge_vert_a, dtype=np.int32)
    edge_vert_b = np.asarray(edge_vert_b, dtype=np.int32)

    # ------------------------------------------------------------------
    # 在走廊子图内求一条贯穿走廊的最小代价路径（硬边优先），
    # 得到单一连续切线，避免碎片化切割。
    # ------------------------------------------------------------------
    if len(corridor_arr) == 1:
        seed_edges = corridor_arr.copy()
    else:
        # 优先用笔迹起止点定走廊两端；否则退回 PCA 主轴。
        if len(stroke_worlds) >= 2:
            start_pt = stroke_worlds[0]
            end_pt = stroke_worlds[-1]
            d_start = np.sum((edge_mids[corridor_arr] - start_pt) ** 2, axis=1)
            d_end = np.sum((edge_mids[corridor_arr] - end_pt) ** 2, axis=1)
            edge_start = int(corridor_arr[int(np.argmin(d_start))])
            edge_goal = int(corridor_arr[int(np.argmin(d_end))])
            if edge_start == edge_goal and len(corridor_arr) > 1:
                # 起止落在同一边时，改用 PCA 拉开两端。
                mids_c = edge_mids[corridor_arr]
                center = mids_c.mean(axis=0)
                deviation = mids_c - center
                try:
                    _u, _s, vt = np.linalg.svd(deviation, full_matrices=False)
                    axis = vt[0]
                except np.linalg.LinAlgError:
                    axis = np.array([1.0, 0.0, 0.0])
                proj = deviation @ axis
                edge_start = int(corridor_arr[int(np.argmin(proj))])
                edge_goal = int(corridor_arr[int(np.argmax(proj))])
        else:
            mids_c = edge_mids[corridor_arr]
            center = mids_c.mean(axis=0)
            deviation = mids_c - center
            try:
                _u, _s, vt = np.linalg.svd(deviation, full_matrices=False)
                axis = vt[0]
            except np.linalg.LinAlgError:
                axis = np.array([1.0, 0.0, 0.0])
            proj = deviation @ axis
            edge_start = int(corridor_arr[int(np.argmin(proj))])
            edge_goal = int(corridor_arr[int(np.argmax(proj))])

        def corridor_bias(edge_index: int) -> float:
            mid = edge_mids[edge_index]
            dists = np.sum((stroke_worlds - mid) ** 2, axis=1)
            nearest = float(np.sqrt(dists.min()))
            return 0.3 * nearest / max(float(max_radius), 1e-6)

        # 走廊内顶点 → 走廊边 邻接
        corridor_set = set(int(e) for e in corridor_arr.tolist())
        vert_to_edges: dict[int, list[int]] = {}
        for edge_index in corridor_arr.tolist():
            va = int(edge_vert_a[edge_index])
            vb = int(edge_vert_b[edge_index])
            vert_to_edges.setdefault(va, []).append(int(edge_index))
            vert_to_edges.setdefault(vb, []).append(int(edge_index))

        sources = {
            int(edge_vert_a[edge_start]),
            int(edge_vert_b[edge_start]),
        }
        targets = {
            int(edge_vert_a[edge_goal]),
            int(edge_vert_b[edge_goal]),
        }
        # 避免起止共享顶点时立刻“到达”导致空路径。
        if sources & targets and edge_start != edge_goal:
            targets = targets - sources
        if not targets:
            targets = {
                int(edge_vert_a[edge_goal]),
                int(edge_vert_b[edge_goal]),
            }

        dist: dict[int, float] = {v: 0.0 for v in sources}
        prev_edge: dict[int, int | None] = {v: None for v in sources}
        prev_vert: dict[int, int | None] = {v: None for v in sources}
        heap = [(0.0, v) for v in sources]
        heapq.heapify(heap)
        reached: int | None = None
        while heap:
            cost_u, u = heapq.heappop(heap)
            if cost_u > dist.get(u, float("inf")) + 1e-12:
                continue
            if u in targets and prev_edge.get(u) is not None:
                reached = u
                break
            if u in targets and edge_start == edge_goal:
                reached = u
                break
            for edge_index in vert_to_edges.get(u, ()):
                if edge_index not in corridor_set:
                    continue
                va = int(edge_vert_a[edge_index])
                vb = int(edge_vert_b[edge_index])
                v_other = vb if va == u else va
                step = float(edge_costs[edge_index]) + corridor_bias(edge_index)
                new_cost = cost_u + step
                if new_cost >= dist.get(v_other, float("inf")):
                    continue
                dist[v_other] = new_cost
                prev_edge[v_other] = int(edge_index)
                prev_vert[v_other] = u
                heapq.heappush(heap, (new_cost, v_other))

        if reached is None:
            # 尝试任意已访问的目标，或退回起止边。
            for t in targets:
                if t in dist and prev_edge.get(t) is not None:
                    reached = t
                    break
        if reached is None:
            return (
                np.empty(0, dtype=np.int32),
                "涂绘区域不连续，请一笔连贯涂过要拆分的位置",
            )

        path_edges: list[int] = []
        cursor: int | None = reached
        while cursor is not None and prev_edge.get(cursor) is not None:
            path_edges.append(int(prev_edge[cursor]))
            cursor = prev_vert.get(cursor)
        if not path_edges:
            path_edges = sorted({edge_start, edge_goal})
        seed_edges = np.unique(np.asarray(path_edges, dtype=np.int32))

    completed = complete_cut_edges_dijkstra(
        topology,
        normals,
        face_centers,
        region_ids,
        target_rid,
        seed_edges,
        stroke_worlds,
        np.empty((0, 2), dtype=np.float64),
        edge_costs,
        edge_mids,
        vert_edge_offsets,
        vert_edge_indices,
        edge_vert_a,
        edge_vert_b,
        max_radius=max_radius,
    )
    if len(completed) == 0:
        return np.empty(0, dtype=np.int32), "未能沿硬边补全切线，请调整笔刷"

    # 验证切边是否能把目标领域拆成 >=2 个分量
    probe_ids, _probe_colors, probe_count = split_region_by_cut_edges(
        region_ids,
        topology,
        completed,
        generate_region_colors(max(int(region_ids.max()) + 1, 1)),
        target_rid=int(target_rid),
    )
    original_count = int(region_ids.max()) + 1 if np.any(region_ids >= 0) else 0
    if probe_count <= original_count and not np.any(
        probe_ids[region_ids == target_rid] != target_rid
    ):
        # 仍可能拆出同 id 压缩前的多分量——再直接检查屏障分量数
        members = np.flatnonzero(region_ids == target_rid)
        if len(members) == 0:
            return completed, "目标领域为空"
        barrier_pairs = set()
        for edge_index in completed.tolist():
            fa = int(face_a[edge_index])
            fb = int(face_b[edge_index])
            barrier_pairs.add((min(fa, fb), max(fa, fb)))
        offsets = np.asarray(topology["adjacency_offsets"], dtype=np.int32)
        adj = np.asarray(topology["adjacency_indices"], dtype=np.int32)
        visited = np.zeros(len(region_ids), dtype=bool)
        components = 0
        for start in members.tolist():
            if visited[start]:
                continue
            components += 1
            stack = [int(start)]
            visited[start] = True
            while stack:
                face = stack.pop()
                begin = int(offsets[face])
                end = int(offsets[face + 1])
                for neighbor in adj[begin:end].tolist():
                    neighbor = int(neighbor)
                    if visited[neighbor] or int(region_ids[neighbor]) != target_rid:
                        continue
                    pair = (min(face, neighbor), max(face, neighbor))
                    if pair in barrier_pairs:
                        continue
                    visited[neighbor] = True
                    stack.append(neighbor)
        if components < 2:
            return (
                completed,
                "切线未能把领域分成两块，请沿硬棱继续涂绘",
            )

    return completed, ""


__all__ = (
    "prepare_edge_costs",
    "candidate_hard_edges",
    "group_candidate_edge_chains",
    "chain_splits_region",
    "filter_bisecting_candidate_chains",
    "filter_internal_cut_edges",
    "grow_ridge_cut_to_boundary",
    "unify_cut_edges_as_line",
    "seal_cut_to_region_boundary",
    "count_components_after_cut",
    "stroke_hits_to_seed_edges",
    "complete_cut_edges_dijkstra",
    "split_region_by_cut_edges",
    "cut_edges_from_paint_corridor",
)
