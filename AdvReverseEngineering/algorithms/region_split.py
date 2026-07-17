# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""画笔种子边、硬边代价补全与领域连通分量拆分。"""

from __future__ import annotations

import colorsys
import heapq
from typing import Iterable

import numpy as np

from .regions import REGION_IGNORED_ID, generate_region_colors


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

    二面角越大（越硬）代价越低；平坦边代价高；跨领域边用作终止目标。
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
    # 二面角（0=共面平坦，π=对折）；硬边 → 低代价
    dihedral = np.arccos(dots)
    hardness = dihedral / np.pi  # 0..1
    flat_penalty = (1.0 - hardness) ** 2

    centers = np.asarray(face_centers, dtype=np.float64)
    mids = 0.5 * (centers[face_a] + centers[face_b])

    # 基础代价：硬边接近 0.05，平坦接近 1.0
    costs = 0.05 + 0.95 * flat_penalty

    rid = np.asarray(region_ids, dtype=np.int32)
    cross = rid[face_a] != rid[face_b]
    # 跨领域边不作为切割路径内部边，代价抬高供搜索规避；但仍可作终止。
    costs = np.where(cross, costs + 50.0, costs)
    return costs.astype(np.float64, copy=False), mids.astype(np.float64, copy=False)


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
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    将补全边作为邻接屏障，在受影响领域内做连通分量拆分。

    最大分量保留原 ID/颜色；其余分量分配新 ID/新色。忽略面不变。
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

    # 边索引 → 是否屏障；同时构建 (min_face,max_face) 快速查询
    barrier_pairs: set[tuple[int, int]] = set()
    for edge_index in cut_set:
        if edge_index < 0 or edge_index >= len(face_a):
            continue
        fa = int(face_a[edge_index])
        fb = int(face_b[edge_index])
        barrier_pairs.add((min(fa, fb), max(fa, fb)))

    # 受影响的原领域：至少一侧被切边触及且属于该领域
    affected: set[int] = set()
    for edge_index in cut_set:
        if edge_index < 0 or edge_index >= len(face_a):
            continue
        for face in (int(face_a[edge_index]), int(face_b[edge_index])):
            rid = int(ids[face])
            if rid >= 0:
                affected.add(rid)

    if not affected:
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
        # 最大分量保留原 rid；其余分配新 id，并用对比色以便预览区分。
        base = colors[rid] if rid < len(colors) else np.array(
            [0.5, 0.5, 0.8, 0.55],
            dtype=np.float32,
        )
        for offset_index, component in enumerate(components[1:]):
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

    # 走廊边评分：硬边优先 + 贴近笔迹中心线。
    scores = []
    for edge_index in corridor_arr.tolist():
        mid = edge_mids[edge_index]
        dists = np.sum((stroke_worlds - mid) ** 2, axis=1)
        nearest = float(np.sqrt(dists.min()))
        hardness = max(0.0, 1.0 - float(edge_costs[edge_index]))
        # 高分更好
        score = hardness * 2.0 - nearest / max(float(max_radius), 1e-6)
        scores.append((score, int(edge_index)))
    scores.sort(reverse=True)

    # 取评分最高的若干边作为种子骨架（至少 1 条，最多走廊 35%）。
    seed_count = max(1, min(len(scores), max(3, len(scores) // 3)))
    seed_edges = np.asarray(
        [edge for _score, edge in scores[:seed_count]],
        dtype=np.int32,
    )

    # 在种子中优先保留真正硬边；若全平坦也继续，靠笔迹约束。
    hard_seeds = [
        edge
        for edge in seed_edges.tolist()
        if float(edge_costs[edge]) < 0.45
    ]
    if hard_seeds:
        seed_edges = np.asarray(hard_seeds, dtype=np.int32)

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
    "stroke_hits_to_seed_edges",
    "complete_cut_edges_dijkstra",
    "split_region_by_cut_edges",
    "cut_edges_from_paint_corridor",
)
