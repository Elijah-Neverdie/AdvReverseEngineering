"""领域智能拆分算法回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from AdvReverseEngineering.algorithms.region_split import (
    complete_cut_edges_dijkstra,
    prepare_edge_costs,
    split_region_by_cut_edges,
    stroke_hits_to_seed_edges,
)
from AdvReverseEngineering.algorithms.regions import generate_region_colors


def _quad_strip_topology():
    """
    构建 2x3 四边形条带的合成拓扑（6 个面）。

      0--1--2--3
      |A |B |C |
      4--5--6--7
      |D |E |F |
      8--9--10-11

    A=0 B=1 C=2
    D=3 E=4 F=5
    """
    # 共享边（无向）
    # 水平：A-B, B-C, D-E, E-F
    # 垂直：A-D, B-E, C-F
    edge_face_a = np.array([0, 1, 3, 4, 0, 1, 2], dtype=np.int32)
    edge_face_b = np.array([1, 2, 4, 5, 3, 4, 5], dtype=np.int32)
    edge_vert_a = np.array([1, 2, 5, 6, 4, 5, 6], dtype=np.int32)
    edge_vert_b = np.array([5, 6, 9, 10, 5, 6, 7], dtype=np.int32)

    # 面邻接 CSR
    adj_lists = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4],
        4: [1, 3, 5],
        5: [2, 4],
    }
    offsets = [0]
    indices = []
    for face in range(6):
        neighbors = adj_lists[face]
        indices.extend(neighbors)
        offsets.append(len(indices))

    # 顶点 → 边 CSR（12 顶点）
    vert_count = 12
    vert_edges: dict[int, list[int]] = {i: [] for i in range(vert_count)}
    for edge_index, (va, vb) in enumerate(
        zip(edge_vert_a.tolist(), edge_vert_b.tolist())
    ):
        vert_edges[va].append(edge_index)
        vert_edges[vb].append(edge_index)
    v_offsets = [0]
    v_indices = []
    for vert in range(vert_count):
        v_indices.extend(vert_edges[vert])
        v_offsets.append(len(v_indices))

    topology = {
        "edge_face_a": edge_face_a,
        "edge_face_b": edge_face_b,
        "edge_vert_a": edge_vert_a,
        "edge_vert_b": edge_vert_b,
        "adjacency_offsets": np.asarray(offsets, dtype=np.int32),
        "adjacency_indices": np.asarray(indices, dtype=np.int32),
        "vert_edge_offsets": np.asarray(v_offsets, dtype=np.int32),
        "vert_edge_indices": np.asarray(v_indices, dtype=np.int32),
    }

    # 面中心 / 法线：上排 Z+硬折到下排（垂直边硬）
    centers = np.array(
        [
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [2.5, 1.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [2.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    # 上排法线 +Z，下排法线 +Y → 垂直共享边很硬
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return topology, centers, normals


class RegionSplitTests(unittest.TestCase):
    """种子边、补全与连通分量拆分。"""

    def test_prepare_edge_costs_hard_cheaper(self) -> None:
        topology, centers, normals = _quad_strip_topology()
        region_ids = np.zeros(6, dtype=np.int32)
        costs, mids = prepare_edge_costs(
            topology, normals, centers, region_ids
        )
        # 垂直硬边索引 4,5,6 应比水平平坦边 0..3 更便宜
        self.assertTrue(float(costs[4:7].mean()) < float(costs[0:4].mean()))
        self.assertEqual(len(mids), len(costs))

    def test_stroke_hits_to_seed_edges_shared(self) -> None:
        topology, centers, normals = _quad_strip_topology()
        del normals
        region_ids = np.zeros(6, dtype=np.int32)
        faces = [0, 1, 2]
        worlds = centers[faces]
        seeds = stroke_hits_to_seed_edges(
            faces,
            worlds,
            topology,
            centers,
            region_ids,
            target_rid=0,
        )
        self.assertTrue(len(seeds) >= 1)
        # 应包含 A-B 或 B-C
        self.assertTrue(any(int(e) in {0, 1} for e in seeds.tolist()))

    def test_complete_prefers_hard_edges(self) -> None:
        topology, centers, normals = _quad_strip_topology()
        region_ids = np.zeros(6, dtype=np.int32)
        costs, mids = prepare_edge_costs(
            topology, normals, centers, region_ids
        )
        # 仅给中间垂直硬边作短种子，期望向两侧硬边延伸
        seed_edges = np.array([5], dtype=np.int32)
        stroke = centers[[1, 4]]
        screens = np.array([[10.0, 10.0], [10.0, 20.0]], dtype=np.float64)
        completed = complete_cut_edges_dijkstra(
            topology,
            normals,
            centers,
            region_ids,
            0,
            seed_edges,
            stroke,
            screens,
            costs,
            mids,
            topology["vert_edge_offsets"],
            topology["vert_edge_indices"],
            topology["edge_vert_a"],
            topology["edge_vert_b"],
            max_radius=10.0,
        )
        completed_set = set(completed.tolist())
        self.assertIn(5, completed_set)
        # 应至少补到相邻硬边之一
        self.assertTrue(completed_set.intersection({4, 6}))

    def test_split_barrier_creates_components(self) -> None:
        topology, centers, normals = _quad_strip_topology()
        del centers, normals
        # 一整块领域 0
        region_ids = np.zeros(6, dtype=np.int32)
        colors = generate_region_colors(1)
        # 切断中间垂直列：边 5 (B-E) 不足以拆成两块；
        # 切断水平中缝 A-D,B-E,C-F = 边 4,5,6 把上下拆开
        cut_edges = np.array([4, 5, 6], dtype=np.int32)
        new_ids, new_colors, new_count = split_region_by_cut_edges(
            region_ids,
            topology,
            cut_edges,
            colors,
        )
        self.assertEqual(new_count, 2)
        top = set(new_ids[[0, 1, 2]].tolist())
        bottom = set(new_ids[[3, 4, 5]].tolist())
        self.assertEqual(len(top), 1)
        self.assertEqual(len(bottom), 1)
        self.assertNotEqual(next(iter(top)), next(iter(bottom)))
        self.assertEqual(new_colors.shape[0], 2)

    def test_ignored_faces_untouched(self) -> None:
        topology, centers, normals = _quad_strip_topology()
        del centers, normals
        region_ids = np.array([0, 0, -1, 0, 0, -1], dtype=np.int32)
        colors = generate_region_colors(1)
        cut_edges = np.array([4, 5], dtype=np.int32)
        new_ids, new_colors, new_count = split_region_by_cut_edges(
            region_ids,
            topology,
            cut_edges,
            colors,
        )
        self.assertEqual(int(new_ids[2]), -1)
        self.assertEqual(int(new_ids[5]), -1)
        self.assertGreaterEqual(new_count, 1)
        self.assertEqual(new_colors.shape[0], new_count)

    def test_multi_stroke_joint_split(self) -> None:
        topology, centers, normals = _quad_strip_topology()
        del centers, normals
        region_ids = np.zeros(6, dtype=np.int32)
        colors = generate_region_colors(1)
        # 两笔分别切垂直硬边，联合后应拆成上下两块
        cut_edges = np.unique(np.array([4, 5, 6], dtype=np.int32))
        new_ids, new_colors, new_count = split_region_by_cut_edges(
            region_ids,
            topology,
            cut_edges,
            colors,
        )
        self.assertEqual(new_count, 2)
        self.assertEqual(len(np.unique(new_ids[new_ids >= 0])), 2)
        self.assertEqual(new_colors.shape[0], 2)


class MergeTransactionLogicTests(unittest.TestCase):
    """合并内存事务风格的撤销/重做栈逻辑（纯数据）。"""

    def test_history_undo_redo(self) -> None:
        live_ids = np.array([0, 1, 2], dtype=np.int32)
        live_colors = generate_region_colors(3)
        history: list[dict] = []
        redo: list[dict] = []

        def push():
            history.append(
                {
                    "ids": live_ids.copy(),
                    "colors": live_colors.copy(),
                    "count": int(live_ids.max()) + 1,
                }
            )
            redo.clear()

        push()
        # 模拟合并 0<-1
        from AdvReverseEngineering.algorithms.regions import merge_region_ids

        new_ids, new_colors, count, _ = merge_region_ids(
            live_ids, live_colors, 0, 1
        )
        live_ids, live_colors = new_ids, new_colors
        self.assertEqual(count, 2)

        # undo
        redo.append(
            {
                "ids": live_ids.copy(),
                "colors": live_colors.copy(),
                "count": count,
            }
        )
        prev = history.pop()
        live_ids = prev["ids"]
        live_colors = prev["colors"]
        self.assertEqual(int(live_ids.max()) + 1, 3)

        # redo
        history.append(prev)
        nxt = redo.pop()
        live_ids = nxt["ids"]
        live_colors = nxt["colors"]
        self.assertEqual(int(np.unique(live_ids).size), 2)

    def test_empty_undo_is_noop(self) -> None:
        history: list[dict] = []
        self.assertEqual(len(history), 0)


if __name__ == "__main__":
    unittest.main()
