"""领域分割纯 NumPy 算法回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from AdvReverseEngineering.algorithms.regions import (
    REGION_IGNORED_ID,
    generate_region_colors,
    segment_regions_by_normal,
)
from AdvReverseEngineering.utils.mesh import FaceTopology


def _topology_from_pairs(
    face_count: int,
    pairs: list[tuple[int, int]],
) -> FaceTopology:
    """由无向邻接边对构造 FaceTopology（测试用）。"""
    if not pairs:
        edge_a = np.empty(0, dtype=np.int32)
        edge_b = np.empty(0, dtype=np.int32)
    else:
        arr = np.asarray(pairs, dtype=np.int32)
        edge_a = np.minimum(arr[:, 0], arr[:, 1])
        edge_b = np.maximum(arr[:, 0], arr[:, 1])

    both_a = np.concatenate((edge_a, edge_b)) if len(edge_a) else edge_a
    both_b = np.concatenate((edge_b, edge_a)) if len(edge_b) else edge_b
    if len(both_a):
        order = np.argsort(both_a, kind="stable")
        both_a = both_a[order]
        both_b = both_b[order]
        offsets = np.zeros(face_count + 1, dtype=np.int32)
        counts = np.bincount(both_a, minlength=face_count)
        offsets[1:] = np.cumsum(counts, dtype=np.int32)
        indices = both_b.astype(np.int32, copy=False)
    else:
        offsets = np.zeros(face_count + 1, dtype=np.int32)
        indices = np.empty(0, dtype=np.int32)

    empty = np.empty(0, dtype=np.int32)
    return FaceTopology(
        loop_start=empty.copy(),
        loop_total=empty.copy(),
        loop_vertex_indices=empty.copy(),
        adjacency_offsets=offsets,
        adjacency_indices=indices,
        edge_face_a=edge_a.astype(np.int32, copy=False),
        edge_face_b=edge_b.astype(np.int32, copy=False),
    )


class RegionSegmentationTests(unittest.TestCase):
    """法线阈值领域分割测试。"""

    def test_coplanar_adjacent_faces_merge(self) -> None:
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        areas = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        topology = _topology_from_pairs(3, [(0, 1), (1, 2)])

        result = segment_regions_by_normal(
            normals,
            areas,
            topology,
            angle_threshold_deg=15.0,
            ignore_discrete=False,
        )

        self.assertEqual(result["region_count"], 1)
        self.assertTrue(np.all(result["region_ids"] == 0))

    def test_large_normal_angle_keeps_separate(self) -> None:
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        areas = np.array([1.0, 1.0], dtype=np.float64)
        topology = _topology_from_pairs(2, [(0, 1)])

        result = segment_regions_by_normal(
            normals,
            areas,
            topology,
            angle_threshold_deg=15.0,
            ignore_discrete=False,
        )

        self.assertEqual(result["region_count"], 2)
        self.assertEqual(int(result["region_ids"][0]), 0)
        self.assertEqual(int(result["region_ids"][1]), 1)

    def test_vertex_only_contact_does_not_merge(self) -> None:
        # 两平面法线相同，但没有共享边邻接，不应合并。
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        areas = np.array([1.0, 1.0], dtype=np.float64)
        topology = _topology_from_pairs(2, [])

        result = segment_regions_by_normal(
            normals,
            areas,
            topology,
            angle_threshold_deg=15.0,
            ignore_discrete=False,
        )

        self.assertEqual(result["region_count"], 2)
        self.assertNotEqual(
            int(result["region_ids"][0]),
            int(result["region_ids"][1]),
        )

    def test_small_area_regions_are_ignored(self) -> None:
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        # 前两面形成大平面，第三面是离散小侧面。
        areas = np.array([10.0, 10.0, 0.01], dtype=np.float64)
        topology = _topology_from_pairs(3, [(0, 1)])

        result = segment_regions_by_normal(
            normals,
            areas,
            topology,
            angle_threshold_deg=15.0,
            ignore_discrete=True,
            min_area_ratio=0.01,
        )

        self.assertEqual(result["region_count"], 1)
        self.assertEqual(int(result["ignored_region_count"]), 1)
        self.assertEqual(int(result["ignored_face_count"]), 1)
        self.assertEqual(int(result["region_ids"][0]), 0)
        self.assertEqual(int(result["region_ids"][1]), 0)
        self.assertEqual(int(result["region_ids"][2]), REGION_IGNORED_ID)

    def test_colors_are_stable(self) -> None:
        first = generate_region_colors(5)
        second = generate_region_colors(5)
        self.assertEqual(first.shape, (5, 4))
        np.testing.assert_allclose(first, second)
        # 相邻编号色相应有明显差异，避免全同色。
        self.assertGreater(
            float(np.linalg.norm(first[0, :3] - first[1, :3])),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
