"""领域分割纯 NumPy 算法回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from AdvReverseEngineering.algorithms.regions import (
    REGION_IGNORED_ID,
    blender_wire_edge_fac,
    blender_wire_step_param,
    generate_region_colors,
    segment_regions_by_normal,
    smooth_face_normals,
    wireframe_threshold_to_cos_limit,
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


class BlenderWireframeFormulaTests(unittest.TestCase):
    """对齐 Blender overlay 线框边因子公式。"""

    def test_coplanar_fac_near_one(self) -> None:
        fac = blender_wire_edge_fac(1.0)
        self.assertAlmostEqual(fac, 254.0 / 255.0, places=6)

    def test_hard_edge_fac_zero(self) -> None:
        # cosine<=0.995 → fac 触底为 0，任意非零线框阈值下都显示
        fac = blender_wire_edge_fac(0.0)
        self.assertEqual(fac, 0.0)
        fac = blender_wire_edge_fac(0.995)
        self.assertEqual(fac, 0.0)

    def test_threshold_0_1_matches_shader_visibility(self) -> None:
        step = blender_wire_step_param(0.1)
        # 平坦边不可见
        self.assertFalse(blender_wire_edge_fac(1.0) <= step)
        # 硬边可见
        self.assertTrue(blender_wire_edge_fac(0.0) <= step)
        # 临界：fac == step 时仍可见（着色器用 <=）
        cos_limit = wireframe_threshold_to_cos_limit(0.1)
        fac_at_limit = blender_wire_edge_fac(cos_limit)
        self.assertAlmostEqual(fac_at_limit, step, places=5)

    def test_lower_threshold_keeps_softer_edges_mergeable(self) -> None:
        cos_low = wireframe_threshold_to_cos_limit(0.05)
        cos_high = wireframe_threshold_to_cos_limit(0.5)
        # 更低的线框阈值 → 更低的合并点积门槛：
        # 软边更易并入同一领域，硬边（fac=0）仍始终切断。
        self.assertLess(cos_low, cos_high)


class RegionSegmentationTests(unittest.TestCase):
    """线框硬边领域分割测试。"""

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
            wireframe_threshold=0.1,
            ignore_discrete=False,
        )

        self.assertEqual(result["region_count"], 1)
        self.assertTrue(np.all(result["region_ids"] == 0))

    def test_wireframe_visible_edge_splits_region(self) -> None:
        # 约 10° 夹角：Blender 线框下 fac=0，必为领域边界
        angle = np.radians(10.0)
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float64,
        )
        areas = np.array([1.0, 1.0], dtype=np.float64)
        topology = _topology_from_pairs(2, [(0, 1)])

        result = segment_regions_by_normal(
            normals,
            areas,
            topology,
            wireframe_threshold=0.1,
            ignore_discrete=False,
        )
        self.assertEqual(result["region_count"], 2)

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
            wireframe_threshold=0.1,
            ignore_discrete=False,
        )

        self.assertEqual(result["region_count"], 2)
        self.assertEqual(int(result["region_ids"][0]), 0)
        self.assertEqual(int(result["region_ids"][1]), 1)

    def test_vertex_only_contact_does_not_merge(self) -> None:
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
            wireframe_threshold=0.1,
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
        areas = np.array([10.0, 10.0, 0.01], dtype=np.float64)
        topology = _topology_from_pairs(3, [(0, 1)])

        result = segment_regions_by_normal(
            normals,
            areas,
            topology,
            wireframe_threshold=0.1,
            ignore_discrete=True,
            min_area_ratio=0.01,
        )

        self.assertEqual(result["region_count"], 1)
        self.assertEqual(int(result["ignored_region_count"]), 1)
        self.assertEqual(int(result["ignored_face_count"]), 1)
        self.assertEqual(int(result["region_ids"][0]), 0)
        self.assertEqual(int(result["region_ids"][1]), 0)
        self.assertEqual(int(result["region_ids"][2]), REGION_IGNORED_ID)

    def test_gradual_curve_does_not_leak(self) -> None:
        # 相邻面两两差 10°：线框下每条边 fac=0，不应链式合成一块
        count = 10
        angles = np.radians(np.arange(count) * 10.0)
        normals = np.stack(
            (
                np.zeros(count),
                -np.sin(angles),
                np.cos(angles),
            ),
            axis=1,
        )
        areas = np.ones(count, dtype=np.float64)
        pairs = [(i, i + 1) for i in range(count - 1)]
        topology = _topology_from_pairs(count, pairs)

        result = segment_regions_by_normal(
            normals,
            areas,
            topology,
            wireframe_threshold=0.1,
            ignore_discrete=False,
            smooth_iterations=0,
        )

        self.assertGreaterEqual(result["region_count"], 2)
        self.assertNotEqual(
            int(result["region_ids"][0]),
            int(result["region_ids"][-1]),
        )

    def test_legacy_angle_threshold_still_works(self) -> None:
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        areas = np.ones(2, dtype=np.float64)
        topology = _topology_from_pairs(2, [(0, 1)])
        result = segment_regions_by_normal(
            normals,
            areas,
            topology,
            wireframe_threshold=None,
            angle_threshold_deg=15.0,
            ignore_discrete=False,
        )
        self.assertEqual(result["region_count"], 1)

    def test_smoothing_preserves_hard_edges(self) -> None:
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        areas = np.ones(2, dtype=np.float64)
        topology = _topology_from_pairs(2, [(0, 1)])

        smoothed = smooth_face_normals(
            normals,
            areas,
            topology,
            iterations=3,
            edge_angle_limit_deg=30.0,
        )

        np.testing.assert_allclose(smoothed, normals, atol=1e-12)

    def test_smoothing_reduces_scan_noise(self) -> None:
        count = 9
        rng = np.random.default_rng(7)
        noise = np.radians((rng.random(count) - 0.5) * 16.0)
        normals = np.stack(
            (
                np.sin(noise),
                np.zeros(count),
                np.cos(noise),
            ),
            axis=1,
        )
        areas = np.ones(count, dtype=np.float64)
        pairs = [(i, i + 1) for i in range(count - 1)]
        topology = _topology_from_pairs(count, pairs)

        smoothed = smooth_face_normals(
            normals,
            areas,
            topology,
            iterations=2,
            edge_angle_limit_deg=30.0,
        )
        spread_before = float(
            np.ptp(np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1))))
        )
        spread_after = float(
            np.ptp(np.degrees(np.arccos(np.clip(smoothed[:, 2], -1, 1))))
        )
        self.assertLess(spread_after, spread_before)

        result = segment_regions_by_normal(
            normals,
            areas,
            topology,
            wireframe_threshold=0.1,
            ignore_discrete=False,
            smooth_iterations=2,
        )
        self.assertEqual(result["region_count"], 1)

    def test_colors_are_stable(self) -> None:
        first = generate_region_colors(5)
        second = generate_region_colors(5)
        self.assertEqual(first.shape, (5, 4))
        np.testing.assert_allclose(first, second)
        self.assertGreater(
            float(np.linalg.norm(first[0, :3] - first[1, :3])),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
