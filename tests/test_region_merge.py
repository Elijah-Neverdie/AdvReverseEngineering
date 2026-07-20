"""领域合并与中心计算回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from AdvReverseEngineering.algorithms.regions import (
    REGION_IGNORED_ID,
    compact_region_ids,
    compute_region_centroids,
    compute_region_label_anchors,
    merge_region_ids,
    remap_region_colors,
    remove_region_ids,
)


class RegionMergeTests(unittest.TestCase):
    """合并、压缩与中心测试。"""

    def test_compute_region_centroids_area_weighted(self) -> None:
        region_ids = np.array([0, 0, 1], dtype=np.int32)
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        areas = np.array([1.0, 3.0, 2.0], dtype=np.float64)

        centroids = compute_region_centroids(region_ids, centers, areas)

        np.testing.assert_allclose(centroids[0], [1.5, 0.0, 0.0])
        np.testing.assert_allclose(centroids[1], [10.0, 0.0, 0.0])

    def test_label_anchor_uses_central_face_normal(self) -> None:
        region_ids = np.array([0, 0, 0], dtype=np.int32)
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        areas = np.array([1.0, 10.0, 1.0], dtype=np.float64)

        anchors = compute_region_label_anchors(
            region_ids,
            centers,
            normals,
            areas,
            offset_ratio=0.0,
            min_offset=0.1,
        )

        self.assertIn(0, anchors)
        # 面积加权中心靠近 [1,0,0]，应选中第 1 个面，沿 +Y 偏移。
        self.assertEqual(int(anchors[0]["face_index"]), 1)
        np.testing.assert_allclose(
            anchors[0]["world_co"],
            [1.0, 0.1, 0.0],
            atol=1e-8,
        )

    def test_merge_then_compact_keeps_anchor_color(self) -> None:
        region_ids = np.array([0, 1, 2, 1], dtype=np.int32)
        colors = np.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.5],
            ],
            dtype=np.float32,
        )

        new_ids, new_colors, count, new_anchor = merge_region_ids(
            region_ids,
            colors,
            anchor_id=1,
            source_id=2,
        )

        self.assertEqual(count, 2)
        self.assertEqual(new_anchor, 1)
        # 原 2 已并入 1
        self.assertTrue(np.all(new_ids[region_ids == 2] == new_anchor))
        # 锚点绿色保留
        np.testing.assert_allclose(new_colors[new_anchor], colors[1])

    def test_sequential_n_merges(self) -> None:
        region_ids = np.array([0, 1, 2, 3], dtype=np.int32)
        colors = np.eye(4, 4, dtype=np.float32)
        colors[:, 3] = 0.5

        ids, colors, count, anchor = merge_region_ids(
            region_ids, colors, 0, 1
        )
        ids, colors, count, anchor = merge_region_ids(ids, colors, anchor, 1)
        # 压缩后原 2 变成 1，原 3 变成 2
        ids, colors, count, anchor = merge_region_ids(ids, colors, anchor, 1)

        self.assertEqual(count, 1)
        self.assertEqual(int(np.unique(ids[ids >= 0]).size), 1)
        # 锚点颜色仍为最初 0 号色
        np.testing.assert_allclose(colors[anchor, :3], [1.0, 0.0, 0.0], atol=1e-5)

    def test_compact_ignores_negative(self) -> None:
        region_ids = np.array([2, -1, 5, 2], dtype=np.int32)
        compacted, remap, count = compact_region_ids(region_ids)
        self.assertEqual(count, 2)
        self.assertEqual(int(compacted[1]), REGION_IGNORED_ID)
        self.assertEqual(int(remap[2]), 0)
        self.assertEqual(int(remap[5]), 1)

    def test_remap_colors_length(self) -> None:
        colors = np.ones((3, 4), dtype=np.float32)
        remap = np.array([-1, 0, 1], dtype=np.int32)
        remapped = remap_region_colors(colors, remap, 2)
        self.assertEqual(remapped.shape, (2, 4))

    def test_remove_region_ids_marks_ignored_and_compacts(self) -> None:
        region_ids = np.array([0, 1, 1, 2], dtype=np.int32)
        colors = np.eye(3, 4, dtype=np.float32)
        colors[:, 3] = 0.5

        new_ids, new_colors, count = remove_region_ids(region_ids, colors, 1)

        self.assertEqual(count, 2)
        self.assertEqual(int(new_ids[1]), REGION_IGNORED_ID)
        self.assertEqual(int(new_ids[2]), REGION_IGNORED_ID)
        self.assertEqual(int(new_ids[0]), 0)
        self.assertEqual(int(new_ids[3]), 1)
        self.assertEqual(new_colors.shape[0], 2)
        np.testing.assert_allclose(new_colors[0, :3], colors[0, :3])
        np.testing.assert_allclose(new_colors[1, :3], colors[2, :3])


if __name__ == "__main__":
    unittest.main()
