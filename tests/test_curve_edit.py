# -*- coding: utf-8 -*-
"""曲线拆分/贝塞尔拟合算法单测。"""

from __future__ import annotations

import unittest

import numpy as np

from AdvReverseEngineering.algorithms.curve_edit import (
    best_closed_alignment,
    best_open_alignment,
    bridge_fit_surface_boundaries,
    compose_patch_from_boundary_polylines,
    default_surface_segments,
    estimate_open_directed_similarity,
    estimate_similarity_transform,
    find_break_indices,
    fit_bezier_n_controls,
    opposite_edge_pairs,
    order_open_curves_as_closed_loop,
    pack_boundary_sides,
    sample_polyline_uniform,
    snap_bezier_endpoints,
    split_polyline_at_breaks,
    stitch_oriented_loop_polylines,
    transform_bezier_points,
    turn_angles_deg,
    unpack_boundary_sides,
    weld_bezier_loop_endpoints,
)


class CurveEditTests(unittest.TestCase):
    def test_square_breaks_at_corners(self) -> None:
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        angles = turn_angles_deg(pts, cyclic=True)
        self.assertTrue(np.all(angles >= 89.0))
        breaks = find_break_indices(pts, 45.0, cyclic=True, min_separation=1)
        self.assertEqual(len(breaks), 4)
        parts = split_polyline_at_breaks(pts, breaks, cyclic=True)
        self.assertEqual(len(parts), 4)

    def test_fit_bezier_open_and_closed(self) -> None:
        line = np.linspace([0, 0, 0], [1, 0, 0], 20)
        bez = fit_bezier_n_controls(line, 4, cyclic=False)
        self.assertEqual(len(bez), 4)
        circle = np.array(
            [
                [np.cos(t), np.sin(t), 0.0]
                for t in np.linspace(0, 2 * np.pi, 36, endpoint=False)
            ],
            dtype=np.float64,
        )
        bez_c = fit_bezier_n_controls(circle, 6, cyclic=True)
        self.assertEqual(len(bez_c), 6)

    def test_fit_bezier_min_two_controls(self) -> None:
        line = np.linspace([0, 0, 0], [1, 0, 0], 12)
        bez = fit_bezier_n_controls(line, 1, cyclic=False)
        self.assertEqual(len(bez), 2)
        bez2 = fit_bezier_n_controls(line, 2, cyclic=False)
        self.assertEqual(len(bez2), 2)

    def test_default_surface_segments_short_edge_controls(self) -> None:
        # U 为短边、控制点 3 → U 细分 3；V 为 2 倍长 → 细分 6
        su, sv = default_surface_segments(1.0, 2.0, 3, 5)
        self.assertEqual(su, 3)
        self.assertEqual(sv, 6)
        # V 为短边、控制点 4 → V 细分 4；U 为 3 倍长 → 细分 12
        su2, sv2 = default_surface_segments(3.0, 1.0, 8, 4)
        self.assertEqual(su2, 12)
        self.assertEqual(sv2, 4)
        # 等长时取 U 为短边侧（≤），细分=各自控制点逻辑：lu<=lv → u=cu
        su3, sv3 = default_surface_segments(1.0, 1.0, 3, 5)
        self.assertEqual(su3, 3)
        self.assertEqual(sv3, 3)

    def test_similarity_roundtrip(self) -> None:
        src = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        # scale 2, translate (3,4,5)
        dst = src * 2.0 + np.array([3.0, 4.0, 5.0])
        scale, rotation, translation = estimate_similarity_transform(src, dst)
        self.assertAlmostEqual(scale, 2.0, places=5)
        recovered = (scale * (src @ rotation.T)) + translation
        self.assertTrue(np.allclose(recovered, dst, atol=1e-5))
        aligned, rmse = best_closed_alignment(src, np.roll(src, 1, axis=0))
        self.assertLess(rmse, 1e-9)

        proto = fit_bezier_n_controls(src, 4, cyclic=True)
        xform = transform_bezier_points(proto, scale, rotation, translation)
        self.assertEqual(len(xform), 4)

    def test_order_open_curves_as_closed_quad(self) -> None:
        # 故意乱序 + 一条反向，端点应能串成单位正方形
        bottom = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
        right = np.array([[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0]])
        top = np.array([[1.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.0, 1.0, 0.0]])
        left = np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        # 乱序：right, left(反向), bottom, top
        left_rev = left[::-1].copy()
        result = order_open_curves_as_closed_loop(
            [right, left_rev, bottom, top]
        )
        self.assertIsNotNone(result)
        order, flipped, gap = result
        self.assertLess(gap, 0.05)
        self.assertEqual(len(order), 4)
        self.assertEqual(set(order), {0, 1, 2, 3})
        self.assertEqual(opposite_edge_pairs(4), [(0, 2), (1, 3)])

    def test_weld_bezier_loop_endpoints_closes(self) -> None:
        def _line_bez(a, b):
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            mid = 0.5 * (a + b)
            return [
                {
                    "co": a.copy(),
                    "handle_left": a - np.array([0.1, 0.0, 0.0]),
                    "handle_right": a + np.array([0.1, 0.0, 0.0]),
                },
                {
                    "co": mid.copy(),
                    "handle_left": mid - np.array([0.05, 0.0, 0.0]),
                    "handle_right": mid + np.array([0.05, 0.0, 0.0]),
                },
                {
                    "co": b.copy(),
                    "handle_left": b - np.array([0.1, 0.0, 0.0]),
                    "handle_right": b + np.array([0.1, 0.0, 0.0]),
                },
            ]

        # 故意留缝隙
        sides = [
            _line_bez([0.0, 0.0, 0.0], [1.02, 0.0, 0.0]),
            _line_bez([1.0, 0.0, 0.0], [1.0, 1.01, 0.0]),
            _line_bez([1.0, 1.0, 0.0], [-0.01, 1.0, 0.0]),
            _line_bez([0.0, 1.0, 0.0], [0.0, -0.02, 0.0]),
        ]
        welded = weld_bezier_loop_endpoints(sides)
        for index in range(4):
            end_co = welded[index][-1]["co"]
            start_co = welded[(index + 1) % 4][0]["co"]
            self.assertTrue(np.allclose(end_co, start_co, atol=1e-9))

    def test_opposite_directed_similarity_preserves_loop_ends(self) -> None:
        """对边环向相反时，有向相似必须保持首尾，不能用反向对齐。"""
        bottom = np.linspace([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], 32)
        # 闭环上的顶边：右→左
        top = np.linspace([2.0, 1.0, 0.0], [0.0, 1.0, 0.0], 32)
        ref = sample_polyline_uniform(bottom, 48, cyclic=False)
        dst = sample_polyline_uniform(top, 48, cyclic=False)

        # 错误做法：反向对齐会把顶边变成左→右，变换后首尾对调
        aligned_bad, _ = best_open_alignment(ref, dst)
        scale_b, rot_b, t_b = estimate_similarity_transform(ref, aligned_bad)
        proto = fit_bezier_n_controls(bottom, 4, cyclic=False)
        bad = transform_bezier_points(proto, scale_b, rot_b, t_b)
        bad_start_err = float(np.linalg.norm(bad[0]["co"] - top[0]))
        bad_end_err = float(np.linalg.norm(bad[-1]["co"] - top[-1]))

        # 正确：有向相似 + 端点钉扎
        scale, rot, trans = estimate_open_directed_similarity(ref, dst)
        good = transform_bezier_points(proto, scale, rot, trans)
        good = snap_bezier_endpoints(good, top[0], top[-1])
        self.assertTrue(np.allclose(good[0]["co"], top[0], atol=1e-9))
        self.assertTrue(np.allclose(good[-1]["co"], top[-1], atol=1e-9))
        # 反向对齐后的首尾误差应明显更大
        self.assertGreater(bad_start_err + bad_end_err, 1.0)

    def test_compose_quad_patch_from_square(self) -> None:
        bottom = np.linspace([0, 0, 0], [1, 0, 0], 16)
        right = np.linspace([1, 0, 0], [1, 1, 0], 16)
        top = np.linspace([1, 1, 0], [0, 1, 0], 16)
        left = np.linspace([0, 1, 0], [0, 0, 0], 16)
        # 乱序传入
        verts, faces, kind = compose_patch_from_boundary_polylines(
            [right, left, bottom, top],
            segments_u=4,
            segments_v=4,
        )
        self.assertEqual(kind, "QUAD")
        self.assertEqual(len(verts), 5 * 5)
        self.assertGreater(len(faces), 0)
        # 角点应接近单位正方形四角
        corners = verts[[0, 4, 20, 24]]
        for expected in (
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ):
            dists = np.linalg.norm(corners - np.asarray(expected), axis=1)
            self.assertLess(float(dists.min()), 0.05)

    def test_compose_tri_patch_from_triangle(self) -> None:
        a = np.linspace([0, 0, 0], [1, 0, 0], 12)
        b = np.linspace([1, 0, 0], [0.5, 0.8, 0], 12)
        c = np.linspace([0.5, 0.8, 0], [0, 0, 0], 12)
        verts, faces, kind = compose_patch_from_boundary_polylines(
            [a, b, c],
            segments_u=4,
            segments_v=4,
        )
        self.assertEqual(kind, "TRI")
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(faces), 0)

    def test_stitch_open_corner_gap(self) -> None:
        """一角有缺口时，切向延伸应汇合到近似直角交点。"""
        bottom = np.linspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 10)
        right = np.linspace([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 10)
        top = np.linspace([1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 10)
        # 左边不到底：止于 (0, 0.15)，与 bottom 起点形成开口
        left = np.linspace([0.0, 1.0, 0.0], [0.0, 0.15, 0.0], 10)
        # 同时 bottom 起点改到 (0.12, 0)，两游离端点不相邻
        bottom = np.linspace([0.12, 0.0, 0.0], [1.0, 0.0, 0.0], 10)
        loop = [bottom, right, top, left]
        # 无缝合时最大间隙应明显
        gap0 = float(np.linalg.norm(loop[3][-1] - loop[0][0]))
        self.assertGreater(gap0, 0.1)
        stitched, count = stitch_oriented_loop_polylines(loop)
        self.assertGreaterEqual(count, 1)
        gap1 = float(np.linalg.norm(stitched[3][-1] - stitched[0][0]))
        self.assertLess(gap1, 1e-6)
        # 交点应接近原点
        hit = stitched[0][0]
        self.assertLess(float(np.linalg.norm(hit - np.array([0.0, 0.0, 0.0]))), 0.05)

    def test_order_allows_large_gaps_when_requested(self) -> None:
        bottom = np.linspace([0.3, 0.0, 0.0], [1.0, 0.0, 0.0], 8)
        right = np.linspace([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 8)
        top = np.linspace([1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 8)
        left = np.linspace([0.0, 1.0, 0.0], [0.0, 0.3, 0.0], 8)
        self.assertIsNone(order_open_curves_as_closed_loop([bottom, right, top, left]))
        ordered = order_open_curves_as_closed_loop(
            [bottom, right, top, left],
            allow_large_gaps=True,
        )
        self.assertIsNotNone(ordered)

    def test_pack_unpack_boundary_sides(self) -> None:
        sides = [
            np.linspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5),
            np.linspace([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 4),
        ]
        flat, counts = pack_boundary_sides(sides)
        restored = unpack_boundary_sides(flat, counts)
        self.assertEqual(len(restored), 2)
        np.testing.assert_allclose(restored[0], sides[0])
        np.testing.assert_allclose(restored[1], sides[1])

    def test_bridge_two_quad_boundaries(self) -> None:
        # 左补丁 x=0..1，右补丁 x=1.4..2.4，中间缺口沿贝塞尔桥接
        left_sides = [
            np.linspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 8),
            np.linspace([1.0, 0.0, 0.0], [1.0, 1.0, 0.2], 8),
            np.linspace([1.0, 1.0, 0.2], [0.0, 1.0, 0.0], 8),
            np.linspace([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], 8),
        ]
        right_sides = [
            np.linspace([1.4, 0.0, 0.0], [2.4, 0.0, 0.0], 8),
            np.linspace([2.4, 0.0, 0.0], [2.4, 1.0, 0.0], 8),
            np.linspace([2.4, 1.0, 0.0], [1.4, 1.0, 0.2], 8),
            np.linspace([1.4, 1.0, 0.2], [1.4, 0.0, 0.0], 8),
        ]
        verts, faces = bridge_fit_surface_boundaries(
            left_sides, right_sides, segments_u=6, segments_v=6
        )
        self.assertGreaterEqual(len(verts), (6 + 1) * (6 + 1))
        self.assertGreater(len(faces), 0)
        # 桥接网格应落在缺口附近（x 大致在 1~1.4）
        xs = verts[:, 0]
        self.assertGreater(float(xs.min()), 0.85)
        self.assertLess(float(xs.max()), 1.55)


if __name__ == "__main__":
    unittest.main()
