# -*- coding: utf-8 -*-
"""曲线拆分/贝塞尔拟合算法单测。"""

from __future__ import annotations

import unittest

import numpy as np

from AdvReverseEngineering.algorithms.curve_edit import (
    best_closed_alignment,
    estimate_similarity_transform,
    find_break_indices,
    fit_bezier_n_controls,
    opposite_edge_pairs,
    order_open_curves_as_closed_loop,
    split_polyline_at_breaks,
    transform_bezier_points,
    turn_angles_deg,
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


if __name__ == "__main__":
    unittest.main()
