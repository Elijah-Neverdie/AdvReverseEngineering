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
    split_polyline_at_breaks,
    transform_bezier_points,
    turn_angles_deg,
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


if __name__ == "__main__":
    unittest.main()
