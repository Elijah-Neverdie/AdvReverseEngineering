"""自动摆正纯 NumPy 算法回归测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from AdvReverseEngineering.algorithms import orientation
from AdvReverseEngineering.algorithms.normal_cluster import (
    cluster_dominant_normal,
)
from AdvReverseEngineering.utils.math import euler_xyz_to_matrix


class NormalClusterTests(unittest.TestCase):
    """法线无向聚类测试。"""

    def test_opposite_normals_share_one_axis_cluster(self) -> None:
        normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        areas = np.array([10.0, 9.0, 1.0, 1.0], dtype=np.float64)

        dominant = cluster_dominant_normal(normals, areas)

        self.assertGreater(abs(float(dominant[0])), 0.999)
        self.assertLess(abs(float(dominant[1])), 1e-6)
        self.assertLess(abs(float(dominant[2])), 1e-6)


class CombinedOrientationTests(unittest.TestCase):
    """组合流程必须累计各阶段修正，而不是覆盖前一步。"""

    def test_combined_strategy_composes_corrections(self) -> None:
        pca_rotation = euler_xyz_to_matrix(0.1, 0.0, 0.0)
        ransac_rotation = euler_xyz_to_matrix(0.0, 0.2, 0.0)
        normal_rotation = euler_xyz_to_matrix(0.0, 0.0, 0.3)
        expected = normal_rotation @ ransac_rotation @ pca_rotation

        mesh_data = {
            "vertices": np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            "normals": np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            "areas": np.array([4.0], dtype=np.float64),
            "face_centers": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            "centroid": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        }
        settings = {
            "use_pca": True,
            "detect_largest_plane": True,
            "normal_clustering": True,
            "obb_refinement": True,
        }

        with (
            patch.object(
                orientation,
                "orientation_matrix_pca",
                return_value=pca_rotation,
            ),
            patch.object(
                orientation,
                "orientation_matrix_ransac",
                return_value=ransac_rotation,
            ),
            patch.object(
                orientation,
                "orientation_matrix_normal_cluster",
                return_value=normal_rotation,
            ),
            patch.object(
                orientation,
                "orientation_matrix_obb",
                side_effect=lambda _vertices, initial: initial,
            ),
        ):
            result = orientation._strategy_combined(mesh_data, settings)

        np.testing.assert_allclose(result, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
