"""自动摆正纯 NumPy 算法回归测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from AdvReverseEngineering.algorithms import orientation
from AdvReverseEngineering.algorithms.ground_snap import fit_plane_svd
from AdvReverseEngineering.algorithms.normal_cluster import (
    cluster_dominant_normal,
)
from AdvReverseEngineering.utils.math import (
    euler_xyz_to_matrix,
    rotation_align_vector_to_axis,
)
from AdvReverseEngineering.utils.versioning import (
    format_version,
    parse_bl_info_version,
)


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


class GroundSnapTests(unittest.TestCase):
    """底面精对齐数学测试。"""

    def test_tilted_plane_aligns_to_xy(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.1],
                [1.0, 1.0, 0.1],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        center, normal = fit_plane_svd(points)
        if normal[2] < 0.0:
            normal = -normal
        correction = rotation_align_vector_to_axis(
            normal,
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        np.testing.assert_allclose(
            correction @ normal,
            np.array([0.0, 0.0, 1.0]),
            atol=1e-10,
        )
        aligned = (points - center) @ correction.T + center
        self.assertLess(float(np.ptp(aligned[:, 2])), 1e-10)


class VersioningTests(unittest.TestCase):
    """GitHub 远端版本解析测试。"""

    def test_parse_bl_info_version(self) -> None:
        source = 'bl_info = {"version": (1, 12, 3), "blender": (4, 2, 0)}'
        version = parse_bl_info_version(source)
        self.assertEqual(version, (1, 12, 3))
        self.assertEqual(format_version(version), "1.12.3")


if __name__ == "__main__":
    unittest.main()
