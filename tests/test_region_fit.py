"""领域三边/四边曲面拟合算法回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from AdvReverseEngineering.algorithms.region_fit import (
    RegionFitError,
    bridge_concave_notches,
    build_quad_patch,
    build_triangular_patch,
    classify_tri_or_quad,
    coons_patch,
    detect_corner_indices,
    extract_region_boundary_loops,
    fit_region_surface,
    polyline_length,
    resample_closed_polyline,
    resample_polyline,
    select_primary_boundary_loop,
)


def _square_mesh():
    """
    单个正方形面：
      0--1
      |  |
      3--2
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    loop_start = np.array([0], dtype=np.int32)
    loop_total = np.array([4], dtype=np.int32)
    loop_vertex_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    region_ids = np.array([0], dtype=np.int32)
    normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    areas = np.array([1.0], dtype=np.float64)
    centers = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)
    return {
        "vertices": vertices,
        "loop_start": loop_start,
        "loop_total": loop_total,
        "loop_vertex_indices": loop_vertex_indices,
        "region_ids": region_ids,
        "normals": normals,
        "areas": areas,
        "centers": centers,
    }


def _two_quad_strip():
    """
    两个共边四边形组成矩形条：
      0--1--2
      |A |B |
      3--4--5
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    # A: 0,1,4,3  B: 1,2,5,4
    loop_start = np.array([0, 4], dtype=np.int32)
    loop_total = np.array([4, 4], dtype=np.int32)
    loop_vertex_indices = np.array(
        [0, 1, 4, 3, 1, 2, 5, 4],
        dtype=np.int32,
    )
    region_ids = np.array([0, 0], dtype=np.int32)
    normals = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    areas = np.array([1.0, 1.0], dtype=np.float64)
    centers = np.array(
        [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    return {
        "vertices": vertices,
        "loop_start": loop_start,
        "loop_total": loop_total,
        "loop_vertex_indices": loop_vertex_indices,
        "region_ids": region_ids,
        "normals": normals,
        "areas": areas,
        "centers": centers,
    }


def _triangle_like_quad_region():
    """
    近似三角形的四边形区域：三条长边 + 极短第四边。
      0----1
       \\  /|
        \\/ |
         3-2   (边 2-3 极短)
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.05, 2.0, 0.0],
            [0.95, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    loop_start = np.array([0], dtype=np.int32)
    loop_total = np.array([4], dtype=np.int32)
    loop_vertex_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    region_ids = np.array([0], dtype=np.int32)
    normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    areas = np.array([2.0], dtype=np.float64)
    centers = np.array([[1.0, 0.8, 0.0]], dtype=np.float64)
    return {
        "vertices": vertices,
        "loop_start": loop_start,
        "loop_total": loop_total,
        "loop_vertex_indices": loop_vertex_indices,
        "region_ids": region_ids,
        "normals": normals,
        "areas": areas,
        "centers": centers,
    }


class BoundaryExtractionTests(unittest.TestCase):
    def test_square_outer_boundary_loop(self) -> None:
        mesh = _square_mesh()
        loops = extract_region_boundary_loops(
            mesh["region_ids"],
            0,
            mesh["loop_start"],
            mesh["loop_total"],
            mesh["loop_vertex_indices"],
        )
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 4)

    def test_internal_shared_edge_cancelled(self) -> None:
        mesh = _two_quad_strip()
        loops = extract_region_boundary_loops(
            mesh["region_ids"],
            0,
            mesh["loop_start"],
            mesh["loop_total"],
            mesh["loop_vertex_indices"],
        )
        self.assertEqual(len(loops), 1)
        # 外环 6 顶点
        self.assertEqual(len(loops[0]), 6)
        primary = select_primary_boundary_loop(loops, mesh["vertices"])
        self.assertEqual(len(primary), 6)

    def test_missing_region_raises(self) -> None:
        mesh = _square_mesh()
        with self.assertRaises(RegionFitError):
            extract_region_boundary_loops(
                mesh["region_ids"],
                9,
                mesh["loop_start"],
                mesh["loop_total"],
                mesh["loop_vertex_indices"],
            )


class TopologyClassificationTests(unittest.TestCase):
    def test_detect_square_corners(self) -> None:
        pts = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        corners = detect_corner_indices(pts, angle_threshold_deg=20.0)
        self.assertEqual(len(corners), 4)

    def test_triangle_ratio_classification(self) -> None:
        long_a = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        long_b = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
        long_c = np.array([[2.0, 0.0], [0.1, 0.0]], dtype=np.float64)
        short = np.array([[0.1, 0.0], [0.0, 0.0]], dtype=np.float64)
        topology, sides = classify_tri_or_quad(
            [long_a, long_b, long_c, short],
            triangle_ratio=0.15,
        )
        self.assertEqual(topology, "TRI")
        self.assertEqual(len(sides), 3)

    def test_quad_when_fourth_edge_long(self) -> None:
        sides = [
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
            np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        ]
        topology, result = classify_tri_or_quad(sides, triangle_ratio=0.15)
        self.assertEqual(topology, "QUAD")
        self.assertEqual(len(result), 4)

    def test_elongated_band_stays_quad(self) -> None:
        # 扁长条带：短边远小于长边的 15%，但两条短边长度接近，
        # 必须保持四边拓扑（回归：曾被误判为三边形成五边形轮廓）
        sides = [
            np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64),
            np.array([[10.0, 0.0], [10.0, 1.0]], dtype=np.float64),
            np.array([[10.0, 1.0], [0.0, 1.2]], dtype=np.float64),
            np.array([[0.0, 1.2], [0.0, 0.0]], dtype=np.float64),
        ]
        topology, result = classify_tri_or_quad(sides, triangle_ratio=0.15)
        self.assertEqual(topology, "QUAD")
        self.assertEqual(len(result), 4)


class ConcaveNotchBridgeTests(unittest.TestCase):
    def test_inward_step_notch_is_bridged(self) -> None:
        # 沿 +x 的边链中部有向内（+y，即领域内侧）的台阶凹口
        side = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [2.2, 0.6],
                [2.6, 0.6],
                [2.8, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float64,
        )
        bridged_2d, bridged_3d = bridge_concave_notches([side])
        self.assertIsNone(bridged_3d)
        result = bridged_2d[0]
        # 凹口被拉直：所有点回到 y≈0 的拟合线上
        self.assertTrue(np.all(np.abs(result[:, 1]) < 1e-9))
        # 端点保持不变
        np.testing.assert_allclose(result[0], side[0])
        np.testing.assert_allclose(result[-1], side[-1])

    def test_outward_bulge_is_preserved(self) -> None:
        # 向外（-y）凸出的边链应保留，不被桥接
        side = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [2.2, -0.6],
                [2.6, -0.6],
                [2.8, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float64,
        )
        bridged_2d, _ = bridge_concave_notches([side])
        result = bridged_2d[0]
        np.testing.assert_allclose(result, side)

    def test_smooth_arc_untouched(self) -> None:
        # 平缓弧形（模拟弯曲条带内弧边）没有锐折角，保持原样
        t = np.linspace(0.0, np.pi, 40)
        side = np.column_stack((t * 2.0, np.sin(t) * 0.4))
        bridged_2d, _ = bridge_concave_notches([side])
        np.testing.assert_allclose(bridged_2d[0], side)


class ClosedResampleTests(unittest.TestCase):
    def test_closed_resample_count_and_range(self) -> None:
        square = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        sampled = resample_closed_polyline(square, 16)
        self.assertEqual(len(sampled), 16)
        # 所有采样点应落在单位正方形边界上
        on_boundary = (
            np.isclose(sampled[:, 0], 0.0)
            | np.isclose(sampled[:, 0], 1.0)
            | np.isclose(sampled[:, 1], 0.0)
            | np.isclose(sampled[:, 1], 1.0)
        )
        self.assertTrue(bool(np.all(on_boundary)))


class ResampleAndCoonsTests(unittest.TestCase):
    def test_resample_preserves_endpoints(self) -> None:
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        sampled = resample_polyline(pts, 5)
        self.assertEqual(len(sampled), 5)
        np.testing.assert_allclose(sampled[0], pts[0])
        np.testing.assert_allclose(sampled[-1], pts[-1])

    def test_coons_plane_rectangle(self) -> None:
        bottom = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        top = np.array(
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        left = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        right = np.array(
            [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        grid = coons_patch(bottom, right, top, left)
        self.assertEqual(grid.shape, (3, 3, 3))
        np.testing.assert_allclose(grid[0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(grid[0, -1], [2.0, 0.0, 0.0])
        np.testing.assert_allclose(grid[-1, 0], [0.0, 2.0, 0.0])
        np.testing.assert_allclose(grid[-1, -1], [2.0, 2.0, 0.0])
        np.testing.assert_allclose(grid[1, 1], [1.0, 1.0, 0.0], atol=1e-8)

    def test_quad_opposite_counts_match(self) -> None:
        sides = [
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64),
            np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
        ]
        verts, faces = build_quad_patch(sides, segments_u=4, segments_v=3)
        self.assertEqual(len(verts), 5 * 4)
        self.assertTrue(all(len(face) == 4 for face in faces))
        self.assertEqual(len(faces), 4 * 3)

    def test_triangle_long_edges_same_count(self) -> None:
        long0 = np.array(
            [[0.0, 0.0, 0.0], [0.5, 1.5, 0.0], [1.0, 3.0, 0.0]],
            dtype=np.float64,
        )
        long1 = np.array(
            [[2.0, 0.0, 0.0], [1.5, 1.5, 0.0], [1.0, 3.0, 0.0]],
            dtype=np.float64,
        )
        base = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        verts, faces = build_triangular_patch(
            long0,
            long1,
            base,
            segments_long=5,
            segments_base=3,
        )
        # body rows = 5, width = 4, plus tip
        self.assertEqual(len(verts), 5 * 4 + 1)
        self.assertTrue(any(len(face) == 3 for face in faces))
        self.assertTrue(any(len(face) == 4 for face in faces))


class FitRegionSurfaceTests(unittest.TestCase):
    def test_fit_square_as_quad(self) -> None:
        mesh = _square_mesh()
        result = fit_region_surface(
            region_ids=mesh["region_ids"],
            target_id=0,
            vertices=mesh["vertices"],
            loop_start=mesh["loop_start"],
            loop_total=mesh["loop_total"],
            loop_vertex_indices=mesh["loop_vertex_indices"],
            face_normals=mesh["normals"],
            face_areas=mesh["areas"],
            face_centers=mesh["centers"],
            segments_u=3,
            segments_v=2,
            triangle_ratio=0.15,
        )
        self.assertEqual(result.topology, "QUAD")
        self.assertEqual(result.segments_u, 3)
        self.assertEqual(result.segments_v, 2)
        self.assertEqual(len(result.vertices), 4 * 3)
        self.assertTrue(all(len(face) == 4 for face in result.faces))

    def test_fit_short_fourth_edge_as_triangle(self) -> None:
        mesh = _triangle_like_quad_region()
        result = fit_region_surface(
            region_ids=mesh["region_ids"],
            target_id=0,
            vertices=mesh["vertices"],
            loop_start=mesh["loop_start"],
            loop_total=mesh["loop_total"],
            loop_vertex_indices=mesh["loop_vertex_indices"],
            face_normals=mesh["normals"],
            face_areas=mesh["areas"],
            face_centers=mesh["centers"],
            segments_u=3,
            segments_v=4,
            triangle_ratio=0.15,
        )
        self.assertEqual(result.topology, "TRI")
        self.assertTrue(any(len(face) == 3 for face in result.faces))

    def test_face_orientation_matches_normal(self) -> None:
        mesh = _square_mesh()
        result = fit_region_surface(
            region_ids=mesh["region_ids"],
            target_id=0,
            vertices=mesh["vertices"],
            loop_start=mesh["loop_start"],
            loop_total=mesh["loop_total"],
            loop_vertex_indices=mesh["loop_vertex_indices"],
            face_normals=mesh["normals"],
            face_areas=mesh["areas"],
            face_centers=mesh["centers"],
            segments_u=2,
            segments_v=2,
        )
        for face in result.faces:
            a, b, c = (result.vertices[i] for i in face[:3])
            normal = np.cross(b - a, c - a)
            self.assertGreater(float(normal[2]), 0.0)

    def test_polyline_length(self) -> None:
        pts = np.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
            dtype=np.float64,
        )
        self.assertAlmostEqual(polyline_length(pts), 7.0)


if __name__ == "__main__":
    unittest.main()
