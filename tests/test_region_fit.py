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
    combine_boundary_islands,
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


def _disconnected_same_region_strip():
    """
    同一 region_id 的两个不连通矩形 island，中间有 0.5 宽断口：

      0--1   4--5
      |A |   |B |
      3--2   7--6
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [2.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return {
        "vertices": vertices,
        "loop_start": np.array([0, 4], dtype=np.int32),
        "loop_total": np.array([4, 4], dtype=np.int32),
        "loop_vertex_indices": np.array(
            [0, 1, 2, 3, 4, 5, 6, 7],
            dtype=np.int32,
        ),
        "region_ids": np.array([0, 0], dtype=np.int32),
        "normals": np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "areas": np.array([1.0, 1.0], dtype=np.float64),
        "centers": np.array(
            [[0.5, 0.5, 0.0], [2.0, 0.5, 0.0]],
            dtype=np.float64,
        ),
    }


def _arc_band_island_loops():
    """
    弯成 150° 弧的条带被分成两个 island（0°–70° 与 80°–150°），
    内弧半径 4、外弧半径 5，位于 z=0 平面。
    返回 (vertices, loops)。
    """

    def band(theta0: float, theta1: float, steps: int) -> np.ndarray:
        thetas = np.linspace(
            np.radians(theta0),
            np.radians(theta1),
            steps,
        )
        inner = np.column_stack(
            (4.0 * np.cos(thetas), 4.0 * np.sin(thetas), np.zeros(steps))
        )
        outer = np.column_stack(
            (5.0 * np.cos(thetas), 5.0 * np.sin(thetas), np.zeros(steps))
        )
        # 闭环：内弧正向 + 外弧反向
        return np.vstack((inner, outer[::-1]))

    island_a = band(0.0, 70.0, 24)
    island_b = band(80.0, 150.0, 24)
    vertices = np.vstack((island_a, island_b))
    loop_a = list(range(len(island_a)))
    loop_b = list(range(len(island_a), len(island_a) + len(island_b)))
    return vertices, [loop_a, loop_b]


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

    def test_union_of_adjacent_regions_single_loop(self) -> None:
        # 两个相邻领域并集：公共边消去，外轮廓为单一闭环
        mesh = _two_quad_strip()
        region_ids = np.array([0, 1], dtype=np.int32)
        loops = extract_region_boundary_loops(
            region_ids,
            [0, 1],
            mesh["loop_start"],
            mesh["loop_total"],
            mesh["loop_vertex_indices"],
        )
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 6)

    def test_curved_band_islands_envelope_follows_arc(self) -> None:
        # 回归：弯曲条带的两个 island 合并时，包络不得在弧的两条臂
        # 之间跳变（旧的主轴分箱会导致拟合面交叉扭曲）
        vertices, loops = _arc_band_island_loops()
        envelope, band_sides = combine_boundary_islands(loops, vertices)
        self.assertIsNotNone(band_sides)
        self.assertEqual(len(band_sides), 4)
        radii = np.linalg.norm(envelope[:, :2], axis=1)
        self.assertTrue(bool(np.all(radii > 3.9)))
        self.assertTrue(bool(np.all(radii < 5.1)))
        # 相邻包络点步长必须远小于两臂间距（约 8.7），
        # 略大于缺口弧长（约 0.8）即为正常连接
        closed = np.vstack((envelope, envelope[:1]))
        steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        self.assertLess(float(steps.max()), 1.5)

    def test_same_region_islands_combined_to_full_envelope(self) -> None:
        mesh = _disconnected_same_region_strip()
        loops = extract_region_boundary_loops(
            mesh["region_ids"],
            0,
            mesh["loop_start"],
            mesh["loop_total"],
            mesh["loop_vertex_indices"],
        )
        self.assertEqual(len(loops), 2)
        envelope, band_sides = combine_boundary_islands(loops, mesh["vertices"])
        self.assertIsNotNone(band_sides)
        self.assertEqual(len(band_sides), 4)
        self.assertGreaterEqual(len(envelope), 4)
        self.assertAlmostEqual(float(envelope[:, 0].min()), 0.0, places=6)
        self.assertAlmostEqual(float(envelope[:, 0].max()), 2.5, places=6)
        self.assertAlmostEqual(float(envelope[:, 1].min()), 0.0, places=6)
        self.assertAlmostEqual(float(envelope[:, 1].max()), 1.0, places=6)


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

    def test_shared_param_keeps_radial_correspondence(self) -> None:
        # 内外弧弧长不同时，按索引重采样保持同一极角对应，避免扭转
        thetas = np.linspace(0.0, np.pi * 0.6, 40)
        inner = np.column_stack(
            (4.0 * np.cos(thetas), 4.0 * np.sin(thetas), np.zeros(40))
        )
        outer = np.column_stack(
            (5.0 * np.cos(thetas), 5.0 * np.sin(thetas), np.zeros(40))
        )
        end = np.vstack((inner[-1], outer[-1]))
        start = np.vstack((outer[0], inner[0]))
        sides = [inner, end, outer[::-1], start]
        verts, faces = build_quad_patch(
            sides, segments_u=8, segments_v=3, shared_param=True
        )
        grid = verts.reshape(4, 9, 3)
        # 同一 u 列的点应近似共径（与原点夹角接近）
        for i in range(9):
            angles = np.arctan2(grid[:, i, 1], grid[:, i, 0])
            self.assertLess(float(angles.max() - angles.min()), 0.08)
        self.assertEqual(len(faces), 8 * 3)

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

    def test_fit_union_of_regions_as_quad(self) -> None:
        # 一条被切成两个领域的矩形条带，合并拟合成完整四边面
        mesh = _two_quad_strip()
        region_ids = np.array([0, 1], dtype=np.int32)
        result = fit_region_surface(
            region_ids=region_ids,
            target_id=[0, 1],
            vertices=mesh["vertices"],
            loop_start=mesh["loop_start"],
            loop_total=mesh["loop_total"],
            loop_vertex_indices=mesh["loop_vertex_indices"],
            face_normals=mesh["normals"],
            face_areas=mesh["areas"],
            face_centers=mesh["centers"],
            segments_u=4,
            segments_v=2,
        )
        self.assertEqual(result.topology, "QUAD")
        self.assertEqual(len(result.vertices), 5 * 3)
        # 拟合面应覆盖 2x1 全域
        self.assertAlmostEqual(float(result.vertices[:, 0].min()), 0.0, places=6)
        self.assertAlmostEqual(float(result.vertices[:, 0].max()), 2.0, places=6)

    def test_fit_disconnected_islands_of_one_region_as_full_quad(self) -> None:
        mesh = _disconnected_same_region_strip()
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
            segments_u=5,
            segments_v=2,
        )
        self.assertEqual(result.topology, "QUAD")
        self.assertEqual(len(result.vertices), 6 * 3)
        self.assertAlmostEqual(float(result.vertices[:, 0].min()), 0.0, places=6)
        self.assertAlmostEqual(float(result.vertices[:, 0].max()), 2.5, places=6)
        self.assertTrue(
            any("island" in warning for warning in result.warnings)
        )

    def test_polyline_length(self) -> None:
        pts = np.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
            dtype=np.float64,
        )
        self.assertAlmostEqual(polyline_length(pts), 7.0)


if __name__ == "__main__":
    unittest.main()
