"""领域三边/四边曲面拟合算法回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from AdvReverseEngineering.algorithms.region_fit import (
    RegionFitError,
    boundary_loop_neighbor_ids,
    bridge_concave_notches,
    build_edge_face_adjacency,
    build_quad_patch,
    build_triangular_patch,
    classify_side_fit_mode,
    classify_tri_or_quad,
    combine_boundary_islands,
    coons_patch,
    detect_concave_fold_indices,
    detect_corner_indices,
    detect_side_fold_indices,
    extract_island_longest_sides,
    loop_has_internal_corridor,
    merge_nearby_loops_to_outer_contours,
    extract_polyline_keypoints,
    extract_region_boundary_loops,
    filter_handle_outliers,
    filter_significant_boundary_loops,
    fit_cubic_bezier_controls,
    fit_region_surface,
    merge_short_boundary_sides,
    neighbor_change_vertex_indices,
    point_to_polyline_distance,
    polyline_length,
    resample_closed_polyline,
    resample_polyline,
    resplit_sides_at_interior_folds,
    sample_cubic_bezier,
    select_primary_boundary_loop,
    side_interior_max_turn_deg,
    split_polyline_at_significant_folds,
)


def _t_junction_shared_edge_mesh():
    """
    上侧领域 A 与下侧 B|C 共线共享边，中间为 T 接缝：
      0--1--2
      |A |A |
      3--4--5
      |B |C |
      6--7--8
    """
    vertices = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    # A: 0,1,4,3 与 1,2,5,4；B: 3,4,7,6；C: 4,5,8,7
    loop_vertex_indices = np.array(
        [
            0, 1, 4, 3,
            1, 2, 5, 4,
            3, 4, 7, 6,
            4, 5, 8, 7,
        ],
        dtype=np.int32,
    )
    loop_start = np.array([0, 4, 8, 12], dtype=np.int32)
    loop_total = np.array([4, 4, 4, 4], dtype=np.int32)
    region_ids = np.array([0, 0, 1, 2], dtype=np.int32)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1))
    areas = np.ones(4, dtype=np.float64)
    centers = np.array(
        [
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
        ],
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
        envelope, band_sides, _samples = combine_boundary_islands(loops, vertices)
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
        envelope, band_sides, _samples = combine_boundary_islands(
            loops, mesh["vertices"]
        )
        self.assertIsNotNone(band_sides)
        self.assertEqual(len(band_sides), 4)
        self.assertGreaterEqual(len(envelope), 4)
        self.assertAlmostEqual(float(envelope[:, 0].min()), 0.0, places=6)
        self.assertAlmostEqual(float(envelope[:, 0].max()), 2.5, places=6)
        self.assertAlmostEqual(float(envelope[:, 1].min()), 0.0, places=6)
        self.assertAlmostEqual(float(envelope[:, 1].max()), 1.0, places=6)

    def test_interior_points_extend_band_envelope(self) -> None:
        # 边界环高度仅到 y=1，但面心探到 y=1.8：包络应向外延伸盖住角部
        vertices, loops = _arc_band_island_loops()
        # 在弧中段外侧放置“角部”内部点
        mid_angles = np.radians(np.array([30.0, 100.0]))
        interior = np.column_stack(
            (
                5.6 * np.cos(mid_angles),
                5.6 * np.sin(mid_angles),
                np.zeros(len(mid_angles)),
            )
        )
        _env_plain, sides_plain, _ = combine_boundary_islands(loops, vertices)
        _env_ext, sides_ext, _ = combine_boundary_islands(
            loops, vertices, interior_points=interior
        )
        self.assertIsNotNone(sides_plain)
        self.assertIsNotNone(sides_ext)
        r_plain = max(
            float(np.linalg.norm(sides_plain[0][:, :2], axis=1).max()),
            float(np.linalg.norm(sides_plain[2][:, :2], axis=1).max()),
        )
        r_ext = max(
            float(np.linalg.norm(sides_ext[0][:, :2], axis=1).max()),
            float(np.linalg.norm(sides_ext[2][:, :2], axis=1).max()),
        )
        self.assertGreater(r_ext, r_plain + 0.3)


class TopologyClassificationTests(unittest.TestCase):
    def test_filter_significant_boundary_loops_drops_tiny(self) -> None:
        """相对主环过短的碎环应被当作离散极小值过滤。"""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                # 面内极小环
                [1.5, 1.4, 0.0],
                [1.7, 1.4, 0.0],
                [1.7, 1.6, 0.0],
                [1.5, 1.6, 0.0],
            ],
            dtype=np.float64,
        )
        big = [0, 1, 2, 3]
        tiny = [4, 5, 6, 7]
        kept = filter_significant_boundary_loops(
            [big, tiny],
            vertices,
            min_perimeter_frac=0.12,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], big)

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

    def test_multi_stride_ignores_edge_sawtooth(self) -> None:
        """
        边上锯齿只在小尺度显尖角；2/4/8/16 共识应抓住四角，不把边中锯齿当角。
        """
        corners_xy = np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 3.0],
                [0.0, 3.0],
            ],
            dtype=np.float64,
        )
        dense: list[np.ndarray] = []
        for index in range(len(corners_xy)):
            a = corners_xy[index]
            b = corners_xy[(index + 1) % len(corners_xy)]
            for t in np.linspace(0.0, 1.0, 40, endpoint=False):
                dense.append(a * (1.0 - t) + b * t)
        pts = np.asarray(dense, dtype=np.float64)
        # 底边中段加入垂直锯齿（小尺度尖、大尺度仍接近直线）
        mid = 20
        pts[mid] = pts[mid] + np.array([0.0, 0.35])
        pts[mid - 1] = pts[mid - 1] + np.array([0.0, 0.18])
        pts[mid + 1] = pts[mid + 1] + np.array([0.0, 0.18])

        found = detect_corner_indices(
            pts,
            angle_threshold_deg=35.0,
            max_corners=8,
            sample_strides=(2, 4, 8, 16),
        )
        self.assertGreaterEqual(len(found), 4)
        # 不应把锯齿点收成主角：主角应靠近四角坐标
        corner_pts = pts[np.asarray(found[:4], dtype=np.int32)]
        for target in corners_xy:
            dist = np.min(np.linalg.norm(corner_pts - target, axis=1))
            self.assertLess(dist, 0.35)
        # 锯齿点本身不应进入最终角点集
        self.assertNotIn(mid, found)
        self.assertEqual(len(found), 4)

    def test_detect_concave_fold_and_ignore_handle_outlier(self) -> None:
        """凹折角应被检出；偏离边线的手柄视为离散噪声忽略。"""
        # CCW 凹多边形：右下有一个明显内凹
        corners_xy = np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 1.0],
                [2.5, 1.0],
                [2.5, 2.0],
                [4.0, 2.0],
                [4.0, 3.0],
                [0.0, 3.0],
            ],
            dtype=np.float64,
        )
        dense: list[np.ndarray] = []
        for index in range(len(corners_xy)):
            a = corners_xy[index]
            b = corners_xy[(index + 1) % len(corners_xy)]
            for t in np.linspace(0.0, 1.0, 24, endpoint=False):
                dense.append(a * (1.0 - t) + b * t)
        pts = np.asarray(dense, dtype=np.float64)
        folds = detect_concave_fold_indices(
            pts,
            fold_angle_deg=45.0,
            sample_strides=(2, 4, 8, 16),
        )
        self.assertGreaterEqual(len(folds), 1)
        fold_pts = pts[np.asarray(folds, dtype=np.int32)]
        # 至少一个折角靠近 (2.5,1) 或 (2.5,2)
        notch = np.array([[2.5, 1.0], [2.5, 2.0]], dtype=np.float64)
        nearest = min(
            float(np.min(np.linalg.norm(fold_pts - target, axis=1)))
            for target in notch
        )
        self.assertLess(nearest, 0.4)

        # 长边上的中段凹折也应检出（closed=False）
        side = np.array(
            [[0.0, 2.0], [2.0, 2.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [4.0, 2.0], [6.0, 2.0]],
            dtype=np.float64,
        )
        # 加密
        dense_side: list[np.ndarray] = []
        for index in range(len(side) - 1):
            a = side[index]
            b = side[index + 1]
            for t in np.linspace(0.0, 1.0, 16, endpoint=False):
                dense_side.append(a * (1.0 - t) + b * t)
        dense_side.append(side[-1])
        side_pts = np.asarray(dense_side, dtype=np.float64)
        mid = detect_concave_fold_indices(
            side_pts,
            fold_angle_deg=35.0,
            closed=False,
            sample_strides=(2, 4, 8, 16),
        )
        self.assertGreaterEqual(len(mid), 1)

        side3 = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        outlier = np.array([1.0, 0.5, 0.0], dtype=np.float64)
        near = np.array([1.0, 0.01, 0.0], dtype=np.float64)
        kept = filter_handle_outliers([outlier, near], side3, max_distance=0.05)
        self.assertEqual(len(kept), 1)
        np.testing.assert_allclose(kept[0], near, atol=1e-9)
        self.assertGreater(
            point_to_polyline_distance(outlier, side3),
            0.05,
        )

    def test_l_shape_corner_not_merged_into_one_side(self) -> None:
        """
        L 形闭环：尖角过多时不应强行并成带折角的「四边」。
        """
        corners_xy = np.array(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [0.0, 2.0],
            ],
            dtype=np.float64,
        )
        dense: list[np.ndarray] = []
        for index in range(len(corners_xy)):
            a = corners_xy[index]
            b = corners_xy[(index + 1) % len(corners_xy)]
            for t in np.linspace(0.0, 1.0, 20, endpoint=False):
                dense.append(a * (1.0 - t) + b * t)
        pts = np.asarray(dense, dtype=np.float64)

        found = detect_corner_indices(
            pts,
            angle_threshold_deg=35.0,
            max_corners=12,
            include_strong_concave=True,
        )
        self.assertGreaterEqual(len(found), 5)

        from AdvReverseEngineering.algorithms.region_fit import (
            merge_collinear_adjacent_sides,
            reduce_sides_to_count,
            split_loop_into_sides,
        )

        sides = split_loop_into_sides(pts, found)
        sides = merge_collinear_adjacent_sides(sides, max_turn_deg=35.0)
        sides4 = reduce_sides_to_count(
            sides,
            4,
            max_merge_turn_deg=55.0,
        )
        # 六个直角尖角：应拒绝合并，保留多于 4 条边
        self.assertGreater(len(sides4), 4)
        for side in sides4:
            self.assertLess(side_interior_max_turn_deg(side), 45.0)

    def test_merge_collinear_reconnects_split_long_edge(self) -> None:
        """平滑长边被误拆成三段后应接回，避免中间缺口。"""
        from AdvReverseEngineering.algorithms.region_fit import (
            merge_collinear_adjacent_sides,
        )

        # 近似共线的三段 + 两条短折回边，构成扁长闭环
        bottom = np.array(
            [[0.0, 0.0], [1.0, 0.02], [2.0, 0.0], [3.0, 0.02], [4.0, 0.0]],
            dtype=np.float64,
        )
        # 人为拆成三段近共线边
        s0 = bottom[:2]
        s1 = bottom[1:4]
        s2 = bottom[3:]
        right = np.array([[4.0, 0.0], [4.0, 1.0]], dtype=np.float64)
        top = np.array([[4.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        left = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        sides = [s0, s1, s2, right, top, left]
        merged = merge_collinear_adjacent_sides(sides, max_turn_deg=35.0)
        self.assertLessEqual(len(merged), 4)
        # 底边应被接成一条，长度接近 4
        lengths = [polyline_length(side) for side in merged]
        self.assertGreaterEqual(max(lengths), 3.5)
        for side in merged:
            self.assertLess(side_interior_max_turn_deg(side), 40.0)

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

    def test_extract_square_longest_sides(self) -> None:
        mesh = _square_mesh()
        debug = extract_island_longest_sides(
            region_ids=mesh["region_ids"],
            target_id=0,
            vertices=mesh["vertices"],
            loop_start=mesh["loop_start"],
            loop_total=mesh["loop_total"],
            loop_vertex_indices=mesh["loop_vertex_indices"],
        )
        self.assertEqual(debug["island_count"], 1)
        island = debug["islands"][0]
        self.assertEqual(len(island["sides"]), 4)
        lengths = island["lengths"]
        self.assertEqual(len(lengths), 4)
        for length in lengths:
            self.assertAlmostEqual(length, 1.0, places=2)
        self.assertGreaterEqual(len(debug["wire_edges"]), 4)
        self.assertGreaterEqual(len(debug["wire_vertices"]), 8)
        beziers = island["beziers"]
        self.assertEqual(len(beziers), 4)
        # 每段独立随机色（确定性种子），不再红绿对边交替
        color_ids = [int(b["color_id"]) for b in beziers]
        self.assertEqual(color_ids, list(range(4)))
        self.assertEqual(len(debug["segment_colors"]), 4)
        for rgba in debug["segment_colors"]:
            self.assertEqual(len(rgba), 4)
            self.assertTrue(all(0.0 <= float(c) <= 1.0 for c in rgba))
        # 相邻段颜色应可区分（不完全相同）
        self.assertNotEqual(
            debug["segment_colors"][0][:3],
            debug["segment_colors"][1][:3],
        )
        # 闭环：相邻边端点重合
        for index, side in enumerate(island["sides"]):
            nxt = island["sides"][(index + 1) % 4]
            np.testing.assert_allclose(side[-1], nxt[0], atol=1e-9)
        for bezier in beziers:
            # 正方形边接近直线，应按折线拟合
            self.assertEqual(bezier["fit_mode"], "POLYLINE")
            polyline = np.asarray(bezier["polyline"], dtype=np.float64)
            self.assertGreaterEqual(len(polyline), 2)
            self.assertIsNone(bezier["spans"])
        self.assertGreater(debug["bevel_depth"], 0.0)
        self.assertGreaterEqual(len(debug["control_points"]), 8)

    def test_classify_zigzag_polyline_not_curve(self) -> None:
        """明显折角边应折线拟合，缓弧仍可为曲线。"""
        # 锯齿噪声叠加的折线：两处约 90° 折角
        zig = []
        for x in np.linspace(0.0, 1.0, 12):
            zig.append([x, 0.02 * ((int(x * 11) % 2) - 0.5), 0.0])
        for y in np.linspace(0.0, 1.0, 12)[1:]:
            zig.append([1.0 + 0.02 * ((int(y * 11) % 2) - 0.5), y, 0.0])
        for x in np.linspace(1.0, 2.0, 12)[1:]:
            zig.append([x, 1.0 + 0.02 * ((int(x * 11) % 2) - 0.5), 0.0])
        zig_pts = np.asarray(zig, dtype=np.float64)
        mode, keys, spans = classify_side_fit_mode(zig_pts, fold_angle_deg=35.0)
        self.assertEqual(mode, "POLYLINE")
        self.assertIsNone(spans)
        self.assertGreaterEqual(len(keys), 3)
        # 折线关键点应贴近真实折角，不应被平滑掉
        corner_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        corner_b = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        self.assertLess(
            min(float(np.linalg.norm(p - corner_a)) for p in keys),
            0.08,
        )
        self.assertLess(
            min(float(np.linalg.norm(p - corner_b)) for p in keys),
            0.08,
        )

        t = np.linspace(0.0, 0.5 * np.pi, 48)
        arc = np.column_stack(
            (np.cos(t), np.sin(t), np.zeros_like(t))
        )
        arc_mode, _arc_pts, arc_spans = classify_side_fit_mode(
            arc,
            fold_angle_deg=35.0,
        )
        self.assertEqual(arc_mode, "CURVE")
        self.assertIsNotNone(arc_spans)
        self.assertGreaterEqual(len(arc_spans), 1)

    def test_split_polyline_keeps_sawtooth_as_one_segment(self) -> None:
        """锯齿关键点不应拆成多段；只有大折角才断开。"""
        # 近似共线锯齿 + 一个 90° 折角，折后继续竖直锯齿
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.01, 0.0],
                [0.6, -0.01, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.01, 1.3, 0.0],
                [0.99, 1.6, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        segments = split_polyline_at_significant_folds(pts, fold_angle_deg=35.0)
        self.assertEqual(len(segments), 2)
        np.testing.assert_allclose(segments[0][-1], pts[3])
        np.testing.assert_allclose(segments[1][0], pts[3])
        np.testing.assert_allclose(segments[1][-1], pts[-1])

    def test_neighbor_t_junction_splits_shared_edge_on_both_logic(self) -> None:
        """上侧 A 在 B|C 的 T 接缝处也应切开，避免一侧 1 段、对侧 2 段。"""
        mesh = _t_junction_shared_edge_mesh()
        edge_faces = build_edge_face_adjacency(
            mesh["loop_start"],
            mesh["loop_total"],
            mesh["loop_vertex_indices"],
        )
        loops = extract_region_boundary_loops(
            mesh["region_ids"],
            0,
            mesh["loop_start"],
            mesh["loop_total"],
            mesh["loop_vertex_indices"],
        )
        self.assertEqual(len(loops), 1)
        neighbor_ids = boundary_loop_neighbor_ids(
            loops[0],
            mesh["region_ids"],
            edge_faces,
            0,
        )
        changes = neighbor_change_vertex_indices(neighbor_ids)
        # 顶点 4 处邻域从 B 变为 C
        self.assertTrue(any(int(loops[0][i]) == 4 for i in changes))

        debug = extract_island_longest_sides(
            region_ids=mesh["region_ids"],
            target_id=0,
            vertices=mesh["vertices"],
            loop_start=mesh["loop_start"],
            loop_total=mesh["loop_total"],
            loop_vertex_indices=mesh["loop_vertex_indices"],
        )
        island = debug["islands"][0]
        # 共享底边被 T 切开后，边数应多于无邻域时的 4
        self.assertGreaterEqual(len(island["beziers"]), 5)
        # 应存在两段贴近 y=1 的水平边（A-B 与 A-C）
        horizontal = []
        for bezier in island["beziers"]:
            if bezier["fit_mode"] == "POLYLINE":
                pts = np.asarray(bezier["polyline"], dtype=np.float64)
            else:
                pts = np.asarray(island["sides"][bezier["side_index"]], dtype=np.float64)
            mid = 0.5 * (pts[0] + pts[-1])
            if abs(float(mid[1]) - 1.0) < 0.15 and abs(float(pts[0, 1] - pts[-1, 1])) < 0.2:
                horizontal.append(mid[0])
        self.assertGreaterEqual(len(horizontal), 2)

    def test_resplit_mid_kink_and_merge_t_junction_stub(self) -> None:
        """中段折弯应再拆；T 接缝短碎段应并回。"""
        # 长边中段 90° 折弯（模拟左侧漏检后仍整段）
        kinked = np.vstack(
            (
                np.column_stack(
                    (np.linspace(0.0, 1.0, 20), np.zeros(20), np.zeros(20))
                ),
                np.column_stack(
                    (np.ones(19), np.linspace(0.0, 1.0, 20)[1:], np.zeros(19))
                ),
            )
        )
        origin = np.zeros(3)
        axis_u = np.array([1.0, 0.0, 0.0])
        axis_v = np.array([0.0, 1.0, 0.0])
        resplit = resplit_sides_at_interior_folds(
            [kinked],
            origin,
            axis_u,
            axis_v,
            fold_angle_deg=35.0,
        )
        self.assertGreaterEqual(len(resplit), 2)

        # 竖直长边中间夹一段极短水平 stub（T 接缝）
        top = np.column_stack(
            (np.zeros(12), np.linspace(1.0, 0.55, 12), np.zeros(12))
        )
        stub = np.array(
            [[0.0, 0.55, 0.0], [0.04, 0.55, 0.0], [0.0, 0.50, 0.0]],
            dtype=np.float64,
        )
        bottom = np.column_stack(
            (np.zeros(12), np.linspace(0.50, 0.0, 12), np.zeros(12))
        )
        merged = merge_short_boundary_sides(
            [top, stub, bottom],
            min_length=0.2,
            min_sides=2,
        )
        self.assertEqual(len(merged), 2)
        self.assertGreater(polyline_length(merged[0]), 0.2)
        self.assertGreater(polyline_length(merged[1]), 0.2)

    def test_detect_side_fold_indices_keeps_sharp_turns(self) -> None:
        pts = []
        for x in np.linspace(0.0, 1.0, 20):
            pts.append([x, 0.0])
        for y in np.linspace(0.0, 1.0, 20)[1:]:
            pts.append([1.0, y])
        for x in np.linspace(1.0, 2.0, 20)[1:]:
            pts.append([x, 1.0])
        pts_2d = np.asarray(pts, dtype=np.float64)
        folds = detect_side_fold_indices(pts_2d, fold_angle_deg=35.0)
        self.assertGreaterEqual(len(folds), 2)
        keys = extract_polyline_keypoints(
            np.column_stack((pts_2d, np.zeros(len(pts_2d)))),
            folds,
        )
        self.assertGreaterEqual(len(keys), 4)

    def test_fit_cubic_bezier_straight_line(self) -> None:
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        controls = fit_cubic_bezier_controls(pts)
        sampled = sample_cubic_bezier(controls, 10)
        self.assertTrue(np.allclose(sampled[:, 1], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(sampled[:, 2], 0.0, atol=1e-6))
        self.assertAlmostEqual(float(sampled[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(sampled[-1, 0]), 3.0, places=6)

    def test_extract_disconnected_islands_longest_sides(self) -> None:
        """同一领域多岛调试边应融成一条外轮廓，不再各画一套内边。"""
        mesh = _disconnected_same_region_strip()
        debug = extract_island_longest_sides(
            region_ids=mesh["region_ids"],
            target_id=0,
            vertices=mesh["vertices"],
            loop_start=mesh["loop_start"],
            loop_total=mesh["loop_total"],
            loop_vertex_indices=mesh["loop_vertex_indices"],
        )
        self.assertEqual(debug["island_count"], 1)
        self.assertEqual(int(debug["islands"][0]["merged_from"]), 2)
        island = debug["islands"][0]
        self.assertGreaterEqual(len(island["sides"]), 2)
        color_ids = [int(b["color_id"]) for b in island["beziers"]]
        self.assertEqual(len(color_ids), len(set(color_ids)))
        # 外轮廓应盖住两岛整体包围盒，中缝不应再出现贯穿内边
        all_pts = np.vstack(island["sides"])
        self.assertAlmostEqual(float(all_pts[:, 0].min()), 0.0, places=5)
        self.assertAlmostEqual(float(all_pts[:, 0].max()), 2.5, places=5)
        mid_interior = all_pts[
            (all_pts[:, 0] > 0.95)
            & (all_pts[:, 0] < 1.55)
            & (all_pts[:, 1] > 0.1)
            & (all_pts[:, 1] < 0.9)
        ]
        self.assertEqual(len(mid_interior), 0)

    def test_nearby_islands_merge_to_outer_contour(self) -> None:
        """间隙很小的两岛融并为外轮廓，岛间内边被消去。"""
        # 两矩形间距 0.05；extent≈2.7 → gap 阈值约 0.13，应融并
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.05, 0.0, 0.0],
                [2.05, 0.0, 0.0],
                [2.05, 1.0, 0.0],
                [1.05, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        loops = [[0, 1, 2, 3], [4, 5, 6, 7]]
        items = merge_nearby_loops_to_outer_contours(
            loops,
            vertices,
            max_gap=0.2,
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(int(items[0]["merged_from"]), 2)
        self.assertIsNone(items[0]["loop_ids"])
        envelope = items[0]["points"]
        self.assertAlmostEqual(float(envelope[:, 0].min()), 0.0, places=5)
        self.assertAlmostEqual(float(envelope[:, 0].max()), 2.05, places=5)
        self.assertAlmostEqual(float(envelope[:, 1].min()), 0.0, places=5)
        self.assertAlmostEqual(float(envelope[:, 1].max()), 1.0, places=5)
        # 岛间竖缝处不应留下贯穿高度的内边点（只允许上下外轮廓）
        mid_interior = envelope[
            (envelope[:, 0] > 0.98)
            & (envelope[:, 0] < 1.07)
            & (envelope[:, 1] > 0.05)
            & (envelope[:, 1] < 0.95)
        ]
        self.assertEqual(len(mid_interior), 0)

    def test_far_islands_stay_separate_after_merge_pass(self) -> None:
        """间距较大的两岛在按间隙聚类时保持各自独立。"""
        mesh = _disconnected_same_region_strip()
        loops = extract_region_boundary_loops(
            mesh["region_ids"],
            0,
            mesh["loop_start"],
            mesh["loop_total"],
            mesh["loop_vertex_indices"],
        )
        items = merge_nearby_loops_to_outer_contours(
            loops,
            mesh["vertices"],
            max_gap=0.15,
        )
        self.assertEqual(len(items), 2)
        self.assertTrue(all(item["merged_from"] == 1 for item in items))
        self.assertTrue(all(item["loop_ids"] is not None for item in items))

    def test_prefer_outer_envelope_keeps_full_bbox(self) -> None:
        """prefer_outer_envelope 始终保留覆盖全部 island 的外包络。"""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.05, 0.2, 0.0],
                [2.4, 0.2, 0.0],
                [2.4, 0.8, 0.0],
                [2.05, 0.8, 0.0],
            ],
            dtype=np.float64,
        )
        loops = [[0, 1, 2, 3], [4, 5, 6, 7]]
        envelope, sides, _band = combine_boundary_islands(
            loops,
            vertices,
            prefer_outer_envelope=True,
        )
        self.assertGreaterEqual(float(envelope[:, 0].max()), 2.35)
        self.assertAlmostEqual(float(envelope[:, 0].min()), 0.0, places=5)
        self.assertIsNotNone(sides)

    def test_notched_loop_corridor_uses_outer_envelope(self) -> None:
        """单环深凹口应识别为内缝，外包络后凹口内边消失。"""
        # U 形：顶部中间开口向下的深凹口
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 0.15, 0.0],
                [1.0, 0.15, 0.0],
                [1.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        loop = [0, 1, 2, 3, 4, 5, 6, 7]
        self.assertTrue(loop_has_internal_corridor(vertices[loop]))
        items = merge_nearby_loops_to_outer_contours(
            [loop],
            vertices,
            max_gap=0.1,
        )
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0]["loop_ids"])
        envelope = items[0]["points"]
        notch_interior = envelope[
            (envelope[:, 0] > 1.05)
            & (envelope[:, 0] < 1.95)
            & (envelope[:, 1] > 0.2)
            & (envelope[:, 1] < 1.8)
        ]
        self.assertEqual(len(notch_interior), 0)


if __name__ == "__main__":
    unittest.main()
