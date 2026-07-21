# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""领域外轮廓拟合操作符：生成可编辑曲线。"""

from __future__ import annotations

import bpy
import numpy as np

from ..algorithms.region_fit import (
    DEFAULT_SEG_U,
    DEFAULT_SEG_V,
    RegionFitError,
    extract_island_longest_sides,
)
from ..algorithms.regions import compute_region_label_anchors
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import (
    LABEL_RADIUS_PX,
    get_merge_label_session,
    register_label_draw_handler,
    set_fit_angle_label_session,
    set_merge_label_session,
    update_merge_label_projections,
)
from ..utils.mesh import extract_face_topology, extract_mesh_data
from ..utils.viewport import hit_test_labels
from .regions import REGION_ID_ATTR, read_region_ids


FIT_COLLECTION_NAME = "拟合面"
FIT_PREVIEW_NAME = "ARE_FitPreview"
FIT_DEBUG_CTRL_NAME = "ARE_FitDebugControls"
FIT_DEBUG_FOLD_NAME = "ARE_FitConcaveFolds"
FIT_BRIDGE_LINK_NAME = "ARE_FitBridgeLinks"
FIT_EDGE_MAT_RED = "ARE_FitEdge_Red"
FIT_EDGE_MAT_GREEN = "ARE_FitEdge_Green"
FIT_EDGE_MAT_SEG_PREFIX = "ARE_FitEdge_Seg_"
FIT_FOLD_MAT = "ARE_FitFold_Marker"
FIT_BRIDGE_MAT = "ARE_FitBridge_Link"
FIT_EDGE_COLOR_RED = (1.0, 0.12, 0.08, 1.0)
FIT_EDGE_COLOR_GREEN = (0.1, 0.95, 0.22, 1.0)
FIT_EDGE_COLOR_FALLBACK = (
    FIT_EDGE_COLOR_RED,
    FIT_EDGE_COLOR_GREEN,
)
FIT_FOLD_COLOR = (0.15, 0.85, 1.0, 1.0)
FIT_BRIDGE_COLOR = (0.55, 0.55, 0.55, 1.0)
FIT_CURVE_STAGES = ("ISLANDS", "STITCH", "BRIDGE")
FIT_STAGE_ORDER = ("ISLANDS", "STITCH", "BRIDGE", "PREVIEW")
FIT_CONTOUR_KIND = "contour"
MODAL_TIMER_STEP = 0.1
_FIT_OP_KEY = "are_active_fit_op"


def _tag_redraw(context: bpy.types.Context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "UI"}:
                area.tag_redraw()


def _add_modal_timer(operator, context: bpy.types.Context) -> None:
    wm = context.window_manager
    operator._timer = wm.event_timer_add(
        MODAL_TIMER_STEP,
        window=context.window,
    )


def _remove_modal_timer(operator, context: bpy.types.Context) -> None:
    timer = getattr(operator, "_timer", None)
    if timer is None:
        return
    try:
        context.window_manager.event_timer_remove(timer)
    except Exception:
        pass
    operator._timer = None


def _set_active_fit_op(operator) -> None:
    bpy.app.driver_namespace[_FIT_OP_KEY] = operator


def _get_active_fit_op():
    return bpy.app.driver_namespace.get(_FIT_OP_KEY)


def _clear_active_fit_op(operator) -> None:
    if bpy.app.driver_namespace.get(_FIT_OP_KEY) is operator:
        bpy.app.driver_namespace[_FIT_OP_KEY] = None


def _schedule_force_exit_check() -> None:
    def _callback():
        scene = getattr(bpy.context, "scene", None)
        scene_props = getattr(scene, SCENE_PROP_NAME, None) if scene else None
        if scene_props is None:
            return None
        stuck = (
            scene_props.fit_mode_active
            and scene_props.fit_confirm_requested
        )
        if stuck:
            scene_props.fit_confirm_requested = False
            scene_props.fit_advance_requested = False
            scene_props.fit_retreat_requested = False
            scene_props.fit_stage_rebuild_requested = False
            scene_props.fit_preview_requested = False
            scene_props.fit_mode_active = False
            scene_props.fit_target_id = -1
            scene_props.fit_hover_id = -1
            scene_props.fit_phase = "IDLE"
            scene_props.fit_status = "拟合模态已失联，已强制退出"
            from .regions import _teardown_labels_then_sync
            from ..ui.overlay import set_fit_angle_label_session

            set_fit_angle_label_session(None)
            _teardown_labels_then_sync(bpy.context)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return None

    bpy.app.timers.register(_callback, first_interval=0.6)


def _bbox_lift(mesh_data) -> float:
    vertices = mesh_data["vertices"]
    if len(vertices) == 0:
        return 0.01
    size = vertices.max(axis=0) - vertices.min(axis=0)
    return float(max(np.linalg.norm(size) * 0.015, 1e-4))


def _build_label_session(
    region_ids: np.ndarray,
    mesh_data,
    colors: np.ndarray | None = None,
) -> dict:
    anchors = compute_region_label_anchors(
        region_ids,
        mesh_data["face_centers"],
        mesh_data["normals"],
        mesh_data["areas"],
    )
    labels = []
    for region_id in sorted(anchors.keys()):
        anchor = anchors[region_id]
        labels.append(
            {
                "id": int(region_id),
                "world_co": np.asarray(anchor["world_co"], dtype=np.float64),
                "face_center": np.asarray(
                    anchor["face_center"],
                    dtype=np.float64,
                ),
                "normal": np.asarray(anchor["normal"], dtype=np.float64),
                "screen_xy": None,
                "face_screen_xy": None,
                "visible": False,
            }
        )
    return {
        "labels": labels,
        "lift": _bbox_lift(mesh_data),
        "object_name": None,
        "region_ids": np.asarray(region_ids, dtype=np.int32).copy(),
        "colors": (
            None
            if colors is None
            else np.asarray(colors, dtype=np.float32).copy()
        ),
        "mesh_data": mesh_data,
    }


def _ensure_fit_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    collection = bpy.data.collections.get(FIT_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(FIT_COLLECTION_NAME)
        scene.collection.children.link(collection)
    return collection


def _delete_object(obj: bpy.types.Object | None) -> None:
    if obj is None:
        return
    try:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            data_type = data.bl_rna.identifier
            if data_type == "Mesh":
                bpy.data.meshes.remove(data)
            elif data_type == "Curve":
                bpy.data.curves.remove(data)
    except ReferenceError:
        pass


def _world_to_object_local(
    matrix_world: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    inv = np.linalg.inv(matrix_world)
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack((_as_float(points), ones))
    return (inv @ homogeneous.T).T[:, :3]


def _as_float(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _ensure_fit_edge_material(name: str, color: tuple[float, ...]) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    # 视口加亮，避免被扫描面淹没
    try:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = color
        emission.inputs["Strength"].default_value = 1.2
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
    except Exception:
        pass
    if hasattr(mat, "roughness"):
        mat.roughness = 0.35
    return mat


def _resolve_segment_colors(debug: dict) -> list[tuple[float, float, float, float]]:
    """从 debug 取每段 RGBA；缺省时回退红/绿交替。"""
    raw = list(debug.get("segment_colors") or [])
    colors: list[tuple[float, float, float, float]] = []
    for item in raw:
        rgba = tuple(float(v) for v in item[:4])
        if len(rgba) < 4:
            rgba = rgba + (1.0,) * (4 - len(rgba))
        colors.append(rgba)  # type: ignore[arg-type]
    if colors:
        return colors
    return list(FIT_EDGE_COLOR_FALLBACK)


def _apply_segment_materials(
    data_block,
    colors: list[tuple[float, float, float, float]],
) -> None:
    """给 Curve/Mesh 挂上每段自定义材质。"""
    data_block.materials.clear()
    for index, color in enumerate(colors):
        mat = _ensure_fit_edge_material(
            f"{FIT_EDGE_MAT_SEG_PREFIX}{index}",
            color,
        )
        data_block.materials.append(mat)
    if not colors:
        data_block.materials.append(
            _ensure_fit_edge_material(FIT_EDGE_MAT_RED, FIT_EDGE_COLOR_RED)
        )
        data_block.materials.append(
            _ensure_fit_edge_material(FIT_EDGE_MAT_GREEN, FIT_EDGE_COLOR_GREEN)
        )


def _link_object_to_scene(
    obj: bpy.types.Object,
    collection: bpy.types.Collection | None,
) -> None:
    if collection is not None:
        if obj.name not in collection.objects:
            collection.objects.link(obj)
        return
    scene_col = bpy.context.scene.collection
    if obj.name not in scene_col.objects:
        scene_col.objects.link(obj)


def _create_or_update_mesh_object(
    name: str,
    vertices_world: np.ndarray,
    faces: list[tuple[int, ...]],
    matrix_world,
    collection: bpy.types.Collection | None,
    existing: bpy.types.Object | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> bpy.types.Object:
    matrix = np.asarray(matrix_world, dtype=np.float64)
    local = _world_to_object_local(matrix, vertices_world)
    edge_list = [] if edges is None else [tuple(e) for e in edges]
    face_list = [tuple(f) for f in faces]
    if (
        existing is not None
        and existing.name in bpy.data.objects
        and existing.type == "MESH"
    ):
        mesh = existing.data
        mesh.clear_geometry()
        mesh.from_pydata(
            [tuple(v) for v in local.tolist()],
            edge_list,
            face_list,
        )
        mesh.update()
        existing.matrix_world = matrix_world.copy()
        return existing

    if existing is not None:
        _delete_object(existing)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [tuple(v) for v in local.tolist()],
        edge_list,
        face_list,
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = matrix_world.copy()
    _link_object_to_scene(obj, collection)
    return obj


def _octahedron_mesh(
    centers: np.ndarray,
    radius: float,
    color_ids: list[int],
) -> tuple[np.ndarray, list[tuple[int, ...]], list[int]]:
    """为每个控制点生成小八面体，便于加粗点显示。"""
    if len(centers) == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            [],
            [],
        )
    offsets = np.array(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float64,
    )
    local_faces = (
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    )
    verts: list[np.ndarray] = []
    faces: list[tuple[int, ...]] = []
    face_colors: list[int] = []
    for index, center in enumerate(centers):
        base = len(verts)
        for offset in offsets:
            verts.append(center + offset)
        color = int(color_ids[index]) if index < len(color_ids) else 0
        for face in local_faces:
            faces.append(tuple(base + i for i in face))
            face_colors.append(color)
    return np.asarray(verts, dtype=np.float64), faces, face_colors


def _join_island_closed_loop(island: dict) -> np.ndarray | None:
    """把岛上各拟合边拼成闭合折线（去掉相邻边重复端点）。"""
    parts: list[np.ndarray] = []
    for side in island.get("sides") or []:
        arr = np.asarray(side, dtype=np.float64)
        if len(arr) >= 2:
            parts.append(arr[:-1])
    if not parts:
        for bezier in island.get("beziers") or []:
            poly = bezier.get("polyline")
            if poly is None:
                continue
            arr = np.asarray(poly, dtype=np.float64)
            if len(arr) >= 2:
                parts.append(arr[:-1])
    if not parts:
        return None
    return np.vstack(parts)


def _create_editable_contour_curve(
    name: str,
    debug: dict,
    matrix_world,
    collection: bpy.types.Collection | None,
) -> bpy.types.Object:
    """以外轮廓闭合折线创建可编辑曲线（每岛一条 cyclic POLY）。"""
    matrix = np.asarray(matrix_world, dtype=np.float64)
    curve = bpy.data.curves.new(name, type="CURVE")
    obj = bpy.data.objects.new(name, curve)
    _link_object_to_scene(obj, collection)

    curve.dimensions = "3D"
    curve.bevel_depth = float(max(float(debug.get("bevel_depth", 0.002)) * 0.5, 1e-4))
    curve.bevel_resolution = 2
    curve.use_fill_caps = True

    for island in debug.get("islands", []):
        loop = _join_island_closed_loop(island)
        if loop is None or len(loop) < 3:
            continue
        local_pts = _world_to_object_local(matrix, loop)
        spline = curve.splines.new("POLY")
        spline.points.add(len(local_pts) - 1)
        for index, point in enumerate(local_pts):
            spline.points[index].co = (
                float(point[0]),
                float(point[1]),
                float(point[2]),
                1.0,
            )
        spline.use_cyclic_u = True
        spline.use_smooth = False

    if not curve.splines:
        _delete_object(obj)
        raise RegionFitError("外轮廓拟合结果为空，无法创建曲线")

    obj.matrix_world = matrix_world.copy()
    obj.hide_select = False
    obj.display_type = "WIRE"
    obj.show_in_front = True
    return obj


def _create_or_update_debug_curve_object(
    name: str,
    debug: dict,
    matrix_world,
    collection: bpy.types.Collection | None,
    existing: bpy.types.Object | None = None,
) -> bpy.types.Object:
    """用加粗曲线显示拟合边：折线边用 POLY，缓弧边用贝塞尔；每段自定义色。"""
    matrix = np.asarray(matrix_world, dtype=np.float64)
    colors = _resolve_segment_colors(debug)

    if (
        existing is not None
        and existing.name in bpy.data.objects
        and existing.type == "CURVE"
    ):
        obj = existing
        curve = obj.data
        curve.splines.clear()
    else:
        if existing is not None:
            _delete_object(existing)
        curve = bpy.data.curves.new(name, type="CURVE")
        obj = bpy.data.objects.new(name, curve)
        _link_object_to_scene(obj, collection)

    curve.dimensions = "3D"
    curve.bevel_depth = float(debug.get("bevel_depth", 0.002))
    curve.bevel_resolution = 3
    curve.use_fill_caps = True
    _apply_segment_materials(curve, colors)
    color_mod = max(len(curve.materials), 1)

    for island in debug.get("islands", []):
        for bezier in island.get("beziers", []):
            fit_mode = str(bezier.get("fit_mode", "CURVE")).upper()
            color_id = int(bezier.get("color_id", 0)) % color_mod

            if fit_mode == "POLYLINE":
                polyline = bezier.get("polyline")
                if polyline is None:
                    continue
                local_pts = _world_to_object_local(
                    matrix,
                    np.asarray(polyline, dtype=np.float64),
                )
                if len(local_pts) < 2:
                    continue
                spline = curve.splines.new("POLY")
                spline.points.add(len(local_pts) - 1)
                for index, point in enumerate(local_pts):
                    spline.points[index].co = (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        1.0,
                    )
                spline.material_index = color_id
                spline.use_smooth = False
                continue

            spans = bezier.get("spans")
            if not spans:
                controls = bezier.get("controls")
                if controls is None:
                    continue
                spans = [controls]
            local_spans = [
                _world_to_object_local(
                    matrix,
                    np.asarray(span, dtype=np.float64),
                )
                for span in spans
            ]
            # 一条边 = 一条连续多点贝塞尔，避免分段样条中间断口
            spline = curve.splines.new("BEZIER")
            point_count = len(local_spans) + 1
            if point_count > 1:
                spline.bezier_points.add(point_count - 1)
            for span_index, span in enumerate(local_spans):
                p0, p1, p2, p3 = span
                bp = spline.bezier_points[span_index]
                bp.co = p0
                bp.handle_left_type = "FREE"
                bp.handle_right_type = "FREE"
                if span_index == 0:
                    bp.handle_left = p0 - (p1 - p0)
                else:
                    prev = local_spans[span_index - 1]
                    bp.handle_left = prev[2]
                bp.handle_right = p1
                if span_index == len(local_spans) - 1:
                    bp_end = spline.bezier_points[span_index + 1]
                    bp_end.co = p3
                    bp_end.handle_left_type = "FREE"
                    bp_end.handle_right_type = "FREE"
                    bp_end.handle_left = p2
                    bp_end.handle_right = p3 + (p3 - p2)
            spline.material_index = color_id
            spline.use_smooth = True

    obj.matrix_world = matrix_world.copy()
    obj.hide_select = True
    obj.display_type = "TEXTURED"
    obj.show_in_front = True
    return obj


def _create_or_update_control_point_object(
    name: str,
    debug: dict,
    matrix_world,
    collection: bpy.types.Collection | None,
    existing: bpy.types.Object | None = None,
) -> bpy.types.Object | None:
    """显示拟合控制点（与所属段同色）。"""
    centers = np.asarray(debug.get("control_points", []), dtype=np.float64)
    if len(centers) == 0:
        if existing is not None:
            _delete_object(existing)
        return None
    color_ids = list(debug.get("control_color_ids", []))
    radius = float(debug.get("control_radius", 0.003))
    verts, faces, face_colors = _octahedron_mesh(centers, radius, color_ids)
    obj = _create_or_update_mesh_object(
        name,
        verts,
        faces,
        matrix_world,
        collection,
        existing=existing,
    )
    colors = _resolve_segment_colors(debug)
    mesh = obj.data
    _apply_segment_materials(mesh, colors)
    color_mod = max(len(mesh.materials), 1)
    if len(mesh.polygons) == len(face_colors):
        for poly, color_id in zip(mesh.polygons, face_colors):
            poly.material_index = int(color_id) % color_mod
    obj.hide_select = True
    obj.display_type = "TEXTURED"
    obj.show_in_front = True
    return obj


def _uv_sphere_mesh(
    centers: np.ndarray,
    radius: float,
    segments: int = 12,
    rings: int = 8,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """为每个中心生成低面数 UV 圆球。"""
    if len(centers) == 0:
        return np.zeros((0, 3), dtype=np.float64), []
    seg = max(int(segments), 3)
    ring = max(int(rings), 2)
    unit_verts: list[np.ndarray] = []
    unit_faces: list[tuple[int, ...]] = []
    unit_verts.append(np.array([0.0, 0.0, radius], dtype=np.float64))
    for ring_index in range(1, ring):
        polar = np.pi * ring_index / ring
        z = radius * np.cos(polar)
        xy = radius * np.sin(polar)
        for seg_index in range(seg):
            azimuth = 2.0 * np.pi * seg_index / seg
            unit_verts.append(
                np.array(
                    [xy * np.cos(azimuth), xy * np.sin(azimuth), z],
                    dtype=np.float64,
                )
            )
    unit_verts.append(np.array([0.0, 0.0, -radius], dtype=np.float64))
    north = 0
    south = len(unit_verts) - 1
    for seg_index in range(seg):
        a = 1 + seg_index
        b = 1 + (seg_index + 1) % seg
        unit_faces.append((north, a, b))
    for ring_index in range(ring - 2):
        row0 = 1 + ring_index * seg
        row1 = 1 + (ring_index + 1) * seg
        for seg_index in range(seg):
            a = row0 + seg_index
            b = row0 + (seg_index + 1) % seg
            c = row1 + (seg_index + 1) % seg
            d = row1 + seg_index
            unit_faces.append((a, b, c, d))
    last_row = 1 + (ring - 2) * seg
    for seg_index in range(seg):
        a = last_row + seg_index
        b = last_row + (seg_index + 1) % seg
        unit_faces.append((a, south, b))

    verts: list[np.ndarray] = []
    faces: list[tuple[int, ...]] = []
    for center in centers:
        base = len(verts)
        for local in unit_verts:
            verts.append(center + local)
        for face in unit_faces:
            faces.append(tuple(base + i for i in face))
    return np.asarray(verts, dtype=np.float64), faces


def _create_or_update_concave_fold_object(
    name: str,
    debug: dict,
    matrix_world,
    collection: bpy.types.Collection | None,
    existing: bpy.types.Object | None = None,
) -> bpy.types.Object | None:
    """在明显凹折角处放置特殊青色圆球标记。"""
    centers = np.asarray(debug.get("concave_fold_points", []), dtype=np.float64)
    if len(centers) == 0:
        if existing is not None:
            _delete_object(existing)
        return None
    radius = float(debug.get("fold_radius", 0.006))
    verts, faces = _uv_sphere_mesh(centers, radius)
    obj = _create_or_update_mesh_object(
        name,
        verts,
        faces,
        matrix_world,
        collection,
        existing=existing,
    )
    mat = _ensure_fit_edge_material(FIT_FOLD_MAT, FIT_FOLD_COLOR)
    mesh = obj.data
    mesh.materials.clear()
    mesh.materials.append(mat)
    obj.hide_select = True
    obj.display_type = "TEXTURED"
    obj.show_in_front = True
    return obj


def _create_or_update_bridge_link_object(
    name: str,
    debug: dict,
    matrix_world,
    existing: bpy.types.Object | None = None,
) -> bpy.types.Object | None:
    """远岛桥接示意：质心之间的灰色折线。"""
    if existing is not None:
        _delete_object(existing)
    links = debug.get("bridge_links") or []
    if not links:
        return None
    verts: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []
    for link in links:
        a = np.asarray(link[0], dtype=np.float64).reshape(3)
        b = np.asarray(link[1], dtype=np.float64).reshape(3)
        base = len(verts)
        verts.append((float(a[0]), float(a[1]), float(a[2])))
        verts.append((float(b[0]), float(b[1]), float(b[2])))
        edges.append((base, base + 1))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = matrix_world.copy()
    obj.hide_select = True
    obj.show_in_front = True
    obj.display_type = "WIRE"
    mat = _ensure_fit_edge_material(FIT_BRIDGE_MAT, FIT_BRIDGE_COLOR)
    mesh.materials.clear()
    mesh.materials.append(mat)
    return obj


class ARE_OT_fit_step_next(bpy.types.Operator):
    """已取消：拟合仅生成外轮廓曲线。"""

    bl_idname = "are.fit_step_next"
    bl_label = "下一步"
    bl_description = "已取消后续拟合步骤；请点击领域编号生成外轮廓曲线"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return False

    def execute(self, context: bpy.types.Context):
        self.report({"WARNING"}, "后续拟合步骤已取消，请点击领域编号生成外轮廓")
        return {"CANCELLED"}


class ARE_OT_fit_step_back(bpy.types.Operator):
    """已取消：拟合仅生成外轮廓曲线。"""

    bl_idname = "are.fit_step_back"
    bl_label = "上一步"
    bl_description = "已取消后续拟合步骤"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return False

    def execute(self, context: bpy.types.Context):
        self.report({"WARNING"}, "后续拟合步骤已取消")
        return {"CANCELLED"}


class ARE_OT_build_fit_surface(bpy.types.Operator):
    """已取消：不再拟合成面。"""

    bl_idname = "are.build_fit_surface"
    bl_label = "拟合成面"
    bl_description = "已取消成面步骤；拟合仅输出可编辑外轮廓曲线"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return False

    def execute(self, context: bpy.types.Context):
        self.report({"WARNING"}, "成面步骤已取消，拟合仅输出外轮廓曲线")
        return {"CANCELLED"}


class ARE_OT_confirm_fit_region(bpy.types.Operator):
    """已取消：点击领域后自动写入外轮廓曲线。"""

    bl_idname = "are.confirm_fit_region"
    bl_label = "确认拟合"
    bl_description = "已取消；点击领域编号即可生成外轮廓曲线"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return False

    def execute(self, context: bpy.types.Context):
        self.report({"WARNING"}, "无需确认：点击领域编号即生成外轮廓曲线")
        return {"CANCELLED"}


class ARE_OT_fit_region(bpy.types.Operator):
    """
    模态拟合领域：点选编号后生成可编辑外轮廓曲线。

    不再自动焊接、不再显示内角，也不再进入缝合/桥接/成面。
    """

    bl_idname = "are.fit_region"
    bl_label = "拟合领域"
    bl_description = (
        "点击领域编号，生成可编辑外轮廓曲线到「拟合面」集合；"
        "Shift+多选后按 Enter 确认；曲线带 are_fit_region_id 属性"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _cleanup_ui(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        _remove_modal_timer(self, context)
        _clear_active_fit_op(self)
        scene_props.fit_mode_active = False
        scene_props.fit_confirm_requested = False
        scene_props.fit_advance_requested = False
        scene_props.fit_retreat_requested = False
        scene_props.fit_stage_rebuild_requested = False
        scene_props.fit_preview_requested = False
        scene_props.fit_target_id = -1
        scene_props.fit_hover_id = -1
        scene_props.fit_phase = "IDLE"
        self._fit_target_ids = []
        from .regions import _teardown_labels_then_sync

        set_fit_angle_label_session(None)
        _teardown_labels_then_sync(context)
        _tag_redraw(context)

    def _discard_preview(self) -> None:
        preview = getattr(self, "_preview_object", None)
        controls = getattr(self, "_debug_control_object", None)
        folds = getattr(self, "_debug_fold_object", None)
        bridges = getattr(self, "_bridge_link_object", None)
        _delete_object(preview)
        _delete_object(controls)
        _delete_object(folds)
        _delete_object(bridges)
        self._preview_object = None
        self._debug_control_object = None
        self._debug_fold_object = None
        self._bridge_link_object = None
        set_fit_angle_label_session(None)

    def confirm_from_panel(self, context: bpy.types.Context) -> None:
        if getattr(self, "_closed", False):
            return
        try:
            if not self._committed:
                self._fit_and_commit_contour(context)
        finally:
            if not getattr(self, "_closed", False):
                self._finish_mode(context, cancelled=not self._committed)
            self._closed = True

    def build_surface_from_panel(self, context: bpy.types.Context) -> bool:
        """兼容旧面板：改为直接生成外轮廓曲线。"""
        return self._fit_and_commit_contour(context)

    def _stage_kwargs(self, scene_props) -> dict:
        return {
            "stitch_gap_frac": float(scene_props.fit_stitch_gap) / 100.0,
            "bridge_gap_frac": float(scene_props.fit_bridge_gap) / 100.0,
            "bridge_enabled": bool(scene_props.fit_bridge_enabled),
            "min_perimeter_frac": float(scene_props.fit_island_min_perimeter)
            / 100.0,
        }

    def _status_text(self, scene_props) -> str:
        targets = self._resolve_fit_targets(scene_props)
        if not targets:
            return (
                "点击领域编号生成可编辑外轮廓曲线；"
                "Shift+点击多选后按 Enter 确认"
            )
        label = "+".join(str(value) for value in targets)
        return f"已选领域 {label} · Enter 生成外轮廓曲线 · Esc 取消"

    def _resolve_fit_targets(self, scene_props) -> list[int]:
        """当前拟合目标：支持 Shift 多选并集。"""
        ids = [
            int(value)
            for value in getattr(self, "_fit_target_ids", []) or []
            if int(value) >= 0
        ]
        if ids:
            return sorted(set(ids))
        target = int(scene_props.fit_target_id)
        return [target] if target >= 0 else []

    def _fit_and_commit_contour(self, context: bpy.types.Context) -> bool:
        """提取外轮廓（不焊接、不内角），写入可编辑曲线并结束模态。"""
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        targets = self._resolve_fit_targets(scene_props)
        if not targets:
            self.report({"WARNING"}, "请先点击领域编号")
            return False
        kwargs = self._stage_kwargs(scene_props)
        try:
            debug = extract_island_longest_sides(
                region_ids=self._region_ids,
                target_id=targets if len(targets) > 1 else targets[0],
                vertices=self._mesh_data["vertices"],
                loop_start=self._topology_data["loop_start"],
                loop_total=self._topology_data["loop_total"],
                loop_vertex_indices=self._topology_data["loop_vertex_indices"],
                stage="ISLANDS",
                skip_weld=True,
                emit_angle_labels=False,
                **kwargs,
            )
        except RegionFitError as exc:
            set_fit_angle_label_session(None)
            scene_props.fit_phase = "SELECT"
            scene_props.fit_topology = ""
            scene_props.fit_status = str(exc)
            scene_props.fit_status_detail = ""
            self.report({"ERROR"}, str(exc))
            _tag_redraw(context)
            return False

        self._discard_preview()
        set_fit_angle_label_session(None)

        collection = _ensure_fit_collection(context.scene)
        label = "+".join(str(value) for value in targets)
        safe_label = label.replace("+", "_")
        final_name = f"FitContour_R{safe_label}"
        try:
            curve_obj = _create_editable_contour_curve(
                final_name,
                debug,
                self._object.matrix_world,
                collection,
            )
        except RegionFitError as exc:
            scene_props.fit_status = str(exc)
            self.report({"ERROR"}, str(exc))
            _tag_redraw(context)
            return False

        if curve_obj.name != final_name:
            curve_obj.name = final_name
        if curve_obj.data is not None and curve_obj.data.name != final_name:
            curve_obj.data.name = final_name

        curve_obj["are_fit_source"] = self._object.name
        curve_obj["are_fit_region_id"] = int(targets[0])
        curve_obj["are_fit_kind"] = FIT_CONTOUR_KIND
        if len(targets) > 1:
            curve_obj["are_fit_region_ids"] = ",".join(
                str(value) for value in targets
            )

        self._preview_object = None
        self._committed = True
        self._debug_result = debug
        self._debug_island_count = int(debug.get("island_count", 0))
        scene_props.fit_status = (
            f"已生成领域 {label} 外轮廓曲线 → {curve_obj.name}"
        )
        scene_props.fit_status_detail = (
            f"{self._debug_island_count} 条闭合轮廓 · 可直接编辑曲线"
        )
        self.report({"INFO"}, scene_props.fit_status)
        self._finish_mode(context, cancelled=False)
        self._closed = True
        return True

    def advance_stage(self, context: bpy.types.Context) -> bool:
        return self._fit_and_commit_contour(context)

    def retreat_stage(self, context: bpy.types.Context) -> bool:
        return False

    def advance_to_preview(self, context: bpy.types.Context) -> bool:
        return self._fit_and_commit_contour(context)

    def _finish_mode(self, context: bpy.types.Context, cancelled: bool) -> set:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if cancelled and not self._committed:
            self._discard_preview()
            scene_props.fit_status = "已取消拟合"
        self._cleanup_ui(context)
        return {"CANCELLED"} if cancelled and not self._committed else {"FINISHED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None:
            return False
        if (
            scene_props.merge_mode_active
            or scene_props.split_mode_active
            or getattr(scene_props, "remove_mode_active", False)
            or scene_props.fit_mode_active
        ):
            return False
        if obj is None or obj.type != "MESH" or obj.mode == "EDIT":
            return False
        return obj.data.attributes.get(REGION_ID_ATTR) is not None

    def invoke(self, context: bpy.types.Context, event):
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        region_ids = read_region_ids(obj.data)
        if region_ids is None or not np.any(region_ids >= 0):
            self.report({"ERROR"}, "请先识别领域")
            return {"CANCELLED"}

        mesh_data = extract_mesh_data(obj)
        topology = extract_face_topology(obj.data)

        self._object = obj
        self._mesh_data = mesh_data
        self._topology_data = topology
        self._region_ids = region_ids.copy()
        self._preview_object = None
        self._debug_control_object = None
        self._debug_fold_object = None
        self._bridge_link_object = None
        self._last_result = None
        self._debug_result = None
        self._debug_island_count = 0
        self._fit_target_ids = []
        self._topology = "QUAD"
        self._committed = False
        self._closed = False
        self._timer = None

        session = _build_label_session(region_ids, mesh_data)
        session["object_name"] = obj.name
        session["preview_version"] = 0
        set_merge_label_session(session)
        register_label_draw_handler()
        set_fit_angle_label_session(None)
        _set_active_fit_op(self)

        scene_props.fit_mode_active = True
        scene_props.fit_confirm_requested = False
        scene_props.fit_advance_requested = False
        scene_props.fit_retreat_requested = False
        scene_props.fit_stage_rebuild_requested = False
        scene_props.fit_preview_requested = False
        scene_props.fit_target_id = -1
        scene_props.fit_hover_id = -1
        scene_props.fit_phase = "SELECT"
        scene_props.fit_segments_u = DEFAULT_SEG_U
        scene_props.fit_segments_v = DEFAULT_SEG_V
        scene_props.fit_topology = ""
        scene_props.fit_status = self._status_text(scene_props)
        scene_props.fit_status_detail = ""
        scene_props.region_object = obj

        _add_modal_timer(self, context)
        context.window_manager.modal_handler_add(self)
        _tag_redraw(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context):
        if getattr(self, "_committed", False):
            self._cleanup_ui(context)
            return {"FINISHED"}
        self._discard_preview()
        self._cleanup_ui(context)
        return {"CANCELLED"}

    def modal(self, context: bpy.types.Context, event):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        obj = self._object

        if getattr(self, "_closed", False):
            return {"FINISHED"}

        try:
            if obj is None or obj.name not in bpy.data.objects:
                return self.cancel(context)
            if obj.mode == "EDIT":
                return self.cancel(context)
        except ReferenceError:
            return self.cancel(context)

        if scene_props.fit_confirm_requested and not self._committed:
            scene_props.fit_confirm_requested = False
            if self._fit_and_commit_contour(context):
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if scene_props.fit_advance_requested:
            scene_props.fit_advance_requested = False
            if self._fit_and_commit_contour(context):
                return {"FINISHED"}
            return {"RUNNING_MODAL"}
        if scene_props.fit_retreat_requested:
            scene_props.fit_retreat_requested = False
            return {"RUNNING_MODAL"}
        if scene_props.fit_preview_requested:
            scene_props.fit_preview_requested = False
            if self._fit_and_commit_contour(context):
                return {"FINISHED"}
            return {"RUNNING_MODAL"}
        if scene_props.fit_stage_rebuild_requested:
            scene_props.fit_stage_rebuild_requested = False
            return {"RUNNING_MODAL"}

        if event.type == "TIMER":
            return {"RUNNING_MODAL"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            if self._committed:
                return self._finish_mode(context, cancelled=False)
            self.report({"INFO"}, "已取消拟合")
            return self._finish_mode(context, cancelled=True)

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if self._resolve_fit_targets(scene_props):
                if self._fit_and_commit_contour(context):
                    return {"FINISHED"}
            else:
                self.report({"INFO"}, "请先点击领域编号")
            return {"RUNNING_MODAL"}

        if context.space_data is None or context.space_data.type != "VIEW_3D":
            return {"PASS_THROUGH"}
        if context.region is None or context.region.type != "WINDOW":
            return {"PASS_THROUGH"}

        if event.type == "MOUSEMOVE":
            update_merge_label_projections(context)
            session = get_merge_label_session() or {"labels": []}
            hover = hit_test_labels(
                event.mouse_region_x,
                event.mouse_region_y,
                session.get("labels", []),
                LABEL_RADIUS_PX,
            )
            new_hover = -1 if hover is None else int(hover)
            if new_hover != int(scene_props.fit_hover_id):
                scene_props.fit_hover_id = new_hover
                _tag_redraw(context)
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            update_merge_label_projections(context)
            session = get_merge_label_session() or {"labels": []}
            hit = hit_test_labels(
                event.mouse_region_x,
                event.mouse_region_y,
                session.get("labels", []),
                LABEL_RADIUS_PX,
            )
            if hit is None:
                return {"PASS_THROUGH"}

            rid = int(hit)
            current = [
                int(value)
                for value in getattr(self, "_fit_target_ids", []) or []
                if int(value) >= 0
            ]
            if event.shift and current:
                if rid in current:
                    current = [value for value in current if value != rid]
                    if not current:
                        current = [rid]
                else:
                    current.append(rid)
                self._fit_target_ids = sorted(
                    set(int(value) for value in current)
                )
                scene_props.fit_target_id = int(rid)
                scene_props.fit_status = self._status_text(scene_props)
                scene_props.fit_status_detail = ""
                _tag_redraw(context)
                return {"RUNNING_MODAL"}

            self._fit_target_ids = [rid]
            scene_props.fit_target_id = int(rid)
            if self._fit_and_commit_contour(context):
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


__all__ = (
    "ARE_OT_fit_region",
    "ARE_OT_fit_step_next",
    "ARE_OT_fit_step_back",
    "ARE_OT_build_fit_surface",
    "ARE_OT_confirm_fit_region",
    "FIT_COLLECTION_NAME",
    "FIT_CONTOUR_KIND",
)
