# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""领域三边/四边曲面拟合操作符。"""

from __future__ import annotations

import bpy
import numpy as np

from ..algorithms.region_fit import (
    DEFAULT_SEG_U,
    DEFAULT_SEG_V,
    MAX_SEGMENTS,
    MIN_SEGMENTS,
    RegionFitError,
    extract_island_longest_sides,
    fit_region_surface,
)
from ..algorithms.regions import compute_region_label_anchors
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import (
    LABEL_RADIUS_PX,
    get_merge_label_session,
    register_label_draw_handler,
    set_merge_label_session,
    unregister_label_draw_handler,
    update_merge_label_projections,
)
from ..utils.mesh import extract_face_topology, extract_mesh_data
from ..utils.viewport import hit_test_labels
from .regions import REGION_ID_ATTR, read_region_ids


FIT_COLLECTION_NAME = "拟合面"
FIT_PREVIEW_NAME = "ARE_FitPreview"
FIT_DEBUG_CTRL_NAME = "ARE_FitDebugControls"
FIT_DEBUG_FOLD_NAME = "ARE_FitConcaveFolds"
FIT_EDGE_MAT_RED = "ARE_FitEdge_Red"
FIT_EDGE_MAT_GREEN = "ARE_FitEdge_Green"
FIT_EDGE_MAT_SEG_PREFIX = "ARE_FitEdge_Seg_"
FIT_FOLD_MAT = "ARE_FitFold_Marker"
FIT_EDGE_COLOR_RED = (1.0, 0.12, 0.08, 1.0)
FIT_EDGE_COLOR_GREEN = (0.1, 0.95, 0.22, 1.0)
FIT_EDGE_COLOR_FALLBACK = (
    FIT_EDGE_COLOR_RED,
    FIT_EDGE_COLOR_GREEN,
)
FIT_FOLD_COLOR = (0.15, 0.85, 1.0, 1.0)
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
            scene_props.fit_mode_active = False
            scene_props.fit_target_id = -1
            scene_props.fit_hover_id = -1
            scene_props.fit_phase = "IDLE"
            scene_props.fit_status = "拟合模态已失联，已强制退出"
            unregister_label_draw_handler()
            set_merge_label_session(None)
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


class ARE_OT_build_fit_surface(bpy.types.Operator):
    """面板按钮：从四边 debug 预览进入曲面拟合。"""

    bl_idname = "are.build_fit_surface"
    bl_label = "拟合成面"
    bl_description = "根据已提取的四条最长边生成三边/四边拟合曲面预览"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return (
            scene_props is not None
            and scene_props.fit_mode_active
            and scene_props.fit_phase == "DEBUG_EDGES"
        )

    def execute(self, context: bpy.types.Context):
        op = _get_active_fit_op()
        if op is None:
            self.report({"ERROR"}, "拟合模态未运行")
            return {"CANCELLED"}
        try:
            if op.build_surface_from_panel(context):
                self.report({"INFO"}, "已生成拟合曲面预览")
            else:
                self.report({"ERROR"}, "拟合成面失败")
        except Exception as exc:
            self.report({"ERROR"}, f"拟合成面失败: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class ARE_OT_confirm_fit_region(bpy.types.Operator):
    """面板确认按钮：通知拟合模态提交。"""

    bl_idname = "are.confirm_fit_region"
    bl_label = "确认"
    bl_description = "确认当前拟合面并退出拟合模式"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return (
            scene_props is not None
            and scene_props.fit_mode_active
            and scene_props.fit_phase == "PREVIEW"
        )

    def execute(self, context: bpy.types.Context):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        op = _get_active_fit_op()
        if op is not None:
            try:
                op.confirm_from_panel(context)
            except Exception as exc:
                self.report({"ERROR"}, f"确认拟合失败: {exc}")
                scene_props.fit_confirm_requested = True
                _schedule_force_exit_check()
                return {"FINISHED"}
            if scene_props.fit_mode_active:
                scene_props.fit_mode_active = False
                scene_props.fit_confirm_requested = False
                scene_props.fit_phase = "IDLE"
                unregister_label_draw_handler()
                set_merge_label_session(None)
            return {"FINISHED"}
        scene_props.fit_confirm_requested = True
        _schedule_force_exit_check()
        return {"FINISHED"}


class ARE_OT_fit_region(bpy.types.Operator):
    """
    模态拟合领域。

    点击编号 → 显示各 island 四条最长边（debug）→ 拟合成面 → 确认写入集合。
    """

    bl_idname = "are.fit_region"
    bl_label = "拟合领域"
    bl_description = (
        "点击领域编号显示外轮廓边；Shift+点击可多选并集（自动消去公共内边）；"
        "确认后拟合成面，再用滚轮/Page 调节控制点并写入「拟合面」集合"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _cleanup_ui(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        _remove_modal_timer(self, context)
        _clear_active_fit_op(self)
        unregister_label_draw_handler()
        set_merge_label_session(None)
        scene_props.fit_mode_active = False
        scene_props.fit_confirm_requested = False
        scene_props.fit_target_id = -1
        scene_props.fit_hover_id = -1
        scene_props.fit_phase = "IDLE"
        self._fit_target_ids = []
        _tag_redraw(context)

    def _discard_preview(self) -> None:
        preview = getattr(self, "_preview_object", None)
        controls = getattr(self, "_debug_control_object", None)
        folds = getattr(self, "_debug_fold_object", None)
        _delete_object(preview)
        _delete_object(controls)
        _delete_object(folds)
        self._preview_object = None
        self._debug_control_object = None
        self._debug_fold_object = None

    def confirm_from_panel(self, context: bpy.types.Context) -> None:
        if getattr(self, "_closed", False):
            return
        try:
            if not self._committed:
                self._commit_fit(context)
        finally:
            self._finish_mode(context, cancelled=False)
            self._closed = True

    def build_surface_from_panel(self, context: bpy.types.Context) -> bool:
        """面板「拟合成面」：从 DEBUG_EDGES 进入曲面预览。"""
        if getattr(self, "_closed", False):
            return False
        return self._rebuild_preview(context)

    def _status_text(self, scene_props) -> str:
        phase = scene_props.fit_phase
        targets = self._resolve_fit_targets(scene_props)
        label = "+".join(str(value) for value in targets) if targets else "?"
        if phase == "DEBUG_EDGES":
            island_n = int(getattr(self, "_debug_island_count", 0))
            return (
                f"领域 {label} · {island_n} 孤岛外轮廓边 · "
                f"Shift+点击加减选 · 核对后点「拟合成面」或按 Enter"
            )
        if phase != "PREVIEW":
            return (
                "点击领域编号显示外轮廓边；"
                "Shift+点击多选并集（消去公共内边）后再拟合成面"
            )
        topo = "三边" if self._topology == "TRI" else "四边"
        if self._topology == "TRI":
            return (
                f"领域 {label} · {topo} · "
                f"长边段数 {scene_props.fit_segments_v} · "
                f"底边段数 {scene_props.fit_segments_u} · "
                f"滚轮调长边 / Page 调底边"
            )
        return (
            f"领域 {label} · {topo} · "
            f"U={scene_props.fit_segments_u} V={scene_props.fit_segments_v} · "
            f"滚轮调 U / Page 调 V"
        )

    @staticmethod
    def _format_debug_detail(debug: dict) -> str:
        parts: list[str] = []
        for island in debug.get("islands", []):
            lengths = island.get("lengths", [])
            length_txt = " / ".join(f"{length:.3f}" for length in lengths)
            fold_n = int(island.get("concave_fold_count", 0))
            fold_txt = f"，凹折{fold_n}" if fold_n else ""
            parts.append(
                f"岛{island.get('island_index', '?')}: "
                f"{len(lengths)}边 [{length_txt}]{fold_txt}"
            )
        return "；".join(parts)

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

    def _rebuild_debug_edges(self, context: bpy.types.Context) -> bool:
        """提取并预览领域（可多选并集）外轮廓边。"""
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        targets = self._resolve_fit_targets(scene_props)
        if not targets:
            return False
        try:
            debug = extract_island_longest_sides(
                region_ids=self._region_ids,
                target_id=targets if len(targets) > 1 else targets[0],
                vertices=self._mesh_data["vertices"],
                loop_start=self._topology_data["loop_start"],
                loop_total=self._topology_data["loop_total"],
                loop_vertex_indices=self._topology_data["loop_vertex_indices"],
            )
        except RegionFitError as exc:
            self._discard_preview()
            scene_props.fit_phase = "SELECT"
            scene_props.fit_topology = ""
            scene_props.fit_status = str(exc)
            scene_props.fit_status_detail = ""
            self.report({"ERROR"}, str(exc))
            _tag_redraw(context)
            return False

        self._debug_result = debug
        self._debug_island_count = int(debug["island_count"])
        self._last_result = None
        self._topology = "QUAD"
        scene_props.fit_topology = ""
        scene_props.fit_phase = "DEBUG_EDGES"
        preview = _create_or_update_debug_curve_object(
            FIT_PREVIEW_NAME,
            debug,
            self._object.matrix_world,
            collection=None,
            existing=self._preview_object,
        )
        controls = _create_or_update_control_point_object(
            FIT_DEBUG_CTRL_NAME,
            debug,
            self._object.matrix_world,
            collection=None,
            existing=getattr(self, "_debug_control_object", None),
        )
        folds = _create_or_update_concave_fold_object(
            FIT_DEBUG_FOLD_NAME,
            debug,
            self._object.matrix_world,
            collection=None,
            existing=getattr(self, "_debug_fold_object", None),
        )
        self._preview_object = preview
        self._debug_control_object = controls
        self._debug_fold_object = folds
        scene_props.fit_status = self._status_text(scene_props)
        scene_props.fit_status_detail = self._format_debug_detail(debug)
        _tag_redraw(context)
        return True

    def _rebuild_preview(self, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        targets = self._resolve_fit_targets(scene_props)
        if not targets:
            return False
        try:
            result = fit_region_surface(
                region_ids=self._region_ids,
                target_id=targets if len(targets) > 1 else targets[0],
                vertices=self._mesh_data["vertices"],
                loop_start=self._topology_data["loop_start"],
                loop_total=self._topology_data["loop_total"],
                loop_vertex_indices=self._topology_data["loop_vertex_indices"],
                face_normals=self._mesh_data["normals"],
                face_areas=self._mesh_data["areas"],
                face_centers=self._mesh_data["face_centers"],
                segments_u=int(scene_props.fit_segments_u),
                segments_v=int(scene_props.fit_segments_v),
                triangle_ratio=float(scene_props.region_fit_triangle_ratio)
                / 100.0,
                adjacency_offsets=self._topology_data["adjacency_offsets"],
                adjacency_indices=self._topology_data["adjacency_indices"],
            )
        except RegionFitError as exc:
            # 拟合失败：撤下旧预览，回到编号选择阶段
            self._discard_preview()
            scene_props.fit_phase = "SELECT"
            scene_props.fit_topology = ""
            scene_props.fit_status = str(exc)
            self.report({"ERROR"}, str(exc))
            _tag_redraw(context)
            return False

        # 曲面预览前清掉 debug 曲线/控制点，避免类型混用
        self._discard_preview()
        self._topology = result.topology
        self._last_result = result
        scene_props.fit_topology = result.topology
        scene_props.fit_phase = "PREVIEW"
        preview = _create_or_update_mesh_object(
            FIT_PREVIEW_NAME,
            result.vertices,
            result.faces,
            self._object.matrix_world,
            collection=None,
            existing=None,
        )
        preview.hide_select = True
        preview.display_type = "WIRE"
        # 预览网格始终显示在扫描面之前，避免被遮挡
        preview.show_in_front = True
        self._preview_object = preview
        scene_props.fit_status = self._status_text(scene_props)
        if result.warnings:
            scene_props.fit_status_detail = "；".join(result.warnings)
        else:
            scene_props.fit_status_detail = ""
        _tag_redraw(context)
        return True

    def _commit_fit(self, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if self._preview_object is None or self._last_result is None:
            self.report({"WARNING"}, "还没有可确认的拟合预览")
            return False

        collection = _ensure_fit_collection(context.scene)
        target = int(scene_props.fit_target_id)
        final_name = f"Fit_R{target}_{self._topology}"
        # 断开预览与场景根，改链到拟合面集合
        preview = self._preview_object
        for col in list(preview.users_collection):
            col.objects.unlink(preview)
        if preview.name not in collection.objects:
            collection.objects.link(preview)
        preview.name = final_name
        if preview.data is not None:
            preview.data.name = final_name
        preview.hide_select = False
        preview.display_type = "TEXTURED"
        # 拟合面与扫描面重合，保持前置显示避免被埋没
        preview.show_in_front = True
        preview["are_fit_source"] = self._object.name
        preview["are_fit_region_id"] = target
        preview["are_fit_topology"] = self._topology

        self._preview_object = None
        self._committed = True
        scene_props.fit_status = (
            f"已拟合领域 {target}（{('三边' if self._topology == 'TRI' else '四边')}）"
            f" → {final_name}"
        )
        scene_props.fit_status_detail = ""
        return True

    def _finish_mode(self, context: bpy.types.Context, cancelled: bool) -> set:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if cancelled and not self._committed:
            self._discard_preview()
            scene_props.fit_status = "已取消拟合"
        self._cleanup_ui(context)
        return {"CANCELLED"} if cancelled and not self._committed else {"FINISHED"}

    def _adjust_segments(
        self,
        context: bpy.types.Context,
        delta_u: int = 0,
        delta_v: int = 0,
    ) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if scene_props.fit_phase != "PREVIEW":
            return
        self._updating_segments = True
        try:
            scene_props.fit_segments_u = int(
                np.clip(
                    int(scene_props.fit_segments_u) + delta_u,
                    MIN_SEGMENTS,
                    MAX_SEGMENTS,
                )
            )
            scene_props.fit_segments_v = int(
                np.clip(
                    int(scene_props.fit_segments_v) + delta_v,
                    MIN_SEGMENTS,
                    MAX_SEGMENTS,
                )
            )
        finally:
            self._updating_segments = False
        self._rebuild_preview(context)

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None:
            return False
        if (
            scene_props.merge_mode_active
            or scene_props.split_mode_active
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
        _set_active_fit_op(self)

        scene_props.fit_mode_active = True
        scene_props.fit_confirm_requested = False
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
            if self._commit_fit(context):
                self.report({"INFO"}, scene_props.fit_status)
                return self._finish_mode(context, cancelled=False)
            return {"RUNNING_MODAL"}

        if event.type == "TIMER":
            return {"RUNNING_MODAL"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            if self._committed:
                return self._finish_mode(context, cancelled=False)
            self.report({"INFO"}, "已取消拟合")
            return self._finish_mode(context, cancelled=True)

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if scene_props.fit_phase == "DEBUG_EDGES":
                if self._rebuild_preview(context):
                    self.report({"INFO"}, "已生成拟合曲面预览")
                return {"RUNNING_MODAL"}
            if scene_props.fit_phase == "PREVIEW":
                if self._commit_fit(context):
                    self.report({"INFO"}, scene_props.fit_status)
                    return self._finish_mode(context, cancelled=False)
            return {"RUNNING_MODAL"}

        # 滚轮：第一组（四边 U / 三边长边→存于 V）
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS":
            if scene_props.fit_phase == "PREVIEW":
                delta = 1 if event.type == "WHEELUPMOUSE" else -1
                if self._topology == "TRI":
                    self._adjust_segments(context, delta_v=delta)
                else:
                    self._adjust_segments(context, delta_u=delta)
                return {"RUNNING_MODAL"}

        # PageUp/PageDown：第二组（四边 V / 三边底边→存于 U）
        if event.type in {"PAGE_UP", "PAGE_DOWN"} and event.value == "PRESS":
            if scene_props.fit_phase == "PREVIEW":
                delta = 1 if event.type == "PAGE_UP" else -1
                if self._topology == "TRI":
                    self._adjust_segments(context, delta_u=delta)
                else:
                    self._adjust_segments(context, delta_v=delta)
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
                return {"RUNNING_MODAL"}

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
            else:
                current = [rid]
            self._fit_target_ids = sorted(set(int(value) for value in current))
            scene_props.fit_target_id = int(rid)
            scene_props.fit_segments_u = DEFAULT_SEG_U
            scene_props.fit_segments_v = DEFAULT_SEG_V
            if self._rebuild_debug_edges(context):
                label = "+".join(str(value) for value in self._fit_target_ids)
                self.report(
                    {"INFO"},
                    f"已选择领域 {label}：显示并集外轮廓边；"
                    "Shift+点击可加减选，核对后点「拟合成面」或按 Enter",
                )
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


__all__ = (
    "ARE_OT_fit_region",
    "ARE_OT_build_fit_surface",
    "ARE_OT_confirm_fit_region",
    "FIT_COLLECTION_NAME",
)
