# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""拟合曲线的拆分与贝塞尔重拟合操作符。"""

from __future__ import annotations

import bpy
import numpy as np

from ..algorithms.curve_edit import (
    RegionFitError,
    best_closed_alignment,
    best_open_alignment,
    estimate_similarity_transform,
    find_break_indices,
    fit_bezier_n_controls,
    opposite_edge_pairs,
    order_open_curves_as_closed_loop,
    sample_polyline_uniform,
    segment_colors_for_count,
    split_polyline_at_breaks,
    transform_bezier_points,
    weld_bezier_loop_endpoints,
)
from ..algorithms.region_fit import sample_cubic_bezier
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import (
    clear_curve_bezier_preview,
    clear_curve_split_preview,
    clear_curve_tool_hud,
    register_curve_tool_hud,
    set_curve_bezier_preview,
    set_curve_split_preview,
    set_curve_tool_hud,
    unregister_curve_tool_hud,
)

FIT_COLLECTION_NAME = "拟合面"
CURVE_SEG_MAT_PREFIX = "ARE_CurveSeg_"
DEFAULT_SPLIT_ANGLE = 35.0
SPLIT_ANGLE_STEP = 2.0
SPLIT_ANGLE_MIN = 5.0
SPLIT_ANGLE_MAX = 170.0
DEFAULT_BEZIER_CONTROLS = 4
BEZIER_CONTROLS_MIN = 3
BEZIER_CONTROLS_MAX = 32
SIMILAR_SAMPLE_COUNT = 64
MODAL_TIMER_STEP = 0.1


def _tag_redraw(context: bpy.types.Context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "UI"}:
                area.tag_redraw()


def _is_object_alive(obj) -> bool:
    """判断 Blender Object 指针是否仍有效（删除/撤销后会失效）。"""
    if obj is None:
        return False
    try:
        name = obj.name
    except ReferenceError:
        return False
    try:
        return name in bpy.data.objects and bpy.data.objects.get(name) is obj
    except ReferenceError:
        return False


def _restore_curve_backups(backups) -> None:
    """安全还原备份；物体已被删除/撤销时跳过。"""
    for entry in backups or []:
        obj = None
        backup = None
        name = None
        if isinstance(entry, dict):
            obj = entry.get("obj")
            backup = entry.get("data")
            name = entry.get("name")
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            obj, backup = entry[0], entry[1]
            if len(entry) >= 3:
                name = entry[2]
        if backup is None:
            continue
        target = None
        if _is_object_alive(obj):
            target = obj
        elif name and name in bpy.data.objects:
            target = bpy.data.objects[name]
        if target is None:
            continue
        try:
            _restore_curve(target, backup)
        except ReferenceError:
            continue
        except Exception:
            continue


def _header(context: bpy.types.Context, text: str) -> None:
    # 只用视口内 HUD（User Perspective 下方），不用 header_text_set
    set_curve_tool_hud(text)
    _tag_redraw(context)


def _clear_header(context: bpy.types.Context) -> None:
    clear_curve_tool_hud()
    clear_curve_split_preview()
    clear_curve_bezier_preview()
    _tag_redraw(context)


def _sample_bezier_polyline(
    bezier_points: list[dict],
    cyclic: bool,
    samples_per_span: int = 20,
) -> np.ndarray:
    """把贝塞尔锚点链采样成光滑折线（世界坐标）。"""
    if not bezier_points:
        return np.zeros((0, 3), dtype=np.float64)
    n = len(bezier_points)
    span_count = n if cyclic else max(n - 1, 0)
    if span_count <= 0:
        co = np.asarray(bezier_points[0]["co"], dtype=np.float64).reshape(1, 3)
        return co
    parts: list[np.ndarray] = []
    for index in range(span_count):
        nxt = (index + 1) % n
        controls = np.vstack(
            (
                np.asarray(bezier_points[index]["co"], dtype=np.float64),
                np.asarray(bezier_points[index]["handle_right"], dtype=np.float64),
                np.asarray(bezier_points[nxt]["handle_left"], dtype=np.float64),
                np.asarray(bezier_points[nxt]["co"], dtype=np.float64),
            )
        )
        sampled = sample_cubic_bezier(controls, max(int(samples_per_span), 4))
        if index > 0:
            sampled = sampled[1:]
        parts.append(sampled)
    return np.vstack(parts)


def _bezier_preview_from_payload(
    payload: list[tuple],
) -> dict | None:
    """从拟合结果收集 GPU 预览：曲线 / 锚点 / 手柄。"""
    curves: list[dict] = []
    anchors: list[np.ndarray] = []
    handles: list[np.ndarray] = []
    handle_edges: list[np.ndarray] = []
    palette = segment_colors_for_count(max(len(payload), 1))
    for color_index, item in enumerate(payload):
        _obj, fitted, cyclic = item
        if not fitted:
            continue
        sampled = _sample_bezier_polyline(fitted, bool(cyclic))
        if len(sampled) >= 2:
            curves.append(
                {
                    "points": sampled,
                    "color": palette[color_index % len(palette)],
                }
            )
        for bp in fitted:
            co = np.asarray(bp["co"], dtype=np.float64).reshape(3)
            hl = np.asarray(bp["handle_left"], dtype=np.float64).reshape(3)
            hr = np.asarray(bp["handle_right"], dtype=np.float64).reshape(3)
            anchors.append(co)
            handles.append(hl)
            handles.append(hr)
            handle_edges.append(np.stack((co, hl), axis=0))
            handle_edges.append(np.stack((co, hr), axis=0))
    if not anchors and not curves:
        return None
    extent = 1.0
    if curves:
        stacked = np.vstack([c["points"] for c in curves])
        extent = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    return {
        "curves": curves,
        "anchors": (
            np.vstack(anchors) if anchors else np.zeros((0, 3), dtype=np.float64)
        ),
        "handles": (
            np.vstack(handles) if handles else np.zeros((0, 3), dtype=np.float64)
        ),
        "handle_edges": (
            np.stack(handle_edges, axis=0)
            if handle_edges
            else np.zeros((0, 2, 3), dtype=np.float64)
        ),
        "line_width": float(max(3.0, min(7.0, extent * 0.008))),
        "handle_line_width": 1.8,
        "anchor_size": 12.0,
        "handle_size": 8.0,
    }


def _selected_curve_objects(context: bpy.types.Context) -> list[bpy.types.Object]:
    result = []
    for obj in context.selected_objects:
        if obj is not None and obj.type == "CURVE":
            result.append(obj)
    if not result:
        active = context.active_object
        if active is not None and active.type == "CURVE":
            result.append(active)
    return result


def _matrix_np(matrix) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)


def _world_points_from_local(matrix_world, local_pts: np.ndarray) -> np.ndarray:
    matrix = _matrix_np(matrix_world)
    ones = np.ones((len(local_pts), 1), dtype=np.float64)
    homo = np.hstack((np.asarray(local_pts, dtype=np.float64), ones))
    return (matrix @ homo.T).T[:, :3]


def _local_points_from_world(matrix_world, world_pts: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(_matrix_np(matrix_world))
    ones = np.ones((len(world_pts), 1), dtype=np.float64)
    homo = np.hstack((np.asarray(world_pts, dtype=np.float64), ones))
    return (inv @ homo.T).T[:, :3]


def _ensure_material(name: str, color: tuple[float, ...]) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
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
    return mat


def _apply_colors(
    curve: bpy.types.Curve,
    colors: list[tuple[float, float, float, float]],
) -> None:
    curve.materials.clear()
    for index, color in enumerate(colors):
        curve.materials.append(
            _ensure_material(f"{CURVE_SEG_MAT_PREFIX}{index}", color)
        )


def _extract_spline_polylines(obj: bpy.types.Object) -> list[dict]:
    curve = obj.data
    matrix = obj.matrix_world
    items: list[dict] = []
    for spline_index, spline in enumerate(curve.splines):
        cyclic = bool(spline.use_cyclic_u)
        local_pts: list[tuple[float, float, float]] = []
        if spline.type == "BEZIER":
            for bp in spline.bezier_points:
                local_pts.append((float(bp.co.x), float(bp.co.y), float(bp.co.z)))
        else:
            for point in spline.points:
                local_pts.append(
                    (float(point.co.x), float(point.co.y), float(point.co.z))
                )
        if len(local_pts) < 2:
            continue
        world = _world_points_from_local(
            matrix, np.asarray(local_pts, dtype=np.float64)
        )
        items.append(
            {
                "spline_index": int(spline_index),
                "points": world,
                "cyclic": cyclic,
                "spline_type": str(spline.type),
            }
        )
    return items


def _serialize_curve(obj: bpy.types.Object) -> dict:
    curve = obj.data
    splines = []
    for spline in curve.splines:
        entry = {
            "type": str(spline.type),
            "cyclic": bool(spline.use_cyclic_u),
            "points": [],
            "bezier": [],
        }
        if spline.type == "BEZIER":
            for bp in spline.bezier_points:
                entry["bezier"].append(
                    {
                        "co": tuple(bp.co),
                        "handle_left": tuple(bp.handle_left),
                        "handle_right": tuple(bp.handle_right),
                    }
                )
        else:
            for point in spline.points:
                entry["points"].append(tuple(point.co))
        splines.append(entry)
    custom = {}
    for key in obj.keys():
        if key == "_RNA_UI":
            continue
        try:
            custom[key] = obj[key]
        except Exception:
            pass
    return {
        "bevel_depth": float(curve.bevel_depth),
        "bevel_resolution": int(curve.bevel_resolution),
        "splines": splines,
        "custom": custom,
        "display_type": str(obj.display_type),
        "show_in_front": bool(obj.show_in_front),
        "color": tuple(float(v) for v in obj.color),
    }


def _clear_splines(curve: bpy.types.Curve) -> None:
    while curve.splines:
        curve.splines.remove(curve.splines[0])


def _restore_curve(obj: bpy.types.Object, backup: dict) -> None:
    curve = obj.data
    _clear_splines(curve)
    curve.bevel_depth = float(backup.get("bevel_depth", curve.bevel_depth))
    curve.bevel_resolution = int(
        backup.get("bevel_resolution", curve.bevel_resolution)
    )
    for entry in backup.get("splines") or []:
        stype = str(entry.get("type") or "POLY")
        if stype == "BEZIER":
            spline = curve.splines.new("BEZIER")
            bez = entry.get("bezier") or []
            if len(bez) > 1:
                spline.bezier_points.add(len(bez) - 1)
            for index, item in enumerate(bez):
                bp = spline.bezier_points[index]
                bp.co = item["co"]
                bp.handle_left_type = "FREE"
                bp.handle_right_type = "FREE"
                bp.handle_left = item["handle_left"]
                bp.handle_right = item["handle_right"]
            spline.use_cyclic_u = bool(entry.get("cyclic", False))
        else:
            pts = entry.get("points") or []
            spline = curve.splines.new("POLY")
            if len(pts) > 1:
                spline.points.add(len(pts) - 1)
            for index, co in enumerate(pts):
                spline.points[index].co = co
            spline.use_cyclic_u = bool(entry.get("cyclic", False))
    for key, value in (backup.get("custom") or {}).items():
        try:
            obj[key] = value
        except Exception:
            pass
    if "display_type" in backup:
        obj.display_type = backup["display_type"]
    if "show_in_front" in backup:
        obj.show_in_front = bool(backup["show_in_front"])
    if "color" in backup:
        obj.color = backup["color"]


def _write_poly_segments(
    obj: bpy.types.Object,
    segments: list[np.ndarray],
    colors: list[tuple[float, float, float, float]],
    cyclic_flags: list[bool] | None = None,
) -> None:
    curve = obj.data
    _clear_splines(curve)
    _apply_colors(curve, colors)
    matrix = obj.matrix_world
    for index, world_pts in enumerate(segments):
        if len(world_pts) < 2:
            continue
        local = _local_points_from_world(matrix, world_pts)
        spline = curve.splines.new("POLY")
        spline.points.add(len(local) - 1)
        for i, point in enumerate(local):
            spline.points[i].co = (
                float(point[0]),
                float(point[1]),
                float(point[2]),
                1.0,
            )
        cyclic = False
        if cyclic_flags is not None and index < len(cyclic_flags):
            cyclic = bool(cyclic_flags[index])
        spline.use_cyclic_u = cyclic
        if curve.materials:
            spline.material_index = index % len(curve.materials)


def _write_bezier_spline(
    obj: bpy.types.Object,
    bezier_points: list[dict],
    cyclic: bool,
) -> None:
    curve = obj.data
    _clear_splines(curve)
    curve.materials.clear()
    curve.bevel_depth = 0.0
    curve.bevel_resolution = 0
    matrix = obj.matrix_world
    spline = curve.splines.new("BEZIER")
    if len(bezier_points) > 1:
        spline.bezier_points.add(len(bezier_points) - 1)
    for index, item in enumerate(bezier_points):
        bp = spline.bezier_points[index]
        co = _local_points_from_world(matrix, item["co"].reshape(1, 3))[0]
        hl = _local_points_from_world(
            matrix, item["handle_left"].reshape(1, 3)
        )[0]
        hr = _local_points_from_world(
            matrix, item["handle_right"].reshape(1, 3)
        )[0]
        bp.handle_left_type = "FREE"
        bp.handle_right_type = "FREE"
        bp.co = (float(co[0]), float(co[1]), float(co[2]))
        bp.handle_left = (float(hl[0]), float(hl[1]), float(hl[2]))
        bp.handle_right = (float(hr[0]), float(hr[1]), float(hr[2]))
    spline.use_cyclic_u = bool(cyclic)


def _ensure_fit_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    col = bpy.data.collections.get(FIT_COLLECTION_NAME)
    if col is None:
        col = bpy.data.collections.new(FIT_COLLECTION_NAME)
        scene.collection.children.link(col)
    return col


def _any_curve_modal_busy(scene_props) -> bool:
    return bool(
        getattr(scene_props, "curve_split_mode_active", False)
        or getattr(scene_props, "curve_fit_mode_active", False)
        or scene_props.merge_mode_active
        or scene_props.split_mode_active
        or getattr(scene_props, "remove_mode_active", False)
        or getattr(scene_props, "fit_mode_active", False)
    )


class ARE_OT_split_fit_curve(bpy.types.Operator):
    """按折角阈值预览分色，确认后拆成多条曲线。"""

    bl_idname = "are.split_fit_curve"
    bl_label = "拆分曲线"
    bl_description = (
        "将选中曲线按折角断开；不同颜色标记连续段；"
        "Ctrl+滚轮调阈值；Enter 确认拆分，Esc 取消"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _update_status(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        n = int(getattr(self, "_segment_count", 0))
        angle = float(getattr(self, "_angle", DEFAULT_SPLIT_ANGLE))
        text = (
            f"拆分曲线：将拆分为 {n} 条 · 折角阈值 {angle:.0f}° · "
            "Ctrl+滚轮调节 · Enter 确认 · Esc 取消"
        )
        scene_props.curve_split_status = text
        _header(context, text)

    def _rebuild_preview(self, context: bpy.types.Context) -> bool:
        angle = float(getattr(self, "_angle", DEFAULT_SPLIT_ANGLE))
        all_segments: list[tuple[bpy.types.Object, list[np.ndarray]]] = []
        overlay_segments: list[dict] = []
        total = 0
        for obj, sources in self._targets:
            segments: list[np.ndarray] = []
            for source in sources:
                breaks = find_break_indices(
                    source["points"],
                    angle_threshold_deg=angle,
                    cyclic=bool(source["cyclic"]),
                )
                parts = split_polyline_at_breaks(
                    source["points"],
                    breaks,
                    cyclic=bool(source["cyclic"]),
                )
                for part in parts:
                    if len(part) >= 2:
                        segments.append(part)
            if not segments:
                continue
            colors = segment_colors_for_count(len(segments))
            # 几何仍写入物体（确认拆分用）；分色靠 GPU 叠加，避免选中橙色盖住材质
            _write_poly_segments(
                obj,
                segments,
                colors,
                cyclic_flags=[False] * len(segments),
            )
            # 预览时弱化物体本身线框，突出分色叠加
            obj.display_type = "WIRE"
            obj.show_in_front = True
            obj.color = (0.15, 0.15, 0.15, 0.15)
            for index, part in enumerate(segments):
                overlay_segments.append(
                    {
                        "points": np.asarray(part, dtype=np.float64).copy(),
                        "color": colors[index % len(colors)],
                    }
                )
            all_segments.append((obj, segments))
            total += len(segments)

        extent = 1.0
        if overlay_segments:
            stacked = np.vstack([item["points"] for item in overlay_segments])
            extent = float(
                np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0))
            )
        set_curve_split_preview(
            {
                "segments": overlay_segments,
                "line_width": float(max(3.5, min(8.0, extent * 0.01))),
            }
        )
        self._preview_segments = all_segments
        self._segment_count = total
        self._update_status(context)
        return total > 0

    def _commit(self, context: bpy.types.Context) -> bool:
        collection = _ensure_fit_collection(context.scene)
        created_objs: list[bpy.types.Object] = []
        for obj, segments in getattr(self, "_preview_segments", []) or []:
            if len(segments) <= 1:
                colors = segment_colors_for_count(1)
                _write_poly_segments(
                    obj, segments, colors, cyclic_flags=[False]
                )
                obj["are_fit_kind"] = "contour_split"
                if obj.data is not None:
                    obj.data.bevel_depth = 0.0
                    obj.data.bevel_resolution = 0
                obj.display_type = "WIRE"
                obj.color = (1.0, 1.0, 1.0, 1.0)
                created_objs.append(obj)
                continue
            base_name = obj.name
            backups_attrs = {
                k: obj[k] for k in obj.keys() if k != "_RNA_UI"
            }
            matrix = obj.matrix_world.copy()
            colors = segment_colors_for_count(len(segments))
            _write_poly_segments(
                obj,
                [segments[0]],
                [colors[0]],
                cyclic_flags=[False],
            )
            obj.name = f"{base_name}_S0"
            if obj.data is not None:
                obj.data.name = obj.name
                obj.data.bevel_depth = 0.0
                obj.data.bevel_resolution = 0
            obj["are_fit_kind"] = "contour_split"
            obj["are_fit_split_index"] = 0
            obj.display_type = "WIRE"
            obj.show_in_front = True
            obj.color = (1.0, 1.0, 1.0, 1.0)
            created_objs.append(obj)
            for index, segment in enumerate(segments[1:], start=1):
                curve = bpy.data.curves.new(
                    f"{base_name}_S{index}", type="CURVE"
                )
                new_obj = bpy.data.objects.new(
                    f"{base_name}_S{index}", curve
                )
                curve.dimensions = "3D"
                curve.bevel_depth = 0.0
                curve.bevel_resolution = 0
                new_obj.matrix_world = matrix
                new_obj.show_in_front = True
                new_obj.display_type = "WIRE"
                new_obj.color = (1.0, 1.0, 1.0, 1.0)
                if new_obj.name not in collection.objects:
                    collection.objects.link(new_obj)
                for key, value in backups_attrs.items():
                    try:
                        new_obj[key] = value
                    except Exception:
                        pass
                new_obj["are_fit_kind"] = "contour_split"
                new_obj["are_fit_split_index"] = index
                _write_poly_segments(
                    new_obj,
                    [segment],
                    [colors[index % len(colors)]],
                    cyclic_flags=[False],
                )
                created_objs.append(new_obj)

        # 默认选中拆分后的全部曲线
        for obj in list(context.selected_objects):
            try:
                obj.select_set(False)
            except ReferenceError:
                pass
        for obj in created_objs:
            try:
                obj.select_set(True)
            except ReferenceError:
                pass
        if created_objs:
            context.view_layer.objects.active = created_objs[0]

        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.curve_split_status = (
            f"已拆分为 {len(created_objs)} 条曲线（已全选）"
        )
        self.report({"INFO"}, scene_props.curve_split_status)
        return len(created_objs) > 0

    def _cleanup(self, context: bpy.types.Context, restore: bool) -> None:
        # 无论还原是否成功，都必须清掉 GPU 预览，避免残影
        try:
            if restore:
                _restore_curve_backups(getattr(self, "_backups", None))
        finally:
            scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
            if scene_props is not None:
                scene_props.curve_split_mode_active = False
                scene_props.curve_split_confirm_requested = False
            _clear_header(context)
            unregister_curve_tool_hud()
            timer = getattr(self, "_timer", None)
            if timer is not None:
                try:
                    context.window_manager.event_timer_remove(timer)
                except Exception:
                    pass
                self._timer = None
            _tag_redraw(context)

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None:
            return False
        if _any_curve_modal_busy(scene_props):
            return False
        return bool(_selected_curve_objects(context))

    def invoke(self, context: bpy.types.Context, event):
        curves = _selected_curve_objects(context)
        if not curves:
            self.report({"ERROR"}, "请先选中曲线物体")
            return {"CANCELLED"}
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._targets = []
        self._backups = []
        for obj in curves:
            sources = _extract_spline_polylines(obj)
            if not sources:
                continue
            self._backups.append(
                {
                    "name": obj.name,
                    "obj": obj,
                    "data": _serialize_curve(obj),
                }
            )
            self._targets.append((obj, sources))
        if not self._targets:
            self.report({"ERROR"}, "选中曲线没有可用样条")
            return {"CANCELLED"}

        self._angle = float(
            getattr(scene_props, "curve_split_angle", DEFAULT_SPLIT_ANGLE)
        )
        self._committed = False
        self._preview_segments = []
        self._segment_count = 0
        scene_props.curve_split_mode_active = True
        scene_props.curve_split_confirm_requested = False
        scene_props.curve_split_angle = self._angle
        clear_curve_bezier_preview()
        register_curve_tool_hud()
        self._rebuild_preview(context)
        self._timer = context.window_manager.event_timer_add(
            MODAL_TIMER_STEP, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        # 目标曲线被删除/撤销后立刻退出，并清预览
        if not all(
            _is_object_alive(obj) for obj, _sources in getattr(self, "_targets", [])
        ):
            self.report({"WARNING"}, "目标曲线已丢失，已退出拆分")
            self._cleanup(context, restore=False)
            return {"CANCELLED"}

        if scene_props.curve_split_confirm_requested:
            scene_props.curve_split_confirm_requested = False
            if self._commit(context):
                self._committed = True
                self._cleanup(context, restore=False)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self.report({"INFO"}, "已取消拆分曲线")
            self._cleanup(context, restore=True)
            return {"CANCELLED"}

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if self._commit(context):
                self._committed = True
                self._cleanup(context, restore=False)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.ctrl and event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            delta = (
                SPLIT_ANGLE_STEP
                if event.type == "WHEELUPMOUSE"
                else -SPLIT_ANGLE_STEP
            )
            self._angle = float(
                min(SPLIT_ANGLE_MAX, max(SPLIT_ANGLE_MIN, self._angle + delta))
            )
            scene_props.curve_split_angle = self._angle
            self._rebuild_preview(context)
            return {"RUNNING_MODAL"}

        panel_angle = float(
            getattr(scene_props, "curve_split_angle", self._angle)
        )
        if abs(panel_angle - self._angle) > 1e-6:
            self._angle = float(
                min(SPLIT_ANGLE_MAX, max(SPLIT_ANGLE_MIN, panel_angle))
            )
            self._rebuild_preview(context)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def cancel(self, context: bpy.types.Context):
        self._cleanup(
            context, restore=not getattr(self, "_committed", False)
        )
        return {"CANCELLED"}


class ARE_OT_confirm_split_fit_curve(bpy.types.Operator):
    bl_idname = "are.confirm_split_fit_curve"
    bl_label = "确认拆分曲线"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return scene_props is not None and bool(
            getattr(scene_props, "curve_split_mode_active", False)
        )

    def execute(self, context: bpy.types.Context):
        getattr(context.scene, SCENE_PROP_NAME).curve_split_confirm_requested = True
        return {"FINISHED"}


class ARE_OT_fit_bezier_curve(bpy.types.Operator):
    """将选中曲线拟合成 n 控制点贝塞尔；S 切换相似模式。"""

    bl_idname = "are.fit_bezier_curve"
    bl_label = "拟合曲线"
    bl_description = (
        "将选中曲线转为 n 个控制点的贝塞尔（默认 4，Ctrl+滚轮调节，最少 3）；"
        "按 S 切换相似模式；四条端点近乎闭合时自动识别对边，"
        "相似模式下对边两两相似并焊接端点保持封闭"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _update_status(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        n = int(getattr(self, "_controls", DEFAULT_BEZIER_CONTROLS))
        similar = bool(getattr(self, "_similar", False))
        mode = "开" if similar else "关"
        quad = bool(getattr(self, "_quad_loop_active", False))
        extra = " · 四边形对边封闭" if quad else ""
        if quad and similar:
            extra = " · 对边两两相似·封闭"
        text = (
            f"拟合曲线：控制点 {n} · 相似模式 {mode}{extra} · "
            "Ctrl+滚轮调点数 · S 切换相似 · Enter 确认 · Esc 取消"
        )
        scene_props.curve_fit_status = text
        _header(context, text)

    def _fit_one(self, points: np.ndarray, cyclic: bool) -> list[dict]:
        return fit_bezier_n_controls(
            points,
            control_count=int(self._controls),
            cyclic=bool(cyclic),
        )

    def _polyline_length(self, points: np.ndarray, cyclic: bool) -> float:
        pts = np.asarray(points, dtype=np.float64)
        if cyclic:
            ring = np.vstack((pts, pts[:1]))
        else:
            ring = pts
        if len(ring) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())

    def _collect_quad_open_edges(self) -> list[tuple] | None:
        """恰好 4 条物体、各一条开环样条时返回 [(obj, source), ...]。"""
        if len(self._targets) != 4:
            return None
        edges: list[tuple] = []
        for obj, sources in self._targets:
            opens = [
                source
                for source in sources
                if (not source["cyclic"]) and len(source["points"]) >= 2
            ]
            if len(opens) != 1:
                return None
            edges.append((obj, opens[0]))
        return edges

    def _rebuild_quad_loop_preview(
        self,
        edges: list[tuple],
        similar: bool,
    ) -> bool:
        """
        四条开环端点近乎闭合时：识别对边；相似则对边两两相似；焊接端点。
        """
        polylines = [source["points"] for _obj, source in edges]
        ordered = order_open_curves_as_closed_loop(polylines)
        if ordered is None:
            return False
        order, flipped, _gap = ordered
        loop_items: list[tuple] = []
        for index, reverse in zip(order, flipped):
            obj, source = edges[index]
            pts = np.asarray(source["points"], dtype=np.float64)
            if reverse:
                pts = pts[::-1].copy()
            loop_items.append((obj, pts))

        beziers: list[list[dict] | None] = [None, None, None, None]
        if similar:
            for pair_id, (a, b) in enumerate(opposite_edge_pairs(4)):
                len_a = self._polyline_length(loop_items[a][1], False)
                len_b = self._polyline_length(loop_items[b][1], False)
                if len_a >= len_b:
                    ref_i, dst_i = a, b
                else:
                    ref_i, dst_i = b, a
                ref_pts = loop_items[ref_i][1]
                dst_pts = loop_items[dst_i][1]
                prototype = self._fit_one(ref_pts, False)
                ref_samples = sample_polyline_uniform(
                    ref_pts, SIMILAR_SAMPLE_COUNT, cyclic=False
                )
                dst_samples = sample_polyline_uniform(
                    dst_pts, SIMILAR_SAMPLE_COUNT, cyclic=False
                )
                aligned, _rmse = best_open_alignment(ref_samples, dst_samples)
                scale, rotation, translation = estimate_similarity_transform(
                    ref_samples, aligned
                )
                fitted = transform_bezier_points(
                    prototype, scale, rotation, translation
                )
                beziers[ref_i] = prototype
                beziers[dst_i] = fitted
                loop_items[ref_i][0]["are_fit_opposite_pair"] = int(pair_id)
                loop_items[dst_i][0]["are_fit_opposite_pair"] = int(pair_id)
        else:
            for index, (_obj, pts) in enumerate(loop_items):
                beziers[index] = self._fit_one(pts, False)

        if any(item is None for item in beziers):
            return False
        welded = weld_bezier_loop_endpoints(beziers)
        group_id = f"quad_{loop_items[0][0].name}"
        for index, (obj, _pts) in enumerate(loop_items):
            fitted = welded[index]
            _write_bezier_spline(obj, fitted, cyclic=False)
            obj["are_fit_kind"] = "contour_bezier"
            obj["are_fit_quad_loop"] = True
            obj["are_fit_similar"] = bool(similar)
            if similar:
                obj["are_fit_similar_group"] = group_id
            self._preview_payload.append((obj, fitted, False))
        self._quad_loop_active = True
        return True

    def _rebuild_preview(self, context: bpy.types.Context) -> bool:
        similar = bool(getattr(self, "_similar", False))
        self._preview_payload = []
        self._quad_loop_active = False
        quad_edges = self._collect_quad_open_edges()
        if quad_edges is not None and self._rebuild_quad_loop_preview(
            quad_edges, similar
        ):
            preview = _bezier_preview_from_payload(self._preview_payload)
            set_curve_bezier_preview(preview)
            self._update_status(context)
            return True

        if similar and len(self._targets) >= 2:
            ranked = []
            for obj, sources in self._targets:
                for source in sources:
                    length = self._polyline_length(
                        source["points"], source["cyclic"]
                    )
                    ranked.append((length, obj, source))
            ranked.sort(key=lambda item: item[0], reverse=True)
            _len0, ref_obj, ref_source = ranked[0]
            cyclic = bool(ref_source["cyclic"])
            ref_samples = sample_polyline_uniform(
                ref_source["points"],
                SIMILAR_SAMPLE_COUNT,
                cyclic=cyclic,
            )
            prototype = self._fit_one(ref_source["points"], cyclic)
            _write_bezier_spline(ref_obj, prototype, cyclic=cyclic)
            ref_obj["are_fit_kind"] = "contour_bezier"
            ref_obj["are_fit_similar"] = True
            group_id = f"sim_{ref_obj.name}"
            ref_obj["are_fit_similar_group"] = group_id
            self._preview_payload.append((ref_obj, prototype, cyclic))

            for _length, obj, source in ranked[1:]:
                samples = sample_polyline_uniform(
                    source["points"],
                    SIMILAR_SAMPLE_COUNT,
                    cyclic=bool(source["cyclic"]) or cyclic,
                )
                if cyclic or source["cyclic"]:
                    aligned, _rmse = best_closed_alignment(ref_samples, samples)
                else:
                    aligned, _rmse = best_open_alignment(ref_samples, samples)
                scale, rotation, translation = estimate_similarity_transform(
                    ref_samples, aligned
                )
                fitted = transform_bezier_points(
                    prototype, scale, rotation, translation
                )
                _write_bezier_spline(obj, fitted, cyclic=cyclic)
                obj["are_fit_kind"] = "contour_bezier"
                obj["are_fit_similar"] = True
                obj["are_fit_similar_group"] = group_id
                self._preview_payload.append((obj, fitted, cyclic))
        else:
            for obj, sources in self._targets:
                if len(sources) == 1:
                    source = sources[0]
                    fitted = self._fit_one(
                        source["points"], source["cyclic"]
                    )
                    _write_bezier_spline(
                        obj, fitted, cyclic=bool(source["cyclic"])
                    )
                    obj["are_fit_kind"] = "contour_bezier"
                    obj["are_fit_similar"] = False
                    self._preview_payload.append(
                        (obj, fitted, bool(source["cyclic"]))
                    )
                    continue
                curve = obj.data
                _clear_splines(curve)
                matrix = obj.matrix_world
                for source in sources:
                    fitted = self._fit_one(
                        source["points"], source["cyclic"]
                    )
                    spline = curve.splines.new("BEZIER")
                    if len(fitted) > 1:
                        spline.bezier_points.add(len(fitted) - 1)
                    for index, item in enumerate(fitted):
                        bp = spline.bezier_points[index]
                        co = _local_points_from_world(
                            matrix, item["co"].reshape(1, 3)
                        )[0]
                        hl = _local_points_from_world(
                            matrix, item["handle_left"].reshape(1, 3)
                        )[0]
                        hr = _local_points_from_world(
                            matrix, item["handle_right"].reshape(1, 3)
                        )[0]
                        bp.handle_left_type = "FREE"
                        bp.handle_right_type = "FREE"
                        bp.co = tuple(float(v) for v in co)
                        bp.handle_left = tuple(float(v) for v in hl)
                        bp.handle_right = tuple(float(v) for v in hr)
                    spline.use_cyclic_u = bool(source["cyclic"])
                    self._preview_payload.append(
                        (obj, fitted, bool(source["cyclic"]))
                    )
                obj["are_fit_kind"] = "contour_bezier"
                obj["are_fit_similar"] = False

        preview = _bezier_preview_from_payload(self._preview_payload)
        set_curve_bezier_preview(preview)
        self._update_status(context)
        return True

    def _commit(self, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        n = int(self._controls)
        similar = "相似" if self._similar else "独立"
        if getattr(self, "_quad_loop_active", False):
            if self._similar:
                similar = "对边两两相似·封闭"
            else:
                similar = "四边形封闭"
        scene_props.curve_fit_status = (
            f"已拟合为 {n} 控制点贝塞尔（{similar}）"
        )
        self.report({"INFO"}, scene_props.curve_fit_status)
        return True

    def _cleanup(self, context: bpy.types.Context, restore: bool) -> None:
        # 无论还原是否成功，都必须清掉 GPU 预览，避免残影
        try:
            if restore:
                _restore_curve_backups(getattr(self, "_backups", None))
        finally:
            scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
            if scene_props is not None:
                scene_props.curve_fit_mode_active = False
                scene_props.curve_fit_confirm_requested = False
            _clear_header(context)
            unregister_curve_tool_hud()
            timer = getattr(self, "_timer", None)
            if timer is not None:
                try:
                    context.window_manager.event_timer_remove(timer)
                except Exception:
                    pass
                self._timer = None
            self._preview_payload = []
            _tag_redraw(context)

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None:
            return False
        if _any_curve_modal_busy(scene_props):
            return False
        return bool(_selected_curve_objects(context))

    def invoke(self, context: bpy.types.Context, event):
        curves = _selected_curve_objects(context)
        if not curves:
            self.report({"ERROR"}, "请先选中曲线物体")
            return {"CANCELLED"}
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._targets = []
        self._backups = []
        for obj in curves:
            sources = _extract_spline_polylines(obj)
            if not sources:
                continue
            self._backups.append(
                {
                    "name": obj.name,
                    "obj": obj,
                    "data": _serialize_curve(obj),
                }
            )
            self._targets.append((obj, sources))
        if not self._targets:
            self.report({"ERROR"}, "选中曲线没有可用样条")
            return {"CANCELLED"}

        self._controls = int(
            getattr(scene_props, "curve_fit_controls", DEFAULT_BEZIER_CONTROLS)
        )
        self._controls = max(
            BEZIER_CONTROLS_MIN, min(BEZIER_CONTROLS_MAX, self._controls)
        )
        self._similar = bool(getattr(scene_props, "curve_fit_similar", False))
        self._committed = False
        self._preview_payload = []
        self._quad_loop_active = False
        scene_props.curve_fit_mode_active = True
        scene_props.curve_fit_confirm_requested = False
        scene_props.curve_fit_controls = self._controls
        scene_props.curve_fit_similar = self._similar
        clear_curve_split_preview()
        register_curve_tool_hud()
        try:
            self._rebuild_preview(context)
        except RegionFitError as exc:
            self._cleanup(context, restore=True)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self._timer = context.window_manager.event_timer_add(
            MODAL_TIMER_STEP, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        # 目标曲线被删除/撤销后立刻退出，并清预览
        if not all(
            _is_object_alive(obj) for obj, _sources in getattr(self, "_targets", [])
        ):
            self.report({"WARNING"}, "目标曲线已丢失，已退出拟合")
            self._cleanup(context, restore=False)
            return {"CANCELLED"}

        if scene_props.curve_fit_confirm_requested:
            scene_props.curve_fit_confirm_requested = False
            if self._commit(context):
                self._committed = True
                self._cleanup(context, restore=False)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self.report({"INFO"}, "已取消拟合曲线")
            self._cleanup(context, restore=True)
            return {"CANCELLED"}

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if self._commit(context):
                self._committed = True
                self._cleanup(context, restore=False)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.type == "S" and event.value == "PRESS" and not event.ctrl:
            self._similar = not self._similar
            scene_props.curve_fit_similar = self._similar
            try:
                self._rebuild_preview(context)
            except RegionFitError as exc:
                self.report({"ERROR"}, str(exc))
            return {"RUNNING_MODAL"}

        if event.ctrl and event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            delta = 1 if event.type == "WHEELUPMOUSE" else -1
            self._controls = int(
                min(
                    BEZIER_CONTROLS_MAX,
                    max(BEZIER_CONTROLS_MIN, self._controls + delta),
                )
            )
            scene_props.curve_fit_controls = self._controls
            try:
                self._rebuild_preview(context)
            except RegionFitError as exc:
                self.report({"ERROR"}, str(exc))
            return {"RUNNING_MODAL"}

        panel_n = int(getattr(scene_props, "curve_fit_controls", self._controls))
        panel_sim = bool(getattr(scene_props, "curve_fit_similar", self._similar))
        if panel_n != self._controls or panel_sim != self._similar:
            self._controls = max(
                BEZIER_CONTROLS_MIN, min(BEZIER_CONTROLS_MAX, panel_n)
            )
            self._similar = panel_sim
            try:
                self._rebuild_preview(context)
            except RegionFitError as exc:
                self.report({"ERROR"}, str(exc))
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def cancel(self, context: bpy.types.Context):
        self._cleanup(
            context, restore=not getattr(self, "_committed", False)
        )
        return {"CANCELLED"}


class ARE_OT_confirm_fit_bezier_curve(bpy.types.Operator):
    bl_idname = "are.confirm_fit_bezier_curve"
    bl_label = "确认拟合曲线"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return scene_props is not None and bool(
            getattr(scene_props, "curve_fit_mode_active", False)
        )

    def execute(self, context: bpy.types.Context):
        getattr(context.scene, SCENE_PROP_NAME).curve_fit_confirm_requested = True
        return {"FINISHED"}


__all__ = (
    "ARE_OT_split_fit_curve",
    "ARE_OT_confirm_split_fit_curve",
    "ARE_OT_fit_bezier_curve",
    "ARE_OT_confirm_fit_bezier_curve",
)
