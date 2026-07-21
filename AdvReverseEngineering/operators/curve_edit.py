# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""拟合曲线的拆分与贝塞尔重拟合操作符。"""

from __future__ import annotations

import bpy
import numpy as np

from ..algorithms.curve_edit import (
    RegionFitError,
    best_closed_alignment,
    estimate_similarity_transform,
    find_break_indices,
    fit_bezier_n_controls,
    sample_polyline_uniform,
    segment_colors_for_count,
    split_polyline_at_breaks,
    transform_bezier_points,
)
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import (
    clear_curve_tool_hud,
    register_curve_tool_hud,
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


def _header(context: bpy.types.Context, text: str) -> None:
    area = getattr(context, "area", None)
    if area is not None:
        try:
            area.header_text_set(text)
        except Exception:
            pass
    set_curve_tool_hud(text)
    _tag_redraw(context)


def _clear_header(context: bpy.types.Context) -> None:
    area = getattr(context, "area", None)
    if area is not None:
        try:
            area.header_text_set(None)
        except Exception:
            pass
    clear_curve_tool_hud()


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
            _write_poly_segments(
                obj,
                segments,
                colors,
                cyclic_flags=[False] * len(segments),
            )
            all_segments.append((obj, segments))
            total += len(segments)
        self._preview_segments = all_segments
        self._segment_count = total
        self._update_status(context)
        return total > 0

    def _commit(self, context: bpy.types.Context) -> bool:
        collection = _ensure_fit_collection(context.scene)
        created = 0
        for obj, segments in getattr(self, "_preview_segments", []) or []:
            if len(segments) <= 1:
                colors = segment_colors_for_count(1)
                _write_poly_segments(
                    obj, segments, colors, cyclic_flags=[False]
                )
                obj["are_fit_kind"] = "contour_split"
                created += 1
                continue
            base_name = obj.name
            backups_attrs = {
                k: obj[k] for k in obj.keys() if k != "_RNA_UI"
            }
            matrix = obj.matrix_world.copy()
            bevel = float(obj.data.bevel_depth)
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
            obj["are_fit_kind"] = "contour_split"
            obj["are_fit_split_index"] = 0
            created += 1
            for index, segment in enumerate(segments[1:], start=1):
                curve = bpy.data.curves.new(
                    f"{base_name}_S{index}", type="CURVE"
                )
                new_obj = bpy.data.objects.new(
                    f"{base_name}_S{index}", curve
                )
                curve.dimensions = "3D"
                curve.bevel_depth = bevel
                curve.bevel_resolution = 2
                new_obj.matrix_world = matrix
                new_obj.show_in_front = True
                new_obj.display_type = "WIRE"
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
                created += 1
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.curve_split_status = f"已拆分为 {created} 条曲线"
        self.report({"INFO"}, scene_props.curve_split_status)
        return created > 0

    def _cleanup(self, context: bpy.types.Context, restore: bool) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if restore:
            for obj, backup in getattr(self, "_backups", []) or []:
                if obj is not None and obj.name in bpy.data.objects:
                    _restore_curve(obj, backup)
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
            self._backups.append((obj, _serialize_curve(obj)))
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
        register_curve_tool_hud()
        self._rebuild_preview(context)
        self._timer = context.window_manager.event_timer_add(
            MODAL_TIMER_STEP, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
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
        "按 S 切换相似模式：多条相似曲线拟合一条原型并变换对齐"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _update_status(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        n = int(getattr(self, "_controls", DEFAULT_BEZIER_CONTROLS))
        similar = bool(getattr(self, "_similar", False))
        mode = "开" if similar else "关"
        text = (
            f"拟合曲线：控制点 {n} · 相似模式 {mode} · "
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

    def _rebuild_preview(self, context: bpy.types.Context) -> bool:
        similar = bool(getattr(self, "_similar", False))
        self._preview_payload = []
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
                    aligned = samples
                    rev = samples[::-1]
                    err = float(
                        np.mean(np.sum((aligned - ref_samples) ** 2, axis=1))
                    )
                    err_r = float(
                        np.mean(np.sum((rev - ref_samples) ** 2, axis=1))
                    )
                    if err_r < err:
                        aligned = rev
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
                obj["are_fit_kind"] = "contour_bezier"
                obj["are_fit_similar"] = False
                self._preview_payload.append((obj, None, False))

        self._update_status(context)
        return True

    def _commit(self, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        n = int(self._controls)
        similar = "相似" if self._similar else "独立"
        scene_props.curve_fit_status = (
            f"已拟合为 {n} 控制点贝塞尔（{similar}）"
        )
        self.report({"INFO"}, scene_props.curve_fit_status)
        return True

    def _cleanup(self, context: bpy.types.Context, restore: bool) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if restore:
            for obj, backup in getattr(self, "_backups", []) or []:
                if obj is not None and obj.name in bpy.data.objects:
                    _restore_curve(obj, backup)
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
            self._backups.append((obj, _serialize_curve(obj)))
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
        scene_props.curve_fit_mode_active = True
        scene_props.curve_fit_confirm_requested = False
        scene_props.curve_fit_controls = self._controls
        scene_props.curve_fit_similar = self._similar
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
