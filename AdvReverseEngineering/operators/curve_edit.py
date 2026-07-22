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
    bridge_fit_surface_boundaries,
    compose_patch_from_boundary_polylines,
    compose_patch_from_boundary_polylines_ex,
    estimate_open_directed_similarity,
    estimate_similarity_transform,
    extract_mesh_boundary_loop_sides,
    find_break_indices,
    fit_bezier_n_controls,
    opposite_edge_pairs,
    opposite_pair_colors,
    order_open_curves_as_closed_loop,
    pack_boundary_sides,
    sample_bezier_anchor_chain,
    sample_polyline_uniform,
    segment_colors_for_count,
    snap_bezier_endpoints,
    split_polyline_at_breaks,
    stitch_oriented_loop_polylines,
    transform_bezier_points,
    unpack_boundary_sides,
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

FIT_COLLECTION_NAME = "拟合曲线"  # 兼容旧导出名
CURVE_COLLECTION_NAME = "拟合曲线"
SURFACE_COLLECTION_NAME = "拟合曲面"
LEGACY_FIT_COLLECTION_NAME = "拟合面"
CURVE_SEG_MAT_PREFIX = "ARE_CurveSeg_"
DEFAULT_SPLIT_ANGLE = 35.0
SPLIT_ANGLE_STEP = 2.0
SPLIT_ANGLE_MIN = 5.0
SPLIT_ANGLE_MAX = 170.0
DEFAULT_BEZIER_CONTROLS = 3
BEZIER_CONTROLS_MIN = 3
BEZIER_CONTROLS_MAX = 32
SIMILAR_SAMPLE_COUNT = 64
MODAL_TIMER_STEP = 0.1
DEFAULT_SURFACE_SEG_U = 12
DEFAULT_SURFACE_SEG_V = 12


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
    """从拟合结果收集 GPU 预览：曲线 / 锚点 / 手柄。

    payload 项: (obj, fitted, cyclic) 或 (obj, fitted, cyclic, color_id)
    color_id 用于对边同色（0/1）。
    """
    curves: list[dict] = []
    anchors: list[np.ndarray] = []
    handles: list[np.ndarray] = []
    handle_edges: list[np.ndarray] = []
    palette = segment_colors_for_count(max(len(payload), 1))
    pair_palette = opposite_pair_colors()
    for color_index, item in enumerate(payload):
        if len(item) >= 4:
            _obj, fitted, cyclic, color_id = item[0], item[1], item[2], item[3]
            color = pair_palette[int(color_id) % len(pair_palette)]
        else:
            _obj, fitted, cyclic = item[0], item[1], item[2]
            color = palette[color_index % len(palette)]
        if not fitted:
            continue
        sampled = _sample_bezier_polyline(fitted, bool(cyclic))
        if len(sampled) >= 2:
            curves.append(
                {
                    "points": sampled,
                    "color": color,
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


def _ensure_named_collection(
    scene: bpy.types.Scene,
    name: str,
    legacy_names: tuple[str, ...] = (),
) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is not None:
        return col
    for legacy in legacy_names:
        old = bpy.data.collections.get(legacy)
        if old is not None:
            old.name = name
            return old
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    return col


def _ensure_curve_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    return _ensure_named_collection(
        scene,
        CURVE_COLLECTION_NAME,
        legacy_names=(LEGACY_FIT_COLLECTION_NAME,),
    )


def _ensure_surface_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    return _ensure_named_collection(scene, SURFACE_COLLECTION_NAME)


def _ensure_fit_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    """兼容旧调用：曲线写入「拟合曲线」。"""
    return _ensure_curve_collection(scene)


def _any_curve_modal_busy(scene_props) -> bool:
    return bool(
        getattr(scene_props, "curve_split_mode_active", False)
        or getattr(scene_props, "curve_fit_mode_active", False)
        or scene_props.merge_mode_active
        or scene_props.split_mode_active
        or getattr(scene_props, "remove_mode_active", False)
        or getattr(scene_props, "fit_mode_active", False)
    )


def _target_objects(targets) -> list[bpy.types.Object]:
    objs: list[bpy.types.Object] = []
    for item in targets or []:
        obj = item[0] if isinstance(item, tuple) else item
        if _is_object_alive(obj):
            objs.append(obj)
    return objs


def _lock_non_target_select(
    allowed: list[bpy.types.Object],
) -> list[tuple[bpy.types.Object, bool]]:
    """
    禁止点选非目标物体（hide_select），返回需还原的 (obj, old_hide_select)。
    """
    allowed_set = {obj for obj in allowed if _is_object_alive(obj)}
    backups: list[tuple[bpy.types.Object, bool]] = []
    for obj in list(bpy.data.objects):
        try:
            if obj in allowed_set:
                if obj.hide_select:
                    backups.append((obj, True))
                    obj.hide_select = False
            else:
                if not obj.hide_select:
                    backups.append((obj, False))
                    obj.hide_select = True
        except ReferenceError:
            continue
    return backups


def _restore_hide_select(
    backups: list[tuple[bpy.types.Object, bool]] | None,
) -> None:
    for obj, was_hidden in backups or []:
        try:
            if _is_object_alive(obj):
                obj.hide_select = bool(was_hidden)
        except ReferenceError:
            continue


def _force_select_targets(
    context: bpy.types.Context,
    targets: list[bpy.types.Object],
) -> None:
    """强制保持目标物体选中，避免取消选择后卡在模态。"""
    alive = [obj for obj in targets if _is_object_alive(obj)]
    if not alive:
        return
    if context.mode == "EDIT_CURVE":
        selected = set()
        for obj in list(context.selected_objects):
            try:
                selected.add(obj)
            except ReferenceError:
                pass
        if all(obj in selected for obj in alive) and len(selected) == len(alive):
            return
    for obj in list(context.selected_objects):
        try:
            if obj not in alive:
                obj.select_set(False)
        except ReferenceError:
            pass
    for obj in alive:
        try:
            obj.select_set(True)
        except ReferenceError:
            pass
    try:
        active = context.view_layer.objects.active
        if active not in alive:
            context.view_layer.objects.active = alive[0]
    except Exception:
        try:
            context.view_layer.objects.active = alive[0]
        except Exception:
            pass


def _ensure_object_mode(context: bpy.types.Context) -> None:
    if getattr(context, "mode", "OBJECT") == "OBJECT":
        return
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def _enter_curve_edit_mode(
    context: bpy.types.Context,
    targets: list[bpy.types.Object],
) -> bool:
    """多选目标曲线后进入编辑模式，便于拖锚点/手柄。"""
    alive = [
        obj for obj in targets if _is_object_alive(obj) and obj.type == "CURVE"
    ]
    if not alive:
        return False
    _ensure_object_mode(context)
    _force_select_targets(context, alive)
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        return True
    except Exception:
        return False


def _region_is_view3d_window(context: bpy.types.Context) -> bool:
    space = getattr(context, "space_data", None)
    region = getattr(context, "region", None)
    return (
        space is not None
        and getattr(space, "type", "") == "VIEW_3D"
        and region is not None
        and getattr(region, "type", "") == "WINDOW"
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
            "Ctrl+滚轮调节 · Enter 确认 · Esc 取消 · 禁止改选物体"
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
            # 保留拟合领域写入的显示色，供后续合成曲面同步
            try:
                region_display_color = tuple(float(v) for v in obj.color[:4])
            except Exception:
                region_display_color = (1.0, 1.0, 1.0, 1.0)
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
                obj.color = region_display_color
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
            obj.color = region_display_color
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
                new_obj.color = region_display_color
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
            _ensure_object_mode(context)
            if restore:
                _restore_curve_backups(getattr(self, "_backups", None))
        finally:
            _restore_hide_select(getattr(self, "_hide_select_backups", None))
            self._hide_select_backups = []
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
        target_objs = _target_objects(self._targets)
        self._hide_select_backups = _lock_non_target_select(target_objs)
        _force_select_targets(context, target_objs)
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
        target_objs = _target_objects(getattr(self, "_targets", []))
        # 目标曲线被删除/撤销后立刻退出，并清预览
        if not target_objs:
            self.report({"WARNING"}, "目标曲线已丢失，已退出拆分")
            self._cleanup(context, restore=False)
            return {"CANCELLED"}

        # 始终保持目标选中；禁止空选/点选其他物体导致无法退出
        _force_select_targets(context, target_objs)

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

        # 拦截视口内点选/框选，避免取消选择；侧栏 UI 仍可点确认
        if event.type in {"LEFTMOUSE", "A"} and event.value == "PRESS":
            if _region_is_view3d_window(context):
                _force_select_targets(context, target_objs)
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
    """预览 3/4 条贝塞尔边界，确认后直接生成曲面。"""

    bl_idname = "are.fit_bezier_curve"
    bl_label = "拟合曲面"
    bl_description = (
        "预览选中的 3/4 条贝塞尔边界，可在编辑模式拖动锚点/手柄微调，"
        "确认后生成曲面并删除边界曲线；"
        "四条闭合对边时：Ctrl/Shift 滚轮分别调两组控制点数，"
        "相似模式对边两两相似（保持环向首尾），对边同色预览；"
        "缝合开口(V)沿切向延伸端点至交点封闭缺口"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _clamp_controls(self, value: int) -> int:
        return int(
            max(BEZIER_CONTROLS_MIN, min(BEZIER_CONTROLS_MAX, int(value)))
        )

    def _update_status(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        na = int(getattr(self, "_controls_a", DEFAULT_BEZIER_CONTROLS))
        nb = int(getattr(self, "_controls_b", DEFAULT_BEZIER_CONTROLS))
        similar = bool(getattr(self, "_similar", False))
        stitch = bool(getattr(self, "_stitch_open", False))
        mode = "开" if similar else "关"
        stitch_s = "开" if stitch else "关"
        quad = bool(getattr(self, "_quad_loop_active", False))
        edit_hint = " · 可拖锚点/手柄微调 · Tab切编辑"
        if quad:
            extra = " · 对边两两相似·封闭" if similar else " · 四边形对边封闭"
            if stitch:
                extra += " · 已缝合开口"
            text = (
                f"拟合曲面预览：组A {na} · 组B {nb} · 相似 {mode} · 缝合 {stitch_s}"
                f"{extra}{edit_hint} · Ctrl滚轮A · Shift滚轮B · S相似 · V缝合 · Enter确认"
            )
        else:
            text = (
                f"拟合曲面预览：控制点 {na} · 相似模式 {mode}"
                f"{edit_hint} · Ctrl+滚轮调点数 · S相似 · Enter确认 · Esc取消"
            )
        scene_props.curve_fit_status = text
        _header(context, text)

    def _sync_preview_from_live_curves(self, context: bpy.types.Context) -> None:
        """从当前贝塞尔物体同步 GPU 预览（用户拖动手柄后）。"""
        payload: list[tuple] = []
        for obj, _sources in getattr(self, "_targets", []):
            if not _is_object_alive(obj) or obj.type != "CURVE" or obj.data is None:
                continue
            pair_id = int(obj.get("are_fit_opposite_pair", -1))
            for spline in obj.data.splines:
                if spline.type != "BEZIER" or len(spline.bezier_points) < 2:
                    continue
                fitted = _bezier_dicts_from_spline(spline, obj.matrix_world)
                cyclic = bool(spline.use_cyclic_u)
                if pair_id >= 0:
                    payload.append((obj, fitted, cyclic, pair_id))
                else:
                    payload.append((obj, fitted, cyclic))
        self._preview_payload = payload
        set_curve_bezier_preview(_bezier_preview_from_payload(payload))
        _tag_redraw(context)

    def _refit_and_enter_edit(self, context: bpy.types.Context) -> bool:
        """退出编辑 → 重拟合 → 再进入编辑模式。"""
        _ensure_object_mode(context)
        ok = self._rebuild_preview(context)
        if ok:
            _enter_curve_edit_mode(context, _target_objects(self._targets))
        return ok

    def _fit_one(
        self,
        points: np.ndarray,
        cyclic: bool,
        control_count: int | None = None,
    ) -> list[dict]:
        n = (
            int(self._controls_a)
            if control_count is None
            else int(control_count)
        )
        return fit_bezier_n_controls(
            points,
            control_count=n,
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

        相似变换严格保持环向首尾对应（禁止反向对齐），避免对边扭曲。
        缝合开口：沿切向延伸缺口两端至交点后再拟合。
        """
        polylines = [source["points"] for _obj, source in edges]
        stitch = bool(getattr(self, "_stitch_open", False))
        ordered = order_open_curves_as_closed_loop(
            polylines,
            allow_large_gaps=stitch,
        )
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

        if stitch:
            stitched_pts, stitch_count = stitch_oriented_loop_polylines(
                [pts for _obj, pts in loop_items]
            )
            self._stitch_count = int(stitch_count)
            loop_items = [
                (loop_items[i][0], stitched_pts[i])
                for i in range(len(loop_items))
            ]
        else:
            self._stitch_count = 0

        counts = [
            self._clamp_controls(self._controls_a),
            self._clamp_controls(self._controls_b),
        ]
        beziers: list[list[dict] | None] = [None, None, None, None]
        pair_ids = [0, 1, 0, 1]
        if similar:
            for pair_id, (a, b) in enumerate(opposite_edge_pairs(4)):
                n_ctrl = counts[pair_id]
                len_a = self._polyline_length(loop_items[a][1], False)
                len_b = self._polyline_length(loop_items[b][1], False)
                if len_a >= len_b:
                    ref_i, dst_i = a, b
                else:
                    ref_i, dst_i = b, a
                ref_pts = loop_items[ref_i][1]
                dst_pts = loop_items[dst_i][1]
                prototype = self._fit_one(ref_pts, False, n_ctrl)
                ref_samples = sample_polyline_uniform(
                    ref_pts, SIMILAR_SAMPLE_COUNT, cyclic=False
                )
                dst_samples = sample_polyline_uniform(
                    dst_pts, SIMILAR_SAMPLE_COUNT, cyclic=False
                )
                # 环向已统一：禁止 reverse，否则首尾对调导致扭曲
                scale, rotation, translation = estimate_open_directed_similarity(
                    ref_samples, dst_samples
                )
                fitted = transform_bezier_points(
                    prototype, scale, rotation, translation
                )
                fitted = snap_bezier_endpoints(
                    fitted, dst_pts[0], dst_pts[-1]
                )
                beziers[ref_i] = prototype
                beziers[dst_i] = fitted
                loop_items[ref_i][0]["are_fit_opposite_pair"] = int(pair_id)
                loop_items[dst_i][0]["are_fit_opposite_pair"] = int(pair_id)
        else:
            for index, (_obj, pts) in enumerate(loop_items):
                pair_id = pair_ids[index]
                beziers[index] = self._fit_one(pts, False, counts[pair_id])

        if any(item is None for item in beziers):
            return False
        welded = weld_bezier_loop_endpoints(beziers)
        group_id = f"quad_{loop_items[0][0].name}"
        for index, (obj, _pts) in enumerate(loop_items):
            fitted = welded[index]
            pair_id = pair_ids[index]
            _write_bezier_spline(obj, fitted, cyclic=False)
            obj["are_fit_kind"] = "contour_bezier"
            obj["are_fit_quad_loop"] = True
            obj["are_fit_opposite_pair"] = int(pair_id)
            obj["are_fit_similar"] = bool(similar)
            obj["are_fit_stitch_open"] = bool(stitch)
            if similar:
                obj["are_fit_similar_group"] = group_id
            self._preview_payload.append((obj, fitted, False, pair_id))
        self._quad_loop_active = True
        return True

    def _rebuild_preview(self, context: bpy.types.Context) -> bool:
        similar = bool(getattr(self, "_similar", False))
        self._preview_payload = []
        self._quad_loop_active = False
        # 兼容旧逻辑：非四边形时用组 A 作为统一控制点数
        self._controls = self._clamp_controls(self._controls_a)
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
        _ensure_object_mode(context)
        # 确认前用当前（可能已手动微调）贝塞尔同步一次
        self._sync_preview_from_live_curves(context)
        curves = [
            obj
            for obj, _sources in getattr(self, "_targets", [])
            if _is_object_alive(obj)
        ]
        if len(curves) not in (3, 4):
            self.report({"ERROR"}, "拟合曲面需要 3 或 4 条有效边界曲线")
            return False

        try:
            obj, kind = _compose_surface_from_curves(context, curves)
        except RegionFitError as exc:
            self.report({"ERROR"}, str(exc))
            return False
        except Exception as exc:
            self.report({"ERROR"}, f"生成拟合曲面失败: {exc}")
            return False

        # 曲面已生成；边界曲线不再保留。
        _delete_curve_objects(curves)

        label = "四边" if kind == "QUAD" else "三边"
        scene_props.curve_fit_status = (
            f"已生成{label}拟合曲面 →「{SURFACE_COLLECTION_NAME}」/{obj.name}"
            "（边界曲线已删除）"
        )
        self.report({"INFO"}, scene_props.curve_fit_status)
        return True

    def _cleanup(self, context: bpy.types.Context, restore: bool) -> None:
        # 无论还原是否成功，都必须清掉 GPU 预览，避免残影
        try:
            _ensure_object_mode(context)
            if restore:
                _restore_curve_backups(getattr(self, "_backups", None))
        finally:
            _restore_hide_select(getattr(self, "_hide_select_backups", None))
            self._hide_select_backups = []
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
        return len(_selected_curve_objects(context)) in (3, 4)

    def invoke(self, context: bpy.types.Context, event):
        curves = _selected_curve_objects(context)
        if len(curves) not in (3, 4):
            self.report({"ERROR"}, "请先选中 3 或 4 条边界曲线")
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
        if len(self._targets) not in (3, 4):
            self.report({"ERROR"}, "请先选中 3 或 4 条边界曲线")
            return {"CANCELLED"}

        self._controls_a = self._clamp_controls(
            getattr(scene_props, "curve_fit_controls", DEFAULT_BEZIER_CONTROLS)
        )
        self._controls_b = self._clamp_controls(
            getattr(
                scene_props,
                "curve_fit_controls_b",
                getattr(
                    scene_props, "curve_fit_controls", DEFAULT_BEZIER_CONTROLS
                ),
            )
        )
        self._controls = self._controls_a
        self._similar = bool(getattr(scene_props, "curve_fit_similar", False))
        self._stitch_open = bool(
            getattr(scene_props, "curve_fit_stitch_open", False)
        )
        self._stitch_count = 0
        self._committed = False
        self._preview_payload = []
        self._quad_loop_active = False
        target_objs = _target_objects(self._targets)
        self._hide_select_backups = _lock_non_target_select(target_objs)
        _force_select_targets(context, target_objs)
        scene_props.curve_fit_mode_active = True
        scene_props.curve_fit_confirm_requested = False
        scene_props.curve_fit_controls = self._controls_a
        scene_props.curve_fit_controls_b = self._controls_b
        scene_props.curve_fit_similar = self._similar
        scene_props.curve_fit_stitch_open = self._stitch_open
        clear_curve_split_preview()
        register_curve_tool_hud()
        try:
            self._refit_and_enter_edit(context)
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
        target_objs = _target_objects(getattr(self, "_targets", []))
        # 目标曲线被删除/撤销后立刻退出，并清预览
        if len(target_objs) not in (3, 4):
            self.report({"WARNING"}, "目标曲线已丢失，已退出拟合")
            self._cleanup(context, restore=False)
            return {"CANCELLED"}

        _force_select_targets(context, target_objs)

        if scene_props.curve_fit_confirm_requested:
            scene_props.curve_fit_confirm_requested = False
            if self._commit(context):
                self._committed = True
                self._cleanup(context, restore=False)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.type in {"ESC"} and event.value == "PRESS":
            self.report({"INFO"}, "已取消拟合曲面")
            self._cleanup(context, restore=True)
            return {"CANCELLED"}

        # 编辑模式中右键用于取消选择控制点，不退出工具
        if (
            event.type == "RIGHTMOUSE"
            and event.value == "PRESS"
            and context.mode != "EDIT_CURVE"
        ):
            self.report({"INFO"}, "已取消拟合曲面")
            self._cleanup(context, restore=True)
            return {"CANCELLED"}

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if self._commit(context):
                self._committed = True
                self._cleanup(context, restore=False)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.type == "TIMER":
            # 用户拖动手柄后刷新预览叠加
            if context.mode == "EDIT_CURVE":
                self._sync_preview_from_live_curves(context)
            else:
                _force_select_targets(context, target_objs)
            return {"PASS_THROUGH"}

        if event.type == "S" and event.value == "PRESS" and not event.ctrl:
            self._similar = not self._similar
            scene_props.curve_fit_similar = self._similar
            try:
                self._refit_and_enter_edit(context)
            except RegionFitError as exc:
                self.report({"ERROR"}, str(exc))
            return {"RUNNING_MODAL"}

        if event.type == "V" and event.value == "PRESS" and not event.ctrl:
            self._stitch_open = not self._stitch_open
            scene_props.curve_fit_stitch_open = self._stitch_open
            try:
                self._refit_and_enter_edit(context)
            except RegionFitError as exc:
                self.report({"ERROR"}, str(exc))
            return {"RUNNING_MODAL"}

        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            delta = 1 if event.type == "WHEELUPMOUSE" else -1
            # Shift（无 Ctrl）：调组 B；Ctrl：调组 A（非四边形时即统一点数）
            if event.shift and not event.ctrl:
                self._controls_b = self._clamp_controls(self._controls_b + delta)
                scene_props.curve_fit_controls_b = self._controls_b
                try:
                    self._refit_and_enter_edit(context)
                except RegionFitError as exc:
                    self.report({"ERROR"}, str(exc))
                return {"RUNNING_MODAL"}
            if event.ctrl:
                self._controls_a = self._clamp_controls(self._controls_a + delta)
                self._controls = self._controls_a
                scene_props.curve_fit_controls = self._controls_a
                try:
                    self._refit_and_enter_edit(context)
                except RegionFitError as exc:
                    self.report({"ERROR"}, str(exc))
                return {"RUNNING_MODAL"}

        panel_a = self._clamp_controls(
            getattr(scene_props, "curve_fit_controls", self._controls_a)
        )
        panel_b = self._clamp_controls(
            getattr(scene_props, "curve_fit_controls_b", self._controls_b)
        )
        panel_sim = bool(getattr(scene_props, "curve_fit_similar", self._similar))
        panel_stitch = bool(
            getattr(scene_props, "curve_fit_stitch_open", self._stitch_open)
        )
        if (
            panel_a != self._controls_a
            or panel_b != self._controls_b
            or panel_sim != self._similar
            or panel_stitch != self._stitch_open
        ):
            self._controls_a = panel_a
            self._controls_b = panel_b
            self._controls = self._controls_a
            self._similar = panel_sim
            self._stitch_open = panel_stitch
            try:
                self._refit_and_enter_edit(context)
            except RegionFitError as exc:
                self.report({"ERROR"}, str(exc))
            return {"RUNNING_MODAL"}

        # 物体模式下拦截点选其他物体；编辑模式放行以拖锚点/手柄
        if (
            context.mode != "EDIT_CURVE"
            and event.type in {"LEFTMOUSE", "A"}
            and event.value == "PRESS"
            and _region_is_view3d_window(context)
        ):
            _force_select_targets(context, target_objs)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def cancel(self, context: bpy.types.Context):
        self._cleanup(
            context, restore=not getattr(self, "_committed", False)
        )
        return {"CANCELLED"}


class ARE_OT_confirm_fit_bezier_curve(bpy.types.Operator):
    bl_idname = "are.confirm_fit_bezier_curve"
    bl_label = "确认生成拟合曲面"
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


def _bezier_dicts_from_spline(spline, matrix_world) -> list[dict]:
    """把 Blender BEZIER spline 转为世界坐标锚点字典列表。"""
    matrix = _matrix_np(matrix_world)
    items: list[dict] = []
    for bp in spline.bezier_points:
        local = np.array(
            [
                [bp.co.x, bp.co.y, bp.co.z],
                [bp.handle_left.x, bp.handle_left.y, bp.handle_left.z],
                [bp.handle_right.x, bp.handle_right.y, bp.handle_right.z],
            ],
            dtype=np.float64,
        )
        world = _world_points_from_local(matrix, local)
        items.append(
            {
                "co": world[0].copy(),
                "handle_left": world[1].copy(),
                "handle_right": world[2].copy(),
            }
        )
    return items


def _boundary_polyline_from_curve_obj(obj: bpy.types.Object) -> np.ndarray:
    """从曲线物体提取一条开环边界折线（优先 BEZIER 采样）。"""
    curve = obj.data
    if curve is None or not curve.splines:
        raise RegionFitError(f"{obj.name} 没有样条")
    # 多段时取最长开环
    candidates: list[np.ndarray] = []
    for spline in curve.splines:
        cyclic = bool(spline.use_cyclic_u)
        if spline.type == "BEZIER" and len(spline.bezier_points) >= 2:
            bez = _bezier_dicts_from_spline(spline, obj.matrix_world)
            pts = sample_bezier_anchor_chain(
                bez, cyclic=False, samples_per_span=24
            )
            # 闭环轮廓不适合作为单边；强制开环用途时去掉环标志
            if cyclic and len(pts) > 2:
                # 仍可作为边界边使用：去掉重复闭合点
                if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
                    pts = pts[:-1]
            candidates.append(pts)
            continue
        local_pts: list[tuple[float, float, float]] = []
        for point in spline.points:
            local_pts.append(
                (float(point.co.x), float(point.co.y), float(point.co.z))
            )
        if len(local_pts) < 2:
            continue
        world = _world_points_from_local(
            obj.matrix_world, np.asarray(local_pts, dtype=np.float64)
        )
        candidates.append(world)
    if not candidates:
        raise RegionFitError(f"{obj.name} 无法提取边界折线")
    candidates.sort(
        key=lambda pts: float(
            np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        )
        if len(pts) >= 2
        else 0.0,
        reverse=True,
    )
    return candidates[0]


def _rgba_from_object_color(obj: bpy.types.Object) -> tuple[float, float, float, float] | None:
    """读取物体 Viewport Display 颜色；近白默认色视为未设置。"""
    try:
        color = tuple(float(v) for v in obj.color[:4])
    except Exception:
        return None
    if len(color) < 3:
        return None
    # 拟合曲线生成时会写入领域色；未写入时多为 (1,1,1,*)
    if color[0] > 0.98 and color[1] > 0.98 and color[2] > 0.98:
        return None
    alpha = float(color[3]) if len(color) > 3 else 0.85
    return (color[0], color[1], color[2], max(alpha, 0.75))


def _resolve_region_display_color(
    curves: list[bpy.types.Object],
) -> tuple[float, float, float, float]:
    """
    从边界曲线同步领域显示色。

    优先用曲线已写入的 ``obj.color``（拟合领域时从源网格调色板拷贝），
    再回退到源网格 ``are_region_colors`` / 算法调色板。
    """
    region_id = -1
    source_name = ""
    curve_color = None
    for curve in curves:
        try:
            rid = int(curve.get("are_fit_region_id", -1))
        except Exception:
            rid = -1
        if rid < 0:
            continue
        region_id = rid
        source_name = str(curve.get("are_fit_source", "") or "")
        curve_color = _rgba_from_object_color(curve)
        if curve_color is not None:
            return curve_color
        break

    if region_id < 0:
        # 无领域标记时仍尝试继承任一条曲线的显示色
        for curve in curves:
            curve_color = _rgba_from_object_color(curve)
            if curve_color is not None:
                return curve_color
        return (0.35, 0.75, 1.0, 0.85)

    palette = None
    source_obj = bpy.data.objects.get(source_name) if source_name else None
    if source_obj is not None:
        # 与领域叠加相同：优先读源网格调色板
        try:
            from .regions import _read_region_colors

            palette = _read_region_colors(source_obj, max(region_id + 1, 1))
        except Exception:
            raw = source_obj.get("are_region_colors")
            if raw is not None:
                flat = np.asarray(list(raw), dtype=np.float32)
                if len(flat) >= (region_id + 1) * 4:
                    palette = flat.reshape(-1, 4)

    if palette is None:
        from ..algorithms.regions import generate_region_colors

        palette = generate_region_colors(max(region_id + 1, 1), alpha=0.85)

    color = palette[int(region_id) % len(palette)]
    rgba = (
        float(color[0]),
        float(color[1]),
        float(color[2]),
        float(color[3]) if len(color) > 3 else 0.85,
    )
    return (rgba[0], rgba[1], rgba[2], max(float(rgba[3]), 0.75))


def _ensure_surface_material(
    name: str,
    color: tuple[float, float, float, float],
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    # Solid/Material 模式下的回退色；真正可视主要靠 GPU 叠加（同领域分色）
    opaque = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    mat.diffuse_color = opaque
    try:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = opaque
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 1.0
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    except Exception:
        pass
    try:
        mat.blend_method = "OPAQUE"
    except Exception:
        pass
    return mat


def _apply_surface_region_color(
    obj: bpy.types.Object,
    color: tuple[float, float, float, float],
) -> None:
    """给合成曲面写入领域色（自定义属性 + 物体色 + 材质回退）。"""
    rgba = (
        float(color[0]),
        float(color[1]),
        float(color[2]),
        max(float(color[3]) if len(color) > 3 else 0.85, 0.75),
    )
    # 视口 GPU 叠加读取此属性（与 are_region_colors 同思路）
    obj["are_fit_display_color"] = list(rgba)
    obj.color = rgba
    try:
        obj.show_transparent = True
    except Exception:
        pass
    mat_name = f"ARE_FitSurf_{obj.name}"
    mat = _ensure_surface_material(mat_name, rgba)
    mesh = obj.data
    if mesh is None:
        return
    mesh.materials.clear()
    mesh.materials.append(mat)


def _hide_curve_collection(scene: bpy.types.Scene) -> None:
    """合成曲面后隐藏「拟合曲线」集合。"""
    col = bpy.data.collections.get(CURVE_COLLECTION_NAME)
    if col is None:
        col = bpy.data.collections.get(LEGACY_FIT_COLLECTION_NAME)
    if col is None:
        return
    col.hide_viewport = True
    try:
        layer = scene.view_layers[0].layer_collection
        stack = [layer]
        while stack:
            lc = stack.pop()
            if lc.collection == col:
                lc.hide_viewport = True
                break
            stack.extend(list(lc.children))
    except Exception:
        pass


def _delete_curve_objects(curves: list[bpy.types.Object]) -> None:
    """删除边界曲线物体及其无主数据块。"""
    for curve_obj in curves:
        try:
            curve_data = curve_obj.data
            bpy.data.objects.remove(curve_obj, do_unlink=True)
            if curve_data is not None and getattr(curve_data, "users", 0) == 0:
                bpy.data.curves.remove(curve_data)
        except (ReferenceError, RuntimeError):
            pass


def _compose_surface_from_curves(
    context: bpy.types.Context,
    curves: list[bpy.types.Object],
) -> tuple[bpy.types.Object, str]:
    """
    由 3/4 条边界曲线生成拟合曲面网格。

    Returns:
        (surface_object, kind)  kind 为 ``QUAD`` 或 ``TRI``。
    """
    if len(curves) not in (3, 4):
        raise RegionFitError("拟合曲面需要选中 3 或 4 条曲线")
    polylines = [_boundary_polyline_from_curve_obj(obj) for obj in curves]
    vertices, faces, kind, loop_sides = compose_patch_from_boundary_polylines_ex(
        polylines,
        segments_u=DEFAULT_SURFACE_SEG_U,
        segments_v=DEFAULT_SURFACE_SEG_V,
    )

    collection = _ensure_surface_collection(context.scene)
    base = curves[0].name.rsplit("_S", 1)[0]
    name = f"FitSurf_{base}_{kind}"
    # 世界坐标写入，物体用单位矩阵
    identity = curves[0].matrix_world.copy()
    identity.identity()
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [tuple(map(float, v)) for v in vertices.tolist()],
        [],
        [tuple(face) for face in faces],
    )
    mesh.update()
    try:
        mesh.calc_normals()
    except Exception:
        pass
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = identity
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    # 避免同时挂在场景根集合造成重复
    scene_col = context.scene.collection
    if obj.name in scene_col.objects and collection != scene_col:
        try:
            scene_col.objects.unlink(obj)
        except Exception:
            pass

    region_color = _resolve_region_display_color(curves)
    _apply_surface_region_color(obj, region_color)

    obj["are_fit_kind"] = "region_surface"
    obj["are_fit_surface_type"] = kind
    obj["are_fit_source_curves"] = [c.name for c in curves]
    # 持久化定向边界，供后续「桥接曲面」沿贝塞尔曲率连接
    flat, counts = pack_boundary_sides(loop_sides)
    obj["are_fit_boundary_flat"] = flat
    obj["are_fit_boundary_counts"] = counts
    try:
        obj["are_fit_region_id"] = int(curves[0].get("are_fit_region_id", -1))
    except Exception:
        obj["are_fit_region_id"] = -1
    source_name = str(curves[0].get("are_fit_source", "") or "")
    if source_name:
        obj["are_fit_source"] = source_name
    obj.display_type = "SOLID"
    obj.show_wire = True

    for item in list(context.selected_objects):
        try:
            item.select_set(False)
        except ReferenceError:
            pass
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj, kind


def _is_fit_surface_object(obj: bpy.types.Object | None) -> bool:
    if obj is None:
        return False
    try:
        return (
            obj.type == "MESH"
            and str(obj.get("are_fit_kind", "") or "") == "region_surface"
        )
    except ReferenceError:
        return False


def _selected_fit_surfaces(context: bpy.types.Context) -> list[bpy.types.Object]:
    return [
        obj
        for obj in list(context.selected_objects)
        if _is_fit_surface_object(obj)
    ]


def _fit_surface_boundary_sides(obj: bpy.types.Object) -> list[np.ndarray]:
    """读取持久化边界；缺失时从网格开放边界回退提取。"""
    flat = obj.get("are_fit_boundary_flat")
    counts = obj.get("are_fit_boundary_counts")
    if flat is not None and counts is not None:
        sides = unpack_boundary_sides(flat, counts)
        if len(sides) >= 3:
            # 变换到世界坐标（合成时多为单位矩阵，仍统一处理）
            matrix = np.asarray(obj.matrix_world, dtype=np.float64)
            world_sides = []
            for side in sides:
                homo = np.hstack(
                    (side, np.ones((len(side), 1), dtype=np.float64))
                )
                world = (matrix @ homo.T).T[:, :3]
                world_sides.append(world)
            return world_sides

    mesh = obj.data
    if mesh is None:
        raise RegionFitError(f"{obj.name} 没有网格数据")
    # 世界坐标顶点
    n = len(mesh.vertices)
    local = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", local)
    local = local.reshape(n, 3)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homo = np.hstack((local, np.ones((n, 1), dtype=np.float64)))
    world = (matrix @ homo.T).T[:, :3]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    return extract_mesh_boundary_loop_sides(world, faces)


def _blend_surface_colors(
    objs: list[bpy.types.Object],
) -> tuple[float, float, float, float]:
    colors = []
    for obj in objs:
        raw = obj.get("are_fit_display_color")
        if raw is not None:
            try:
                values = [float(v) for v in list(raw)[:4]]
                if len(values) >= 3:
                    alpha = values[3] if len(values) > 3 else 0.85
                    colors.append(
                        (values[0], values[1], values[2], max(alpha, 0.75))
                    )
                    continue
            except Exception:
                pass
        got = _rgba_from_object_color(obj)
        if got is not None:
            colors.append(got)
    if not colors:
        return (0.35, 0.75, 1.0, 0.85)
    arr = np.asarray(colors, dtype=np.float64)
    mean = arr.mean(axis=0)
    return (
        float(mean[0]),
        float(mean[1]),
        float(mean[2]),
        max(float(mean[3]), 0.75),
    )


class ARE_OT_bridge_fit_surfaces(bpy.types.Operator):
    """将两张拟合曲面沿最近对边做贝塞尔曲率桥接。"""

    bl_idname = "are.bridge_fit_surfaces"
    bl_label = "桥接曲面"
    bl_description = (
        "选中两张「拟合曲面」，自动找最近对边，"
        "用继承邻边切向的贝塞尔连接线做 Coons 桥接，写入「拟合曲面」集合"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None:
            return False
        if _any_curve_modal_busy(scene_props):
            return False
        return len(_selected_fit_surfaces(context)) == 2

    def execute(self, context: bpy.types.Context):
        surfaces = _selected_fit_surfaces(context)
        if len(surfaces) != 2:
            self.report({"ERROR"}, "请选中两张拟合曲面")
            return {"CANCELLED"}
        try:
            sides_a = _fit_surface_boundary_sides(surfaces[0])
            sides_b = _fit_surface_boundary_sides(surfaces[1])
            vertices, faces = bridge_fit_surface_boundaries(
                sides_a,
                sides_b,
                segments_u=DEFAULT_SURFACE_SEG_U,
                segments_v=DEFAULT_SURFACE_SEG_V,
            )
        except RegionFitError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"桥接曲面失败: {exc}")
            return {"CANCELLED"}

        collection = _ensure_surface_collection(context.scene)
        name = f"FitBridge_{surfaces[0].name}_{surfaces[1].name}"
        identity = surfaces[0].matrix_world.copy()
        identity.identity()
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(
            [tuple(map(float, v)) for v in vertices.tolist()],
            [],
            [tuple(face) for face in faces],
        )
        mesh.update()
        try:
            mesh.calc_normals()
        except Exception:
            pass
        obj = bpy.data.objects.new(name, mesh)
        obj.matrix_world = identity
        if obj.name not in collection.objects:
            collection.objects.link(obj)
        scene_col = context.scene.collection
        if obj.name in scene_col.objects and collection != scene_col:
            try:
                scene_col.objects.unlink(obj)
            except Exception:
                pass

        color = _blend_surface_colors(surfaces)
        _apply_surface_region_color(obj, color)
        obj["are_fit_kind"] = "region_surface"
        obj["are_fit_surface_type"] = "BRIDGE"
        obj["are_fit_bridge_of"] = [surfaces[0].name, surfaces[1].name]
        try:
            obj["are_fit_region_id"] = int(
                surfaces[0].get("are_fit_region_id", -1)
            )
        except Exception:
            obj["are_fit_region_id"] = -1
        source_name = str(surfaces[0].get("are_fit_source", "") or "")
        if source_name:
            obj["are_fit_source"] = source_name
        # 桥接面也写入四边边界，便于继续桥接
        try:
            bridge_sides = extract_mesh_boundary_loop_sides(vertices, faces)
            flat, counts = pack_boundary_sides(bridge_sides)
            obj["are_fit_boundary_flat"] = flat
            obj["are_fit_boundary_counts"] = counts
        except Exception:
            pass
        obj.display_type = "SOLID"
        obj.show_wire = True

        for item in list(context.selected_objects):
            try:
                item.select_set(False)
            except ReferenceError:
                pass
        obj.select_set(True)
        context.view_layer.objects.active = obj
        self.report(
            {"INFO"},
            f"已桥接曲面 →「{SURFACE_COLLECTION_NAME}」/{obj.name}",
        )
        return {"FINISHED"}


class ARE_OT_compose_region_surface(bpy.types.Operator):
    """将选中的 3/4 条贝塞尔边界拟合成区面网格（内部/兼容入口）。"""

    bl_idname = "are.compose_region_surface"
    bl_label = "合成区面"
    bl_description = (
        "选中 3 或 4 条端点近乎闭合的贝塞尔曲线，"
        "用 Coons/三边规则网格合成区面，写入「拟合曲面」集合；"
        "沿用领域显示色，并隐藏「拟合曲线」集合"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None:
            return False
        if _any_curve_modal_busy(scene_props):
            return False
        curves = _selected_curve_objects(context)
        return len(curves) in (3, 4)

    def execute(self, context: bpy.types.Context):
        curves = _selected_curve_objects(context)
        if len(curves) not in (3, 4):
            self.report({"ERROR"}, "请选中 3 或 4 条曲线")
            return {"CANCELLED"}
        try:
            obj, kind = _compose_surface_from_curves(context, curves)
        except RegionFitError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"合成区面失败: {exc}")
            return {"CANCELLED"}

        _hide_curve_collection(context.scene)
        label = "四边" if kind == "QUAD" else "三边"
        self.report(
            {"INFO"},
            f"已合成{label}区面 →「{SURFACE_COLLECTION_NAME}」/{obj.name}"
            f"（已隐藏「{CURVE_COLLECTION_NAME}」）",
        )
        return {"FINISHED"}


__all__ = (
    "ARE_OT_split_fit_curve",
    "ARE_OT_confirm_split_fit_curve",
    "ARE_OT_fit_bezier_curve",
    "ARE_OT_confirm_fit_bezier_curve",
    "ARE_OT_compose_region_surface",
    "ARE_OT_bridge_fit_surfaces",
    "CURVE_COLLECTION_NAME",
    "SURFACE_COLLECTION_NAME",
    "FIT_COLLECTION_NAME",
)
