# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""底面紫色高亮与领域多色视口叠加绘制。"""

from __future__ import annotations

import time

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from ..registration import SCENE_PROP_NAME

# 视口绘制句柄
_DRAW_HANDLE = None
_LABEL_DRAW_HANDLE = None
_SPLIT_DRAW_HANDLE = None
DRAW_HANDLE_KEY = "AdvReverseEngineering.bottom_draw_handle"
LABEL_DRAW_HANDLE_KEY = "AdvReverseEngineering.region_label_draw_handle"
SPLIT_DRAW_HANDLE_KEY = "AdvReverseEngineering.split_stroke_draw_handle"
OVERLAY_UI_KEY = "AdvReverseEngineering.overlay_ui_callback"

# 世界坐标 / 颜色缓存：signature 记录矩阵、网格与领域版本。
_BOTTOM_CACHE: dict[int, dict] = {}
_REGION_CACHE: dict[int, dict] = {}
# 合并预览缓存：按 (对象指针, 预览版本) 复用三角缓冲，避免每帧重建。
_MERGE_PREVIEW_CACHE: dict = {}
_HIGHLIGHT_CACHE: dict = {}

# 合并模式标签会话（由 merge operator 写入）
_MERGE_LABEL_SESSION: dict | None = None
# 拟合边界内角标注会话
_FIT_ANGLE_LABEL_SESSION: dict | None = None
# 拆分模式笔迹会话
_SPLIT_STROKE_SESSION: dict | None = None
# 曲线拆分/拟合工具左上角 HUD
_CURVE_TOOL_HUD: str | None = None
_CURVE_TOOL_HUD_HANDLE = None
_CURVE_SPLIT_PREVIEW_HANDLE = None
_CURVE_SPLIT_PREVIEW: dict | None = None
_CURVE_BEZIER_PREVIEW: dict | None = None
CURVE_TOOL_HUD_KEY = "AdvReverseEngineering.curve_tool_hud_handle"
CURVE_SPLIT_PREVIEW_KEY = "AdvReverseEngineering.curve_split_preview_handle"
# 视口左下角，避开顶部 User Perspective / 物体名
_CURVE_HUD_BOTTOM_MARGIN_PX = 18.0
_CURVE_HUD_LEFT_MARGIN_PX = 16.0

# 紫色半透明 (R, G, B, A)
HIGHLIGHT_COLOR = (0.78, 0.22, 1.0, 0.55)
ANCHOR_HIGHLIGHT_COLOR = (1.0, 0.55, 0.12, 0.42)
HOVER_HIGHLIGHT_COLOR = (0.35, 0.75, 1.0, 0.28)
PAINT_FACE_COLOR = (1.0, 0.12, 0.08, 0.55)
LABEL_RADIUS_PX = 16.0
ANGLE_LABEL_RADIUS_PX = 18.0
LABEL_OFFSET_Y = 28.0
LEADER_ENDPOINT_RADIUS_PX = 3.5
_ULTRA_REFLEX_DEG = 355.0
# 覆盖层沿法线外移比例（相对包围盒对角线），消除共面 Z-fighting。
OVERLAY_NORMAL_OFFSET_RATIO = 0.0015
OVERLAY_NORMAL_OFFSET_MIN = 1e-4

# 对象自定义属性键
BOTTOM_FACES_ATTR = "are_bottom_faces"
REGION_ID_ATTR = "are_region_id"
REGION_VERSION_ATTR = "are_region_version"
REGION_COLORS_ATTR = "are_region_colors"


def set_bottom_face_highlight(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    face_indices: list[int],
) -> None:
    """保存底面索引并触发视口重绘。"""
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    old_obj = scene_props.highlight_object
    if old_obj is not None and old_obj != obj:
        if BOTTOM_FACES_ATTR in old_obj:
            del old_obj[BOTTOM_FACES_ATTR]
        _BOTTOM_CACHE.pop(old_obj.as_pointer(), None)

    obj[BOTTOM_FACES_ATTR] = list(face_indices)
    _BOTTOM_CACHE.pop(obj.as_pointer(), None)
    scene_props.highlight_object = obj
    _tag_view3d_redraw(context)


def clear_bottom_face_highlight(context: bpy.types.Context) -> None:
    """清除底面高亮。"""
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    old_obj = scene_props.highlight_object
    if old_obj and BOTTOM_FACES_ATTR in old_obj:
        del old_obj[BOTTOM_FACES_ATTR]
        _BOTTOM_CACHE.pop(old_obj.as_pointer(), None)
    scene_props.highlight_object = None
    _tag_view3d_redraw(context)


def set_region_highlight(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    region_ids: np.ndarray,
    colors: np.ndarray,
) -> None:
    """刷新领域覆盖层缓存并触发重绘。"""
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    old_obj = scene_props.region_object
    if old_obj is not None and old_obj != obj:
        _REGION_CACHE.pop(old_obj.as_pointer(), None)

    obj[REGION_COLORS_ATTR] = (
        np.asarray(colors, dtype=np.float32).ravel().tolist()
    )
    _REGION_CACHE.pop(obj.as_pointer(), None)
    scene_props.region_object = obj
    _tag_view3d_redraw(context)


def clear_region_highlight(
    context: bpy.types.Context,
    obj: bpy.types.Object | None = None,
) -> None:
    """清除领域覆盖层缓存。"""
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    target = obj if obj is not None else scene_props.region_object
    if target is not None:
        _REGION_CACHE.pop(target.as_pointer(), None)
    if obj is None or scene_props.region_object == obj:
        # 调用方负责清理 scene 状态时可不强行置空。
        pass
    _tag_view3d_redraw(context)


def set_merge_label_session(session: dict | None) -> None:
    """设置或清除合并模式编号标签会话。"""
    global _MERGE_LABEL_SESSION
    _MERGE_LABEL_SESSION = session
    if session is None:
        _MERGE_PREVIEW_CACHE.clear()
        _HIGHLIGHT_CACHE.clear()


def get_merge_label_session() -> dict | None:
    """读取合并模式编号标签会话。"""
    return _MERGE_LABEL_SESSION


def set_fit_angle_label_session(session: dict | None) -> None:
    """设置或清除拟合内角标注会话。"""
    global _FIT_ANGLE_LABEL_SESSION
    _FIT_ANGLE_LABEL_SESSION = session


def get_fit_angle_label_session() -> dict | None:
    """读取拟合内角标注会话。"""
    return _FIT_ANGLE_LABEL_SESSION


def active_label_hover_id(scene_props) -> int:
    """当前应高亮的悬停领域编号（模态优先，否则空闲标签悬停）。"""
    if scene_props is None:
        return -1
    if scene_props.merge_mode_active:
        return int(scene_props.merge_hover_id)
    if getattr(scene_props, "remove_mode_active", False):
        return int(getattr(scene_props, "remove_hover_id", -1))
    if getattr(scene_props, "fit_mode_active", False):
        return int(getattr(scene_props, "fit_hover_id", -1))
    if scene_props.split_mode_active:
        return int(getattr(scene_props, "split_hover_id", -1))
    return int(getattr(scene_props, "label_hover_id", -1))


def update_merge_label_projections(context: bpy.types.Context) -> None:
    """根据当前视角刷新领域编号标签屏幕坐标。"""
    _update_label_session_projections(context, _MERGE_LABEL_SESSION)


def update_fit_angle_label_projections(context: bpy.types.Context) -> None:
    """根据当前视角刷新拟合内角标签屏幕坐标。"""
    _update_label_session_projections(context, _FIT_ANGLE_LABEL_SESSION)


def _update_label_session_projections(
    context: bpy.types.Context,
    session: dict | None,
) -> None:
    """刷新单个标签会话的屏幕投影。"""
    from mathutils import Vector

    from ..utils.viewport import (
        is_label_facing_camera,
        is_world_point_occluded,
        project_world_to_region,
        view_direction_to_point,
    )

    if session is None:
        return
    region = context.region
    rv3d = getattr(context, "region_data", None)
    if region is None or rv3d is None or region.type != "WINDOW":
        space = getattr(context, "space_data", None)
        if space is None or space.type != "VIEW_3D":
            return
        rv3d = space.region_3d
        if region is None or region.type != "WINDOW":
            for area_region in getattr(space, "regions", []):
                if area_region.type == "WINDOW":
                    region = area_region
                    break
    if region is None or rv3d is None:
        return

    view_signature = (
        tuple(round(v, 6) for row in rv3d.view_matrix for v in row),
        region.width,
        region.height,
        session.get("preview_version", 0),
    )
    if session.get("_proj_signature") == view_signature:
        return
    session["_proj_signature"] = view_signature

    depsgraph = context.evaluated_depsgraph_get()
    if getattr(rv3d, "is_perspective", True):
        view_origin = rv3d.view_matrix.inverted().translation
    else:
        view_origin = None

    exclude_obj = None
    object_name = session.get("object_name")
    if object_name:
        exclude_obj = bpy.data.objects.get(object_name)

    labels = session.get("labels", [])
    skip_facing = bool(session.get("skip_facing", False))
    skip_occlusion = bool(session.get("skip_occlusion", False))
    for label in labels:
        world = label.get("world_co")
        normal = label.get("normal")
        if world is None:
            label["visible"] = False
            label["screen_xy"] = None
            label["face_screen_xy"] = None
            continue
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        view_dir = view_direction_to_point(rv3d, world)
        if not skip_facing and not is_label_facing_camera(normal, view_dir):
            label["visible"] = False
            label["screen_xy"] = None
            label["face_screen_xy"] = None
            continue

        if view_origin is None:
            ray_origin = Vector(world) - view_dir * 1000.0
        else:
            ray_origin = view_origin

        if (not skip_occlusion) and is_world_point_occluded(
            depsgraph,
            ray_origin,
            world,
            exclude_obj=exclude_obj,
        ):
            label["visible"] = False
            label["screen_xy"] = None
            label["face_screen_xy"] = None
            continue

        screen = project_world_to_region(region, rv3d, world)
        if screen is None:
            label["visible"] = False
            label["screen_xy"] = None
            label["face_screen_xy"] = None
            continue

        label["screen_xy"] = screen
        label["visible"] = True

        face_center = label.get("face_center")
        if face_center is None:
            label["face_screen_xy"] = screen
        else:
            face_screen = project_world_to_region(region, rv3d, face_center)
            label["face_screen_xy"] = (
                face_screen if face_screen is not None else screen
            )


def _draw_circle_2d(center_x: float, center_y: float, radius: float, color) -> None:
    """绘制填充圆（像素空间）。"""
    import math

    segments = 24
    coords = [(center_x, center_y)]
    for index in range(segments + 1):
        angle = (index / segments) * math.tau
        coords.append(
            (
                center_x + math.cos(angle) * radius,
                center_y + math.sin(angle) * radius,
            )
        )
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRI_FAN", {"pos": coords})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_polyline_2d(points, color, width: float = 1.0) -> None:
    """屏幕空间折线。"""
    if len(points) < 2:
        return
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": points})
    shader.bind()
    shader.uniform_float("color", color)
    gpu.state.line_width_set(width)
    try:
        batch.draw(shader)
    finally:
        gpu.state.line_width_set(1.0)


def _leader_edge_point(cx: float, cy: float, fx: float, fy: float, radius: float):
    """从圆中心指向中心面的圆边缘点。"""
    dx = fx - cx
    dy = fy - cy
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return cx, cy - radius
    scale = radius / length
    return cx + dx * scale, cy + dy * scale


def draw_region_merge_labels() -> None:
    """POST_PIXEL：编号圆、引线与中心面端点（已过滤遮挡）。"""
    import blf

    context = bpy.context
    if context is None or context.scene is None:
        return
    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None or not scene_props.show_region_highlight:
        return
    # 空闲不显示编号：仅合并/拆分/移除/拟合
    if not _region_label_modes_active(scene_props):
        return
    session = _MERGE_LABEL_SESSION
    if session is None:
        return

    region_obj = None
    obj_name = session.get("object_name")
    if obj_name:
        region_obj = bpy.data.objects.get(obj_name)
    if region_obj is None:
        region_obj = getattr(scene_props, "region_object", None)
    if not _is_region_mesh_overlay_visible(context, region_obj):
        return

    update_merge_label_projections(context)
    if scene_props.merge_mode_active:
        anchor_id = int(scene_props.merge_anchor_id)
    elif getattr(scene_props, "fit_mode_active", False):
        anchor_id = int(getattr(scene_props, "fit_target_id", -1))
    elif scene_props.split_mode_active:
        anchor_id = int(getattr(scene_props, "split_target_id", -1))
    else:
        anchor_id = -1
    hover_id = active_label_hover_id(scene_props)

    selected_ids = session.get("selected_ids") or ()

    gpu.state.blend_set("ALPHA")
    try:
        for label in session.get("labels", []):
            if not label.get("visible", False):
                continue
            screen = label.get("screen_xy")
            if screen is None:
                continue
            region_id = int(label["id"])
            if region_id == anchor_id or region_id in selected_ids:
                fill = (1.0, 0.55, 0.12, 0.92)
                leader = (1.0, 0.65, 0.25, 0.95)
            elif region_id == hover_id:
                fill = (0.35, 0.75, 1.0, 0.90)
                leader = (0.45, 0.82, 1.0, 0.90)
            else:
                fill = (0.12, 0.12, 0.14, 0.82)
                leader = (0.85, 0.85, 0.88, 0.75)

            face_screen = label.get("face_screen_xy")
            if face_screen is None:
                face_screen = screen
            edge_x, edge_y = _leader_edge_point(
                screen.x,
                screen.y,
                face_screen.x,
                face_screen.y,
                LABEL_RADIUS_PX,
            )
            _draw_polyline_2d(
                [(edge_x, edge_y), (face_screen.x, face_screen.y)],
                leader,
                width=1.5,
            )
            _draw_circle_2d(
                face_screen.x,
                face_screen.y,
                LEADER_ENDPOINT_RADIUS_PX,
                leader,
            )

            _draw_circle_2d(screen.x, screen.y, LABEL_RADIUS_PX, fill)
            _draw_circle_2d(
                screen.x,
                screen.y,
                LABEL_RADIUS_PX + 2.0,
                (1.0, 1.0, 1.0, 0.35),
            )
            _draw_circle_2d(screen.x, screen.y, LABEL_RADIUS_PX, fill)

            text = str(region_id)
            font_id = 0
            blf.size(font_id, 14)
            width, height = blf.dimensions(font_id, text)
            blf.position(
                font_id,
                screen.x - width * 0.5,
                screen.y - height * 0.35,
                0,
            )
            blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            blf.draw(font_id, text)
    finally:
        gpu.state.blend_set("NONE")


def draw_fit_angle_labels() -> None:
    """POST_PIXEL：拟合边界内角度数圆标（风格同领域编号）。"""
    import blf

    context = bpy.context
    if context is None or context.scene is None:
        return
    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None:
        return
    if not getattr(scene_props, "fit_mode_active", False):
        return
    session = _FIT_ANGLE_LABEL_SESSION
    if session is None:
        return
    region_obj = getattr(scene_props, "region_object", None)
    obj_name = session.get("object_name")
    if obj_name:
        region_obj = bpy.data.objects.get(obj_name) or region_obj
    if not _is_region_mesh_overlay_visible(context, region_obj):
        return

    update_fit_angle_label_projections(context)
    gpu.state.blend_set("ALPHA")
    try:
        for label in session.get("labels", []):
            if not label.get("visible", False):
                continue
            screen = label.get("screen_xy")
            if screen is None:
                continue
            angle = float(label.get("angle_deg", 0.0))
            if angle >= _ULTRA_REFLEX_DEG:
                fill = (0.92, 0.22, 0.18, 0.94)
                leader = (1.0, 0.45, 0.35, 0.95)
            elif angle >= 300.0:
                fill = (1.0, 0.55, 0.12, 0.92)
                leader = (1.0, 0.65, 0.25, 0.95)
            else:
                fill = (0.12, 0.12, 0.14, 0.82)
                leader = (0.85, 0.85, 0.88, 0.75)

            face_screen = label.get("face_screen_xy")
            if face_screen is None:
                face_screen = screen
            edge_x, edge_y = _leader_edge_point(
                screen.x,
                screen.y,
                face_screen.x,
                face_screen.y,
                ANGLE_LABEL_RADIUS_PX,
            )
            _draw_polyline_2d(
                [(edge_x, edge_y), (face_screen.x, face_screen.y)],
                leader,
                width=1.5,
            )
            _draw_circle_2d(
                face_screen.x,
                face_screen.y,
                LEADER_ENDPOINT_RADIUS_PX,
                leader,
            )
            _draw_circle_2d(
                screen.x, screen.y, ANGLE_LABEL_RADIUS_PX, fill
            )
            _draw_circle_2d(
                screen.x,
                screen.y,
                ANGLE_LABEL_RADIUS_PX + 2.0,
                (1.0, 1.0, 1.0, 0.35),
            )
            _draw_circle_2d(
                screen.x, screen.y, ANGLE_LABEL_RADIUS_PX, fill
            )

            text = str(label.get("text") or f"{int(round(angle))}")
            font_id = 0
            blf.size(font_id, 12)
            width, height = blf.dimensions(font_id, text)
            blf.position(
                font_id,
                screen.x - width * 0.5,
                screen.y - height * 0.35,
                0,
            )
            blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            blf.draw(font_id, text)
    finally:
        gpu.state.blend_set("NONE")


def draw_all_region_labels() -> None:
    """POST_PIXEL：领域编号 + 拟合内角标注。"""
    draw_region_merge_labels()
    draw_fit_angle_labels()


def set_split_stroke_session(session: dict | None) -> None:
    """设置或清除拆分笔迹预览会话。"""
    global _SPLIT_STROKE_SESSION
    _SPLIT_STROKE_SESSION = session
    if session is None:
        _MERGE_PREVIEW_CACHE.pop("split_preview", None)
        _HIGHLIGHT_CACHE.pop("paint", None)


def get_split_stroke_session() -> dict | None:
    """读取拆分笔迹预览会话。"""
    return _SPLIT_STROKE_SESSION


def draw_split_brush_cursor() -> None:
    """POST_PIXEL：旧版笔刷光标（硬边点选模式不再使用）。"""
    return


def _draw_edge_lines_world(edges_world, color, width: float) -> None:
    """
    绘制 (N,2,3) 世界空间线段。

    关闭深度测试：候选硬边与网格/领域色面共面，LESS_EQUAL 会被
    外移后的领域覆盖层完全挡住，导致「候选 N 条」却看不见线。
    """
    if edges_world is None or len(edges_world) == 0:
        return
    segments = []
    for edge in np.asarray(edges_world):
        if len(edge) < 2:
            continue
        segments.append(tuple(map(float, edge[0])))
        segments.append(tuple(map(float, edge[1])))
    if not segments:
        return
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINES", {"pos": segments})
    gpu.state.blend_set("ALPHA")
    # 引导线始终盖在网格之上，避免与领域色面 Z-fighting / 深度遮挡。
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)
    gpu.state.line_width_set(float(width))
    try:
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)
    finally:
        gpu.state.line_width_set(1.0)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


def _draw_circle_outline_2d(
    center_x: float,
    center_y: float,
    radius: float,
    color,
    segments: int = 32,
) -> None:
    """屏幕空间圆轮廓。"""
    import math

    coords = []
    for index in range(segments + 1):
        angle = (index / segments) * math.tau
        coords.append(
            (
                center_x + math.cos(angle) * radius,
                center_y + math.sin(angle) * radius,
            )
        )
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": coords})
    shader.bind()
    shader.uniform_float("color", color)
    gpu.state.line_width_set(2.0)
    try:
        batch.draw(shader)
    finally:
        gpu.state.line_width_set(1.0)


def draw_split_stroke_preview() -> None:
    """POST_VIEW：候选硬边、悬停/已选边与补全切线。"""
    context = bpy.context
    if context is None or context.scene is None:
        return
    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None or not scene_props.split_mode_active:
        return
    session = _SPLIT_STROKE_SESSION
    if session is None:
        return

    object_name = session.get("object_name")
    obj = bpy.data.objects.get(object_name) if object_name else None
    if obj is None or obj.type != "MESH":
        return

    # 已选完整切线时隐藏候选碎边，避免预览阶段「看起来又变短」
    has_cut = False
    completed = session.get("completed_edges_world")
    selected = session.get("selected_edges_world")
    if completed is not None and len(completed) > 0:
        has_cut = True
    if selected is not None and len(selected) > 1:
        has_cut = True
    if not has_cut:
        _draw_edge_lines_world(
            session.get("candidate_edges_world"),
            (0.1, 1.0, 0.82, 0.95),
            3.5,
        )
    hover = session.get("hover_edge_world")
    if hover is not None and len(hover) > 0 and not has_cut:
        _draw_edge_lines_world(hover, (1.0, 0.95, 0.2, 1.0), 5.0)
    # 完整分割棱：红色加粗（选中与预览同一条）
    cut_edges = completed if (completed is not None and len(completed) > 0) else selected
    _draw_edge_lines_world(
        cut_edges,
        (1.0, 0.08, 0.08, 1.0),
        6.0,
    )


def register_split_draw_handler() -> None:
    """注册拆分笔迹 POST_VIEW + 笔刷 POST_PIXEL 绘制。"""
    global _SPLIT_DRAW_HANDLE
    namespace = bpy.app.driver_namespace
    old = namespace.get(SPLIT_DRAW_HANDLE_KEY)
    # 句柄存为 (POST_VIEW, POST_PIXEL) tuple，须逐个移除。
    for item in old if isinstance(old, (tuple, list)) else (old,):
        if item is None:
            continue
        try:
            bpy.types.SpaceView3D.draw_handler_remove(item, "WINDOW")
        except (ReferenceError, ValueError, TypeError):
            pass
    view_handle = bpy.types.SpaceView3D.draw_handler_add(
        draw_split_stroke_preview,
        (),
        "WINDOW",
        "POST_VIEW",
    )
    pixel_handle = bpy.types.SpaceView3D.draw_handler_add(
        draw_split_brush_cursor,
        (),
        "WINDOW",
        "POST_PIXEL",
    )
    _SPLIT_DRAW_HANDLE = (view_handle, pixel_handle)
    namespace[SPLIT_DRAW_HANDLE_KEY] = _SPLIT_DRAW_HANDLE


def unregister_split_draw_handler() -> None:
    """注销拆分笔迹绘制句柄。"""
    global _SPLIT_DRAW_HANDLE, _SPLIT_STROKE_SESSION
    namespace = bpy.app.driver_namespace
    handle = namespace.pop(SPLIT_DRAW_HANDLE_KEY, None) or _SPLIT_DRAW_HANDLE
    handles = handle if isinstance(handle, (tuple, list)) else (handle,)
    for item in handles:
        if item is None:
            continue
        try:
            bpy.types.SpaceView3D.draw_handler_remove(item, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _SPLIT_DRAW_HANDLE = None
    _SPLIT_STROKE_SESSION = None
    _HIGHLIGHT_CACHE.pop("paint", None)


def set_curve_tool_hud(text: str | None) -> None:
    """设置曲线工具提示文案（视口左下角）。"""
    global _CURVE_TOOL_HUD
    _CURVE_TOOL_HUD = None if text is None else str(text)


def clear_curve_tool_hud() -> None:
    """清除曲线工具 HUD 文案。"""
    set_curve_tool_hud(None)


def set_curve_split_preview(session: dict | None) -> None:
    """
    设置拆分曲线分色预览。

    session = {
      "segments": [{"points": (N,3) ndarray, "color": (r,g,b,a)}, ...],
      "line_width": float,
    }
    """
    global _CURVE_SPLIT_PREVIEW
    _CURVE_SPLIT_PREVIEW = session


def clear_curve_split_preview() -> None:
    set_curve_split_preview(None)


def set_curve_bezier_preview(session: dict | None) -> None:
    """
    设置贝塞尔拟合预览（曲线采样 + 锚点 + 手柄）。

    session = {
      "curves": [{"points": (N,3), "color": rgba}, ...],
      "anchors": (M,3),
      "handles": (K,3),
      "handle_edges": (H,2,3),
      "line_width": float,
      "anchor_size": float,
      "handle_size": float,
    }
    """
    global _CURVE_BEZIER_PREVIEW
    _CURVE_BEZIER_PREVIEW = session


def clear_curve_bezier_preview() -> None:
    set_curve_bezier_preview(None)


def _draw_points_world(points, color, size: float) -> None:
    """世界空间点（关闭深度测试，始终可见）。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return
    coords = [tuple(map(float, p)) for p in pts]
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "POINTS", {"pos": coords})
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)
    try:
        gpu.state.point_size_set(float(size))
    except Exception:
        pass
    try:
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)
    finally:
        try:
            gpu.state.point_size_set(1.0)
        except Exception:
            pass
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


def draw_curve_split_preview() -> None:
    """POST_VIEW：拆分分色折线 + 贝塞尔控制点/手柄预览。"""
    session = _CURVE_SPLIT_PREVIEW
    if session:
        segments = session.get("segments") or []
        width = float(session.get("line_width", 4.0))
        for item in segments:
            pts = np.asarray(item.get("points"), dtype=np.float64)
            if len(pts) < 2:
                continue
            color = tuple(
                float(v) for v in (item.get("color") or (1, 0.2, 0.1, 1))[:4]
            )
            if len(color) < 4:
                color = color + (1.0,) * (4 - len(color))
            edges = np.stack((pts[:-1], pts[1:]), axis=1)
            _draw_edge_lines_world(edges, color, width)

    bez = _CURVE_BEZIER_PREVIEW
    if not bez:
        return
    # 拟合后的光滑曲线
    for item in bez.get("curves") or []:
        pts = np.asarray(item.get("points"), dtype=np.float64)
        if len(pts) < 2:
            continue
        color = tuple(
            float(v) for v in (item.get("color") or (0.2, 0.85, 1.0, 1.0))[:4]
        )
        if len(color) < 4:
            color = color + (1.0,) * (4 - len(color))
        edges = np.stack((pts[:-1], pts[1:]), axis=1)
        _draw_edge_lines_world(
            edges, color, float(bez.get("line_width", 3.5))
        )
    # 手柄连杆
    handle_edges = bez.get("handle_edges")
    if handle_edges is not None and len(handle_edges) > 0:
        _draw_edge_lines_world(
            handle_edges,
            (0.35, 0.95, 0.55, 0.95),
            float(bez.get("handle_line_width", 1.8)),
        )
    # 手柄端点
    handles = bez.get("handles")
    if handles is not None and len(handles) > 0:
        _draw_points_world(
            handles,
            (0.25, 1.0, 0.55, 1.0),
            float(bez.get("handle_size", 8.0)),
        )
    # 锚点（控制点）
    anchors = bez.get("anchors")
    if anchors is not None and len(anchors) > 0:
        _draw_points_world(
            anchors,
            (1.0, 0.85, 0.15, 1.0),
            float(bez.get("anchor_size", 12.0)),
        )


def draw_curve_tool_hud() -> None:
    """POST_PIXEL：视口左下角显示状态。"""
    import blf

    text = _CURVE_TOOL_HUD
    if not text:
        return
    context = bpy.context
    if context is None or context.region is None:
        return
    font_id = 0
    blf.size(font_id, 15)
    width, height = blf.dimensions(font_id, text)
    x = float(_CURVE_HUD_LEFT_MARGIN_PX)
    y = float(_CURVE_HUD_BOTTOM_MARGIN_PX)
    gpu.state.blend_set("ALPHA")
    try:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        pad_x = 10.0
        pad_y = 6.0
        verts = (
            (x - pad_x, y - pad_y),
            (x + width + pad_x, y - pad_y),
            (x + width + pad_x, y + height + pad_y),
            (x - pad_x, y + height + pad_y),
        )
        batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": (
                    verts[0],
                    verts[1],
                    verts[2],
                    verts[0],
                    verts[2],
                    verts[3],
                )
            },
        )
        shader.bind()
        shader.uniform_float("color", (0.02, 0.02, 0.05, 0.62))
        batch.draw(shader)
        blf.position(font_id, x, y, 0)
        blf.color(font_id, 1.0, 0.95, 0.55, 1.0)
        blf.draw(font_id, text)
    finally:
        gpu.state.blend_set("NONE")


def register_curve_tool_hud() -> None:
    """注册曲线工具 HUD + 拆分/贝塞尔预览绘制。"""
    global _CURVE_TOOL_HUD_HANDLE, _CURVE_SPLIT_PREVIEW_HANDLE
    # 重新进入工具时先清掉可能残留的旧预览
    clear_curve_split_preview()
    clear_curve_bezier_preview()
    namespace = bpy.app.driver_namespace

    old_hud = namespace.get(CURVE_TOOL_HUD_KEY)
    if old_hud is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(old_hud, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _CURVE_TOOL_HUD_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        draw_curve_tool_hud,
        (),
        "WINDOW",
        "POST_PIXEL",
    )
    namespace[CURVE_TOOL_HUD_KEY] = _CURVE_TOOL_HUD_HANDLE

    old_prev = namespace.get(CURVE_SPLIT_PREVIEW_KEY)
    if old_prev is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(old_prev, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _CURVE_SPLIT_PREVIEW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        draw_curve_split_preview,
        (),
        "WINDOW",
        "POST_VIEW",
    )
    namespace[CURVE_SPLIT_PREVIEW_KEY] = _CURVE_SPLIT_PREVIEW_HANDLE


def unregister_curve_tool_hud() -> None:
    """注销曲线工具 HUD 与预览。"""
    global _CURVE_TOOL_HUD_HANDLE, _CURVE_TOOL_HUD
    global _CURVE_SPLIT_PREVIEW_HANDLE, _CURVE_SPLIT_PREVIEW, _CURVE_BEZIER_PREVIEW
    namespace = bpy.app.driver_namespace

    handle = namespace.pop(CURVE_TOOL_HUD_KEY, None) or _CURVE_TOOL_HUD_HANDLE
    if handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _CURVE_TOOL_HUD_HANDLE = None
    _CURVE_TOOL_HUD = None

    prev = namespace.pop(CURVE_SPLIT_PREVIEW_KEY, None) or _CURVE_SPLIT_PREVIEW_HANDLE
    if prev is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(prev, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _CURVE_SPLIT_PREVIEW_HANDLE = None
    _CURVE_SPLIT_PREVIEW = None
    _CURVE_BEZIER_PREVIEW = None


def register_label_draw_handler() -> None:
    """按需注册领域编号 POST_PIXEL 绘制。"""
    global _LABEL_DRAW_HANDLE
    namespace = bpy.app.driver_namespace
    old = namespace.get(LABEL_DRAW_HANDLE_KEY)
    if old is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(old, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _LABEL_DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        draw_all_region_labels,
        (),
        "WINDOW",
        "POST_PIXEL",
    )
    namespace[LABEL_DRAW_HANDLE_KEY] = _LABEL_DRAW_HANDLE


def unregister_label_draw_handler() -> None:
    """注销领域编号绘制句柄。"""
    global _LABEL_DRAW_HANDLE, _MERGE_LABEL_SESSION, _FIT_ANGLE_LABEL_SESSION
    namespace = bpy.app.driver_namespace
    handle = namespace.pop(LABEL_DRAW_HANDLE_KEY, None) or _LABEL_DRAW_HANDLE
    if handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _LABEL_DRAW_HANDLE = None
    _MERGE_LABEL_SESSION = None
    _FIT_ANGLE_LABEL_SESSION = None


def _tag_view3d_redraw(context: bpy.types.Context) -> None:
    """标记所有 3D 视口重绘。"""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _is_object_selected(context: bpy.types.Context, obj: bpy.types.Object) -> bool:
    """仅当对象被选中时显示高亮。"""
    try:
        return bool(obj.select_get())
    except ReferenceError:
        return False


def _is_region_mesh_overlay_visible(
    context: bpy.types.Context,
    obj: bpy.types.Object | None,
) -> bool:
    """
    领域分色/编号是否应绘制：对象须在当前视图层（及视口）可见。

    使用 visible_get，从而与 Collection / LayerCollection 隐藏、
    Local Collections、视图层排除保持一致；仅 hide_get 不够。
    """
    if obj is None:
        return False
    try:
        if obj.type != "MESH":
            return False
        if obj.mode == "EDIT":
            return False
        if obj.hide_viewport:
            return False
        view_layer = getattr(context, "view_layer", None)
        if view_layer is None:
            return not bool(obj.hide_get())
        space = getattr(context, "space_data", None)
        if space is not None and getattr(space, "type", "") == "VIEW_3D":
            return bool(obj.visible_get(view_layer=view_layer, viewport=space))
        return bool(obj.visible_get(view_layer=view_layer))
    except ReferenceError:
        return False
    except TypeError:
        # 旧版 Blender 可能不支持 viewport= 参数
        try:
            view_layer = getattr(context, "view_layer", None)
            if view_layer is None:
                return not bool(obj.hide_get())
            return bool(obj.visible_get(view_layer=view_layer))
        except Exception:
            return False


def _region_label_modes_active(scene_props) -> bool:
    """编号标签仅在合并/拆分/移除/拟合模态中显示。"""
    return bool(
        getattr(scene_props, "merge_mode_active", False)
        or getattr(scene_props, "remove_mode_active", False)
        or getattr(scene_props, "split_mode_active", False)
        or getattr(scene_props, "fit_mode_active", False)
    )


def _matrix_signature(obj: bpy.types.Object) -> tuple:
    """对象世界矩阵签名。"""
    matrix = obj.matrix_world
    return tuple(
        round(float(matrix[row][column]), 9)
        for row in range(4)
        for column in range(4)
    )


def _mesh_signature(obj: bpy.types.Object) -> tuple:
    """网格数据签名。"""
    mesh = obj.data
    return (
        mesh.as_pointer(),
        len(mesh.vertices),
        len(mesh.polygons),
        len(mesh.loops),
    )


def _bottom_signature(obj: bpy.types.Object) -> tuple:
    """底面覆盖层签名。"""
    return _matrix_signature(obj) + _mesh_signature(obj)


def _region_signature(obj: bpy.types.Object) -> tuple:
    """领域覆盖层签名，包含结果版本。"""
    version = int(obj.get(REGION_VERSION_ATTR, 0))
    return _matrix_signature(obj) + _mesh_signature(obj) + (version,)


def _read_local_vertices(mesh: bpy.types.Mesh) -> np.ndarray:
    """批量读取本地顶点坐标。"""
    vert_count = len(mesh.vertices)
    if vert_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    local = np.empty(vert_count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", local)
    return local.reshape(vert_count, 3)


def _local_to_world(obj: bpy.types.Object, local: np.ndarray) -> np.ndarray:
    """本地坐标转世界坐标。"""
    if len(local) == 0:
        return np.empty((0, 3), dtype=np.float32)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    world = local @ matrix[:3, :3].T + matrix[:3, 3]
    return world.astype(np.float32)


def _overlay_offset_distance(obj: bpy.types.Object) -> float:
    """按包围盒对角线计算法线外移距离。"""
    mesh = obj.data
    if len(mesh.vertices) == 0:
        return OVERLAY_NORMAL_OFFSET_MIN
    dims = np.asarray(obj.dimensions, dtype=np.float64)
    extent = float(np.linalg.norm(dims))
    if extent < 1e-12:
        # dimensions 不可用时退回本地包围盒。
        local = _read_local_vertices(mesh)
        if len(local) == 0:
            return OVERLAY_NORMAL_OFFSET_MIN
        extent = float(np.linalg.norm(local.max(axis=0) - local.min(axis=0)))
    return float(max(extent * OVERLAY_NORMAL_OFFSET_RATIO, OVERLAY_NORMAL_OFFSET_MIN))


def _world_face_normals(obj: bpy.types.Object) -> np.ndarray:
    """批量读取面法线并变换到世界空间。"""
    mesh = obj.data
    face_count = len(mesh.polygons)
    if face_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    local = np.empty(face_count * 3, dtype=np.float64)
    mesh.polygons.foreach_get("normal", local)
    normals = local.reshape(face_count, 3)
    matrix3 = np.asarray(obj.matrix_world, dtype=np.float64)[:3, :3]
    # 法线用逆转置近似；均匀缩放下等价于旋转。
    try:
        normal_matrix = np.linalg.inv(matrix3).T
    except np.linalg.LinAlgError:
        normal_matrix = matrix3
    world = normals @ normal_matrix.T
    lengths = np.linalg.norm(world, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    return world / lengths


def _offset_world_by_face_normals(
    world: np.ndarray,
    face_per_tri: np.ndarray,
    face_normals_world: np.ndarray,
    distance: float,
) -> np.ndarray:
    """把每个三角形的三个顶点沿所属面法线外移。"""
    if len(world) == 0 or distance <= 0.0:
        return world
    normals = face_normals_world[face_per_tri]
    offset = np.repeat(normals, 3, axis=0) * float(distance)
    return (world.astype(np.float64) + offset).astype(np.float32)


def _triangulate_face_indices(
    mesh: bpy.types.Mesh,
    face_indices: list[int] | np.ndarray,
) -> np.ndarray:
    """将指定面三角化为顶点索引数组 (T*3,)。"""
    polygon_count = len(mesh.polygons)
    triangle_indices: list[int] = []
    for face_index in face_indices:
        index = int(face_index)
        if index < 0 or index >= polygon_count:
            continue
        vert_indices = mesh.polygons[index].vertices
        if len(vert_indices) < 3:
            continue
        anchor = vert_indices[0]
        for tri in range(1, len(vert_indices) - 1):
            triangle_indices.append(anchor)
            triangle_indices.append(vert_indices[tri])
            triangle_indices.append(vert_indices[tri + 1])
    if not triangle_indices:
        return np.empty(0, dtype=np.int64)
    return np.asarray(triangle_indices, dtype=np.int64)


def _build_bottom_world_coords(
    obj: bpy.types.Object,
    face_indices: list[int],
) -> np.ndarray:
    """构建底面三角形世界坐标（沿法线外移）。"""
    mesh = obj.data
    if len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
        return np.empty((0, 3), dtype=np.float32)

    faces = np.asarray(face_indices, dtype=np.int64)
    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.float32)

    loop_start, loop_total, loop_verts = _read_loop_arrays(mesh)
    tri_verts, face_per_tri = _triangle_fan_arrays(
        loop_start,
        loop_total,
        loop_verts,
        faces,
    )
    if len(tri_verts) == 0:
        return np.empty((0, 3), dtype=np.float32)

    local_all = _read_local_vertices(mesh)
    world = _local_to_world(obj, local_all[tri_verts])
    normals = _world_face_normals(obj)
    return _offset_world_by_face_normals(
        world,
        face_per_tri,
        normals,
        _overlay_offset_distance(obj),
    )


def _cached_bottom_coords(
    obj: bpy.types.Object,
    face_indices: list[int],
) -> np.ndarray:
    """按签名获取或重建底面世界坐标。"""
    pointer = obj.as_pointer()
    signature = _bottom_signature(obj)
    entry = _BOTTOM_CACHE.get(pointer)
    if entry is not None and entry["signature"] == signature:
        return entry["coords"]

    coords = _build_bottom_world_coords(obj, face_indices)
    _BOTTOM_CACHE[pointer] = {
        "signature": signature,
        "coords": coords,
    }
    return coords


def _read_region_ids(mesh: bpy.types.Mesh) -> np.ndarray | None:
    """读取 FACE 域领域标签。"""
    attribute = mesh.attributes.get(REGION_ID_ATTR)
    if attribute is None or attribute.domain != "FACE":
        return None
    face_count = len(mesh.polygons)
    if face_count == 0:
        return np.empty(0, dtype=np.int32)
    values = np.empty(face_count, dtype=np.int32)
    attribute.data.foreach_get("value", values)
    return values


def _read_region_colors(obj: bpy.types.Object, region_count: int) -> np.ndarray:
    """从对象属性读取调色板；缺失时按算法重新生成。"""
    from ..algorithms.regions import generate_region_colors

    raw = obj.get(REGION_COLORS_ATTR)
    if raw is not None:
        flat = np.asarray(list(raw), dtype=np.float32)
        if len(flat) >= region_count * 4:
            return flat[: region_count * 4].reshape(region_count, 4)
    return generate_region_colors(region_count)


def _read_loop_arrays(
    mesh: bpy.types.Mesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量读取 loop_start / loop_total / loop 顶点索引。"""
    face_count = len(mesh.polygons)
    loop_count = len(mesh.loops)
    loop_start = np.empty(face_count, dtype=np.int32)
    loop_total = np.empty(face_count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loop_verts = np.empty(loop_count, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    return loop_start, loop_total, loop_verts


def _triangle_fan_arrays(
    loop_start: np.ndarray,
    loop_total: np.ndarray,
    loop_verts: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    向量化三角扇展开。

    返回 (顶点索引 (T*3,), 每个三角形所属面 (T,))。
    """
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    totals = loop_total[faces].astype(np.int64) - 2
    keep = totals > 0
    faces = faces[keep]
    totals = totals[keep]
    if len(faces) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    tri_count = int(totals.sum())
    face_per_tri = np.repeat(faces, totals)
    start_per_tri = np.repeat(loop_start[faces].astype(np.int64), totals)
    offsets = np.cumsum(totals) - totals
    local_tri = np.arange(tri_count, dtype=np.int64) - np.repeat(offsets, totals)

    tri_verts = np.empty(tri_count * 3, dtype=np.int64)
    tri_verts[0::3] = loop_verts[start_per_tri]
    tri_verts[1::3] = loop_verts[start_per_tri + local_tri + 1]
    tri_verts[2::3] = loop_verts[start_per_tri + local_tri + 2]
    return tri_verts, face_per_tri


def _build_region_draw_arrays(
    obj: bpy.types.Object,
    region_ids: np.ndarray | None = None,
    palette: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建领域绘制用世界坐标与逐顶点颜色（全程 NumPy 向量化）。

    忽略标签（<0）的面不进入绘制缓冲。
    可传入内存中的 region_ids/palette（合并预览）。
    """
    coords, face_per_tri, colors = _build_region_draw_parts(
        obj,
        region_ids=region_ids,
        palette=palette,
    )
    return coords, colors


def _build_region_draw_parts(
    obj: bpy.types.Object,
    region_ids: np.ndarray | None = None,
    palette: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (world_coords, face_per_tri, colors)。

    合并预览可缓存 coords/face_per_tri，仅在标签变化时重算 colors。
    """
    mesh = obj.data
    if region_ids is None:
        region_ids = _read_region_ids(mesh)
    else:
        region_ids = np.asarray(region_ids, dtype=np.int32)

    empty_pos = np.empty((0, 3), dtype=np.float32)
    empty_faces = np.empty(0, dtype=np.int64)
    empty_col = np.empty((0, 4), dtype=np.float32)
    if region_ids is None or len(region_ids) == 0:
        return empty_pos, empty_faces, empty_col

    polygon_count = len(mesh.polygons)
    if len(region_ids) != polygon_count:
        region_ids = region_ids[:polygon_count]

    valid_faces = np.flatnonzero(region_ids >= 0)
    if len(valid_faces) == 0:
        return empty_pos, empty_faces, empty_col

    region_count = int(region_ids.max()) + 1
    if palette is None:
        palette = _read_region_colors(obj, region_count)
    else:
        palette = np.asarray(palette, dtype=np.float32)
        if len(palette) < region_count:
            from ..algorithms.regions import generate_region_colors

            extra = generate_region_colors(region_count - len(palette))
            palette = np.vstack((palette, extra))

    loop_start, loop_total, loop_verts = _read_loop_arrays(mesh)
    tri_verts, face_per_tri = _triangle_fan_arrays(
        loop_start,
        loop_total,
        loop_verts,
        valid_faces,
    )
    if len(tri_verts) == 0:
        return empty_pos, empty_faces, empty_col

    local_all = _read_local_vertices(mesh)
    world = _local_to_world(obj, local_all[tri_verts])
    normals = _world_face_normals(obj)
    world = _offset_world_by_face_normals(
        world,
        face_per_tri,
        normals,
        _overlay_offset_distance(obj),
    )
    color_array = np.repeat(
        palette[region_ids[face_per_tri]].astype(np.float32),
        3,
        axis=0,
    )
    return world, face_per_tri, color_array


def _region_tri_colors(
    region_ids: np.ndarray,
    face_per_tri: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """按已缓存的 face_per_tri 仅重算逐顶点颜色。"""
    ids = np.asarray(region_ids, dtype=np.int32)
    faces = np.asarray(face_per_tri, dtype=np.int64)
    colors = np.asarray(palette, dtype=np.float32)
    if len(faces) == 0:
        return np.empty((0, 4), dtype=np.float32)
    region_count = int(ids.max()) + 1 if len(ids) else 0
    if len(colors) < region_count:
        from ..algorithms.regions import generate_region_colors

        extra = generate_region_colors(region_count - len(colors))
        colors = np.vstack((colors, extra))
    return np.repeat(colors[ids[faces]].astype(np.float32), 3, axis=0)


def _build_selection_highlight_coords(
    obj: bpy.types.Object,
    region_ids: np.ndarray,
    highlight_id: int,
) -> np.ndarray:
    """构建指定领域的三角面世界坐标，用于选中/悬停高亮。"""
    if highlight_id < 0:
        return np.empty((0, 3), dtype=np.float32)
    mesh = obj.data
    faces = np.flatnonzero(
        np.asarray(region_ids, dtype=np.int32) == highlight_id
    )
    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.float32)
    # 超大领域高亮会卡死主线程；跳过几何构建，仅保留编号交互
    if len(faces) > 250_000:
        from ..utils.debug import are_debug

        are_debug(
            f"skip highlight id={highlight_id} faces={len(faces)} (>250k)"
        )
        return np.empty((0, 3), dtype=np.float32)

    loop_start, loop_total, loop_verts = _read_loop_arrays(mesh)
    tri_verts, face_per_tri = _triangle_fan_arrays(
        loop_start,
        loop_total,
        loop_verts,
        faces,
    )
    if len(tri_verts) == 0:
        return np.empty((0, 3), dtype=np.float32)
    local_all = _read_local_vertices(mesh)
    world = _local_to_world(obj, local_all[tri_verts])
    normals = _world_face_normals(obj)
    return _offset_world_by_face_normals(
        world,
        face_per_tri,
        normals,
        _overlay_offset_distance(obj),
    )


def _build_paint_face_coords(
    obj: bpy.types.Object,
    face_indices: np.ndarray | list[int],
) -> np.ndarray:
    """构建涂红面的三角世界坐标（沿法线外移）。"""
    faces = np.asarray(list(face_indices), dtype=np.int64)
    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.float32)
    mesh = obj.data
    loop_start, loop_total, loop_verts = _read_loop_arrays(mesh)
    tri_verts, face_per_tri = _triangle_fan_arrays(
        loop_start,
        loop_total,
        loop_verts,
        faces,
    )
    if len(tri_verts) == 0:
        return np.empty((0, 3), dtype=np.float32)
    local_all = _read_local_vertices(mesh)
    world = _local_to_world(obj, local_all[tri_verts])
    normals = _world_face_normals(obj)
    return _offset_world_by_face_normals(
        world,
        face_per_tri,
        normals,
        _overlay_offset_distance(obj) * 1.25,
    )


def _cached_region_draw_arrays(
    obj: bpy.types.Object,
) -> tuple[np.ndarray, np.ndarray]:
    """按签名获取或重建领域绘制缓冲。"""
    pointer = obj.as_pointer()
    signature = _region_signature(obj)
    entry = _REGION_CACHE.get(pointer)
    if entry is not None and entry["signature"] == signature:
        return entry["coords"], entry["colors"]

    coords, colors = _build_region_draw_arrays(obj)
    _REGION_CACHE[pointer] = {
        "signature": signature,
        "coords": coords,
        "colors": colors,
    }
    return coords, colors


def _draw_uniform_tris(coords: np.ndarray, color: tuple) -> None:
    """单色三角形绘制。"""
    if len(coords) == 0:
        return
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": coords})
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    try:
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)
    finally:
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


def _draw_colored_tris(coords: np.ndarray, colors: np.ndarray) -> None:
    """逐顶点颜色三角形绘制。"""
    if len(coords) == 0:
        return
    # Blender 4.x 内置着色器：优先 SMOOTH_COLOR，回退 FLAT_COLOR。
    try:
        shader = gpu.shader.from_builtin("SMOOTH_COLOR")
    except ValueError:
        shader = gpu.shader.from_builtin("FLAT_COLOR")
    batch = batch_for_shader(
        shader,
        "TRIS",
        {
            "pos": coords,
            "color": colors,
        },
    )
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    try:
        shader.bind()
        batch.draw(shader)
    finally:
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


def _draw_label_hover_highlight(
    obj: bpy.types.Object,
    region_ids: np.ndarray,
    hover_id: int,
    *,
    skip_id: int = -1,
    cache_token=None,
) -> None:
    """绘制悬停领域半透明高亮（跳过与锚点/目标相同的编号）。"""
    if hover_id < 0 or hover_id == skip_id:
        return
    pointer = obj.as_pointer()
    slot_key = (pointer, cache_token, hover_id, "label_hover")
    entry = _HIGHLIGHT_CACHE.get("label_hover")
    if entry is None or entry.get("key") != slot_key:
        entry = {
            "key": slot_key,
            "coords": _build_selection_highlight_coords(
                obj,
                region_ids,
                hover_id,
            ),
        }
        _HIGHLIGHT_CACHE["label_hover"] = entry
    coords = entry["coords"]
    if coords is not None and len(coords) > 0:
        _draw_uniform_tris(coords, HOVER_HIGHLIGHT_COLOR)


def draw_overlays() -> None:
    """视口叠加回调：领域多色与底面紫色统一由“显示领域”控制。"""
    context = bpy.context
    if context is None or context.scene is None:
        return

    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None:
        return
    if not scene_props.show_region_highlight:
        return

    # 先画普通领域，再画底面紫色（底面是固定前置领域，不参与编号）。
    region_obj = scene_props.region_object
    merge_session = _MERGE_LABEL_SESSION
    split_session = _SPLIT_STROKE_SESSION
    use_merge_preview = (
        (
            bool(scene_props.merge_mode_active)
            or bool(getattr(scene_props, "remove_mode_active", False))
        )
        and merge_session is not None
        and merge_session.get("region_ids") is not None
    )
    use_split_preview = (
        bool(scene_props.split_mode_active)
        and split_session is not None
        and split_session.get("preview_ids") is not None
    )

    if (
        region_obj is not None
        and region_obj.type == "MESH"
        and _is_object_selected(context, region_obj)
        and _is_region_mesh_overlay_visible(context, region_obj)
        and (
            use_merge_preview
            or use_split_preview
            or region_obj.data.attributes.get(REGION_ID_ATTR) is not None
        )
    ):
        hover_id = active_label_hover_id(scene_props)
        if use_merge_preview:
            from ..utils.debug import are_debug

            live_ids = np.asarray(merge_session["region_ids"], dtype=np.int32)
            live_colors = merge_session.get("colors")
            version = merge_session.get("preview_version", 0)
            pointer = region_obj.as_pointer()
            valid_count = int(np.count_nonzero(live_ids >= 0))
            geom_key = (pointer, valid_count, len(live_ids))
            color_key = (pointer, version, valid_count)

            if _MERGE_PREVIEW_CACHE.get("geom_key") != geom_key:
                t0 = time.perf_counter()
                are_debug(
                    f"merge_preview geom rebuild faces={len(live_ids)} "
                    f"valid={valid_count}"
                )
                coords, face_per_tri, colors = _build_region_draw_parts(
                    region_obj,
                    region_ids=live_ids,
                    palette=live_colors,
                )
                _MERGE_PREVIEW_CACHE["geom_key"] = geom_key
                _MERGE_PREVIEW_CACHE["coords"] = coords
                _MERGE_PREVIEW_CACHE["face_per_tri"] = face_per_tri
                _MERGE_PREVIEW_CACHE["colors"] = colors
                _MERGE_PREVIEW_CACHE["color_key"] = color_key
                are_debug(
                    f"merge_preview geom done tris={len(face_per_tri)} "
                    f"ms={(time.perf_counter() - t0) * 1000.0:.1f}"
                )
            elif _MERGE_PREVIEW_CACHE.get("color_key") != color_key:
                t0 = time.perf_counter()
                are_debug(f"merge_preview colors only version={version}")
                face_per_tri = _MERGE_PREVIEW_CACHE.get("face_per_tri")
                if face_per_tri is None or live_colors is None:
                    coords, face_per_tri, colors = _build_region_draw_parts(
                        region_obj,
                        region_ids=live_ids,
                        palette=live_colors,
                    )
                    _MERGE_PREVIEW_CACHE["coords"] = coords
                    _MERGE_PREVIEW_CACHE["face_per_tri"] = face_per_tri
                    _MERGE_PREVIEW_CACHE["colors"] = colors
                else:
                    _MERGE_PREVIEW_CACHE["colors"] = _region_tri_colors(
                        live_ids,
                        face_per_tri,
                        live_colors,
                    )
                _MERGE_PREVIEW_CACHE["color_key"] = color_key
                are_debug(
                    f"merge_preview colors done "
                    f"ms={(time.perf_counter() - t0) * 1000.0:.1f}"
                )

            _draw_colored_tris(
                _MERGE_PREVIEW_CACHE["coords"],
                _MERGE_PREVIEW_CACHE["colors"],
            )

            anchor_id = (
                int(scene_props.merge_anchor_id)
                if scene_props.merge_mode_active
                else -1
            )
            if anchor_id >= 0:
                slot_key = (pointer, version, anchor_id)
                entry = _HIGHLIGHT_CACHE.get("anchor")
                if entry is None or entry.get("key") != slot_key:
                    t0 = time.perf_counter()
                    are_debug(f"merge_preview anchor highlight id={anchor_id}")
                    entry = {
                        "key": slot_key,
                        "coords": _build_selection_highlight_coords(
                            region_obj,
                            live_ids,
                            anchor_id,
                        ),
                    }
                    _HIGHLIGHT_CACHE["anchor"] = entry
                    are_debug(
                        f"merge_preview anchor done "
                        f"ms={(time.perf_counter() - t0) * 1000.0:.1f}"
                    )
                _draw_uniform_tris(entry["coords"], ANCHOR_HIGHLIGHT_COLOR)
            _draw_label_hover_highlight(
                region_obj,
                live_ids,
                hover_id,
                skip_id=anchor_id,
                cache_token=("merge", version),
            )
        elif use_split_preview:
            live_ids = np.asarray(split_session["preview_ids"], dtype=np.int32)
            live_colors = split_session.get("preview_colors")
            version = split_session.get("preview_version", 0)
            pointer = region_obj.as_pointer()
            preview_key = ("split", pointer, version)
            cached = _MERGE_PREVIEW_CACHE.get("split_preview")
            if cached is None or cached.get("key") != preview_key:
                coords, colors = _build_region_draw_arrays(
                    region_obj,
                    region_ids=live_ids,
                    palette=live_colors,
                )
                _MERGE_PREVIEW_CACHE["split_preview"] = {
                    "key": preview_key,
                    "coords": coords,
                    "colors": colors,
                }
                cached = _MERGE_PREVIEW_CACHE["split_preview"]
            _draw_colored_tris(cached["coords"], cached["colors"])
            target_id = int(getattr(scene_props, "split_target_id", -1))
            if target_id >= 0:
                # 高亮仍基于进入时的目标领域面集合（原始 id）
                base_ids = split_session.get("base_ids")
                if base_ids is not None:
                    coords = _build_selection_highlight_coords(
                        region_obj,
                        np.asarray(base_ids, dtype=np.int32),
                        target_id,
                    )
                    _draw_uniform_tris(coords, ANCHOR_HIGHLIGHT_COLOR)
            base_ids = split_session.get("base_ids")
            hover_source = (
                np.asarray(base_ids, dtype=np.int32)
                if base_ids is not None
                else live_ids
            )
            _draw_label_hover_highlight(
                region_obj,
                hover_source,
                hover_id,
                skip_id=target_id,
                cache_token=("split", version),
            )
        else:
            coords, colors = _cached_region_draw_arrays(region_obj)
            _draw_colored_tris(coords, colors)
            target_id = -1
            region_ids = _read_region_ids(region_obj.data)
            if (
                scene_props.split_mode_active
                and int(getattr(scene_props, "split_target_id", -1)) >= 0
            ):
                target_id = int(scene_props.split_target_id)
                if region_ids is not None:
                    coords = _build_selection_highlight_coords(
                        region_obj,
                        region_ids,
                        target_id,
                    )
                    _draw_uniform_tris(coords, ANCHOR_HIGHLIGHT_COLOR)
            elif getattr(scene_props, "fit_mode_active", False):
                target_id = int(getattr(scene_props, "fit_target_id", -1))
                if target_id >= 0 and region_ids is not None:
                    coords = _build_selection_highlight_coords(
                        region_obj,
                        region_ids,
                        target_id,
                    )
                    _draw_uniform_tris(coords, ANCHOR_HIGHLIGHT_COLOR)
            if region_ids is not None:
                _draw_label_hover_highlight(
                    region_obj,
                    region_ids,
                    hover_id,
                    skip_id=target_id,
                    cache_token=(
                        "idle",
                        int(getattr(scene_props, "region_version", 0)),
                    ),
                )

    obj = scene_props.highlight_object
    if (
        obj is not None
        and obj.type == "MESH"
        and _is_object_selected(context, obj)
        and _is_region_mesh_overlay_visible(context, obj)
        and BOTTOM_FACES_ATTR in obj
    ):
        face_indices = list(obj[BOTTOM_FACES_ATTR])
        if face_indices:
            coords = _cached_bottom_coords(obj, face_indices)
            _draw_uniform_tris(coords, HIGHLIGHT_COLOR)


def draw_overlay_controls(self, context: bpy.types.Context) -> None:
    """向 Viewport Overlays 弹窗追加逆向工具控制项。"""
    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None:
        return

    layout = self.layout
    layout.separator()
    column = layout.column(align=True)
    column.label(text="逆向工具")
    column.prop(
        scene_props,
        "show_region_highlight",
        text="显示领域",
    )


def register_draw_handler() -> None:
    """注册视口绘制句柄，并清理由旧模块遗留的句柄。"""
    global _DRAW_HANDLE

    namespace = bpy.app.driver_namespace
    old_handle = namespace.get(DRAW_HANDLE_KEY)
    if old_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(
                old_handle,
                "WINDOW",
            )
        except (ReferenceError, ValueError):
            pass

    _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        draw_overlays,
        (),
        "WINDOW",
        "POST_VIEW",
    )
    namespace[DRAW_HANDLE_KEY] = _DRAW_HANDLE


def register_overlay_ui() -> None:
    """把覆盖层复选框追加到 Viewport Overlays。"""
    namespace = bpy.app.driver_namespace
    old_callback = namespace.get(OVERLAY_UI_KEY)
    if old_callback is not None:
        try:
            bpy.types.VIEW3D_PT_overlay.remove(old_callback)
        except (AttributeError, ValueError):
            pass

    bpy.types.VIEW3D_PT_overlay.append(draw_overlay_controls)
    namespace[OVERLAY_UI_KEY] = draw_overlay_controls


def unregister_draw_handler() -> None:
    """注销视口绘制句柄。"""
    global _DRAW_HANDLE
    unregister_label_draw_handler()
    unregister_split_draw_handler()
    namespace = bpy.app.driver_namespace
    handle = namespace.pop(DRAW_HANDLE_KEY, None) or _DRAW_HANDLE
    if handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _DRAW_HANDLE = None
    _BOTTOM_CACHE.clear()
    _REGION_CACHE.clear()


def unregister_overlay_ui() -> None:
    """从 Viewport Overlays 移除逆向工具控制项。"""
    namespace = bpy.app.driver_namespace
    callback = namespace.pop(OVERLAY_UI_KEY, None)
    if callback is None:
        callback = draw_overlay_controls
    try:
        bpy.types.VIEW3D_PT_overlay.remove(callback)
    except (AttributeError, ValueError):
        pass


# 兼容旧名称
draw_bottom_faces_overlay = draw_overlays
