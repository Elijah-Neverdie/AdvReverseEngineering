# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""底面紫色高亮与领域多色视口叠加绘制。"""

from __future__ import annotations

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

# 合并模式标签会话（由 merge operator 写入）
_MERGE_LABEL_SESSION: dict | None = None
# 拆分模式笔迹会话
_SPLIT_STROKE_SESSION: dict | None = None

# 紫色半透明 (R, G, B, A)
HIGHLIGHT_COLOR = (0.78, 0.22, 1.0, 0.55)
ANCHOR_HIGHLIGHT_COLOR = (1.0, 0.55, 0.12, 0.42)
HOVER_HIGHLIGHT_COLOR = (0.35, 0.75, 1.0, 0.28)
LABEL_RADIUS_PX = 16.0
LABEL_OFFSET_Y = 28.0
LEADER_ENDPOINT_RADIUS_PX = 3.5

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


def get_merge_label_session() -> dict | None:
    """读取合并模式编号标签会话。"""
    return _MERGE_LABEL_SESSION


def update_merge_label_projections(context: bpy.types.Context) -> None:
    """根据当前视角刷新标签屏幕坐标，并过滤背面/遮挡编号。"""
    from mathutils import Vector

    from ..utils.viewport import (
        is_label_facing_camera,
        is_world_point_occluded,
        project_world_to_region,
        view_direction_to_point,
    )

    session = _MERGE_LABEL_SESSION
    if session is None:
        return
    region = context.region
    rv3d = getattr(context, "region_data", None)
    if region is None or rv3d is None or region.type != "WINDOW":
        space = getattr(context, "space_data", None)
        if space is None or space.type != "VIEW_3D":
            return
        rv3d = space.region_3d
        # draw handler 下尽量取当前 WINDOW region
        if region is None or region.type != "WINDOW":
            for area_region in getattr(space, "regions", []):
                if area_region.type == "WINDOW":
                    region = area_region
                    break
    if region is None or rv3d is None:
        return

    depsgraph = context.evaluated_depsgraph_get()
    if getattr(rv3d, "is_perspective", True):
        view_origin = rv3d.view_matrix.inverted().translation
    else:
        # 正交：从标签沿视线反方向退一段作为射线起点。
        view_origin = None

    exclude_obj = None
    object_name = session.get("object_name")
    if object_name:
        exclude_obj = bpy.data.objects.get(object_name)

    labels = session.get("labels", [])
    for label in labels:
        world = label.get("world_co")
        normal = label.get("normal")
        if world is None or normal is None:
            label["visible"] = False
            label["screen_xy"] = None
            label["face_screen_xy"] = None
            continue

        view_dir = view_direction_to_point(rv3d, world)
        if not is_label_facing_camera(normal, view_dir):
            label["visible"] = False
            label["screen_xy"] = None
            label["face_screen_xy"] = None
            continue

        if view_origin is None:
            # 正交视图：从标签朝向相机方向退回一段距离。
            ray_origin = Vector(world) - view_dir * 1000.0
        else:
            ray_origin = view_origin

        if is_world_point_occluded(
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
            label["face_screen_xy"] = face_screen if face_screen is not None else screen


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
    if scene_props is None or not scene_props.merge_mode_active:
        return
    session = _MERGE_LABEL_SESSION
    if session is None:
        return

    update_merge_label_projections(context)
    anchor_id = int(scene_props.merge_anchor_id)
    hover_id = int(scene_props.merge_hover_id)

    gpu.state.blend_set("ALPHA")
    try:
        for label in session.get("labels", []):
            if not label.get("visible", False):
                continue
            screen = label.get("screen_xy")
            if screen is None:
                continue
            region_id = int(label["id"])
            if region_id == anchor_id:
                fill = (1.0, 0.55, 0.12, 0.92)
                leader = (1.0, 0.65, 0.25, 0.95)
            elif region_id == hover_id:
                fill = (0.35, 0.75, 1.0, 0.90)
                leader = (0.45, 0.82, 1.0, 0.90)
            else:
                fill = (0.12, 0.12, 0.14, 0.82)
                leader = (0.85, 0.85, 0.88, 0.75)

            face_screen = label.get("face_screen_xy") or screen
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


def set_split_stroke_session(session: dict | None) -> None:
    """设置或清除拆分笔迹预览会话。"""
    global _SPLIT_STROKE_SESSION
    _SPLIT_STROKE_SESSION = session


def get_split_stroke_session() -> dict | None:
    """读取拆分笔迹预览会话。"""
    return _SPLIT_STROKE_SESSION


def draw_split_stroke_preview() -> None:
    """POST_VIEW：深度感知红色笔迹与智能补全边。"""
    context = bpy.context
    if context is None or context.scene is None:
        return
    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None or not scene_props.split_mode_active:
        return
    session = _SPLIT_STROKE_SESSION
    if session is None:
        return

    strokes = session.get("strokes") or []
    if not strokes:
        return

    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    try:
        live_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        for stroke in strokes:
            poly = stroke.get("polyline_world")
            if poly is not None and len(poly) >= 2:
                coords = [tuple(map(float, p)) for p in np.asarray(poly)]
                batch = batch_for_shader(
                    live_shader,
                    "LINE_STRIP",
                    {"pos": coords},
                )
                live_shader.bind()
                if stroke.get("live"):
                    live_shader.uniform_float(
                        "color",
                        (1.0, 0.25, 0.2, 0.85),
                    )
                    gpu.state.line_width_set(1.5)
                else:
                    live_shader.uniform_float(
                        "color",
                        (1.0, 0.2, 0.15, 0.70),
                    )
                    gpu.state.line_width_set(1.5)
                batch.draw(live_shader)

            completed = stroke.get("completed_edges_world")
            if completed is not None and len(completed) > 0:
                segments = []
                for edge in np.asarray(completed):
                    if len(edge) < 2:
                        continue
                    segments.append(tuple(map(float, edge[0])))
                    segments.append(tuple(map(float, edge[1])))
                if segments:
                    batch = batch_for_shader(
                        live_shader,
                        "LINES",
                        {"pos": segments},
                    )
                    live_shader.bind()
                    live_shader.uniform_float(
                        "color",
                        (1.0, 0.05, 0.05, 0.95),
                    )
                    gpu.state.line_width_set(3.0)
                    batch.draw(live_shader)
    finally:
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


def register_split_draw_handler() -> None:
    """注册拆分笔迹 POST_VIEW 绘制。"""
    global _SPLIT_DRAW_HANDLE
    namespace = bpy.app.driver_namespace
    old = namespace.get(SPLIT_DRAW_HANDLE_KEY)
    if old is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(old, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _SPLIT_DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        draw_split_stroke_preview,
        (),
        "WINDOW",
        "POST_VIEW",
    )
    namespace[SPLIT_DRAW_HANDLE_KEY] = _SPLIT_DRAW_HANDLE


def unregister_split_draw_handler() -> None:
    """注销拆分笔迹绘制句柄。"""
    global _SPLIT_DRAW_HANDLE, _SPLIT_STROKE_SESSION
    namespace = bpy.app.driver_namespace
    handle = namespace.pop(SPLIT_DRAW_HANDLE_KEY, None) or _SPLIT_DRAW_HANDLE
    if handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _SPLIT_DRAW_HANDLE = None
    _SPLIT_STROKE_SESSION = None


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
        draw_region_merge_labels,
        (),
        "WINDOW",
        "POST_PIXEL",
    )
    namespace[LABEL_DRAW_HANDLE_KEY] = _LABEL_DRAW_HANDLE


def unregister_label_draw_handler() -> None:
    """注销领域编号绘制句柄。"""
    global _LABEL_DRAW_HANDLE, _MERGE_LABEL_SESSION
    namespace = bpy.app.driver_namespace
    handle = namespace.pop(LABEL_DRAW_HANDLE_KEY, None) or _LABEL_DRAW_HANDLE
    if handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except (ReferenceError, ValueError):
            pass
    _LABEL_DRAW_HANDLE = None
    _MERGE_LABEL_SESSION = None


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
    """构建底面三角形世界坐标。"""
    mesh = obj.data
    if len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
        return np.empty((0, 3), dtype=np.float32)

    index_array = _triangulate_face_indices(mesh, face_indices)
    if len(index_array) == 0:
        return np.empty((0, 3), dtype=np.float32)

    local_all = _read_local_vertices(mesh)
    valid = index_array < len(local_all)
    if not np.all(valid):
        index_array = index_array[valid]
        remainder = len(index_array) - len(index_array) % 3
        index_array = index_array[:remainder]
        if len(index_array) == 0:
            return np.empty((0, 3), dtype=np.float32)

    return _local_to_world(obj, local_all[index_array])


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


def _build_region_draw_arrays(
    obj: bpy.types.Object,
    region_ids: np.ndarray | None = None,
    palette: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建领域绘制用世界坐标与逐顶点颜色。

    忽略标签（<0）的面不进入绘制缓冲。
    可传入内存中的 region_ids/palette（合并预览）。
    """
    mesh = obj.data
    if region_ids is None:
        region_ids = _read_region_ids(mesh)
    else:
        region_ids = np.asarray(region_ids, dtype=np.int32)

    if region_ids is None or len(region_ids) == 0:
        empty_pos = np.empty((0, 3), dtype=np.float32)
        empty_col = np.empty((0, 4), dtype=np.float32)
        return empty_pos, empty_col

    valid_faces = np.flatnonzero(region_ids >= 0)
    if len(valid_faces) == 0:
        empty_pos = np.empty((0, 3), dtype=np.float32)
        empty_col = np.empty((0, 4), dtype=np.float32)
        return empty_pos, empty_col

    region_count = int(region_ids.max()) + 1
    if palette is None:
        palette = _read_region_colors(obj, region_count)
    else:
        palette = np.asarray(palette, dtype=np.float32)
        if len(palette) < region_count:
            from ..algorithms.regions import generate_region_colors

            extra = generate_region_colors(region_count - len(palette))
            palette = np.vstack((palette, extra))

    # 按面展开三角扇，并为每个三角形顶点写入对应领域颜色。
    positions_local: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    local_all = _read_local_vertices(mesh)
    polygon_count = len(mesh.polygons)

    for face_index in valid_faces.tolist():
        if face_index < 0 or face_index >= polygon_count:
            continue
        region_id = int(region_ids[face_index])
        if region_id < 0 or region_id >= len(palette):
            continue
        vert_indices = mesh.polygons[face_index].vertices
        if len(vert_indices) < 3:
            continue
        color = palette[region_id]
        anchor = int(vert_indices[0])
        for tri in range(1, len(vert_indices) - 1):
            i1 = int(vert_indices[tri])
            i2 = int(vert_indices[tri + 1])
            tri_local = local_all[[anchor, i1, i2]]
            positions_local.append(tri_local)
            colors.append(np.broadcast_to(color, (3, 4)).copy())

    if not positions_local:
        empty_pos = np.empty((0, 3), dtype=np.float32)
        empty_col = np.empty((0, 4), dtype=np.float32)
        return empty_pos, empty_col

    local = np.vstack(positions_local)
    world = _local_to_world(obj, local)
    color_array = np.vstack(colors).astype(np.float32)
    return world, color_array


def _build_selection_highlight_coords(
    obj: bpy.types.Object,
    region_ids: np.ndarray,
    highlight_id: int,
) -> np.ndarray:
    """构建指定领域的三角面世界坐标，用于选中/悬停高亮。"""
    if highlight_id < 0:
        return np.empty((0, 3), dtype=np.float32)
    mesh = obj.data
    faces = np.flatnonzero(np.asarray(region_ids, dtype=np.int32) == highlight_id)
    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.float32)

    local_all = _read_local_vertices(mesh)
    positions_local: list[np.ndarray] = []
    polygon_count = len(mesh.polygons)
    for face_index in faces.tolist():
        if face_index < 0 or face_index >= polygon_count:
            continue
        vert_indices = mesh.polygons[face_index].vertices
        if len(vert_indices) < 3:
            continue
        anchor = int(vert_indices[0])
        for tri in range(1, len(vert_indices) - 1):
            i1 = int(vert_indices[tri])
            i2 = int(vert_indices[tri + 1])
            positions_local.append(local_all[[anchor, i1, i2]])
    if not positions_local:
        return np.empty((0, 3), dtype=np.float32)
    return _local_to_world(obj, np.vstack(positions_local))


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
    use_merge_preview = (
        bool(scene_props.merge_mode_active)
        and merge_session is not None
        and merge_session.get("region_ids") is not None
    )

    if (
        region_obj is not None
        and region_obj.type == "MESH"
        and _is_object_selected(context, region_obj)
        and not region_obj.hide_get()
        and not region_obj.hide_viewport
        and region_obj.mode != "EDIT"
        and (
            use_merge_preview
            or region_obj.data.attributes.get(REGION_ID_ATTR) is not None
        )
    ):
        if use_merge_preview:
            live_ids = np.asarray(merge_session["region_ids"], dtype=np.int32)
            live_colors = merge_session.get("colors")
            coords, colors = _build_region_draw_arrays(
                region_obj,
                region_ids=live_ids,
                palette=live_colors,
            )
            _draw_colored_tris(coords, colors)

            anchor_id = int(scene_props.merge_anchor_id)
            hover_id = int(scene_props.merge_hover_id)
            if hover_id >= 0 and hover_id != anchor_id:
                hover_coords = _build_selection_highlight_coords(
                    region_obj,
                    live_ids,
                    hover_id,
                )
                _draw_uniform_tris(hover_coords, HOVER_HIGHLIGHT_COLOR)
            if anchor_id >= 0:
                anchor_coords = _build_selection_highlight_coords(
                    region_obj,
                    live_ids,
                    anchor_id,
                )
                _draw_uniform_tris(anchor_coords, ANCHOR_HIGHLIGHT_COLOR)
        else:
            coords, colors = _cached_region_draw_arrays(region_obj)
            _draw_colored_tris(coords, colors)

    obj = scene_props.highlight_object
    if (
        obj is not None
        and obj.type == "MESH"
        and _is_object_selected(context, obj)
        and not obj.hide_get()
        and not obj.hide_viewport
        and obj.mode != "EDIT"
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
