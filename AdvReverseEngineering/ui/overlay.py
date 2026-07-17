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
DRAW_HANDLE_KEY = "AdvReverseEngineering.bottom_draw_handle"
OVERLAY_UI_KEY = "AdvReverseEngineering.overlay_ui_callback"

# 世界坐标 / 颜色缓存：signature 记录矩阵、网格与领域版本。
_BOTTOM_CACHE: dict[int, dict] = {}
_REGION_CACHE: dict[int, dict] = {}

# 紫色半透明 (R, G, B, A)
HIGHLIGHT_COLOR = (0.78, 0.22, 1.0, 0.55)

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
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建领域绘制用世界坐标与逐顶点颜色。

    忽略标签（<0）的面不进入绘制缓冲。
    """
    mesh = obj.data
    region_ids = _read_region_ids(mesh)
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
    palette = _read_region_colors(obj, region_count)

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
    """视口叠加回调：绘制领域多色与底面紫色。"""
    context = bpy.context
    if context is None or context.scene is None:
        return

    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None:
        return

    # 先画领域，再画底面，保证紫色底面叠在上层更易辨认。
    if scene_props.show_region_highlight:
        region_obj = scene_props.region_object
        if (
            region_obj is not None
            and region_obj.type == "MESH"
            and _is_object_selected(context, region_obj)
            and not region_obj.hide_get()
            and not region_obj.hide_viewport
            and region_obj.mode != "EDIT"
            and region_obj.data.attributes.get(REGION_ID_ATTR) is not None
        ):
            coords, colors = _cached_region_draw_arrays(region_obj)
            _draw_colored_tris(coords, colors)

    if scene_props.show_bottom_highlight:
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
        "show_bottom_highlight",
        text="显示底面",
    )
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
