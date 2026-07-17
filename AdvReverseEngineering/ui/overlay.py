# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""底面紫色高亮视口叠加绘制。"""

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

# 世界坐标缓存：signature 记录矩阵与网格状态，任一变化立即重建，
# 避免撤销/重做、数据更新后绘制陈旧坐标。
_OVERLAY_CACHE: dict[int, dict] = {}

# 紫色半透明 (R, G, B, A)
HIGHLIGHT_COLOR = (0.78, 0.22, 1.0, 0.55)

# 对象自定义属性键：底面索引列表
BOTTOM_FACES_ATTR = "are_bottom_faces"


def set_bottom_face_highlight(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    face_indices: list[int],
) -> None:
    """保存底面索引并缓存本地坐标，触发视口重绘。"""
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    old_obj = scene_props.highlight_object
    if old_obj is not None and old_obj != obj:
        if BOTTOM_FACES_ATTR in old_obj:
            del old_obj[BOTTOM_FACES_ATTR]
        _OVERLAY_CACHE.pop(old_obj.as_pointer(), None)

    obj[BOTTOM_FACES_ATTR] = list(face_indices)
    _OVERLAY_CACHE.pop(obj.as_pointer(), None)
    scene_props.highlight_object = obj
    _tag_view3d_redraw(context)


def clear_bottom_face_highlight(context: bpy.types.Context) -> None:
    """清除底面高亮。"""
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    old_obj = scene_props.highlight_object
    if old_obj and BOTTOM_FACES_ATTR in old_obj:
        del old_obj[BOTTOM_FACES_ATTR]
        _OVERLAY_CACHE.pop(old_obj.as_pointer(), None)
    scene_props.highlight_object = None
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


def _overlay_signature(obj: bpy.types.Object) -> tuple:
    """对象矩阵 + 网格状态签名，任一变化都需重建世界坐标。"""
    mesh = obj.data
    matrix = obj.matrix_world
    return (
        tuple(
            round(float(matrix[row][column]), 9)
            for row in range(4)
            for column in range(4)
        ),
        mesh.as_pointer(),
        len(mesh.vertices),
        len(mesh.polygons),
    )


def _build_world_coords(
    obj: bpy.types.Object,
    face_indices: list[int],
) -> np.ndarray:
    """
    用当前网格数据即时构建底面三角形世界坐标。

    顶点通过 foreach_get 一次性读取，再按三角扇索引取值，
    避免逐顶点访问 Blender API。
    """
    mesh = obj.data
    vert_count = len(mesh.vertices)
    polygon_count = len(mesh.polygons)
    if vert_count == 0 or polygon_count == 0:
        return np.empty((0, 3), dtype=np.float32)

    triangle_indices: list[int] = []
    for face_index in face_indices:
        if face_index < 0 or face_index >= polygon_count:
            continue
        vert_indices = mesh.polygons[face_index].vertices
        if len(vert_indices) < 3:
            continue
        anchor = vert_indices[0]
        for tri in range(1, len(vert_indices) - 1):
            triangle_indices.append(anchor)
            triangle_indices.append(vert_indices[tri])
            triangle_indices.append(vert_indices[tri + 1])

    if not triangle_indices:
        return np.empty((0, 3), dtype=np.float32)

    local_all = np.empty(vert_count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", local_all)
    local_all = local_all.reshape(vert_count, 3)

    index_array = np.asarray(triangle_indices, dtype=np.int64)
    valid = index_array < vert_count
    if not np.all(valid):
        index_array = index_array[valid]
        remainder = len(index_array) - len(index_array) % 3
        index_array = index_array[:remainder]
        if len(index_array) == 0:
            return np.empty((0, 3), dtype=np.float32)

    local = local_all[index_array]
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    world = local @ matrix[:3, :3].T + matrix[:3, 3]
    return world.astype(np.float32)


def _cached_world_coords(
    obj: bpy.types.Object,
    face_indices: list[int],
) -> np.ndarray:
    """按签名获取或重建世界坐标缓存。"""
    pointer = obj.as_pointer()
    signature = _overlay_signature(obj)
    entry = _OVERLAY_CACHE.get(pointer)
    if entry is not None and entry["signature"] == signature:
        return entry["coords"]

    coords = _build_world_coords(obj, face_indices)
    _OVERLAY_CACHE[pointer] = {
        "signature": signature,
        "coords": coords,
    }
    return coords


def draw_bottom_faces_overlay() -> None:
    """视口叠加回调：以紫色绘制当前选中物体的底面。"""
    context = bpy.context
    if context is None or context.scene is None:
        return

    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None:
        return
    if not scene_props.show_bottom_highlight:
        return

    obj = scene_props.highlight_object
    if obj is None or obj.type != "MESH":
        return
    if not _is_object_selected(context, obj):
        return
    if obj.hide_get() or obj.hide_viewport:
        return
    if BOTTOM_FACES_ATTR not in obj:
        return

    face_indices = list(obj[BOTTOM_FACES_ATTR])
    if not face_indices:
        return

    # 编辑模式下 obj.data 尚未同步 bmesh 修改，暂不绘制以免坐标错位。
    if obj.mode == "EDIT":
        return

    world_coords = _cached_world_coords(obj, face_indices)
    if len(world_coords) == 0:
        return

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": world_coords})

    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    try:
        shader.bind()
        shader.uniform_float("color", HIGHLIGHT_COLOR)
        batch.draw(shader)
    finally:
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


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
        draw_bottom_faces_overlay,
        (),
        "WINDOW",
        "POST_VIEW",
    )
    namespace[DRAW_HANDLE_KEY] = _DRAW_HANDLE


def register_overlay_ui() -> None:
    """把“显示底面”复选框追加到 Viewport Overlays。"""
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
    _OVERLAY_CACHE.clear()


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
