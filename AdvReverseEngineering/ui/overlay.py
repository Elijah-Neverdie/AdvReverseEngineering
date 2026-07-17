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

# 缓存对象本地坐标三角面，绘制时再乘 matrix_world，确保跟随物体变换
_OVERLAY_CACHE: dict[int, np.ndarray] = {}

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
    _OVERLAY_CACHE[obj.as_pointer()] = np.asarray(
        _collect_bottom_face_local_coords(obj, face_indices),
        dtype=np.float32,
    )
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


def _collect_bottom_face_local_coords(
    obj: bpy.types.Object,
    face_indices: list[int],
) -> list[tuple[float, float, float]]:
    """将底面多边形三角化，缓存本地空间坐标。"""
    mesh = obj.data
    coords: list[tuple[float, float, float]] = []
    polygon_count = len(mesh.polygons)

    for face_index in face_indices:
        if face_index < 0 or face_index >= polygon_count:
            continue
        polygon = mesh.polygons[face_index]
        vert_indices = polygon.vertices
        if len(vert_indices) < 3:
            continue

        v0 = mesh.vertices[vert_indices[0]].co
        for tri in range(1, len(vert_indices) - 1):
            v1 = mesh.vertices[vert_indices[tri]].co
            v2 = mesh.vertices[vert_indices[tri + 1]].co
            coords.append((v0.x, v0.y, v0.z))
            coords.append((v1.x, v1.y, v1.z))
            coords.append((v2.x, v2.y, v2.z))

    return coords


def _is_object_selected(context: bpy.types.Context, obj: bpy.types.Object) -> bool:
    """仅当对象被选中时显示高亮。"""
    try:
        return bool(obj.select_get())
    except ReferenceError:
        return False


def _local_to_world(
    obj: bpy.types.Object,
    local_coords: np.ndarray,
) -> np.ndarray:
    """按对象当前 matrix_world 将缓存本地坐标转换为世界坐标。"""
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    local = np.asarray(local_coords, dtype=np.float64)
    ones = np.ones((len(local), 1), dtype=np.float64)
    homogeneous = np.hstack((local, ones))
    return (matrix @ homogeneous.T).T[:, :3].astype(np.float32)


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

    pointer = obj.as_pointer()
    coords = _OVERLAY_CACHE.get(pointer)
    if coords is None:
        coords = np.asarray(
            _collect_bottom_face_local_coords(obj, face_indices),
            dtype=np.float32,
        )
        _OVERLAY_CACHE[pointer] = coords
    if len(coords) == 0:
        return

    world_coords = _local_to_world(obj, coords)
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
