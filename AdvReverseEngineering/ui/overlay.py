# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""底面紫色高亮视口叠加绘制。"""

from __future__ import annotations

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from ..registration import SCENE_PROP_NAME

# 视口绘制句柄
_DRAW_HANDLE = None
_OVERLAY_CACHE: dict[int, list[tuple[float, float, float]]] = {}

# 紫色半透明 (R, G, B, A)
HIGHLIGHT_COLOR = (0.78, 0.22, 1.0, 0.45)

# 对象自定义属性键：底面索引列表
BOTTOM_FACES_ATTR = "are_bottom_faces"


def set_bottom_face_highlight(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    face_indices: list[int],
) -> None:
    """保存底面索引并绑定到场景，触发视口重绘。"""
    obj[BOTTOM_FACES_ATTR] = face_indices
    _OVERLAY_CACHE[obj.as_pointer()] = _collect_bottom_face_coords(
        obj,
        face_indices,
    )
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    scene_props.highlight_object = obj

    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def clear_bottom_face_highlight(context: bpy.types.Context) -> None:
    """清除底面高亮。"""
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    old_obj = scene_props.highlight_object
    if old_obj and BOTTOM_FACES_ATTR in old_obj:
        del old_obj[BOTTOM_FACES_ATTR]
        _OVERLAY_CACHE.pop(old_obj.as_pointer(), None)
    scene_props.highlight_object = None


def _collect_bottom_face_coords(
    obj: bpy.types.Object,
    face_indices: list[int],
) -> list[tuple[float, float, float]]:
    """将底面多边形三角化，收集世界空间顶点坐标。"""
    mesh = obj.data
    matrix = obj.matrix_world
    coords: list[tuple[float, float, float]] = []
    polygon_count = len(mesh.polygons)
    for face_index in face_indices:
        if face_index < 0 or face_index >= polygon_count:
            continue
        polygon = mesh.polygons[face_index]

        vert_indices = polygon.vertices
        if len(vert_indices) < 3:
            continue

        v0 = matrix @ mesh.vertices[vert_indices[0]].co
        for tri in range(1, len(vert_indices) - 1):
            v1 = matrix @ mesh.vertices[vert_indices[tri]].co
            v2 = matrix @ mesh.vertices[vert_indices[tri + 1]].co
            coords.append((v0.x, v0.y, v0.z))
            coords.append((v1.x, v1.y, v1.z))
            coords.append((v2.x, v2.y, v2.z))

    return coords


def draw_bottom_faces_overlay() -> None:
    """视口叠加回调：以紫色绘制底面。"""
    context = bpy.context
    if context is None or context.scene is None:
        return

    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None:
        return

    obj = scene_props.highlight_object
    if obj is None or obj.type != "MESH":
        return
    if BOTTOM_FACES_ATTR not in obj:
        return

    face_indices = obj[BOTTOM_FACES_ATTR]
    if not face_indices:
        return

    coords = _OVERLAY_CACHE.get(obj.as_pointer())
    if coords is None:
        coords = _collect_bottom_face_coords(obj, list(face_indices))
        _OVERLAY_CACHE[obj.as_pointer()] = coords
    if not coords:
        return

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": coords})

    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    shader.bind()
    shader.uniform_float("color", HIGHLIGHT_COLOR)
    batch.draw(shader)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def register_draw_handler() -> None:
    """注册视口绘制句柄。"""
    global _DRAW_HANDLE
    if _DRAW_HANDLE is not None:
        return
    _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        draw_bottom_faces_overlay,
        (),
        "WINDOW",
        "POST_VIEW",
    )


def unregister_draw_handler() -> None:
    """注销视口绘制句柄。"""
    global _DRAW_HANDLE
    if _DRAW_HANDLE is None:
        return
    bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
    _DRAW_HANDLE = None
    _OVERLAY_CACHE.clear()
