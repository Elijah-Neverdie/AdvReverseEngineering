# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""视口投影与屏幕空间命中测试。"""

from __future__ import annotations

from mathutils import Vector


def project_world_to_region(region, rv3d, world_co) -> Vector | None:
    """世界坐标投影到 region 像素坐标；不可见时返回 None。"""
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    co = location_3d_to_region_2d(region, rv3d, Vector(world_co))
    if co is None:
        return None
    if co.x < -40 or co.y < -40:
        return None
    if co.x > region.width + 40 or co.y > region.height + 40:
        return None
    return co


def view_direction_to_point(rv3d, world_co) -> Vector:
    """从观察点指向世界坐标的单位方向。"""
    target = Vector(world_co)
    if getattr(rv3d, "is_perspective", True):
        origin = rv3d.view_matrix.inverted().translation
        direction = target - origin
    else:
        # 正交视图：沿视线方向前进。
        direction = rv3d.view_matrix.inverted().to_3x3() @ Vector((0.0, 0.0, -1.0))
    if direction.length < 1e-12:
        return Vector((0.0, 0.0, -1.0))
    return direction.normalized()


def is_label_facing_camera(normal, view_dir, min_dot: float = 0.05) -> bool:
    """面法线朝向镜头时才显示编号。"""
    return float(Vector(normal).dot(-Vector(view_dir))) >= float(min_dot)


def is_world_point_occluded(
    depsgraph,
    origin,
    target,
    exclude_obj=None,
    epsilon: float = 1e-4,
) -> bool:
    """
    从观察点到标签点做场景射线，若先撞到其他几何则视为遮挡。

    终点附近的命中不算遮挡，避免标签自身所在面误判。
    """
    import bpy

    start = Vector(origin)
    end = Vector(target)
    direction = end - start
    distance = float(direction.length)
    if distance < 1e-8:
        return True
    direction.normalize()
    hit, location, _normal, _index, hit_obj, _matrix = bpy.context.scene.ray_cast(
        depsgraph,
        start,
        direction,
        distance=distance,
    )
    if not hit:
        return False
    remaining = float((end - Vector(location)).length)
    if remaining <= max(epsilon, distance * 0.02):
        return False
    if exclude_obj is not None and hit_obj == exclude_obj and remaining <= distance * 0.08:
        # 擦到目标物体表面但仍接近标签锚点时，视为可见。
        return False
    return True


def hit_test_labels(
    mouse_x: float,
    mouse_y: float,
    labels: list[dict],
    radius_px: float = 16.0,
) -> int | None:
    """
    屏幕空间命中最近标签。

    labels 元素需含: id, screen_xy(Vector|None), visible(bool)
    """
    best_id: int | None = None
    best_dist = radius_px * radius_px
    for label in labels:
        if not label.get("visible", False):
            continue
        screen = label.get("screen_xy")
        if screen is None:
            continue
        dx = float(screen.x) - float(mouse_x)
        dy = float(screen.y) - float(mouse_y)
        dist = dx * dx + dy * dy
        if dist <= best_dist:
            best_dist = dist
            best_id = int(label["id"])
    return best_id
