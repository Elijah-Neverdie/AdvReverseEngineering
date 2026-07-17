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
