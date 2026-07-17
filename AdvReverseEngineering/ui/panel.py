# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""逆向工具侧边栏面板。"""

from __future__ import annotations

import bpy

from ..registration import SCENE_PROP_NAME, TAB_CATEGORY


def _prefs(context: bpy.types.Context):
    """读取插件偏好。"""
    addon = context.preferences.addons.get("AdvReverseEngineering")
    if addon is None:
        return None
    return addon.preferences


class ARE_PT_main(bpy.types.Panel):
    """逆向工具主面板，始终显示以确保侧边栏标签可见。"""

    bl_label = "逆向工具"
    bl_idname = "ARE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = TAB_CATEGORY

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        obj = context.active_object

        if obj is None or obj.type != "MESH":
            layout.label(text="请选中网格对象", icon="INFO")

        row = layout.row()
        row.scale_y = 1.4
        row.enabled = obj is not None and obj.type == "MESH"
        row.operator(
            "are.auto_orient",
            text="自动摆正",
            icon="ORIENTATION_GLOBAL",
        )

        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is not None and scene_props.orientation_status:
            status_box = layout.box()
            status_box.label(
                text=scene_props.orientation_status,
                icon="INFO",
            )
            status_box.label(text=scene_props.orientation_status_detail)
            status_box.label(text=scene_props.orientation_status_next)

        layout.separator()
        box = layout.box()
        box.label(text="GitHub 同步", icon="URL")
        prefs = _prefs(context)
        if prefs is not None:
            box.prop(prefs, "github_owner")
            box.prop(prefs, "github_repo")
            box.prop(prefs, "github_branch")
        box.operator(
            "are.update_from_github",
            text="从 GitHub 更新",
            icon="FILE_REFRESH",
        )


classes = (ARE_PT_main,)
