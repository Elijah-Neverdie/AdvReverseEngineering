# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""逆向工具侧边栏面板。"""

from __future__ import annotations

import bpy

from .. import bl_info
from ..registration import SCENE_PROP_NAME, TAB_CATEGORY


VERSION_TEXT = ".".join(str(value) for value in bl_info["version"])


def _prefs(context: bpy.types.Context):
    """读取插件偏好。"""
    addon = context.preferences.addons.get("AdvReverseEngineering")
    if addon is None:
        return None
    return addon.preferences


class ARE_PT_main(bpy.types.Panel):
    """逆向工具主面板，始终显示以确保侧边栏标签可见。"""

    bl_label = f"逆向工具  v{VERSION_TEXT}"
    bl_idname = "ARE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = TAB_CATEGORY

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        obj = context.active_object
        prefs = _prefs(context)

        # GitHub 同步位于工具顶部，默认收起。
        if prefs is not None:
            foldout = layout.row(align=True)
            foldout.prop(
                prefs,
                "show_github_sync",
                text="GitHub 同步",
                icon=(
                    "TRIA_DOWN"
                    if prefs.show_github_sync
                    else "TRIA_RIGHT"
                ),
                emboss=False,
            )

            if prefs.show_github_sync:
                sync_box = layout.box()
                sync_box.prop(prefs, "github_owner")
                sync_box.prop(prefs, "github_repo")
                sync_box.prop(prefs, "github_branch")

                if prefs.update_check_message:
                    sync_box.label(
                        text=prefs.update_check_message,
                        icon=(
                            "ERROR"
                            if prefs.update_check_state == "ERROR"
                            else "INFO"
                        ),
                    )

                update_row = sync_box.row()
                if prefs.update_check_state == "AVAILABLE":
                    update_row.enabled = True
                    update_text = f"更新到 v{prefs.latest_version}"
                elif prefs.update_check_state == "CHECKING":
                    update_row.enabled = False
                    update_text = "正在检查更新…"
                elif prefs.update_check_state == "UPDATED":
                    update_row.enabled = False
                    update_text = "已更新，请重启 Blender"
                elif prefs.update_check_state == "ERROR":
                    update_row.enabled = False
                    update_text = "检查更新失败"
                elif prefs.update_check_state == "CURRENT":
                    update_row.enabled = False
                    update_text = "当前为最新版"
                else:
                    update_row.enabled = False
                    update_text = "尚未检查更新"

                update_row.operator(
                    "are.update_from_github",
                    text=update_text,
                    icon="IMPORT",
                )

                retry_row = sync_box.row()
                retry_row.enabled = (
                    prefs.update_check_state != "CHECKING"
                )
                retry_row.operator(
                    "are.check_github_update",
                    text="重新检查更新",
                    icon="FILE_REFRESH",
                )

            layout.separator()

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
        region_box = layout.box()
        region_box.label(text="领域分割", icon="FACESEL")
        mesh_ready = obj is not None and obj.type == "MESH"
        if scene_props is not None:
            region_box.prop(
                scene_props,
                "region_normal_threshold",
                text="法线阈值 (°)",
            )
            region_box.prop(
                scene_props,
                "region_smooth_iterations",
                text="法线平滑",
            )
            region_box.prop(
                scene_props,
                "region_ignore_discrete",
                text="忽略离散面",
            )
            if scene_props.region_ignore_discrete:
                region_box.prop(
                    scene_props,
                    "region_min_area_ratio",
                    text="最小面积占比 (%)",
                )

            button_row = region_box.row(align=True)
            button_row.scale_y = 1.2
            button_row.enabled = mesh_ready and (
                obj is None or obj.mode != "EDIT"
            )
            button_row.operator(
                "are.segment_regions",
                text="识别领域",
                icon="MOD_EDGESPLIT",
            )
            clear_row = region_box.row(align=True)
            clear_row.enabled = mesh_ready
            clear_row.operator(
                "are.clear_regions",
                text="清除领域",
                icon="X",
            )

            if scene_props.region_status:
                region_box.label(
                    text=scene_props.region_status,
                    icon="INFO",
                )
                if scene_props.region_status_detail:
                    region_box.label(text=scene_props.region_status_detail)
        else:
            region_box.label(text="属性未就绪", icon="ERROR")


classes = (ARE_PT_main,)
