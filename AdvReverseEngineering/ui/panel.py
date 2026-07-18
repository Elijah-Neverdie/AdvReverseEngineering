# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""逆向工具侧边栏面板。"""

from __future__ import annotations

import bpy

from ..operators.update import get_display_version_text
from ..registration import SCENE_PROP_NAME, TAB_CATEGORY


def _prefs(context: bpy.types.Context):
    """读取插件偏好。"""
    addon = context.preferences.addons.get("AdvReverseEngineering")
    if addon is None:
        return None
    return addon.preferences


def _draw_foldout(layout, props, prop_name: str, title: str):
    """绘制可折叠卷展栏标题行，返回是否展开。"""
    row = layout.row(align=True)
    expanded = bool(getattr(props, prop_name))
    row.prop(
        props,
        prop_name,
        text=title,
        icon="TRIA_DOWN" if expanded else "TRIA_RIGHT",
        emboss=False,
    )
    return expanded


class ARE_PT_main(bpy.types.Panel):
    """逆向工具主面板，始终显示以确保侧边栏标签可见。"""

    bl_label = "逆向工具"
    bl_idname = "ARE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = TAB_CATEGORY

    def draw_header(self, context: bpy.types.Context) -> None:
        # 每次绘制读磁盘版本，避免「文件已更新、标题仍显示旧号」
        self.layout.label(text=f"v{get_display_version_text()}")

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        obj = context.active_object
        prefs = _prefs(context)
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)

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
                sync_box.prop(prefs, "github_branch", text="分支")

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
                elif prefs.update_check_state == "AHEAD":
                    update_row.enabled = False
                    server = prefs.latest_version or "?"
                    update_text = f"服务端版本为{server}"
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

        if scene_props is None:
            return

        layout.separator()
        if _draw_foldout(layout, scene_props, "show_simplify_section", "简化"):
            simplify_box = layout.box()
            mesh_ready = obj is not None and obj.type == "MESH"
            modal_busy = bool(
                scene_props.merge_mode_active
                or scene_props.split_mode_active
                or getattr(scene_props, "fit_mode_active", False)
            )
            simplify_box.enabled = not modal_busy
            simplify_box.prop(
                scene_props,
                "viewport_simplify_percent",
                text="视图简化 (%)",
            )
            if scene_props.simplify_active or scene_props.simplify_status:
                if scene_props.simplify_original_faces:
                    simplify_box.label(
                        text=(
                            f"面数 {scene_props.simplify_original_faces}"
                            f" → {scene_props.simplify_current_faces}"
                        ),
                        icon="MESH_DATA",
                    )
                if scene_props.simplify_status:
                    simplify_box.label(
                        text=scene_props.simplify_status,
                        icon=(
                            "TIME"
                            if scene_props.simplify_rebuild_pending
                            else "INFO"
                        ),
                    )

            apply_row = simplify_box.row()
            apply_row.scale_y = 1.2
            apply_row.enabled = (
                scene_props.simplify_active
                and not scene_props.simplify_rebuild_pending
            )
            apply_row.operator(
                "are.simplify_apply",
                text="应用",
                icon="CHECKMARK",
            )
            if not mesh_ready:
                simplify_box.label(text="请选中网格对象", icon="INFO")

        layout.separator()
        if _draw_foldout(layout, scene_props, "show_region_section", "领域"):
            region_box = layout.box()
            mesh_ready = obj is not None and obj.type == "MESH"
            merging = bool(scene_props.merge_mode_active)
            splitting = bool(scene_props.split_mode_active)
            fitting = bool(getattr(scene_props, "fit_mode_active", False))
            modal_busy = merging or splitting or fitting

            region_box.prop(
                scene_props,
                "region_wireframe_threshold",
                text="线框阈值",
            )
            region_box.prop(
                scene_props,
                "region_smooth_iterations",
                text="法线平滑",
            )
            region_box.prop(
                scene_props,
                "region_ignore_discrete",
                text="合并碎屑领域",
            )
            if scene_props.region_ignore_discrete:
                region_box.prop(
                    scene_props,
                    "region_min_area_ratio",
                    text="碎屑面积占比 (%)",
                )
            region_box.prop(
                scene_props,
                "region_fit_triangle_ratio",
                text="三边判定阈值 (%)",
            )

            button_row = region_box.row(align=True)
            button_row.scale_y = 1.2
            button_row.enabled = (
                mesh_ready
                and not modal_busy
                and (obj is None or obj.mode != "EDIT")
            )
            button_row.operator(
                "are.segment_regions",
                text="识别领域",
                icon="MOD_EDGESPLIT",
            )

            merge_row = region_box.row(align=True)
            merge_row.enabled = mesh_ready and not modal_busy
            merge_row.operator(
                "are.merge_regions",
                text="合并领域",
                icon="AUTOMERGE_ON",
            )

            split_row = region_box.row(align=True)
            split_row.enabled = mesh_ready and not modal_busy
            split_row.operator(
                "are.split_regions",
                text="拆分领域",
                icon="MOD_BOOLEAN",
            )

            fit_row = region_box.row(align=True)
            fit_row.enabled = mesh_ready and not modal_busy
            fit_row.operator(
                "are.fit_region",
                text="拟合领域",
                icon="MOD_SMOOTH",
            )

            clear_row = region_box.row(align=True)
            clear_row.enabled = mesh_ready and not modal_busy
            clear_row.operator(
                "are.clear_regions",
                text="清除领域",
                icon="X",
            )

            if merging:
                tip = region_box.box()
                tip.label(text="合并模式", icon="INFO")
                if scene_props.merge_status:
                    tip.label(text=scene_props.merge_status)
                confirm_row = tip.row()
                confirm_row.scale_y = 1.2
                confirm_row.operator(
                    "are.confirm_merge_regions",
                    text="确认",
                    icon="CHECKMARK",
                )
                if _draw_foldout(
                    tip,
                    scene_props,
                    "show_merge_help",
                    "操作说明",
                ):
                    help_box = tip.box()
                    help_box.label(text="首击设锚点，续击立即合并")
                    help_box.label(text="Ctrl+Z 撤销上次合并")
                    help_box.label(text="点击确认或 Enter 写入并退出")
                    help_box.label(text="Esc 取消")

            if splitting:
                tip = region_box.box()
                tip.label(text="拆分模式", icon="INFO")
                tip.prop(
                    scene_props,
                    "split_brush_radius",
                    text="笔刷半径 (px)",
                )
                if scene_props.split_status:
                    tip.label(text=scene_props.split_status)
                confirm_row = tip.row()
                confirm_row.scale_y = 1.2
                confirm_row.enabled = scene_props.split_phase == "PREVIEW"
                confirm_row.operator(
                    "are.confirm_split_regions",
                    text="确认拆分",
                    icon="CHECKMARK",
                )
                if _draw_foldout(
                    tip,
                    scene_props,
                    "show_split_help",
                    "操作说明",
                ):
                    help_box = tip.box()
                    help_box.label(text="1. 点击编号选择要拆分的领域")
                    help_box.label(text="2. [ ] 调节圆形笔刷粗细并涂红")
                    help_box.label(text="3. 松开 0.5 秒后自动分色预览")
                    help_box.label(text="Ctrl+Z 清除涂绘")
                    help_box.label(text="点击确认拆分或 Enter 写入并退出")
                    help_box.label(text="Esc 取消")

            if fitting:
                tip = region_box.box()
                tip.label(text="拟合模式", icon="INFO")
                if scene_props.fit_status:
                    tip.label(text=scene_props.fit_status)
                if scene_props.fit_status_detail:
                    tip.label(text=scene_props.fit_status_detail)
                if scene_props.fit_phase == "DEBUG_EDGES":
                    tip.label(text="Debug：各孤岛四条最长边")
                    build_row = tip.row()
                    build_row.scale_y = 1.2
                    build_row.operator(
                        "are.build_fit_surface",
                        text="拟合成面",
                        icon="MESH_GRID",
                    )
                if scene_props.fit_phase == "PREVIEW":
                    tip.label(
                        text=(
                            "三边曲面"
                            if scene_props.fit_topology == "TRI"
                            else "四边曲面"
                        )
                    )
                    tip.prop(
                        scene_props,
                        "fit_segments_u",
                        text=(
                            "底边段数"
                            if scene_props.fit_topology == "TRI"
                            else "U 向段数"
                        ),
                    )
                    tip.prop(
                        scene_props,
                        "fit_segments_v",
                        text=(
                            "长边段数"
                            if scene_props.fit_topology == "TRI"
                            else "V 向段数"
                        ),
                    )
                    confirm_row = tip.row()
                    confirm_row.scale_y = 1.2
                    confirm_row.operator(
                        "are.confirm_fit_region",
                        text="确认",
                        icon="CHECKMARK",
                    )
                if _draw_foldout(
                    tip,
                    scene_props,
                    "show_fit_help",
                    "操作说明",
                ):
                    help_box = tip.box()
                    help_box.label(text="1. 点击编号：显示各孤岛最长边")
                    help_box.label(text="2. 核对边线后点「拟合成面」或 Enter")
                    help_box.label(text="3. 滚轮 / Page 调节控制点")
                    help_box.label(text="4. 确认或 Enter 写入「拟合面」集合")
                    help_box.label(text="Esc 取消")

            if scene_props.region_status:
                region_box.label(
                    text=scene_props.region_status,
                    icon="INFO",
                )
                if scene_props.region_status_detail:
                    region_box.label(text=scene_props.region_status_detail)


classes = (ARE_PT_main,)
