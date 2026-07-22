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
                or getattr(scene_props, "remove_mode_active", False)
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
            removing = bool(getattr(scene_props, "remove_mode_active", False))
            splitting = bool(scene_props.split_mode_active)
            fitting = bool(getattr(scene_props, "fit_mode_active", False))
            curve_splitting = bool(
                getattr(scene_props, "curve_split_mode_active", False)
            )
            curve_fitting = bool(
                getattr(scene_props, "curve_fit_mode_active", False)
            )
            modal_busy = (
                merging
                or removing
                or splitting
                or fitting
                or curve_splitting
                or curve_fitting
            )

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

            remove_row = region_box.row(align=True)
            remove_row.enabled = mesh_ready and not modal_busy
            remove_row.operator(
                "are.remove_regions",
                text="移除领域",
                icon="TRASH",
            )

            fit_row = region_box.row(align=True)
            fit_row.enabled = mesh_ready and not modal_busy
            fit_row.operator(
                "are.fit_region",
                text="拟合领域",
                icon="MOD_SMOOTH",
            )

            curve_row = region_box.row(align=True)
            has_curve = any(
                getattr(item, "type", "") == "CURVE"
                for item in context.selected_objects
            )
            curve_count = sum(
                1
                for item in context.selected_objects
                if getattr(item, "type", "") == "CURVE"
            )
            edit_row = curve_row.row(align=True)
            split_row = edit_row.row(align=True)
            split_row.enabled = bool(has_curve) and not modal_busy
            split_row.operator(
                "are.split_fit_curve",
                text="拆分曲线",
                icon="MOD_EDGESPLIT",
            )
            fit_surf_row = edit_row.row(align=True)
            fit_surf_row.enabled = curve_count in (3, 4) and not modal_busy
            fit_surf_row.operator(
                "are.fit_bezier_curve",
                text="拟合曲面",
                icon="MESH_GRID",
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

            if removing:
                tip = region_box.box()
                tip.label(text="移除模式", icon="INFO")
                if scene_props.remove_status:
                    tip.label(text=scene_props.remove_status)
                confirm_row = tip.row()
                confirm_row.scale_y = 1.2
                confirm_row.operator(
                    "are.confirm_remove_regions",
                    text="确认",
                    icon="CHECKMARK",
                )
                if _draw_foldout(
                    tip,
                    scene_props,
                    "show_remove_help",
                    "操作说明",
                ):
                    help_box = tip.box()
                    help_box.label(text="点击编号移除该领域")
                    help_box.label(text="被移除面变为未标记")
                    help_box.label(text="Ctrl+Z 撤销上次移除")
                    help_box.label(text="点击确认或 Enter 写入并退出")
                    help_box.label(text="Esc 取消")

            if splitting:
                tip = region_box.box()
                tip.label(text="拆分模式", icon="INFO")
                tip.prop(
                    scene_props,
                    "split_hard_threshold",
                    text="线框阈值",
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
                    help_box.label(text="2. Ctrl+滚轮调线框阈值（同识别领域）")
                    help_box.label(text="3. 增大阈值可显示曲面上的缓棱")
                    help_box.label(text="4. 多次点选领域内部折棱（勿点分界）")
                    help_box.label(text="5. 选中段会合并成一条切线")
                    help_box.label(text="6. 出现分色预览后再确认拆分")
                    help_box.label(text="再点已选红线可去掉该段")
                    help_box.label(text="Ctrl+Z 清除全部选中边")
                    help_box.label(text="点击确认拆分或 Enter 写入并退出")
                    help_box.label(text="Esc 取消")

            if fitting:
                tip = region_box.box()
                tip.label(text="拟合模式（外轮廓曲线）", icon="INFO")
                if scene_props.fit_status:
                    tip.label(text=scene_props.fit_status)
                if scene_props.fit_status_detail:
                    tip.label(text=scene_props.fit_status_detail)
                tip.prop(
                    scene_props,
                    "fit_island_min_perimeter",
                    text="碎岛周长阈值 (%)",
                )
                if _draw_foldout(
                    tip,
                    scene_props,
                    "show_fit_help",
                    "操作说明",
                ):
                    help_box = tip.box()
                    help_box.label(text="1. 点击领域编号 → 生成可编辑外轮廓曲线")
                    help_box.label(text="2. 曲线写入「拟合曲线」集合，可直接编辑")
                    help_box.label(text="3. 生成后自动选中该曲线")
                    help_box.label(text="3. 物体属性 are_fit_region_id 标记领域")
                    help_box.label(text="Shift+点击多选后按 Enter 确认")
                    help_box.label(text="Esc 取消 · 不再焊接/内角/成面")

            if curve_splitting:
                tip = region_box.box()
                tip.label(text="拆分曲线", icon="INFO")
                tip.prop(
                    scene_props,
                    "curve_split_angle",
                    text="折角阈值 (°)",
                )
                if scene_props.curve_split_status:
                    tip.label(text=scene_props.curve_split_status)
                confirm_row = tip.row()
                confirm_row.scale_y = 1.2
                confirm_row.operator(
                    "are.confirm_split_fit_curve",
                    text="确认拆分",
                    icon="CHECKMARK",
                )
                if _draw_foldout(
                    tip,
                    scene_props,
                    "show_curve_split_help",
                    "操作说明",
                ):
                    help_box = tip.box()
                    help_box.label(text="1. 选中拟合外轮廓曲线后点「拆分曲线」")
                    help_box.label(text="2. 折角处断开，连续段用不同颜色标记")
                    help_box.label(text="3. Ctrl+滚轮调节折角阈值")
                    help_box.label(text="4. 左下角显示将拆分为几条")
                    help_box.label(text="5. 确认前禁止点选/取消选中其他物体")
                    help_box.label(text="6. 确认后全选拆分结果（无粗细）")
                    help_box.label(text="Enter / 确认拆分 · Esc 取消")

            if curve_fitting:
                tip = region_box.box()
                tip.label(text="拟合曲面（贝塞尔边界预览）", icon="INFO")
                tip.prop(
                    scene_props,
                    "curve_fit_controls",
                    text="控制点数A",
                )
                tip.prop(
                    scene_props,
                    "curve_fit_controls_b",
                    text="控制点数B",
                )
                mode_row = tip.row(align=True)
                mode_row.prop(
                    scene_props,
                    "curve_fit_similar",
                    text="相似模式",
                )
                mode_row.prop(
                    scene_props,
                    "curve_fit_stitch_open",
                    text="缝合开口",
                )
                if scene_props.curve_fit_status:
                    tip.label(text=scene_props.curve_fit_status)
                confirm_row = tip.row()
                confirm_row.scale_y = 1.2
                confirm_row.operator(
                    "are.confirm_fit_bezier_curve",
                    text="确认生成曲面",
                    icon="CHECKMARK",
                )
                if _draw_foldout(
                    tip,
                    scene_props,
                    "show_curve_fit_help",
                    "操作说明",
                ):
                    help_box = tip.box()
                    help_box.label(text="1. 选中 3/4 条曲线后点「拟合曲面」")
                    help_box.label(text="2. 自动进入编辑模式，可拖锚点/手柄微调")
                    help_box.label(text="3. Ctrl+滚轮调组A点数；Shift+滚轮调组B")
                    help_box.label(text="4. 按 S 切换相似；按 V 切换缝合开口")
                    help_box.label(text="四条端点近闭合：对边同色，相似=对边两两")
                    help_box.label(text="缝合开口：切向延伸缺口两端至交点封闭")
                    help_box.label(text="调点数/相似/缝合会重新拟合并覆盖手动微调")
                    help_box.label(text="确认后按当前贝塞尔生成拟合曲面并删除边界")
                    help_box.label(text="曲面沿用领域显示色")
                    help_box.label(text="Enter / 确认生成曲面 · Esc 取消")

            if scene_props.region_status:
                region_box.label(
                    text=scene_props.region_status,
                    icon="INFO",
                )
                if scene_props.region_status_detail:
                    region_box.label(text=scene_props.region_status_detail)


classes = (ARE_PT_main,)
