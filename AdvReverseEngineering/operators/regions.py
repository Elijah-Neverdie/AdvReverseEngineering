# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""领域自动识别、清除与模态合并操作符。"""

from __future__ import annotations

import bpy
import numpy as np

from ..algorithms.regions import (
    compute_region_centroids,
    merge_region_ids,
    segment_regions_by_normal,
)
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import (
    LABEL_RADIUS_PX,
    clear_region_highlight,
    get_merge_label_session,
    register_label_draw_handler,
    set_merge_label_session,
    set_region_highlight,
    unregister_label_draw_handler,
    update_merge_label_projections,
)
from ..utils.mesh import extract_face_topology, extract_mesh_data
from ..utils.progress import progress_scope
from ..utils.viewport import hit_test_labels


REGION_ID_ATTR = "are_region_id"
REGION_VERSION_ATTR = "are_region_version"
REGION_COLORS_ATTR = "are_region_colors"


def _ensure_region_attribute(mesh: bpy.types.Mesh):
    """获取或创建 FACE 域整型属性 are_region_id。"""
    attribute = mesh.attributes.get(REGION_ID_ATTR)
    if attribute is not None and attribute.domain == "FACE":
        return attribute
    if attribute is not None:
        mesh.attributes.remove(attribute)
    return mesh.attributes.new(
        name=REGION_ID_ATTR,
        type="INT",
        domain="FACE",
    )


def write_region_ids(
    mesh: bpy.types.Mesh,
    region_ids: np.ndarray,
) -> None:
    """将领域标签写入 Mesh FACE 属性。"""
    face_count = len(mesh.polygons)
    if face_count == 0:
        return
    values = np.asarray(region_ids, dtype=np.int32)
    if len(values) != face_count:
        raise ValueError("领域标签数量与面数量不一致")

    attribute = _ensure_region_attribute(mesh)
    flat = np.ascontiguousarray(values, dtype=np.int32)
    attribute.data.foreach_set("value", flat)
    mesh.update()


def clear_region_ids(mesh: bpy.types.Mesh) -> None:
    """移除领域 FACE 属性。"""
    attribute = mesh.attributes.get(REGION_ID_ATTR)
    if attribute is not None:
        mesh.attributes.remove(attribute)


def read_region_ids(mesh: bpy.types.Mesh) -> np.ndarray | None:
    """读取 FACE 域领域标签；不存在时返回 None。"""
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
    """读取对象上保存的领域调色板。"""
    from ..algorithms.regions import generate_region_colors

    raw = obj.get(REGION_COLORS_ATTR)
    if raw is not None:
        flat = np.asarray(list(raw), dtype=np.float32)
        if len(flat) >= region_count * 4:
            return flat[: region_count * 4].reshape(region_count, 4)
    return generate_region_colors(region_count)


def _bbox_lift(mesh_data) -> float:
    """标签上浮高度，取包围盒对角线的一小部分。"""
    vertices = mesh_data["vertices"]
    if len(vertices) == 0:
        return 0.01
    size = vertices.max(axis=0) - vertices.min(axis=0)
    return float(max(np.linalg.norm(size) * 0.02, 1e-4))


def _build_label_session(region_ids: np.ndarray, mesh_data) -> dict:
    """根据领域标签构建编号会话。"""
    centroids = compute_region_centroids(
        region_ids,
        mesh_data["face_centers"],
        mesh_data["areas"],
    )
    labels = []
    for region_id in sorted(centroids.keys()):
        labels.append(
            {
                "id": int(region_id),
                "world_co": centroids[region_id].copy(),
                "screen_xy": None,
                "visible": False,
            }
        )
    return {
        "labels": labels,
        "lift": _bbox_lift(mesh_data),
    }


class ARE_OT_segment_regions(bpy.types.Operator):
    """按法线阈值自动识别领域并以随机色标记。"""

    bl_idname = "are.segment_regions"
    bl_label = "识别领域"
    bl_description = (
        "按相邻面法线阈值分割领域，并可忽略面积过小的离散领域"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is not None and scene_props.merge_mode_active:
            return False
        return (
            obj is not None
            and obj.type == "MESH"
            and obj.mode != "EDIT"
        )

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)

        try:
            with progress_scope(context, "识别领域", 5) as step:
                step("读取网格几何")
                mesh_data = extract_mesh_data(obj)

                step("构建面邻接拓扑")
                topology = extract_face_topology(obj.data)

                step("法线阈值区域生长")
                min_ratio = (
                    float(scene_props.region_min_area_ratio) / 100.0
                    if scene_props.region_ignore_discrete
                    else 0.0
                )
                result = segment_regions_by_normal(
                    normals=mesh_data["normals"],
                    areas=mesh_data["areas"],
                    topology=topology,
                    angle_threshold_deg=float(
                        scene_props.region_normal_threshold
                    ),
                    ignore_discrete=bool(
                        scene_props.region_ignore_discrete
                    ),
                    min_area_ratio=min_ratio,
                    smooth_iterations=int(
                        scene_props.region_smooth_iterations
                    ),
                )

                step("写入领域属性")
                write_region_ids(obj.data, result["region_ids"])
                version = int(scene_props.region_version) + 1
                obj[REGION_VERSION_ATTR] = version
                obj[REGION_COLORS_ATTR] = (
                    result["colors"].astype(np.float32).ravel().tolist()
                )

                step("更新高亮与状态")
                scene_props.region_version = version
                scene_props.region_object = obj
                scene_props.region_count = int(result["region_count"])
                scene_props.region_ignored_face_count = int(
                    result["ignored_face_count"]
                )
                scene_props.region_ignored_region_count = int(
                    result["ignored_region_count"]
                )
                scene_props.region_status = (
                    f"已识别 {result['region_count']} 个领域"
                )
                if scene_props.region_ignore_discrete:
                    scene_props.region_status_detail = (
                        f"忽略 {result['ignored_region_count']} 个离散领域"
                        f"（{result['ignored_face_count']} 个面）"
                    )
                else:
                    scene_props.region_status_detail = "未启用忽略离散面"

                set_region_highlight(
                    context,
                    obj,
                    result["region_ids"],
                    result["colors"],
                )

        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"领域识别失败: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            (
                f"{scene_props.region_status}，"
                f"{scene_props.region_status_detail}"
            ),
        )
        return {"FINISHED"}


class ARE_OT_clear_regions(bpy.types.Operator):
    """清除当前对象的领域分割结果与高亮。"""

    bl_idname = "are.clear_regions"
    bl_label = "清除领域"
    bl_description = "清除领域标签、颜色缓存与视口高亮"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is not None and scene_props.merge_mode_active:
            return False
        return obj is not None and obj.type == "MESH"

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)

        clear_region_ids(obj.data)
        if REGION_VERSION_ATTR in obj:
            del obj[REGION_VERSION_ATTR]
        if REGION_COLORS_ATTR in obj:
            del obj[REGION_COLORS_ATTR]

        if scene_props.region_object == obj:
            scene_props.region_object = None
            scene_props.region_count = 0
            scene_props.region_ignored_face_count = 0
            scene_props.region_ignored_region_count = 0
            scene_props.region_status = ""
            scene_props.region_status_detail = ""
            scene_props.region_version = int(scene_props.region_version) + 1

        clear_region_highlight(context, obj)
        self.report({"INFO"}, "已清除领域标记")
        return {"FINISHED"}


class ARE_OT_confirm_merge_regions(bpy.types.Operator):
    """面板确认按钮：通知合并模态提交。"""

    bl_idname = "are.confirm_merge_regions"
    bl_label = "确认"
    bl_description = "结束合并领域并保留当前结果"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return scene_props is not None and scene_props.merge_mode_active

    def execute(self, context: bpy.types.Context):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.merge_confirm_requested = True
        return {"FINISHED"}


class ARE_OT_merge_regions(bpy.types.Operator):
    """
    模态合并领域。

    首击设锚点；续击立即合入锚点；点击空白结束当前组；
    Enter/确认提交；Esc/右键取消并恢复快照。
    """

    bl_idname = "are.merge_regions"
    bl_label = "合并领域"
    bl_description = (
        "在视口显示领域编号；首击为锚点，续击立即合并，"
        "空白换组，确认结束，Esc 取消"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _finish_mode(self, context: bpy.types.Context, cancelled: bool) -> set:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        unregister_label_draw_handler()
        set_merge_label_session(None)
        scene_props.merge_mode_active = False
        scene_props.merge_anchor_id = -1
        scene_props.merge_hover_id = -1
        scene_props.merge_confirm_requested = False
        if cancelled:
            scene_props.merge_status = "已取消合并"
        else:
            scene_props.merge_status = (
                f"合并完成，当前 {scene_props.region_count} 个领域"
            )
            scene_props.region_status = (
                f"已识别 {scene_props.region_count} 个领域"
            )
        self._tag_redraw(context)
        return {"CANCELLED"} if cancelled else {"FINISHED"}

    def _tag_redraw(self, context: bpy.types.Context) -> None:
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type in {"VIEW_3D", "UI"}:
                    area.tag_redraw()

    def _restore_snapshot(self, context: bpy.types.Context) -> None:
        obj = self._object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        write_region_ids(obj.data, self._snapshot_ids)
        obj[REGION_COLORS_ATTR] = (
            self._snapshot_colors.astype(np.float32).ravel().tolist()
        )
        obj[REGION_VERSION_ATTR] = int(self._snapshot_version)
        scene_props.region_version = int(self._snapshot_version)
        scene_props.region_count = int(self._snapshot_count)
        scene_props.region_object = obj
        set_region_highlight(
            context,
            obj,
            self._snapshot_ids,
            self._snapshot_colors,
        )

    def _apply_live(
        self,
        context: bpy.types.Context,
        region_ids: np.ndarray,
        colors: np.ndarray,
        region_count: int,
        anchor_id: int,
    ) -> None:
        obj = self._object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        write_region_ids(obj.data, region_ids)
        version = int(scene_props.region_version) + 1
        obj[REGION_VERSION_ATTR] = version
        obj[REGION_COLORS_ATTR] = colors.astype(np.float32).ravel().tolist()
        scene_props.region_version = version
        scene_props.region_count = int(region_count)
        scene_props.region_object = obj
        scene_props.merge_anchor_id = int(anchor_id)
        self._live_ids = region_ids
        self._live_colors = colors

        mesh_data = extract_mesh_data(obj)
        session = _build_label_session(region_ids, mesh_data)
        set_merge_label_session(session)
        set_region_highlight(context, obj, region_ids, colors)
        scene_props.merge_status = (
            f"锚点 {anchor_id} · 当前 {region_count} 个领域"
            if anchor_id >= 0
            else f"请选择锚点 · 当前 {region_count} 个领域"
        )
        self._tag_redraw(context)

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is not None and scene_props.merge_mode_active:
            return False
        if obj is None or obj.type != "MESH" or obj.mode == "EDIT":
            return False
        return obj.data.attributes.get(REGION_ID_ATTR) is not None

    def invoke(self, context: bpy.types.Context, event):
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        region_ids = read_region_ids(obj.data)
        if region_ids is None or not np.any(region_ids >= 0):
            self.report({"ERROR"}, "请先识别领域")
            return {"CANCELLED"}

        region_count = int(region_ids.max()) + 1
        colors = _read_region_colors(obj, region_count)
        self._object = obj
        self._snapshot_ids = region_ids.copy()
        self._snapshot_colors = colors.copy()
        self._snapshot_version = int(scene_props.region_version)
        self._snapshot_count = int(scene_props.region_count) or region_count
        self._live_ids = region_ids.copy()
        self._live_colors = colors.copy()

        mesh_data = extract_mesh_data(obj)
        session = _build_label_session(region_ids, mesh_data)
        set_merge_label_session(session)
        register_label_draw_handler()

        scene_props.merge_mode_active = True
        scene_props.merge_anchor_id = -1
        scene_props.merge_hover_id = -1
        scene_props.merge_confirm_requested = False
        scene_props.merge_status = (
            "首击设锚点，续击合并；空白换组；确认结束；Esc 取消"
        )
        scene_props.region_object = obj

        context.window_manager.modal_handler_add(self)
        self._tag_redraw(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context):
        self._restore_snapshot(context)
        return self._finish_mode(context, cancelled=True)

    def modal(self, context: bpy.types.Context, event):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        obj = self._object

        try:
            if obj is None or obj.name not in bpy.data.objects:
                return self.cancel(context)
            if obj.mode == "EDIT":
                return self.cancel(context)
        except ReferenceError:
            return self.cancel(context)

        if scene_props.merge_confirm_requested:
            scene_props.merge_confirm_requested = False
            self.report({"INFO"}, scene_props.merge_status or "合并完成")
            return self._finish_mode(context, cancelled=False)

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self.report({"INFO"}, "已取消合并")
            return self.cancel(context)

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self.report({"INFO"}, "合并完成")
            return self._finish_mode(context, cancelled=False)

        if context.space_data is None or context.space_data.type != "VIEW_3D":
            return {"PASS_THROUGH"}
        if context.region is None or context.region.type != "WINDOW":
            return {"PASS_THROUGH"}

        if event.type == "MOUSEMOVE":
            update_merge_label_projections(context)
            session = get_merge_label_session() or {"labels": []}
            hover = hit_test_labels(
                event.mouse_region_x,
                event.mouse_region_y,
                session.get("labels", []),
                LABEL_RADIUS_PX,
            )
            scene_props.merge_hover_id = -1 if hover is None else int(hover)
            self._tag_redraw(context)
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            update_merge_label_projections(context)
            session = get_merge_label_session() or {"labels": []}
            hit = hit_test_labels(
                event.mouse_region_x,
                event.mouse_region_y,
                session.get("labels", []),
                LABEL_RADIUS_PX,
            )
            if hit is None:
                scene_props.merge_anchor_id = -1
                scene_props.merge_status = (
                    f"请选择锚点 · 当前 {scene_props.region_count} 个领域"
                )
                self._tag_redraw(context)
                return {"RUNNING_MODAL"}

            anchor = int(scene_props.merge_anchor_id)
            if anchor < 0:
                scene_props.merge_anchor_id = int(hit)
                scene_props.merge_status = (
                    f"锚点 {hit} · 继续点击其他领域合并"
                )
                self._tag_redraw(context)
                return {"RUNNING_MODAL"}

            if int(hit) == anchor:
                return {"RUNNING_MODAL"}

            try:
                new_ids, new_colors, new_count, new_anchor = merge_region_ids(
                    self._live_ids,
                    self._live_colors,
                    anchor,
                    int(hit),
                )
            except ValueError as exc:
                self.report({"ERROR"}, str(exc))
                return {"RUNNING_MODAL"}

            self._apply_live(
                context,
                new_ids,
                new_colors,
                new_count,
                new_anchor,
            )
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


__all__ = (
    "ARE_OT_segment_regions",
    "ARE_OT_clear_regions",
    "ARE_OT_merge_regions",
    "ARE_OT_confirm_merge_regions",
    "REGION_ID_ATTR",
    "REGION_VERSION_ATTR",
    "REGION_COLORS_ATTR",
    "write_region_ids",
    "clear_region_ids",
    "read_region_ids",
)
