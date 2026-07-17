# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""领域自动识别、清除、内存事务合并与智能拆分操作符。"""

from __future__ import annotations

import bpy
import numpy as np

from ..algorithms.regions import (
    compute_region_label_anchors,
    merge_region_ids,
    segment_regions_by_normal,
)
from ..algorithms.region_split import (
    complete_cut_edges_dijkstra,
    prepare_edge_costs,
    split_region_by_cut_edges,
    stroke_hits_to_seed_edges,
)
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import (
    LABEL_RADIUS_PX,
    clear_region_highlight,
    get_merge_label_session,
    register_label_draw_handler,
    register_split_draw_handler,
    set_merge_label_session,
    set_region_highlight,
    set_split_stroke_session,
    unregister_label_draw_handler,
    unregister_split_draw_handler,
    update_merge_label_projections,
)
from ..utils.mesh import extract_face_topology, extract_mesh_data
from ..utils.progress import progress_scope
from ..utils.viewport import hit_test_labels


REGION_ID_ATTR = "are_region_id"
REGION_VERSION_ATTR = "are_region_version"
REGION_COLORS_ATTR = "are_region_colors"
STROKE_SAMPLE_PX = 5.0


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
    """标签法线偏移参考尺度。"""
    vertices = mesh_data["vertices"]
    if len(vertices) == 0:
        return 0.01
    size = vertices.max(axis=0) - vertices.min(axis=0)
    return float(max(np.linalg.norm(size) * 0.015, 1e-4))


def _build_label_session(
    region_ids: np.ndarray,
    mesh_data,
    colors: np.ndarray | None = None,
) -> dict:
    """根据最中心面及法线正方向构建编号锚点会话（含内存标签供预览）。"""
    anchors = compute_region_label_anchors(
        region_ids,
        mesh_data["face_centers"],
        mesh_data["normals"],
        mesh_data["areas"],
    )
    labels = []
    for region_id in sorted(anchors.keys()):
        anchor = anchors[region_id]
        labels.append(
            {
                "id": int(region_id),
                "world_co": np.asarray(anchor["world_co"], dtype=np.float64),
                "face_center": np.asarray(
                    anchor["face_center"],
                    dtype=np.float64,
                ),
                "normal": np.asarray(anchor["normal"], dtype=np.float64),
                "screen_xy": None,
                "face_screen_xy": None,
                "visible": False,
            }
        )
    return {
        "labels": labels,
        "lift": _bbox_lift(mesh_data),
        "object_name": None,
        "region_ids": np.asarray(region_ids, dtype=np.int32).copy(),
        "colors": (
            None
            if colors is None
            else np.asarray(colors, dtype=np.float32).copy()
        ),
        "mesh_data": mesh_data,
    }


def _modal_busy(scene_props) -> bool:
    """合并或拆分模态进行中。"""
    return bool(
        scene_props.merge_mode_active or scene_props.split_mode_active
    )


def _is_undo_event(event) -> bool:
    """Ctrl/Cmd + Z（无 Shift）表示撤销。"""
    if event.type != "Z" or event.value != "PRESS":
        return False
    if not (event.ctrl or event.oskey):
        return False
    return not event.shift


def _is_redo_event(event) -> bool:
    """Ctrl/Cmd + Shift + Z 或 Ctrl + Y 表示重做。"""
    if event.value != "PRESS":
        return False
    if event.type == "Y" and (event.ctrl or event.oskey) and not event.shift:
        return True
    if event.type == "Z" and (event.ctrl or event.oskey) and event.shift:
        return True
    return False


def _tag_redraw(context: bpy.types.Context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "UI"}:
                area.tag_redraw()


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
        if scene_props is not None and _modal_busy(scene_props):
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
        if scene_props is not None and _modal_busy(scene_props):
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
    模态合并领域（内存事务）。

    模态期间不写 Mesh；Ctrl+Z 撤销最近一次合并；
    确认时一次写入并形成 Blender Undo。
    """

    bl_idname = "are.merge_regions"
    bl_label = "合并领域"
    bl_description = (
        "首击锚点、续击合并；Ctrl+Z 撤销上次合并；确认写入，Esc 取消"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _cleanup_ui(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        unregister_label_draw_handler()
        set_merge_label_session(None)
        scene_props.merge_mode_active = False
        scene_props.merge_anchor_id = -1
        scene_props.merge_hover_id = -1
        scene_props.merge_confirm_requested = False
        _tag_redraw(context)

    def _commit_to_mesh(self, context: bpy.types.Context) -> None:
        """仅在确认时写入一次 Mesh。"""
        obj = self._object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        write_region_ids(obj.data, self._live_ids)
        version = int(self._snapshot_version) + 1
        obj[REGION_VERSION_ATTR] = version
        obj[REGION_COLORS_ATTR] = (
            self._live_colors.astype(np.float32).ravel().tolist()
        )
        scene_props.region_version = version
        scene_props.region_count = int(self._live_count)
        scene_props.region_object = obj
        set_region_highlight(
            context,
            obj,
            self._live_ids,
            self._live_colors,
        )

    def _finish_mode(self, context: bpy.types.Context, cancelled: bool) -> set:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._cleanup_ui(context)
        if cancelled:
            scene_props.merge_status = "已取消合并"
            # 恢复进入前的覆盖层显示（mesh 未改，只需刷新指针）
            try:
                set_region_highlight(
                    context,
                    self._object,
                    self._snapshot_ids,
                    self._snapshot_colors,
                )
                scene_props.region_count = int(self._snapshot_count)
            except Exception:
                pass
        else:
            scene_props.merge_status = (
                f"合并完成，当前 {scene_props.region_count} 个领域"
            )
            scene_props.region_status = (
                f"已识别 {scene_props.region_count} 个领域"
            )
        return {"CANCELLED"} if cancelled else {"FINISHED"}

    def _refresh_preview(self, context: bpy.types.Context) -> None:
        """用内存标签刷新编号会话与覆盖层预览（不写 Mesh）。"""
        obj = self._object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        mesh_data = self._mesh_data
        session = _build_label_session(
            self._live_ids,
            mesh_data,
            self._live_colors,
        )
        session["object_name"] = obj.name
        set_merge_label_session(session)
        scene_props.region_count = int(self._live_count)
        scene_props.region_object = obj
        # 覆盖层从 session 内存读取，避免写 mesh
        scene_props.merge_status = (
            f"锚点 {scene_props.merge_anchor_id} · "
            f"当前 {self._live_count} 个领域 · "
            f"历史 {len(self._history)}"
            if scene_props.merge_anchor_id >= 0
            else (
                f"请选择锚点 · 当前 {self._live_count} 个领域 · "
                f"Ctrl+Z 撤销"
            )
        )
        _tag_redraw(context)

    def _push_history(self) -> None:
        self._history.append(
            {
                "ids": self._live_ids.copy(),
                "colors": self._live_colors.copy(),
                "count": int(self._live_count),
                "anchor": int(
                    getattr(self, "_anchor_before_merge", -1)
                ),
            }
        )
        self._redo.clear()

    def _undo_merge(self, context: bpy.types.Context) -> None:
        if not self._history:
            scene_props = getattr(context.scene, SCENE_PROP_NAME)
            scene_props.merge_status = "没有可撤销的合并"
            _tag_redraw(context)
            return
        current = {
            "ids": self._live_ids.copy(),
            "colors": self._live_colors.copy(),
            "count": int(self._live_count),
            "anchor": int(
                getattr(context.scene, SCENE_PROP_NAME).merge_anchor_id
            ),
        }
        self._redo.append(current)
        previous = self._history.pop()
        self._live_ids = previous["ids"]
        self._live_colors = previous["colors"]
        self._live_count = int(previous["count"])
        getattr(context.scene, SCENE_PROP_NAME).merge_anchor_id = int(
            previous.get("anchor", -1)
        )
        self._refresh_preview(context)

    def _redo_merge(self, context: bpy.types.Context) -> None:
        if not self._redo:
            return
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._history.append(
            {
                "ids": self._live_ids.copy(),
                "colors": self._live_colors.copy(),
                "count": int(self._live_count),
                "anchor": int(scene_props.merge_anchor_id),
            }
        )
        nxt = self._redo.pop()
        self._live_ids = nxt["ids"]
        self._live_colors = nxt["colors"]
        self._live_count = int(nxt["count"])
        scene_props.merge_anchor_id = int(nxt.get("anchor", -1))
        self._refresh_preview(context)

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is not None and _modal_busy(scene_props):
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
        mesh_data = extract_mesh_data(obj)

        self._object = obj
        self._mesh_data = mesh_data
        self._snapshot_ids = region_ids.copy()
        self._snapshot_colors = colors.copy()
        self._snapshot_version = int(scene_props.region_version)
        self._snapshot_count = int(scene_props.region_count) or region_count
        self._live_ids = region_ids.copy()
        self._live_colors = colors.copy()
        self._live_count = int(region_count)
        self._history: list[dict] = []
        self._redo: list[dict] = []
        self._closing_by_system = False

        session = _build_label_session(region_ids, mesh_data, colors)
        session["object_name"] = obj.name
        set_merge_label_session(session)
        register_label_draw_handler()

        scene_props.merge_mode_active = True
        scene_props.merge_anchor_id = -1
        scene_props.merge_hover_id = -1
        scene_props.merge_confirm_requested = False
        scene_props.merge_status = (
            "首击锚点，续击合并；Ctrl+Z 撤销上次；确认结束；Esc 取消"
        )
        scene_props.region_object = obj

        context.window_manager.modal_handler_add(self)
        _tag_redraw(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context):
        # 系统 Undo 触发时禁止二次写 Mesh，只清理 UI。
        self._cleanup_ui(context)
        return {"CANCELLED"}

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

        # 必须在透传前拦截全局撤销，防止崩溃。
        if _is_undo_event(event):
            self._undo_merge(context)
            return {"RUNNING_MODAL"}
        if _is_redo_event(event):
            self._redo_merge(context)
            return {"RUNNING_MODAL"}

        if scene_props.merge_confirm_requested:
            scene_props.merge_confirm_requested = False
            self._commit_to_mesh(context)
            self.report(
                {"INFO"},
                f"合并完成，当前 {self._live_count} 个领域",
            )
            return self._finish_mode(context, cancelled=False)

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self.report({"INFO"}, "已取消合并")
            return self._finish_mode(context, cancelled=True)

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._commit_to_mesh(context)
            self.report(
                {"INFO"},
                f"合并完成，当前 {self._live_count} 个领域",
            )
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
            _tag_redraw(context)
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
                self._refresh_preview(context)
                return {"RUNNING_MODAL"}

            anchor = int(scene_props.merge_anchor_id)
            if anchor < 0:
                scene_props.merge_anchor_id = int(hit)
                self._refresh_preview(context)
                return {"RUNNING_MODAL"}

            if int(hit) == anchor:
                return {"RUNNING_MODAL"}

            try:
                self._anchor_before_merge = anchor
                self._push_history()
                new_ids, new_colors, new_count, new_anchor = merge_region_ids(
                    self._live_ids,
                    self._live_colors,
                    anchor,
                    int(hit),
                )
            except ValueError as exc:
                if self._history:
                    self._history.pop()
                self.report({"ERROR"}, str(exc))
                return {"RUNNING_MODAL"}

            self._live_ids = new_ids
            self._live_colors = new_colors
            self._live_count = int(new_count)
            scene_props.merge_anchor_id = int(new_anchor)
            self._refresh_preview(context)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


# ---------------------------------------------------------------------------
# 拆分领域
# ---------------------------------------------------------------------------


class ARE_OT_confirm_split_regions(bpy.types.Operator):
    """面板确认拆分。"""

    bl_idname = "are.confirm_split_regions"
    bl_label = "确认拆分"
    bl_description = "应用全部预览拆分线并写入领域"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return scene_props is not None and scene_props.split_mode_active

    def execute(self, context: bpy.types.Context):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.split_confirm_requested = True
        return {"FINISHED"}


class ARE_OT_split_regions(bpy.types.Operator):
    """
    画笔式智能拆分领域。

    绘制不完整硬边笔迹，松开后自动补全预览；可多笔；
    Ctrl+Z 撤销最近一笔；确认统一拆分。
    """

    bl_idname = "are.split_regions"
    bl_label = "拆分领域"
    bl_description = (
        "画笔绘制拆分线，松开后智能补全硬边；确认后拆分领域"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _cleanup_ui(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        unregister_split_draw_handler()
        set_split_stroke_session(None)
        scene_props.split_mode_active = False
        scene_props.split_confirm_requested = False
        _tag_redraw(context)

    def _refresh_split_preview(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        strokes = []
        for item in self._stroke_stack:
            strokes.append(
                {
                    "polyline_world": item["polyline_world"],
                    "completed_edges_world": item["completed_edges_world"],
                    "live": False,
                }
            )
        if self._painting and len(self._live_hits) >= 1:
            live_poly = np.asarray(
                [hit["world"] for hit in self._live_hits],
                dtype=np.float64,
            )
            strokes.append(
                {
                    "polyline_world": live_poly,
                    "completed_edges_world": np.empty((0, 2, 3)),
                    "live": True,
                }
            )
        set_split_stroke_session(
            {
                "object_name": self._object.name,
                "strokes": strokes,
            }
        )
        scene_props.split_status = (
            f"已预览 {len(self._stroke_stack)} 笔 · "
            "继续绘制或确认拆分 · Ctrl+Z 撤销笔迹"
        )
        _tag_redraw(context)

    def _undo_stroke(self, context: bpy.types.Context) -> None:
        if not self._stroke_stack:
            getattr(context.scene, SCENE_PROP_NAME).split_status = (
                "没有可撤销的笔迹"
            )
            _tag_redraw(context)
            return
        self._redo_stack.append(self._stroke_stack.pop())
        self._refresh_split_preview(context)

    def _redo_stroke(self, context: bpy.types.Context) -> None:
        if not self._redo_stack:
            return
        self._stroke_stack.append(self._redo_stack.pop())
        self._refresh_split_preview(context)

    def _sample_hit(self, context: bpy.types.Context, event) -> dict | None:
        from bpy_extras import view3d_utils
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree

        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return None

        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        if direction.length < 1e-12:
            return None
        direction.normalize()

        if self._bvh is None:
            mesh = self._object.data
            verts = [v.co.copy() for v in mesh.vertices]
            polys = [tuple(p.vertices) for p in mesh.polygons]
            self._bvh = BVHTree.FromPolygons(verts, polys)

        matrix = self._object.matrix_world
        inv = matrix.inverted()
        local_origin = inv @ origin
        local_direction = (inv.to_3x3() @ direction).normalized()
        location, normal, index, distance = self._bvh.ray_cast(
            local_origin,
            local_direction,
        )
        if location is None or index is None:
            return None

        # 变换到世界坐标
        world = matrix @ location
        world_n = (matrix.to_3x3() @ normal).normalized()
        view_dir = (world - origin).normalized()
        if world_n.dot(-view_dir) < 0.05:
            return None

        rid = int(self._region_ids[int(index)])
        if rid < 0:
            return None

        return {
            "face": int(index),
            "world": np.array(
                (world.x, world.y, world.z),
                dtype=np.float64,
            ),
            "screen": np.array(
                (float(coord[0]), float(coord[1])),
                dtype=np.float64,
            ),
            "region_id": rid,
        }

    def _finalize_stroke(self, context: bpy.types.Context) -> None:
        if len(self._live_hits) < 2:
            self._live_hits = []
            self._painting = False
            self._refresh_split_preview(context)
            return

        faces = [h["face"] for h in self._live_hits]
        worlds = np.asarray(
            [h["world"] for h in self._live_hits],
            dtype=np.float64,
        )
        screens = np.asarray(
            [h["screen"] for h in self._live_hits],
            dtype=np.float64,
        )
        # 众数领域作为目标
        rids = [int(self._region_ids[f]) for f in faces]
        target = max(set(rids), key=rids.count)

        seed_edges = stroke_hits_to_seed_edges(
            faces,
            worlds,
            self._topology,
            self._mesh_data["face_centers"],
            self._region_ids,
            target,
        )
        if len(seed_edges) == 0:
            self.report({"WARNING"}, "未捕捉到可拆分的边，请沿硬边重画")
            self._live_hits = []
            self._painting = False
            self._refresh_split_preview(context)
            return

        completed = complete_cut_edges_dijkstra(
            self._topology,
            self._mesh_data["normals"],
            self._mesh_data["face_centers"],
            self._region_ids,
            target,
            seed_edges,
            worlds,
            screens,
            self._edge_costs,
            self._edge_mids,
            self._vert_edge_offsets,
            self._vert_edge_indices,
            self._edge_vert_a,
            self._edge_vert_b,
            max_radius=self._max_radius,
        )

        # 预览线段世界坐标
        verts = self._mesh_data["vertices"]
        # 本地顶点需转世界——mesh_data vertices 已是世界坐标
        edge_lines = []
        for edge_index in completed.tolist():
            va = int(self._edge_vert_a[edge_index])
            vb = int(self._edge_vert_b[edge_index])
            # edge verts 是 mesh 本地索引，但我们存的是从 loop 提取的索引；
            # vertices 数组已是世界坐标且与 mesh 顶点顺序一致。
            if 0 <= va < len(verts) and 0 <= vb < len(verts):
                edge_lines.append([verts[va], verts[vb]])
        completed_world = (
            np.asarray(edge_lines, dtype=np.float64)
            if edge_lines
            else np.empty((0, 2, 3), dtype=np.float64)
        )

        self._stroke_stack.append(
            {
                "target_rid": int(target),
                "seed_edges": seed_edges,
                "completed_edges": completed,
                "polyline_world": worlds,
                "completed_edges_world": completed_world,
            }
        )
        self._redo_stack.clear()
        self._live_hits = []
        self._painting = False
        self._refresh_split_preview(context)
        self.report(
            {"INFO"},
            f"已补全拆分线（{len(completed)} 条边），可继续绘制",
        )

    def _commit_splits(self, context: bpy.types.Context) -> None:
        if not self._stroke_stack:
            self.report({"WARNING"}, "没有可确认的拆分笔迹")
            return

        all_cuts = []
        for item in self._stroke_stack:
            all_cuts.extend(item["completed_edges"].tolist())
        cut_edges = np.unique(np.asarray(all_cuts, dtype=np.int32))

        new_ids, new_colors, new_count = split_region_by_cut_edges(
            self._region_ids,
            self._topology,
            cut_edges,
            self._colors,
        )
        write_region_ids(self._object.data, new_ids)
        version = int(self._snapshot_version) + 1
        self._object[REGION_VERSION_ATTR] = version
        self._object[REGION_COLORS_ATTR] = (
            new_colors.astype(np.float32).ravel().tolist()
        )
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.region_version = version
        scene_props.region_count = int(new_count)
        scene_props.region_object = self._object
        scene_props.region_status = f"已识别 {new_count} 个领域"
        scene_props.region_status_detail = (
            f"拆分写入 {len(self._stroke_stack)} 笔"
        )
        set_region_highlight(context, self._object, new_ids, new_colors)

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is not None and _modal_busy(scene_props):
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

        mesh_data = extract_mesh_data(obj)
        topology = extract_face_topology(obj.data)
        region_count = int(region_ids.max()) + 1
        colors = _read_region_colors(obj, region_count)

        # 边顶点与顶点-边 CSR（优先使用拓扑缓存）
        edge_vert_a = topology["edge_vert_a"]
        edge_vert_b = topology["edge_vert_b"]
        if "vert_edge_offsets" in topology and "vert_edge_indices" in topology:
            offsets = topology["vert_edge_offsets"]
            both_e = topology["vert_edge_indices"]
        else:
            vert_count = len(obj.data.vertices)
            both_v = np.concatenate((edge_vert_a, edge_vert_b))
            both_e = np.concatenate(
                (
                    np.arange(len(edge_vert_a), dtype=np.int32),
                    np.arange(len(edge_vert_a), dtype=np.int32),
                )
            )
            order = np.argsort(both_v, kind="stable")
            both_v = both_v[order]
            both_e = both_e[order]
            offsets = np.zeros(vert_count + 1, dtype=np.int32)
            if len(both_v):
                counts = np.bincount(both_v, minlength=vert_count)
                offsets[1:] = np.cumsum(counts, dtype=np.int32)

        edge_costs, edge_mids = prepare_edge_costs(
            topology,
            mesh_data["normals"],
            mesh_data["face_centers"],
            region_ids,
        )

        verts = mesh_data["vertices"]
        extent = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        self._max_radius = max(extent * 0.08, 1e-3)

        self._object = obj
        self._mesh_data = mesh_data
        self._topology = topology
        self._region_ids = region_ids.copy()
        self._colors = colors.copy()
        self._snapshot_version = int(scene_props.region_version)
        self._edge_costs = edge_costs
        self._edge_mids = edge_mids
        self._edge_vert_a = edge_vert_a
        self._edge_vert_b = edge_vert_b
        self._vert_edge_offsets = offsets
        self._vert_edge_indices = both_e.astype(np.int32, copy=False)
        self._bvh = None
        self._stroke_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._painting = False
        self._live_hits: list[dict] = []
        self._last_sample_xy = None

        scene_props.split_mode_active = True
        scene_props.split_confirm_requested = False
        scene_props.split_status = (
            "按住左键沿硬边绘制；松开自动补全；确认拆分；Esc 取消"
        )
        register_split_draw_handler()
        self._refresh_split_preview(context)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context):
        self._cleanup_ui(context)
        return {"CANCELLED"}

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

        if _is_undo_event(event):
            self._undo_stroke(context)
            return {"RUNNING_MODAL"}
        if _is_redo_event(event):
            self._redo_stroke(context)
            return {"RUNNING_MODAL"}

        if scene_props.split_confirm_requested:
            scene_props.split_confirm_requested = False
            try:
                self._commit_splits(context)
            except Exception as exc:
                self.report({"ERROR"}, f"拆分失败: {exc}")
                return self.cancel(context)
            self._cleanup_ui(context)
            self.report({"INFO"}, scene_props.region_status)
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self._cleanup_ui(context)
            scene_props.split_status = "已取消拆分"
            self.report({"INFO"}, "已取消拆分")
            return {"CANCELLED"}

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            try:
                self._commit_splits(context)
            except Exception as exc:
                self.report({"ERROR"}, f"拆分失败: {exc}")
                return self.cancel(context)
            self._cleanup_ui(context)
            self.report({"INFO"}, scene_props.region_status)
            return {"FINISHED"}

        if context.space_data is None or context.space_data.type != "VIEW_3D":
            return {"PASS_THROUGH"}
        if context.region is None or context.region.type != "WINDOW":
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self._painting = True
            self._live_hits = []
            self._last_sample_xy = None
            hit = self._sample_hit(context, event)
            if hit is not None:
                self._live_hits.append(hit)
                self._last_sample_xy = (
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
            self._refresh_split_preview(context)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self._painting:
                self._finalize_stroke(context)
            return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE" and self._painting:
            if self._last_sample_xy is not None:
                dx = event.mouse_region_x - self._last_sample_xy[0]
                dy = event.mouse_region_y - self._last_sample_xy[1]
                if (dx * dx + dy * dy) < STROKE_SAMPLE_PX * STROKE_SAMPLE_PX:
                    return {"RUNNING_MODAL"}
            hit = self._sample_hit(context, event)
            if hit is not None:
                self._live_hits.append(hit)
                self._last_sample_xy = (
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
                self._refresh_split_preview(context)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


__all__ = (
    "ARE_OT_segment_regions",
    "ARE_OT_clear_regions",
    "ARE_OT_merge_regions",
    "ARE_OT_confirm_merge_regions",
    "ARE_OT_split_regions",
    "ARE_OT_confirm_split_regions",
    "REGION_ID_ATTR",
    "REGION_VERSION_ATTR",
    "REGION_COLORS_ATTR",
    "write_region_ids",
    "clear_region_ids",
    "read_region_ids",
)
