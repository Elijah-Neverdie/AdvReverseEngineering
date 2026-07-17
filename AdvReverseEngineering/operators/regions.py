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
    cut_edges_from_paint_corridor,
    prepare_edge_costs,
    split_region_by_cut_edges,
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
MODAL_TIMER_STEP = 0.1
SPLIT_DEBOUNCE_SEC = 0.5
BRUSH_RADIUS_DEFAULT = 40.0
BRUSH_RADIUS_MIN = 8.0
BRUSH_RADIUS_MAX = 200.0
BRUSH_RADIUS_STEP = 6.0


def _add_modal_timer(operator, context: bpy.types.Context):
    """为模态算子注册低频 TIMER，确保侧栏确认能被消费。"""
    wm = context.window_manager
    operator._timer = wm.event_timer_add(
        MODAL_TIMER_STEP,
        window=context.window,
    )


def _remove_modal_timer(operator, context: bpy.types.Context) -> None:
    timer = getattr(operator, "_timer", None)
    if timer is None:
        return
    try:
        context.window_manager.event_timer_remove(timer)
    except Exception:
        pass
    operator._timer = None


# 当前活跃的模态实例，供侧栏确认/取消按钮直接调用，
# 不依赖 TIMER 事件的派发时机。
# 存放在 driver_namespace 中，插件热重载后按钮仍能取到旧模态实例。
_MERGE_OP_KEY = "are_active_merge_op"
_SPLIT_OP_KEY = "are_active_split_op"


def _set_active_merge_op(operator) -> None:
    bpy.app.driver_namespace[_MERGE_OP_KEY] = operator


def _get_active_merge_op():
    return bpy.app.driver_namespace.get(_MERGE_OP_KEY)


def _clear_active_merge_op(operator) -> None:
    if bpy.app.driver_namespace.get(_MERGE_OP_KEY) is operator:
        bpy.app.driver_namespace[_MERGE_OP_KEY] = None


def _set_active_split_op(operator) -> None:
    bpy.app.driver_namespace[_SPLIT_OP_KEY] = operator


def _get_active_split_op():
    return bpy.app.driver_namespace.get(_SPLIT_OP_KEY)


def _clear_active_split_op(operator) -> None:
    if bpy.app.driver_namespace.get(_SPLIT_OP_KEY) is operator:
        bpy.app.driver_namespace[_SPLIT_OP_KEY] = None


def _schedule_force_exit_check(kind: str) -> None:
    """
    确认标志置位后若迟迟无人消费（模态实例已丢失），
    超时强制清理模式标志与覆盖层，避免面板永远卡在模态提示。
    """

    def _callback():
        scene = getattr(bpy.context, "scene", None)
        scene_props = getattr(scene, SCENE_PROP_NAME, None) if scene else None
        if scene_props is None:
            return None
        if kind == "merge":
            stuck = (
                scene_props.merge_mode_active
                and scene_props.merge_confirm_requested
            )
            if stuck:
                scene_props.merge_confirm_requested = False
                scene_props.merge_mode_active = False
                scene_props.merge_anchor_id = -1
                scene_props.merge_hover_id = -1
                scene_props.merge_status = "合并模态已失联，已强制退出"
                unregister_label_draw_handler()
                set_merge_label_session(None)
        else:
            stuck = (
                scene_props.split_mode_active
                and scene_props.split_confirm_requested
            )
            if stuck:
                scene_props.split_confirm_requested = False
                scene_props.split_mode_active = False
                scene_props.split_target_id = -1
                scene_props.split_hover_id = -1
                scene_props.split_phase = "IDLE"
                scene_props.split_status = "拆分模态已失联，已强制退出"
                unregister_split_draw_handler()
                unregister_label_draw_handler()
                set_split_stroke_session(None)
                set_merge_label_session(None)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return None

    bpy.app.timers.register(_callback, first_interval=0.6)


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
    """按 Blender 线框硬边阈值自动识别领域并以随机色标记。"""

    bl_idname = "are.segment_regions"
    bl_label = "识别领域"
    bl_description = (
        "按视图叠加层线框阈值分割领域，橘色硬边作为默认领域边界"
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

                step("线框硬边区域生长")
                min_ratio = (
                    float(scene_props.region_min_area_ratio) / 100.0
                    if scene_props.region_ignore_discrete
                    else 0.0
                )
                wire_threshold = float(
                    getattr(scene_props, "region_wireframe_threshold", 0.1)
                )
                result = segment_regions_by_normal(
                    normals=mesh_data["normals"],
                    areas=mesh_data["areas"],
                    topology=topology,
                    wireframe_threshold=wire_threshold,
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
                    absorbed = int(result["ignored_region_count"])
                    if absorbed:
                        scene_props.region_status_detail = (
                            f"已并入 {absorbed} 个碎屑领域"
                        )
                    else:
                        scene_props.region_status_detail = "无碎屑需要合并"
                else:
                    scene_props.region_status_detail = "未启用碎屑合并"

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
        # 优先直接驱动活跃模态实例，确保立即提交并退出。
        op = _get_active_merge_op()
        if op is not None:
            try:
                op.confirm_from_panel(context)
            except Exception as exc:
                self.report({"ERROR"}, f"确认合并失败: {exc}")
                scene_props.merge_confirm_requested = True
                _schedule_force_exit_check("merge")
                return {"FINISHED"}
            # 按钮点击应等效于回车：提交后立即退出模式显示。
            if scene_props.merge_mode_active:
                scene_props.merge_mode_active = False
                scene_props.merge_confirm_requested = False
                unregister_label_draw_handler()
                set_merge_label_session(None)
            return {"FINISHED"}
        # 后备：置位标志，等待模态在下一次事件消费；
        # 若无人消费（模态已丢失），超时后强制清理残留状态。
        scene_props.merge_confirm_requested = True
        _schedule_force_exit_check("merge")
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
        _remove_modal_timer(self, context)
        _clear_active_merge_op(self)
        unregister_label_draw_handler()
        set_merge_label_session(None)
        scene_props.merge_mode_active = False
        scene_props.merge_anchor_id = -1
        scene_props.merge_hover_id = -1
        scene_props.merge_confirm_requested = False
        _tag_redraw(context)

    def confirm_from_panel(self, context: bpy.types.Context) -> None:
        """侧栏确认按钮直接调用：提交并结束模态显示。"""
        if getattr(self, "_closed", False):
            return
        try:
            if not self._committed:
                self._commit_to_mesh(context)
        finally:
            # 无论提交是否异常，都保证退出合并模式。
            self._finish_mode(context, cancelled=False)
            self._closed = True

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
        self._committed = True

    def _finish_mode(self, context: bpy.types.Context, cancelled: bool) -> set:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        # 已提交后再按 Esc 也不回滚。
        if cancelled and getattr(self, "_committed", False):
            cancelled = False
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
        # 预览版本号驱动 overlay 缓存，仅标签变化时重建三角缓冲。
        self._preview_serial += 1
        session["preview_version"] = self._preview_serial
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
        self._preview_serial = 0
        self._committed = False
        self._closed = False
        self._timer = None

        session = _build_label_session(region_ids, mesh_data, colors)
        session["object_name"] = obj.name
        session["preview_version"] = 0
        set_merge_label_session(session)
        register_label_draw_handler()
        _set_active_merge_op(self)

        scene_props.merge_mode_active = True
        scene_props.merge_anchor_id = -1
        scene_props.merge_hover_id = -1
        scene_props.merge_confirm_requested = False
        scene_props.merge_status = (
            "首击锚点，续击合并；Ctrl+Z 撤销上次；确认结束；Esc 取消"
        )
        scene_props.region_object = obj

        _add_modal_timer(self, context)
        context.window_manager.modal_handler_add(self)
        _tag_redraw(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context):
        # 系统 Undo 触发时禁止二次写 Mesh，只清理 UI。
        if getattr(self, "_committed", False):
            self._cleanup_ui(context)
            return {"FINISHED"}
        self._cleanup_ui(context)
        return {"CANCELLED"}

    def modal(self, context: bpy.types.Context, event):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        obj = self._object

        # 侧栏确认已直接完成提交并清理，模态只需静默结束。
        if getattr(self, "_closed", False):
            return {"FINISHED"}

        try:
            if obj is None or obj.name not in bpy.data.objects:
                return self.cancel(context)
            if obj.mode == "EDIT":
                return self.cancel(context)
        except ReferenceError:
            return self.cancel(context)

        # TIMER / 任意事件都先消费确认请求，避免侧栏确认后卡在模态。
        if scene_props.merge_confirm_requested and not self._committed:
            scene_props.merge_confirm_requested = False
            self._commit_to_mesh(context)
            self.report(
                {"INFO"},
                f"合并完成，当前 {self._live_count} 个领域",
            )
            return self._finish_mode(context, cancelled=False)

        if event.type == "TIMER":
            return {"RUNNING_MODAL"}

        # 必须在透传前拦截全局撤销，防止崩溃。
        if _is_undo_event(event):
            if self._committed:
                return {"RUNNING_MODAL"}
            self._undo_merge(context)
            return {"RUNNING_MODAL"}
        if _is_redo_event(event):
            if self._committed:
                return {"RUNNING_MODAL"}
            self._redo_merge(context)
            return {"RUNNING_MODAL"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            if self._committed:
                return self._finish_mode(context, cancelled=False)
            self.report({"INFO"}, "已取消合并")
            return self._finish_mode(context, cancelled=True)

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if not self._committed:
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
            new_hover = -1 if hover is None else int(hover)
            # 悬停未变化则不触发重绘，避免拖动视角时反复刷新覆盖层。
            if new_hover != int(scene_props.merge_hover_id):
                scene_props.merge_hover_id = new_hover
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
    bl_description = "应用当前分色预览并写入领域"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return scene_props is not None and scene_props.split_mode_active

    def execute(self, context: bpy.types.Context):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        op = _get_active_split_op()
        if op is not None:
            try:
                ok = op.confirm_from_panel(context)
            except Exception as exc:
                self.report({"ERROR"}, f"确认拆分失败: {exc}")
                scene_props.split_confirm_requested = True
                _schedule_force_exit_check("split")
                return {"FINISHED"}
            if not ok:
                return {"CANCELLED"}
            # 按钮点击应等效于回车：提交后立即退出模式显示。
            if scene_props.split_mode_active:
                scene_props.split_mode_active = False
                scene_props.split_confirm_requested = False
                scene_props.split_phase = "IDLE"
                unregister_split_draw_handler()
                unregister_label_draw_handler()
                set_split_stroke_session(None)
                set_merge_label_session(None)
            return {"FINISHED"}
        scene_props.split_confirm_requested = True
        _schedule_force_exit_check("split")
        return {"FINISHED"}


class ARE_OT_split_regions(bpy.types.Operator):
    """
    先选编号领域，再用可调圆形笔刷涂红；
    松开后 0.5 秒自动硬边补全并分色预览；确认后写入。
    """

    bl_idname = "are.split_regions"
    bl_label = "拆分领域"
    bl_description = (
        "点击编号选择领域，[] 调笔刷粗细，涂红后等待预览，确认拆分"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _cleanup_ui(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._cancel_debounce_timer()
        _remove_modal_timer(self, context)
        _clear_active_split_op(self)
        unregister_split_draw_handler()
        unregister_label_draw_handler()
        set_split_stroke_session(None)
        set_merge_label_session(None)
        scene_props.split_mode_active = False
        scene_props.split_confirm_requested = False
        scene_props.split_target_id = -1
        scene_props.split_hover_id = -1
        scene_props.split_phase = "IDLE"
        _tag_redraw(context)

    def confirm_from_panel(self, context: bpy.types.Context) -> bool:
        """侧栏确认按钮直接调用：提交并结束模态显示。"""
        if getattr(self, "_closed", False):
            return True
        committed = False
        try:
            committed = self._commit_splits(context)
        except Exception:
            committed = False
        if not committed:
            return False
        self._alive = False
        self._cleanup_ui(context)
        self._closed = True
        return True

    def _cancel_debounce_timer(self) -> None:
        timer = getattr(self, "_debounce_timer", None)
        if timer is None:
            return
        try:
            bpy.app.timers.unregister(timer)
        except Exception:
            pass
        self._debounce_timer = None
        self._debounce_token = None

    def _schedule_debounce(self, context: bpy.types.Context) -> None:
        self._cancel_debounce_timer()
        token = object()
        self._debounce_token = token
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.split_status = (
            f"目标 {scene_props.split_target_id} · "
            f"已涂 {len(self._paint_faces)} 面 · "
            f"{SPLIT_DEBOUNCE_SEC:.1f}s 后自动预览"
        )
        _tag_redraw(context)

        def _callback():
            if getattr(self, "_debounce_token", None) is not token:
                return None
            if not getattr(self, "_alive", False):
                return None
            try:
                self._run_corridor_preview(bpy.context)
            except Exception as exc:
                scene = getattr(bpy.context.scene, SCENE_PROP_NAME, None)
                if scene is not None:
                    scene.split_status = f"预览失败: {exc}"
            return None

        self._debounce_timer = _callback
        bpy.app.timers.register(_callback, first_interval=SPLIT_DEBOUNCE_SEC)

    def _refresh_session(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        paint = (
            np.asarray(sorted(self._paint_faces), dtype=np.int32)
            if self._paint_faces
            else np.empty(0, dtype=np.int32)
        )
        session = {
            "object_name": self._object.name,
            "paint_faces": paint,
            "completed_edges_world": self._completed_edges_world,
            "preview_ids": self._preview_ids,
            "preview_colors": self._preview_colors,
            "base_ids": self._region_ids,
            "preview_version": self._preview_serial,
            "brush": {
                "screen_xy": self._brush_xy,
                "radius_px": float(self._brush_radius),
            },
        }
        set_split_stroke_session(session)
        scene_props.split_brush_radius = float(self._brush_radius)
        _tag_redraw(context)

    def _set_phase_select(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.split_phase = "SELECT"
        scene_props.split_target_id = -1
        scene_props.split_status = (
            "点击编号选择要拆分的领域；Esc 取消"
        )
        self._paint_faces.clear()
        self._preview_ids = None
        self._preview_colors = None
        self._completed_edges = np.empty(0, dtype=np.int32)
        self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
        self._preview_serial += 1
        session = _build_label_session(
            self._region_ids,
            self._mesh_data,
            self._colors,
        )
        session["object_name"] = self._object.name
        session["preview_version"] = self._preview_serial
        set_merge_label_session(session)
        register_label_draw_handler()
        self._refresh_session(context)

    def _set_phase_brush(self, context: bpy.types.Context, target_id: int) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.split_phase = "BRUSH"
        scene_props.split_target_id = int(target_id)
        scene_props.split_status = (
            f"已选领域 {target_id} · [] 调半径 · "
            "按住左键涂红 · 松开 0.5s 后预览"
        )
        self._paint_faces.clear()
        self._preview_ids = None
        self._preview_colors = None
        self._completed_edges = np.empty(0, dtype=np.int32)
        self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
        self._preview_serial += 1
        # 保留标签以便换目标；也可隐藏。这里保留并高亮目标。
        session = get_merge_label_session()
        if session is not None:
            session["preview_version"] = self._preview_serial
        self._refresh_session(context)

    def _sample_hit(self, context: bpy.types.Context, event) -> dict | None:
        from bpy_extras import view3d_utils
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
        location, normal, index, _distance = self._bvh.ray_cast(
            local_origin,
            local_direction,
        )
        if location is None or index is None:
            return None

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
            "world": np.array((world.x, world.y, world.z), dtype=np.float64),
            "normal": np.array(
                (world_n.x, world_n.y, world_n.z),
                dtype=np.float64,
            ),
            "screen": np.array(
                (float(coord[0]), float(coord[1])),
                dtype=np.float64,
            ),
            "region_id": rid,
            "view_origin": np.array(
                (origin.x, origin.y, origin.z),
                dtype=np.float64,
            ),
        }

    def _world_brush_radius(
        self,
        context: bpy.types.Context,
        hit: dict,
    ) -> float:
        """把屏幕像素半径换算为命中点切平面上的世界半径。"""
        from bpy_extras import view3d_utils
        from mathutils import Vector

        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return float(self._max_radius * 0.05)

        center = Vector(hit["world"])
        screen = hit["screen"]
        offset = (
            float(screen[0]) + float(self._brush_radius),
            float(screen[1]),
        )
        # 与中心同深度的偏移点
        depth_loc = view3d_utils.location_3d_to_region_2d(
            region,
            rv3d,
            center,
        )
        if depth_loc is None:
            return float(self._max_radius * 0.05)
        # region_2d_to_location_3d 需要深度参考点
        offset_world = view3d_utils.region_2d_to_location_3d(
            region,
            rv3d,
            offset,
            center,
        )
        radius = float((Vector(offset_world) - center).length)
        return max(radius, 1e-6)

    def _paint_at(self, context: bpy.types.Context, event) -> None:
        hit = self._sample_hit(context, event)
        self._brush_xy = (
            float(event.mouse_region_x),
            float(event.mouse_region_y),
        )
        if hit is None:
            self._refresh_session(context)
            return

        target = int(
            getattr(context.scene, SCENE_PROP_NAME).split_target_id
        )
        if int(hit["region_id"]) != target:
            self._refresh_session(context)
            return

        world_radius = self._world_brush_radius(context, hit)
        center = hit["world"]
        centers = self._mesh_data["face_centers"]
        deltas = centers - center
        dist_sq = np.sum(deltas * deltas, axis=1)
        radius_sq = world_radius * world_radius
        candidates = np.flatnonzero(dist_sq <= radius_sq * 1.05)
        if len(candidates) == 0:
            candidates = np.asarray([hit["face"]], dtype=np.int32)

        added = False
        for face in candidates.tolist():
            if int(self._region_ids[face]) != target:
                continue
            if face not in self._paint_faces:
                self._paint_faces.add(int(face))
                added = True
            self._stroke_worlds.append(centers[face].copy())

        # 保证中心命中面一定在内
        if hit["face"] not in self._paint_faces:
            self._paint_faces.add(int(hit["face"]))
            added = True

        if added:
            # 新涂绘使旧预览失效
            self._preview_ids = None
            self._preview_colors = None
            self._completed_edges = np.empty(0, dtype=np.int32)
            self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
            self._preview_serial += 1
        self._refresh_session(context)

    def _run_corridor_preview(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        target = int(scene_props.split_target_id)
        if target < 0 or not self._paint_faces:
            scene_props.split_status = "请先涂绘拆分区域"
            self._refresh_session(context)
            return

        stroke = (
            np.asarray(self._stroke_worlds, dtype=np.float64)
            if self._stroke_worlds
            else self._mesh_data["face_centers"][
                np.asarray(sorted(self._paint_faces), dtype=np.int32)
            ]
        )
        completed, message = cut_edges_from_paint_corridor(
            np.asarray(sorted(self._paint_faces), dtype=np.int32),
            self._topology,
            self._mesh_data["normals"],
            self._mesh_data["face_centers"],
            self._region_ids,
            target,
            stroke,
            self._edge_costs,
            self._edge_mids,
            self._vert_edge_offsets,
            self._vert_edge_indices,
            self._edge_vert_a,
            self._edge_vert_b,
            max_radius=self._max_radius,
        )
        if message:
            scene_props.split_status = message
            scene_props.split_phase = "BRUSH"
            self._preview_ids = None
            self._preview_colors = None
            self._completed_edges = completed
            # 仍显示已找到的边提示
            self._completed_edges_world = self._edges_to_world(completed)
            self._preview_serial += 1
            self._refresh_session(context)
            return

        new_ids, new_colors, new_count = split_region_by_cut_edges(
            self._region_ids,
            self._topology,
            completed,
            self._colors,
        )
        self._completed_edges = completed
        self._completed_edges_world = self._edges_to_world(completed)
        self._preview_ids = new_ids
        self._preview_colors = new_colors
        self._preview_count = int(new_count)
        self._preview_serial += 1
        scene_props.split_phase = "PREVIEW"
        scene_props.split_status = (
            f"预览完成：{self._live_base_count} → {new_count} 个领域 · "
            "确认拆分保存 · Ctrl+Z 清除涂绘 · Esc 取消"
        )
        self._refresh_session(context)
        self.report({"INFO"}, scene_props.split_status)

    def _edges_to_world(self, edges: np.ndarray) -> np.ndarray:
        verts = self._mesh_data["vertices"]
        lines = []
        for edge_index in np.asarray(edges, dtype=np.int32).tolist():
            va = int(self._edge_vert_a[edge_index])
            vb = int(self._edge_vert_b[edge_index])
            if 0 <= va < len(verts) and 0 <= vb < len(verts):
                lines.append([verts[va], verts[vb]])
        if not lines:
            return np.empty((0, 2, 3), dtype=np.float64)
        return np.asarray(lines, dtype=np.float64)

    def _undo_paint(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._cancel_debounce_timer()
        if self._preview_ids is not None or self._paint_faces:
            self._paint_faces.clear()
            self._stroke_worlds.clear()
            self._preview_ids = None
            self._preview_colors = None
            self._completed_edges = np.empty(0, dtype=np.int32)
            self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
            self._preview_serial += 1
            scene_props.split_phase = (
                "BRUSH" if scene_props.split_target_id >= 0 else "SELECT"
            )
            scene_props.split_status = "已清除涂绘/预览，可重新涂绘"
            self._refresh_session(context)
            return
        scene_props.split_status = "没有可撤销的涂绘"
        _tag_redraw(context)

    def _commit_splits(self, context: bpy.types.Context) -> bool:
        if self._preview_ids is None or self._preview_colors is None:
            self.report({"WARNING"}, "还没有可确认的分色预览")
            return False
        write_region_ids(self._object.data, self._preview_ids)
        version = int(self._snapshot_version) + 1
        self._object[REGION_VERSION_ATTR] = version
        self._object[REGION_COLORS_ATTR] = (
            self._preview_colors.astype(np.float32).ravel().tolist()
        )
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.region_version = version
        scene_props.region_count = int(self._preview_count)
        scene_props.region_object = self._object
        scene_props.region_status = (
            f"已识别 {self._preview_count} 个领域"
        )
        scene_props.region_status_detail = "拆分已确认写入"
        set_region_highlight(
            context,
            self._object,
            self._preview_ids,
            self._preview_colors,
        )
        self._committed = True
        return True

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
        self._max_radius = max(extent * 0.12, 1e-3)

        self._object = obj
        self._mesh_data = mesh_data
        self._topology = topology
        self._region_ids = region_ids.copy()
        self._colors = colors.copy()
        self._live_base_count = int(region_count)
        self._snapshot_version = int(scene_props.region_version)
        self._snapshot_ids = region_ids.copy()
        self._snapshot_colors = colors.copy()
        self._edge_costs = edge_costs
        self._edge_mids = edge_mids
        self._edge_vert_a = edge_vert_a
        self._edge_vert_b = edge_vert_b
        self._vert_edge_offsets = offsets
        self._vert_edge_indices = both_e.astype(np.int32, copy=False)
        self._bvh = None
        self._paint_faces: set[int] = set()
        self._stroke_worlds: list[np.ndarray] = []
        self._preview_ids = None
        self._preview_colors = None
        self._preview_count = region_count
        self._completed_edges = np.empty(0, dtype=np.int32)
        self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
        self._preview_serial = 0
        self._brush_radius = float(
            getattr(scene_props, "split_brush_radius", BRUSH_RADIUS_DEFAULT)
            or BRUSH_RADIUS_DEFAULT
        )
        self._brush_xy = None
        self._painting = False
        self._last_sample_xy = None
        self._committed = False
        self._closed = False
        self._alive = True
        self._timer = None
        self._debounce_timer = None
        self._debounce_token = None

        scene_props.split_mode_active = True
        scene_props.split_confirm_requested = False
        scene_props.split_hover_id = -1
        scene_props.split_brush_radius = float(self._brush_radius)
        scene_props.region_object = obj
        register_split_draw_handler()
        self._set_phase_select(context)
        _add_modal_timer(self, context)
        _set_active_split_op(self)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context):
        self._alive = False
        if getattr(self, "_committed", False):
            self._cleanup_ui(context)
            return {"FINISHED"}
        self._cleanup_ui(context)
        try:
            set_region_highlight(
                context,
                self._object,
                self._snapshot_ids,
                self._snapshot_colors,
            )
        except Exception:
            pass
        return {"CANCELLED"}

    def modal(self, context: bpy.types.Context, event):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        obj = self._object

        # 侧栏确认已直接完成提交并清理，模态只需静默结束。
        if getattr(self, "_closed", False):
            return {"FINISHED"}

        try:
            if obj is None or obj.name not in bpy.data.objects:
                return self.cancel(context)
            if obj.mode == "EDIT":
                return self.cancel(context)
        except ReferenceError:
            return self.cancel(context)

        if scene_props.split_confirm_requested and not self._committed:
            scene_props.split_confirm_requested = False
            if self._commit_splits(context):
                self._alive = False
                self._cleanup_ui(context)
                self._closed = True
                self.report({"INFO"}, scene_props.region_status)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        if event.type == "TIMER":
            return {"RUNNING_MODAL"}

        if _is_undo_event(event):
            self._undo_paint(context)
            return {"RUNNING_MODAL"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            if self._committed:
                self._alive = False
                self._cleanup_ui(context)
                return {"FINISHED"}
            self._alive = False
            self._cleanup_ui(context)
            try:
                set_region_highlight(
                    context,
                    self._object,
                    self._snapshot_ids,
                    self._snapshot_colors,
                )
            except Exception:
                pass
            scene_props.split_status = "已取消拆分"
            self.report({"INFO"}, "已取消拆分")
            return {"CANCELLED"}

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if self._commit_splits(context):
                self._alive = False
                self._cleanup_ui(context)
                self.report({"INFO"}, scene_props.region_status)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        # [] 调整笔刷半径
        if event.value == "PRESS" and event.type in {
            "LEFT_BRACKET",
            "RIGHT_BRACKET",
        }:
            if event.type == "LEFT_BRACKET":
                self._brush_radius = max(
                    BRUSH_RADIUS_MIN,
                    self._brush_radius - BRUSH_RADIUS_STEP,
                )
            else:
                self._brush_radius = min(
                    BRUSH_RADIUS_MAX,
                    self._brush_radius + BRUSH_RADIUS_STEP,
                )
            scene_props.split_brush_radius = float(self._brush_radius)
            scene_props.split_status = (
                f"笔刷半径 {self._brush_radius:.0f}px · "
                f"目标 {scene_props.split_target_id}"
            )
            self._refresh_session(context)
            return {"RUNNING_MODAL"}

        if context.space_data is None or context.space_data.type != "VIEW_3D":
            return {"PASS_THROUGH"}
        if context.region is None or context.region.type != "WINDOW":
            return {"PASS_THROUGH"}

        phase = str(scene_props.split_phase)

        if phase == "SELECT":
            if event.type == "MOUSEMOVE":
                update_merge_label_projections(context)
                session = get_merge_label_session() or {"labels": []}
                hover = hit_test_labels(
                    event.mouse_region_x,
                    event.mouse_region_y,
                    session.get("labels", []),
                    LABEL_RADIUS_PX,
                )
                new_hover = -1 if hover is None else int(hover)
                if new_hover != int(scene_props.split_hover_id):
                    scene_props.split_hover_id = new_hover
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
                    return {"RUNNING_MODAL"}
                self._set_phase_brush(context, int(hit))
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        # BRUSH / PREVIEW
        self._brush_xy = (
            float(event.mouse_region_x),
            float(event.mouse_region_y),
        )

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self._cancel_debounce_timer()
            self._painting = True
            self._last_sample_xy = None
            self._paint_at(context, event)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self._painting:
                self._painting = False
                if self._paint_faces:
                    self._schedule_debounce(context)
                else:
                    self._refresh_session(context)
            return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE":
            if self._painting:
                if self._last_sample_xy is not None:
                    dx = event.mouse_region_x - self._last_sample_xy[0]
                    dy = event.mouse_region_y - self._last_sample_xy[1]
                    if (dx * dx + dy * dy) < STROKE_SAMPLE_PX * STROKE_SAMPLE_PX:
                        self._refresh_session(context)
                        return {"RUNNING_MODAL"}
                self._last_sample_xy = (
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
                self._paint_at(context, event)
            else:
                self._refresh_session(context)
            return {"PASS_THROUGH"}

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
