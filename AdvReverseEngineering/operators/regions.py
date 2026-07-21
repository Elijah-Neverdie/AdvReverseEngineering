# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""领域自动识别、清除、内存事务合并与智能拆分操作符。"""

from __future__ import annotations

import bpy
import numpy as np

from ..algorithms.regions import (
    compute_region_label_anchors,
    merge_region_ids,
    remove_region_ids,
    segment_regions_by_normal,
)
from ..algorithms.region_split import (
    candidate_hard_edges,
    count_components_after_cut,
    filter_internal_cut_edges,
    group_candidate_edge_chains,
    grow_ridge_cut_to_boundary,
    prepare_edge_costs,
    refine_cut_to_hard_ridge,
    seal_cut_to_region_boundary,
    split_region_by_cut_edges,
    unify_cut_edges_as_line,
)
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import (
    LABEL_RADIUS_PX,
    clear_region_highlight,
    get_merge_label_session,
    register_label_draw_handler,
    register_split_draw_handler,
    set_fit_angle_label_session,
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
MODAL_TIMER_STEP = 0.1
EDGE_PICK_PX = 14.0
HARD_THRESHOLD_DEFAULT = 0.25
HARD_THRESHOLD_MIN = 0.02
HARD_THRESHOLD_MAX = 1.0
HARD_THRESHOLD_STEP = 0.05
SPLIT_PREVIEW_DEBOUNCE_SEC = 0.12
SPLIT_CANDIDATE_DEBOUNCE_SEC = 0.08


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
_REMOVE_OP_KEY = "are_active_remove_op"


def _set_active_merge_op(operator) -> None:
    bpy.app.driver_namespace[_MERGE_OP_KEY] = operator


def _get_active_merge_op():
    return bpy.app.driver_namespace.get(_MERGE_OP_KEY)


def _clear_active_merge_op(operator) -> None:
    if bpy.app.driver_namespace.get(_MERGE_OP_KEY) is operator:
        bpy.app.driver_namespace[_MERGE_OP_KEY] = None


def _set_active_remove_op(operator) -> None:
    bpy.app.driver_namespace[_REMOVE_OP_KEY] = operator


def _get_active_remove_op():
    return bpy.app.driver_namespace.get(_REMOVE_OP_KEY)


def _clear_active_remove_op(operator) -> None:
    if bpy.app.driver_namespace.get(_REMOVE_OP_KEY) is operator:
        bpy.app.driver_namespace[_REMOVE_OP_KEY] = None


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
                _teardown_labels_then_sync(bpy.context)
        elif kind == "remove":
            stuck = (
                getattr(scene_props, "remove_mode_active", False)
                and getattr(scene_props, "remove_confirm_requested", False)
            )
            if stuck:
                scene_props.remove_confirm_requested = False
                scene_props.remove_mode_active = False
                scene_props.remove_hover_id = -1
                scene_props.remove_status = "移除模态已失联，已强制退出"
                _teardown_labels_then_sync(bpy.context)
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
                set_split_stroke_session(None)
                _teardown_labels_then_sync(bpy.context)
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
        # 大网格上逐标签 scene.ray_cast 会让合并/悬停卡死；朝向过滤仍保留
        "skip_occlusion": True,
    }


def _modal_busy(scene_props) -> bool:
    """合并、移除、拆分或拟合模态进行中。"""
    return bool(
        scene_props.merge_mode_active
        or getattr(scene_props, "remove_mode_active", False)
        or scene_props.split_mode_active
        or getattr(scene_props, "fit_mode_active", False)
    )


def _resolve_region_object(context: bpy.types.Context, scene_props):
    """解析当前应显示标签的领域对象。"""
    obj = getattr(scene_props, "region_object", None)
    if (
        obj is not None
        and obj.name in bpy.data.objects
        and obj.type == "MESH"
        and obj.data.attributes.get(REGION_ID_ATTR) is not None
    ):
        return obj
    obj = context.active_object
    if (
        obj is not None
        and obj.type == "MESH"
        and obj.data.attributes.get(REGION_ID_ATTR) is not None
    ):
        return obj
    return None


def sync_region_label_overlay(context: bpy.types.Context | None = None) -> None:
    """
    空闲状态下同步编号标签覆盖层。

    有领域且开启「显示领域」时常驻显示编号；否则拆除。
    模态进行中不改写会话（由模态算子接管）。
    """
    context = context or bpy.context
    if context is None or context.scene is None:
        return
    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None:
        return
    if hasattr(scene_props, "label_hover_id"):
        scene_props.label_hover_id = -1
    if _modal_busy(scene_props):
        return
    if not scene_props.show_region_highlight:
        unregister_label_draw_handler()
        set_merge_label_session(None)
        return
    obj = _resolve_region_object(context, scene_props)
    if obj is None:
        unregister_label_draw_handler()
        set_merge_label_session(None)
        return
    region_ids = read_region_ids(obj.data)
    if region_ids is None or not np.any(region_ids >= 0):
        unregister_label_draw_handler()
        set_merge_label_session(None)
        return
    valid = region_ids[region_ids >= 0]
    region_count = max(int(valid.max()) + 1, int(scene_props.region_count))
    colors = _read_region_colors(obj, region_count)
    mesh_data = extract_mesh_data(obj)
    session = _build_label_session(region_ids, mesh_data, colors)
    session["object_name"] = obj.name
    session["idle"] = True
    session["preview_version"] = int(scene_props.region_version)
    set_merge_label_session(session)
    register_label_draw_handler()
    scene_props.region_object = obj
    _tag_redraw(context)


def _teardown_labels_then_sync(context: bpy.types.Context) -> None:
    """拆除当前标签会话后按空闲状态重建。"""
    unregister_label_draw_handler()
    set_merge_label_session(None)
    sync_region_label_overlay(context)


_addon_keymaps: list[tuple] = []


def register_label_hover_keymap() -> None:
    """注册视口鼠标移动时的标签悬停更新。"""
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc is None:
        return
    km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
    kmi = km.keymap_items.new(
        "are.update_label_hover",
        type="MOUSEMOVE",
        value="ANY",
    )
    _addon_keymaps.append((km, kmi))


def unregister_label_hover_keymap() -> None:
    """注销标签悬停快捷键。"""
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _addon_keymaps.clear()


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
                sync_region_label_overlay(context)

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
        sync_region_label_overlay(context)
        self.report({"INFO"}, "已清除领域标记")
        return {"FINISHED"}


class ARE_OT_update_label_hover(bpy.types.Operator):
    """空闲状态下根据鼠标位置更新编号悬停高亮。"""

    bl_idname = "are.update_label_hover"
    bl_label = "更新领域标签悬停"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None or not scene_props.show_region_highlight:
            return False
        if _modal_busy(scene_props):
            return False
        return get_merge_label_session() is not None or (
            _resolve_region_object(context, scene_props) is not None
        )

    def invoke(self, context: bpy.types.Context, event):
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None:
            return {"PASS_THROUGH"}
        if context.area is None or context.area.type != "VIEW_3D":
            return {"PASS_THROUGH"}
        if context.region is None or context.region.type != "WINDOW":
            return {"PASS_THROUGH"}

        if get_merge_label_session() is None:
            sync_region_label_overlay(context)
        session = get_merge_label_session()
        if session is None:
            if int(getattr(scene_props, "label_hover_id", -1)) != -1:
                scene_props.label_hover_id = -1
                _tag_redraw(context)
            return {"PASS_THROUGH"}

        update_merge_label_projections(context)
        hover = hit_test_labels(
            event.mouse_region_x,
            event.mouse_region_y,
            session.get("labels", []),
            LABEL_RADIUS_PX,
        )
        new_hover = -1 if hover is None else int(hover)
        if new_hover != int(getattr(scene_props, "label_hover_id", -1)):
            scene_props.label_hover_id = new_hover
            _tag_redraw(context)
        return {"PASS_THROUGH"}


class ARE_OT_confirm_remove_regions(bpy.types.Operator):
    """面板确认按钮：通知移除模态提交。"""

    bl_idname = "are.confirm_remove_regions"
    bl_label = "确认"
    bl_description = "结束移除领域并保留当前结果"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return scene_props is not None and bool(
            getattr(scene_props, "remove_mode_active", False)
        )

    def execute(self, context: bpy.types.Context):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        op = _get_active_remove_op()
        if op is not None:
            try:
                op.confirm_from_panel(context)
            except Exception as exc:
                self.report({"ERROR"}, f"确认移除失败: {exc}")
                scene_props.remove_confirm_requested = True
                _schedule_force_exit_check("remove")
                return {"FINISHED"}
            if scene_props.remove_mode_active:
                scene_props.remove_mode_active = False
                scene_props.remove_confirm_requested = False
                _teardown_labels_then_sync(context)
            return {"FINISHED"}
        scene_props.remove_confirm_requested = True
        _schedule_force_exit_check("remove")
        return {"FINISHED"}


class ARE_OT_remove_regions(bpy.types.Operator):
    """
    模态移除领域（内存事务）。

    点击编号移除该领域（面标为忽略）；Ctrl+Z 撤销；
    确认时一次写入 Mesh。
    """

    bl_idname = "are.remove_regions"
    bl_label = "移除领域"
    bl_description = (
        "点击编号移除领域；Ctrl+Z 撤销上次；确认写入，Esc 取消"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _cleanup_ui(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        _remove_modal_timer(self, context)
        _clear_active_remove_op(self)
        scene_props.remove_mode_active = False
        scene_props.remove_hover_id = -1
        scene_props.remove_confirm_requested = False
        _teardown_labels_then_sync(context)
        _tag_redraw(context)

    def confirm_from_panel(self, context: bpy.types.Context) -> None:
        if getattr(self, "_closed", False):
            return
        try:
            if not self._committed:
                self._commit_to_mesh(context)
        finally:
            self._finish_mode(context, cancelled=False)
            self._closed = True

    def _commit_to_mesh(self, context: bpy.types.Context) -> None:
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
        ignored = int(np.count_nonzero(self._live_ids < 0))
        scene_props.region_ignored_face_count = ignored
        set_region_highlight(
            context,
            obj,
            self._live_ids,
            self._live_colors,
        )
        self._committed = True

    def _finish_mode(self, context: bpy.types.Context, cancelled: bool) -> set:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if cancelled and getattr(self, "_committed", False):
            cancelled = False
        self._cleanup_ui(context)
        if cancelled:
            scene_props.remove_status = "已取消移除"
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
            scene_props.remove_status = (
                f"移除完成，当前 {scene_props.region_count} 个领域"
            )
            scene_props.region_status = (
                f"已识别 {scene_props.region_count} 个领域"
            )
        return {"CANCELLED"} if cancelled else {"FINISHED"}

    def _refresh_preview(self, context: bpy.types.Context) -> None:
        obj = self._object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        session = _build_label_session(
            self._live_ids,
            self._mesh_data,
            self._live_colors,
        )
        session["object_name"] = obj.name
        self._preview_serial += 1
        session["preview_version"] = self._preview_serial
        set_merge_label_session(session)
        scene_props.region_count = int(self._live_count)
        scene_props.region_object = obj
        scene_props.remove_status = (
            f"点击编号移除 · 当前 {self._live_count} 个领域 · "
            f"已操作 {len(self._history)} 次 · Ctrl+Z 撤销"
        )
        _tag_redraw(context)

    def _push_history(self) -> None:
        self._history.append(
            {
                "ids": self._live_ids.copy(),
                "colors": self._live_colors.copy(),
                "count": int(self._live_count),
            }
        )
        self._redo.clear()

    def _undo_remove(self, context: bpy.types.Context) -> None:
        if not self._history:
            getattr(context.scene, SCENE_PROP_NAME).remove_status = (
                "没有可撤销的移除"
            )
            _tag_redraw(context)
            return
        self._redo.append(
            {
                "ids": self._live_ids.copy(),
                "colors": self._live_colors.copy(),
                "count": int(self._live_count),
            }
        )
        previous = self._history.pop()
        self._live_ids = previous["ids"]
        self._live_colors = previous["colors"]
        self._live_count = int(previous["count"])
        self._refresh_preview(context)

    def _redo_remove(self, context: bpy.types.Context) -> None:
        if not self._redo:
            return
        self._history.append(
            {
                "ids": self._live_ids.copy(),
                "colors": self._live_colors.copy(),
                "count": int(self._live_count),
            }
        )
        nxt = self._redo.pop()
        self._live_ids = nxt["ids"]
        self._live_colors = nxt["colors"]
        self._live_count = int(nxt["count"])
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
        self._preview_serial = 0
        self._committed = False
        self._closed = False
        self._timer = None

        session = _build_label_session(region_ids, mesh_data, colors)
        session["object_name"] = obj.name
        session["preview_version"] = 0
        set_merge_label_session(session)
        register_label_draw_handler()
        _set_active_remove_op(self)

        scene_props.remove_mode_active = True
        scene_props.remove_hover_id = -1
        scene_props.remove_confirm_requested = False
        scene_props.remove_status = (
            "点击编号移除领域；Ctrl+Z 撤销；确认写入；Esc 取消"
        )
        scene_props.region_object = obj

        _add_modal_timer(self, context)
        context.window_manager.modal_handler_add(self)
        _tag_redraw(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context):
        if getattr(self, "_committed", False):
            self._cleanup_ui(context)
            return {"FINISHED"}
        self._cleanup_ui(context)
        return {"CANCELLED"}

    def modal(self, context: bpy.types.Context, event):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        obj = self._object

        if getattr(self, "_closed", False):
            return {"FINISHED"}

        try:
            if obj is None or obj.name not in bpy.data.objects:
                return self.cancel(context)
            if obj.mode == "EDIT":
                return self.cancel(context)
        except ReferenceError:
            return self.cancel(context)

        if scene_props.remove_confirm_requested and not self._committed:
            scene_props.remove_confirm_requested = False
            self._commit_to_mesh(context)
            self.report(
                {"INFO"},
                f"移除完成，当前 {self._live_count} 个领域",
            )
            return self._finish_mode(context, cancelled=False)

        if event.type == "TIMER":
            return {"RUNNING_MODAL"}

        if _is_undo_event(event):
            if self._committed:
                return {"RUNNING_MODAL"}
            self._undo_remove(context)
            return {"RUNNING_MODAL"}
        if _is_redo_event(event):
            if self._committed:
                return {"RUNNING_MODAL"}
            self._redo_remove(context)
            return {"RUNNING_MODAL"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            if self._committed:
                return self._finish_mode(context, cancelled=False)
            self.report({"INFO"}, "已取消移除")
            return self._finish_mode(context, cancelled=True)

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if not self._committed:
                self._commit_to_mesh(context)
            self.report(
                {"INFO"},
                f"移除完成，当前 {self._live_count} 个领域",
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
            if new_hover != int(scene_props.remove_hover_id):
                scene_props.remove_hover_id = new_hover
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
            rid = int(hit)
            if not np.any(self._live_ids == rid):
                scene_props.remove_status = f"领域 {rid} 已不存在"
                _tag_redraw(context)
                return {"RUNNING_MODAL"}
            try:
                self._push_history()
                self._live_ids, self._live_colors, self._live_count = (
                    remove_region_ids(
                        self._live_ids,
                        self._live_colors,
                        rid,
                    )
                )
            except Exception as exc:
                if self._history:
                    previous = self._history.pop()
                    self._live_ids = previous["ids"]
                    self._live_colors = previous["colors"]
                    self._live_count = int(previous["count"])
                scene_props.remove_status = f"移除失败: {exc}"
                _tag_redraw(context)
                return {"RUNNING_MODAL"}
            self._refresh_preview(context)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


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
                _teardown_labels_then_sync(context)
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
        scene_props.merge_mode_active = False
        scene_props.merge_anchor_id = -1
        scene_props.merge_hover_id = -1
        scene_props.merge_confirm_requested = False
        _teardown_labels_then_sync(context)
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

        # 进入合并时清掉拟合内角标，避免残留会话拖慢投影/绘制
        set_fit_angle_label_session(None)

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
                set_split_stroke_session(None)
                _teardown_labels_then_sync(context)
            return {"FINISHED"}
        scene_props.split_confirm_requested = True
        _schedule_force_exit_check("split")
        return {"FINISHED"}


class ARE_OT_split_regions(bpy.types.Operator):
    """
    先选编号领域，再显示候选硬边粗线；
    可多次点选硬边，选中段合并为一条切线后再分色预览确认。
    """

    bl_idname = "are.split_regions"
    bl_label = "拆分领域"
    bl_description = (
        "点击编号选择领域，Ctrl+滚轮调线框阈值，多次点选硬边合并切线"
    )
    # 不用 UNDO：确认写入后若模态被 cancel，Blender 会回滚 Mesh 属性导致「拆了又没了」
    bl_options = {"REGISTER"}

    def _cleanup_ui(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._cancel_debounce_timer()
        self._cancel_candidate_timer()
        _remove_modal_timer(self, context)
        _clear_active_split_op(self)
        unregister_split_draw_handler()
        set_split_stroke_session(None)
        scene_props.split_mode_active = False
        scene_props.split_confirm_requested = False
        scene_props.split_target_id = -1
        scene_props.split_hover_id = -1
        scene_props.split_phase = "IDLE"
        _teardown_labels_then_sync(context)
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

    def _cancel_candidate_timer(self) -> None:
        timer = getattr(self, "_candidate_timer", None)
        if timer is None:
            return
        try:
            bpy.app.timers.unregister(timer)
        except Exception:
            pass
        self._candidate_timer = None
        self._candidate_token = None

    def _schedule_candidate_rebuild(self, context: bpy.types.Context) -> None:
        """滚轮连调时防抖重建候选，避免每帧对大网格做连通分组。"""
        self._cancel_candidate_timer()
        token = object()
        self._candidate_token = token
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.split_status = (
            f"线框阈值 {self._hard_threshold:.2f} · 正在更新候选…"
        )
        _tag_redraw(context)

        def _callback():
            if getattr(self, "_candidate_token", None) is not token:
                return None
            if not getattr(self, "_alive", False):
                return None
            try:
                ctx = bpy.context
                self._rebuild_candidates(ctx)
                props = getattr(ctx.scene, SCENE_PROP_NAME, None)
                if props is not None:
                    props.split_status = (
                        f"线框阈值 {self._hard_threshold:.2f} · "
                        f"候选 {len(self._candidate_edges)} 条边 / "
                        f"{len(self._candidate_chains)} 条棱线 · "
                        f"已选 {self._selected_chain_count()} 条"
                    )
                self._preview_serial += 1
                if self._selected_cut_edges:
                    self._schedule_preview(ctx)
                else:
                    self._clear_preview_keep_selection(ctx)
                self._refresh_session(ctx)
            except Exception as exc:
                scene = getattr(bpy.context.scene, SCENE_PROP_NAME, None)
                if scene is not None:
                    scene.split_status = f"候选更新失败: {exc}"
            return None

        self._candidate_timer = _callback
        bpy.app.timers.register(
            _callback,
            first_interval=SPLIT_CANDIDATE_DEBOUNCE_SEC,
        )

    def _schedule_preview(self, context: bpy.types.Context) -> None:
        self._cancel_debounce_timer()
        token = object()
        self._debounce_token = token

        def _callback():
            if getattr(self, "_debounce_token", None) is not token:
                return None
            if not getattr(self, "_alive", False):
                return None
            try:
                self._run_edge_preview(bpy.context)
            except Exception as exc:
                scene = getattr(bpy.context.scene, SCENE_PROP_NAME, None)
                if scene is not None:
                    scene.split_status = f"预览失败: {exc}"
            return None

        self._debounce_timer = _callback
        bpy.app.timers.register(
            _callback,
            first_interval=SPLIT_PREVIEW_DEBOUNCE_SEC,
        )

    def _refresh_session(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        hover_world = np.empty((0, 2, 3), dtype=np.float64)
        hover = getattr(self, "_hover_edge", None)
        if hover is not None:
            # 悬停只高亮当前小边，避免大棱线每帧重建世界坐标
            hover_world = self._edges_to_world(
                np.asarray([int(hover)], dtype=np.int32)
            )
        selected = (
            np.asarray(sorted(self._selected_cut_edges), dtype=np.int32)
            if self._selected_cut_edges
            else np.empty(0, dtype=np.int32)
        )
        session = {
            "object_name": self._object.name,
            "candidate_edges_world": self._candidate_edges_world,
            "selected_edges_world": self._edges_to_world(selected),
            "hover_edge_world": hover_world,
            "completed_edges_world": self._completed_edges_world,
            "preview_ids": self._preview_ids,
            "preview_colors": self._preview_colors,
            "base_ids": self._region_ids,
            "preview_version": self._preview_serial,
        }
        set_split_stroke_session(session)
        scene_props.split_hard_threshold = float(self._hard_threshold)
        _tag_redraw(context)

    def _set_phase_select(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.split_phase = "SELECT"
        scene_props.split_target_id = -1
        scene_props.split_status = (
            "点击编号选择要拆分的领域；Esc 取消"
        )
        self._selected_cut_edges.clear()
        self._ridge_cut_edges = np.empty(0, dtype=np.int32)
        self._candidate_edges = np.empty(0, dtype=np.int32)
        self._candidate_edges_world = np.empty((0, 2, 3), dtype=np.float64)
        self._hover_edge = None
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

    def _set_phase_edge(self, context: bpy.types.Context, target_id: int) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        scene_props.split_phase = "EDGE"
        scene_props.split_target_id = int(target_id)
        self._selected_cut_edges.clear()
        self._ridge_cut_edges = np.empty(0, dtype=np.int32)
        self._hover_edge = None
        self._preview_ids = None
        self._preview_colors = None
        self._completed_edges = np.empty(0, dtype=np.int32)
        self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
        self._preview_serial += 1
        session = get_merge_label_session()
        if session is not None:
            session["preview_version"] = self._preview_serial
        self._rebuild_candidates(context)
        scene_props.split_status = (
            f"已选领域 {target_id} · "
            f"线框阈值 {self._hard_threshold:.2f} · "
            f"候选 {len(self._candidate_edges)} 条边 / "
            f"{len(self._candidate_chains)} 条棱线 · "
            "可多次点选折棱，合并成一条切线后再预览"
        )
        self._refresh_session(context)

    def _chain_edges_for(self, edge_index: int) -> set[int]:
        chain_id = self._edge_to_chain.get(int(edge_index))
        if chain_id is None:
            return {int(edge_index)}
        return {int(e) for e in self._candidate_chains[chain_id].tolist()}

    def _rebuild_candidates(self, context: bpy.types.Context) -> None:
        target = int(getattr(context.scene, SCENE_PROP_NAME).split_target_id)
        if target < 0:
            self._candidate_edges = np.empty(0, dtype=np.int32)
            self._candidate_edges_world = np.empty((0, 2, 3), dtype=np.float64)
            self._candidate_chains = []
            self._edge_to_chain = {}
            return
        raw = candidate_hard_edges(
            self._topology,
            self._mesh_data["normals"],
            self._region_ids,
            target,
            float(self._hard_threshold),
        )
        # 只做连通分组，不做「能否一分为二」预筛——后者在大领域上极慢，
        # 且模糊硬棱带常被误杀成 0 候选。
        chains = group_candidate_edge_chains(
            raw,
            self._edge_vert_a,
            self._edge_vert_b,
        )
        self._candidate_edges = np.asarray(raw, dtype=np.int32)
        self._candidate_chains = chains
        self._edge_to_chain = {}
        for chain_id, chain in enumerate(chains):
            for edge_index in chain.tolist():
                self._edge_to_chain[int(edge_index)] = chain_id

        # 已延伸的完整分割棱不得被候选集合裁短（棱线中段常不在候选里）
        ridge = getattr(self, "_ridge_cut_edges", None)
        if ridge is not None and len(ridge) > 0:
            self._selected_cut_edges = {int(e) for e in ridge.tolist()}
            self._completed_edges = np.asarray(ridge, dtype=np.int32)
            self._completed_edges_world = self._edges_to_world(self._completed_edges)
        self._candidate_edges_world = self._edges_to_world(self._candidate_edges)
        if self._hover_edge is not None and int(self._hover_edge) not in {
            int(e) for e in self._candidate_edges.tolist()
        }:
            self._hover_edge = None

    def _adjust_threshold(self, context: bpy.types.Context, delta: float) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._hard_threshold = float(
            min(
                HARD_THRESHOLD_MAX,
                max(HARD_THRESHOLD_MIN, self._hard_threshold + delta),
            )
        )
        scene_props.split_hard_threshold = float(self._hard_threshold)
        # 立即反馈数值，候选重建防抖，避免滚轮连调卡死
        self._schedule_candidate_rebuild(context)

    def _clear_preview_keep_selection(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._preview_ids = None
        self._preview_colors = None
        self._completed_edges = np.empty(0, dtype=np.int32)
        self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
        if scene_props.split_target_id >= 0:
            scene_props.split_phase = "EDGE"

    def _pick_candidate_edge(self, context: bpy.types.Context, event):
        """屏幕空间最近候选边中点；超出拾取半径返回 None。"""
        from bpy_extras import view3d_utils

        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return None
        if len(self._candidate_edges) == 0:
            return None
        mx = float(event.mouse_region_x)
        my = float(event.mouse_region_y)
        best_edge = None
        best_dist = float(EDGE_PICK_PX)
        for edge_index in self._candidate_edges.tolist():
            mid = self._edge_mids[int(edge_index)]
            screen = view3d_utils.location_3d_to_region_2d(
                region,
                rv3d,
                mid,
            )
            if screen is None:
                continue
            dx = float(screen.x) - mx
            dy = float(screen.y) - my
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_edge = int(edge_index)
        return best_edge

    def _selected_chain_count(self) -> int:
        chain_ids: set[int] = set()
        for edge_index in self._selected_cut_edges:
            chain_id = self._edge_to_chain.get(int(edge_index))
            if chain_id is not None:
                chain_ids.add(chain_id)
        return len(chain_ids)

    def _selected_component_edges(self, edge_index: int) -> set[int]:
        """当前选中切边里，与给定边同连通分量的边集合。"""
        edge_index = int(edge_index)
        if edge_index not in self._selected_cut_edges:
            return set()
        vert_to_edges: dict[int, list[int]] = {}
        for e in self._selected_cut_edges:
            e = int(e)
            for vert in (
                int(self._edge_vert_a[e]),
                int(self._edge_vert_b[e]),
            ):
                vert_to_edges.setdefault(vert, []).append(e)
        visited: set[int] = set()
        stack = [edge_index]
        visited.add(edge_index)
        while stack:
            cur = stack.pop()
            for vert in (
                int(self._edge_vert_a[cur]),
                int(self._edge_vert_b[cur]),
            ):
                for neighbor in vert_to_edges.get(vert, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        return visited

    def _apply_unified_cut(self, context: bpy.types.Context) -> None:
        """把已点选边合并为一条切线并尝试软预览（失败不清空）。"""
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        target = int(scene_props.split_target_id)
        if target < 0 or not self._selected_cut_edges:
            self._ridge_cut_edges = np.empty(0, dtype=np.int32)
            self._completed_edges = np.empty(0, dtype=np.int32)
            self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
            self._preview_ids = None
            self._preview_colors = None
            self._preview_serial += 1
            self._refresh_session(context)
            return

        seeds = np.asarray(sorted(self._selected_cut_edges), dtype=np.int32)
        unified = unify_cut_edges_as_line(
            seeds,
            self._topology,
            self._region_ids,
            target,
            self._edge_costs,
            self._vert_edge_offsets,
            self._vert_edge_indices,
            self._edge_vert_a,
            self._edge_vert_b,
        )
        if len(unified) == 0:
            unified = filter_internal_cut_edges(
                seeds,
                self._topology,
                self._region_ids,
                target,
            )
        # unify 内已 seal；再封一次确保多点选后悬空端贴周界
        if len(unified):
            unified = seal_cut_to_region_boundary(
                unified,
                self._topology,
                self._region_ids,
                target,
                self._edge_costs,
                self._vert_edge_offsets,
                self._vert_edge_indices,
                self._edge_vert_a,
                self._edge_vert_b,
            )
            # 吸附到硬棱测地线，使边界接近「识别领域」的整齐硬边
            unified = refine_cut_to_hard_ridge(
                unified,
                self._topology,
                self._region_ids,
                target,
                self._edge_costs,
                self._edge_mids,
                self._vert_edge_offsets,
                self._vert_edge_indices,
                self._edge_vert_a,
                self._edge_vert_b,
            )
        self._ridge_cut_edges = np.asarray(unified, dtype=np.int32)
        self._selected_cut_edges = {int(e) for e in self._ridge_cut_edges.tolist()}
        self._completed_edges = self._ridge_cut_edges.copy()
        self._completed_edges_world = self._edges_to_world(self._completed_edges)
        self._preview_serial += 1
        n_sel = len(self._selected_cut_edges)
        scene_props.split_status = (
            f"已选切线 {n_sel} 段 · 线框阈值 {self._hard_threshold:.2f} · "
            "可继续点选补全 · 正在尝试预览…"
        )
        self._schedule_preview(context)
        self._refresh_session(context)

    def _toggle_edge(self, context: bpy.types.Context, edge_index: int) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        edge_index = int(edge_index)

        # 再点已选分量 → 仅去掉该段，保留其它点选
        if edge_index in self._selected_cut_edges:
            remove = self._selected_component_edges(edge_index)
            self._selected_cut_edges -= remove
            self._preview_ids = None
            self._preview_colors = None
            if not self._selected_cut_edges:
                self._ridge_cut_edges = np.empty(0, dtype=np.int32)
                self._clear_preview_keep_selection(context)
                scene_props.split_phase = "EDGE"
                scene_props.split_status = (
                    f"已选领域 {scene_props.split_target_id} · "
                    f"线框阈值 {self._hard_threshold:.2f} · "
                    f"候选 {len(self._candidate_edges)} 条边 · 继续点选折棱"
                )
                self._preview_serial += 1
                self._refresh_session(context)
                return
            self._apply_unified_cut(context)
            return

        target = int(scene_props.split_target_id)
        # 从短边延伸，并与已有点选合并为一条切线
        grown = grow_ridge_cut_to_boundary(
            edge_index,
            self._topology,
            self._region_ids,
            target,
            self._edge_costs,
            self._edge_mids,
            self._vert_edge_offsets,
            self._vert_edge_indices,
            self._edge_vert_a,
            self._edge_vert_b,
            vertices=self._mesh_data["vertices"],
        )
        grown = filter_internal_cut_edges(
            grown,
            self._topology,
            self._region_ids,
            target,
        )
        if len(grown) == 0:
            grown = filter_internal_cut_edges(
                np.asarray([edge_index], dtype=np.int32),
                self._topology,
                self._region_ids,
                target,
            )
        if len(grown) == 0:
            scene_props.split_status = (
                "点到的是已有领域分界，不能加入切线。"
                "请点选当前领域内部的折棱（青绿候选）；已保留现有选中"
            )
            self.report({"WARNING"}, scene_props.split_status)
            self._refresh_session(context)
            return

        self._selected_cut_edges |= {int(e) for e in grown.tolist()}
        self._apply_unified_cut(context)

    def _run_edge_preview(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        target = int(scene_props.split_target_id)
        ridge = getattr(self, "_ridge_cut_edges", None)
        if target < 0 or ridge is None or len(ridge) == 0:
            if not self._selected_cut_edges:
                self._clear_preview_keep_selection(context)
                self._refresh_session(context)
                return
            ridge = np.asarray(sorted(self._selected_cut_edges), dtype=np.int32)
            self._ridge_cut_edges = ridge

        # 预览必须用点选合并后的完整切线，禁止 Dijkstra 改短/改道
        completed = np.asarray(ridge, dtype=np.int32)
        self._selected_cut_edges = {int(e) for e in completed.tolist()}
        new_ids, new_colors, new_count = split_region_by_cut_edges(
            self._region_ids,
            self._topology,
            completed,
            self._colors,
            target_rid=target,
            smooth_iterations=2,
            edge_costs=self._edge_costs,
            face_centers=self._mesh_data["face_centers"],
        )
        changed = bool(np.any(new_ids != self._region_ids))

        self._completed_edges = completed
        self._completed_edges_world = self._edges_to_world(completed)
        self._preview_ids = new_ids if changed else None
        self._preview_colors = new_colors if changed else None
        self._preview_count = int(new_count)
        self._preview_serial += 1
        if changed:
            scene_props.split_phase = "PREVIEW"
            scene_props.split_status = (
                f"预览：{self._live_base_count} → {new_count} 个领域 · "
                f"切线 {len(completed)} 段 · "
                "确认拆分保存 · Ctrl+Z 清除选中 · Esc 取消"
            )
            self.report({"INFO"}, scene_props.split_status)
        else:
            # 不清空红线：说明真实原因，便于继续补刀
            scene_props.split_phase = "EDGE"
            n_comp = count_components_after_cut(
                self._region_ids,
                self._topology,
                completed,
                target,
            )
            if n_comp <= 1:
                scene_props.split_status = (
                    f"切线 {len(completed)} 段看起来贯通，但拓扑上领域仍连成一块"
                    "（两端未封死或背面/侧面仍相连）。"
                    "请在缺口或另一侧继续点选折棱补刀"
                )
            else:
                scene_props.split_status = (
                    f"切线已分成 {n_comp} 块但未能写入预览，请再点一次或 Ctrl+Z 后重选"
                )
            self._preview_ids = None
            self._preview_colors = None
        self._refresh_session(context)

    def _edges_to_world(self, edges: np.ndarray) -> np.ndarray:
        edge_arr = np.asarray(edges, dtype=np.int32)
        if len(edge_arr) == 0:
            return np.empty((0, 2, 3), dtype=np.float64)
        verts = self._mesh_data["vertices"]
        va = self._edge_vert_a[edge_arr]
        vb = self._edge_vert_b[edge_arr]
        return np.stack((verts[va], verts[vb]), axis=1).astype(
            np.float64, copy=False
        )

    def _undo_selection(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self._cancel_debounce_timer()
        if self._preview_ids is not None or self._selected_cut_edges:
            self._selected_cut_edges.clear()
            self._ridge_cut_edges = np.empty(0, dtype=np.int32)
            self._hover_edge = None
            self._preview_ids = None
            self._preview_colors = None
            self._completed_edges = np.empty(0, dtype=np.int32)
            self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
            self._preview_serial += 1
            scene_props.split_phase = (
                "EDGE" if scene_props.split_target_id >= 0 else "SELECT"
            )
            scene_props.split_status = "已清除选中边/预览，可重新点选"
            self._refresh_session(context)
            return
        scene_props.split_status = "没有可撤销的选中边"
        _tag_redraw(context)

    def _commit_splits(self, context: bpy.types.Context) -> bool:
        if self._preview_ids is None or self._preview_colors is None:
            self.report({"WARNING"}, "还没有可确认的分色预览")
            return False
        # 先标记已提交，避免后续异常导致 cancel 回滚显示
        self._committed = True
        write_region_ids(self._object.data, self._preview_ids)
        # 强制刷新依赖图，确保属性写入立即可见
        try:
            self._object.data.update()
            context.view_layer.update()
        except Exception:
            pass
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
        try:
            bpy.ops.ed.undo_push(message="拆分领域")
        except Exception:
            pass
        # 校验写入是否生效
        written = read_region_ids(self._object.data)
        if written is None or not np.array_equal(
            written, np.asarray(self._preview_ids, dtype=np.int32)
        ):
            self.report({"ERROR"}, "拆分写入校验失败，请重试")
            self._committed = False
            return False
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
        # 补全搜索覆盖大半模型，避免长棱线被半径截断
        self._max_radius = max(extent * 0.75, 1e-3)
        self._mesh_extent = extent

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
        self._candidate_edges = np.empty(0, dtype=np.int32)
        self._candidate_edges_world = np.empty((0, 2, 3), dtype=np.float64)
        self._candidate_chains: list[np.ndarray] = []
        self._edge_to_chain: dict[int, int] = {}
        self._selected_cut_edges: set[int] = set()
        self._ridge_cut_edges = np.empty(0, dtype=np.int32)
        self._hover_edge = None
        self._preview_ids = None
        self._preview_colors = None
        self._preview_count = region_count
        self._completed_edges = np.empty(0, dtype=np.int32)
        self._completed_edges_world = np.empty((0, 2, 3), dtype=np.float64)
        self._preview_serial = 0
        self._hard_threshold = float(
            getattr(scene_props, "split_hard_threshold", HARD_THRESHOLD_DEFAULT)
            or HARD_THRESHOLD_DEFAULT
        )
        # 旧版「硬度 0~1」默认偏高；若明显高于识别线框阈值，回落到识别参数以便调出缓棱。
        identify_wire = float(
            getattr(scene_props, "region_wireframe_threshold", 0.1) or 0.1
        )
        if self._hard_threshold > 0.6 and identify_wire <= 0.5:
            self._hard_threshold = float(
                max(identify_wire, HARD_THRESHOLD_DEFAULT)
            )
        self._hard_threshold = float(
            min(
                HARD_THRESHOLD_MAX,
                max(HARD_THRESHOLD_MIN, self._hard_threshold),
            )
        )
        self._committed = False
        self._closed = False
        self._alive = True
        self._timer = None
        self._debounce_timer = None
        self._debounce_token = None
        self._candidate_timer = None
        self._candidate_token = None

        scene_props.split_mode_active = True
        scene_props.split_confirm_requested = False
        scene_props.split_hover_id = -1
        scene_props.split_hard_threshold = float(self._hard_threshold)
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
            self._closed = True
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
            self._undo_selection(context)
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

        # Ctrl + 滚轮：调节线框阈值（与识别领域相同语义）
        if event.ctrl and event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            phase = str(scene_props.split_phase)
            if phase in {"EDGE", "PREVIEW"}:
                delta = (
                    HARD_THRESHOLD_STEP
                    if event.type == "WHEELUPMOUSE"
                    else -HARD_THRESHOLD_STEP
                )
                self._adjust_threshold(context, delta)
                return {"RUNNING_MODAL"}

        # 面板滑条改阈值时同步（任意事件轮询）
        panel_thr = float(
            getattr(scene_props, "split_hard_threshold", self._hard_threshold)
        )
        if abs(panel_thr - self._hard_threshold) > 1e-6 and str(
            scene_props.split_phase
        ) in {"EDGE", "PREVIEW"}:
            self._hard_threshold = float(
                min(HARD_THRESHOLD_MAX, max(HARD_THRESHOLD_MIN, panel_thr))
            )
            self._schedule_candidate_rebuild(context)

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
                self._set_phase_edge(context, int(hit))
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        # EDGE / PREVIEW：硬边点选
        if event.type == "MOUSEMOVE":
            hover = self._pick_candidate_edge(context, event)
            if hover != self._hover_edge:
                self._hover_edge = hover
                self._refresh_session(context)
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = self._pick_candidate_edge(context, event)
            if hit is None:
                return {"RUNNING_MODAL"}
            self._toggle_edge(context, int(hit))
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


__all__ = (
    "ARE_OT_segment_regions",
    "ARE_OT_clear_regions",
    "ARE_OT_update_label_hover",
    "ARE_OT_merge_regions",
    "ARE_OT_confirm_merge_regions",
    "ARE_OT_remove_regions",
    "ARE_OT_confirm_remove_regions",
    "ARE_OT_split_regions",
    "ARE_OT_confirm_split_regions",
    "REGION_ID_ATTR",
    "REGION_VERSION_ATTR",
    "REGION_COLORS_ATTR",
    "write_region_ids",
    "clear_region_ids",
    "read_region_ids",
    "sync_region_label_overlay",
    "register_label_hover_keymap",
    "unregister_label_hover_keymap",
)
