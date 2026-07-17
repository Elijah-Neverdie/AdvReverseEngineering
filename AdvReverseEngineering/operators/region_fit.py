# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""领域三边/四边曲面拟合操作符。"""

from __future__ import annotations

import bpy
import numpy as np

from ..algorithms.region_fit import (
    DEFAULT_SEG_U,
    DEFAULT_SEG_V,
    MAX_SEGMENTS,
    MIN_SEGMENTS,
    RegionFitError,
    fit_region_surface,
)
from ..algorithms.regions import compute_region_label_anchors
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import (
    LABEL_RADIUS_PX,
    get_merge_label_session,
    register_label_draw_handler,
    set_merge_label_session,
    unregister_label_draw_handler,
    update_merge_label_projections,
)
from ..utils.mesh import extract_face_topology, extract_mesh_data
from ..utils.viewport import hit_test_labels
from .regions import REGION_ID_ATTR, read_region_ids


FIT_COLLECTION_NAME = "拟合面"
FIT_PREVIEW_NAME = "ARE_FitPreview"
MODAL_TIMER_STEP = 0.1
_FIT_OP_KEY = "are_active_fit_op"


def _tag_redraw(context: bpy.types.Context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "UI"}:
                area.tag_redraw()


def _add_modal_timer(operator, context: bpy.types.Context) -> None:
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


def _set_active_fit_op(operator) -> None:
    bpy.app.driver_namespace[_FIT_OP_KEY] = operator


def _get_active_fit_op():
    return bpy.app.driver_namespace.get(_FIT_OP_KEY)


def _clear_active_fit_op(operator) -> None:
    if bpy.app.driver_namespace.get(_FIT_OP_KEY) is operator:
        bpy.app.driver_namespace[_FIT_OP_KEY] = None


def _schedule_force_exit_check() -> None:
    def _callback():
        scene = getattr(bpy.context, "scene", None)
        scene_props = getattr(scene, SCENE_PROP_NAME, None) if scene else None
        if scene_props is None:
            return None
        stuck = (
            scene_props.fit_mode_active
            and scene_props.fit_confirm_requested
        )
        if stuck:
            scene_props.fit_confirm_requested = False
            scene_props.fit_mode_active = False
            scene_props.fit_target_id = -1
            scene_props.fit_hover_id = -1
            scene_props.fit_phase = "IDLE"
            scene_props.fit_status = "拟合模态已失联，已强制退出"
            unregister_label_draw_handler()
            set_merge_label_session(None)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return None

    bpy.app.timers.register(_callback, first_interval=0.6)


def _bbox_lift(mesh_data) -> float:
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


def _ensure_fit_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    collection = bpy.data.collections.get(FIT_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(FIT_COLLECTION_NAME)
        scene.collection.children.link(collection)
    return collection


def _delete_object(obj: bpy.types.Object | None) -> None:
    if obj is None:
        return
    try:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    except ReferenceError:
        pass


def _world_to_object_local(
    matrix_world: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    inv = np.linalg.inv(matrix_world)
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack((_as_float(points), ones))
    return (inv @ homogeneous.T).T[:, :3]


def _as_float(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _create_or_update_mesh_object(
    name: str,
    vertices_world: np.ndarray,
    faces: list[tuple[int, ...]],
    matrix_world,
    collection: bpy.types.Collection | None,
    existing: bpy.types.Object | None = None,
) -> bpy.types.Object:
    matrix = np.asarray(matrix_world, dtype=np.float64)
    local = _world_to_object_local(matrix, vertices_world)
    if existing is not None and existing.name in bpy.data.objects:
        mesh = existing.data
        mesh.clear_geometry()
        mesh.from_pydata(
            [tuple(v) for v in local.tolist()],
            [],
            [tuple(f) for f in faces],
        )
        mesh.update()
        existing.matrix_world = matrix_world.copy()
        return existing

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [tuple(v) for v in local.tolist()],
        [],
        [tuple(f) for f in faces],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = matrix_world.copy()
    if collection is not None:
        collection.objects.link(obj)
    else:
        bpy.context.scene.collection.objects.link(obj)
    return obj


class ARE_OT_confirm_fit_region(bpy.types.Operator):
    """面板确认按钮：通知拟合模态提交。"""

    bl_idname = "are.confirm_fit_region"
    bl_label = "确认"
    bl_description = "确认当前拟合面并退出拟合模式"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return (
            scene_props is not None
            and scene_props.fit_mode_active
            and scene_props.fit_phase == "PREVIEW"
        )

    def execute(self, context: bpy.types.Context):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        op = _get_active_fit_op()
        if op is not None:
            try:
                op.confirm_from_panel(context)
            except Exception as exc:
                self.report({"ERROR"}, f"确认拟合失败: {exc}")
                scene_props.fit_confirm_requested = True
                _schedule_force_exit_check()
                return {"FINISHED"}
            if scene_props.fit_mode_active:
                scene_props.fit_mode_active = False
                scene_props.fit_confirm_requested = False
                scene_props.fit_phase = "IDLE"
                unregister_label_draw_handler()
                set_merge_label_session(None)
            return {"FINISHED"}
        scene_props.fit_confirm_requested = True
        _schedule_force_exit_check()
        return {"FINISHED"}


class ARE_OT_fit_region(bpy.types.Operator):
    """
    模态拟合领域。

    点击编号选择领域 → 滚轮/Page 调节控制点 → 确认写入「拟合面」集合。
    """

    bl_idname = "are.fit_region"
    bl_label = "拟合领域"
    bl_description = (
        "点击领域编号拟合三边/四边规则曲面；"
        "滚轮与 PageUp/PageDown 调节控制点；确认放入「拟合面」集合"
    )
    bl_options = {"REGISTER", "UNDO"}

    def _cleanup_ui(self, context: bpy.types.Context) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        _remove_modal_timer(self, context)
        _clear_active_fit_op(self)
        unregister_label_draw_handler()
        set_merge_label_session(None)
        scene_props.fit_mode_active = False
        scene_props.fit_confirm_requested = False
        scene_props.fit_target_id = -1
        scene_props.fit_hover_id = -1
        scene_props.fit_phase = "IDLE"
        _tag_redraw(context)

    def _discard_preview(self) -> None:
        preview = getattr(self, "_preview_object", None)
        _delete_object(preview)
        self._preview_object = None

    def confirm_from_panel(self, context: bpy.types.Context) -> None:
        if getattr(self, "_closed", False):
            return
        try:
            if not self._committed:
                self._commit_fit(context)
        finally:
            self._finish_mode(context, cancelled=False)
            self._closed = True

    def _target_label(self) -> str:
        return "+".join(str(value) for value in self._selected_ids)

    def _status_text(self, scene_props) -> str:
        if scene_props.fit_phase != "PREVIEW":
            return "点击领域编号开始拟合，可连点多个编号合并；Esc 取消"
        topo = "三边" if self._topology == "TRI" else "四边"
        label = self._target_label()
        if self._topology == "TRI":
            return (
                f"领域 {label} · {topo} · "
                f"长边段数 {scene_props.fit_segments_v} · "
                f"底边段数 {scene_props.fit_segments_u} · "
                f"滚轮调长边 / Page 调底边"
            )
        return (
            f"领域 {label} · {topo} · "
            f"U={scene_props.fit_segments_u} V={scene_props.fit_segments_v} · "
            f"滚轮调 U / Page 调 V"
        )

    def _sync_selected_to_session(self, context: bpy.types.Context) -> None:
        """把多选编号写入标签会话，供覆盖层高亮。"""
        session = get_merge_label_session()
        if session is not None:
            session["selected_ids"] = set(self._selected_ids)
        _tag_redraw(context)

    def _rebuild_preview(self, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if not self._selected_ids:
            return False
        try:
            result = fit_region_surface(
                region_ids=self._region_ids,
                target_id=list(self._selected_ids),
                vertices=self._mesh_data["vertices"],
                loop_start=self._topology_data["loop_start"],
                loop_total=self._topology_data["loop_total"],
                loop_vertex_indices=self._topology_data["loop_vertex_indices"],
                face_normals=self._mesh_data["normals"],
                face_areas=self._mesh_data["areas"],
                face_centers=self._mesh_data["face_centers"],
                segments_u=int(scene_props.fit_segments_u),
                segments_v=int(scene_props.fit_segments_v),
                triangle_ratio=float(scene_props.region_fit_triangle_ratio)
                / 100.0,
            )
        except RegionFitError as exc:
            # 拟合失败：撤下旧预览，回到选择阶段，保留当前多选供继续调整
            self._discard_preview()
            scene_props.fit_phase = "SELECT"
            scene_props.fit_topology = ""
            scene_props.fit_status = str(exc)
            self.report({"ERROR"}, str(exc))
            _tag_redraw(context)
            return False

        self._topology = result.topology
        self._last_result = result
        scene_props.fit_topology = result.topology
        scene_props.fit_phase = "PREVIEW"
        preview = _create_or_update_mesh_object(
            FIT_PREVIEW_NAME,
            result.vertices,
            result.faces,
            self._object.matrix_world,
            collection=None,
            existing=self._preview_object,
        )
        # 预览先挂到场景根，确认时再移入「拟合面」
        if preview.name not in context.scene.collection.objects:
            try:
                context.scene.collection.objects.link(preview)
            except RuntimeError:
                pass
        preview.hide_select = True
        preview.display_type = "WIRE"
        # 预览网格始终显示在扫描面之前，避免被遮挡
        preview.show_in_front = True
        self._preview_object = preview
        scene_props.fit_status = self._status_text(scene_props)
        if result.warnings:
            scene_props.fit_status_detail = "；".join(result.warnings)
        else:
            scene_props.fit_status_detail = ""
        _tag_redraw(context)
        return True

    def _commit_fit(self, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if self._preview_object is None or self._last_result is None:
            self.report({"WARNING"}, "还没有可确认的拟合预览")
            return False

        collection = _ensure_fit_collection(context.scene)
        label = self._target_label()
        final_name = f"Fit_R{label}_{self._topology}"
        # 断开预览与场景根，改链到拟合面集合
        preview = self._preview_object
        for col in list(preview.users_collection):
            col.objects.unlink(preview)
        if preview.name not in collection.objects:
            collection.objects.link(preview)
        preview.name = final_name
        if preview.data is not None:
            preview.data.name = final_name
        preview.hide_select = False
        preview.display_type = "TEXTURED"
        # 拟合面与扫描面重合，保持前置显示避免被埋没
        preview.show_in_front = True
        preview["are_fit_source"] = self._object.name
        preview["are_fit_region_ids"] = label
        preview["are_fit_topology"] = self._topology

        self._preview_object = None
        self._committed = True
        scene_props.fit_status = (
            f"已拟合领域 {label}（{('三边' if self._topology == 'TRI' else '四边')}）"
            f" → {final_name}"
        )
        scene_props.fit_status_detail = ""
        return True

    def _finish_mode(self, context: bpy.types.Context, cancelled: bool) -> set:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if cancelled and not self._committed:
            self._discard_preview()
            scene_props.fit_status = "已取消拟合"
        self._cleanup_ui(context)
        return {"CANCELLED"} if cancelled and not self._committed else {"FINISHED"}

    def _adjust_segments(
        self,
        context: bpy.types.Context,
        delta_u: int = 0,
        delta_v: int = 0,
    ) -> None:
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        if scene_props.fit_phase != "PREVIEW":
            return
        self._updating_segments = True
        try:
            scene_props.fit_segments_u = int(
                np.clip(
                    int(scene_props.fit_segments_u) + delta_u,
                    MIN_SEGMENTS,
                    MAX_SEGMENTS,
                )
            )
            scene_props.fit_segments_v = int(
                np.clip(
                    int(scene_props.fit_segments_v) + delta_v,
                    MIN_SEGMENTS,
                    MAX_SEGMENTS,
                )
            )
        finally:
            self._updating_segments = False
        self._rebuild_preview(context)

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None:
            return False
        if (
            scene_props.merge_mode_active
            or scene_props.split_mode_active
            or scene_props.fit_mode_active
        ):
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

        self._object = obj
        self._mesh_data = mesh_data
        self._topology_data = topology
        self._region_ids = region_ids.copy()
        self._selected_ids: list[int] = []
        self._preview_object = None
        self._last_result = None
        self._topology = "QUAD"
        self._committed = False
        self._closed = False
        self._timer = None

        session = _build_label_session(region_ids, mesh_data)
        session["object_name"] = obj.name
        session["preview_version"] = 0
        set_merge_label_session(session)
        register_label_draw_handler()
        _set_active_fit_op(self)

        scene_props.fit_mode_active = True
        scene_props.fit_confirm_requested = False
        scene_props.fit_target_id = -1
        scene_props.fit_hover_id = -1
        scene_props.fit_phase = "SELECT"
        scene_props.fit_segments_u = DEFAULT_SEG_U
        scene_props.fit_segments_v = DEFAULT_SEG_V
        scene_props.fit_topology = ""
        scene_props.fit_status = self._status_text(scene_props)
        scene_props.fit_status_detail = ""
        scene_props.region_object = obj

        _add_modal_timer(self, context)
        context.window_manager.modal_handler_add(self)
        _tag_redraw(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context):
        if getattr(self, "_committed", False):
            self._cleanup_ui(context)
            return {"FINISHED"}
        self._discard_preview()
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

        if scene_props.fit_confirm_requested and not self._committed:
            scene_props.fit_confirm_requested = False
            if self._commit_fit(context):
                self.report({"INFO"}, scene_props.fit_status)
                return self._finish_mode(context, cancelled=False)
            return {"RUNNING_MODAL"}

        if event.type == "TIMER":
            return {"RUNNING_MODAL"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            if self._committed:
                return self._finish_mode(context, cancelled=False)
            self.report({"INFO"}, "已取消拟合")
            return self._finish_mode(context, cancelled=True)

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if self._commit_fit(context):
                self.report({"INFO"}, scene_props.fit_status)
                return self._finish_mode(context, cancelled=False)
            return {"RUNNING_MODAL"}

        # 滚轮：第一组（四边 U / 三边长边→存于 V）
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS":
            if scene_props.fit_phase == "PREVIEW":
                delta = 1 if event.type == "WHEELUPMOUSE" else -1
                if self._topology == "TRI":
                    self._adjust_segments(context, delta_v=delta)
                else:
                    self._adjust_segments(context, delta_u=delta)
                return {"RUNNING_MODAL"}

        # PageUp/PageDown：第二组（四边 V / 三边底边→存于 U）
        if event.type in {"PAGE_UP", "PAGE_DOWN"} and event.value == "PRESS":
            if scene_props.fit_phase == "PREVIEW":
                delta = 1 if event.type == "PAGE_UP" else -1
                if self._topology == "TRI":
                    self._adjust_segments(context, delta_u=delta)
                else:
                    self._adjust_segments(context, delta_v=delta)
                return {"RUNNING_MODAL"}

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
            if new_hover != int(scene_props.fit_hover_id):
                scene_props.fit_hover_id = new_hover
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
            first_pick = not self._selected_ids
            # 点击已选编号则取消该领域，否则并入拟合域
            if rid in self._selected_ids:
                self._selected_ids.remove(rid)
            else:
                self._selected_ids.append(rid)
            scene_props.fit_target_id = (
                self._selected_ids[-1] if self._selected_ids else -1
            )
            self._sync_selected_to_session(context)

            if not self._selected_ids:
                self._discard_preview()
                scene_props.fit_phase = "SELECT"
                scene_props.fit_topology = ""
                scene_props.fit_status = self._status_text(scene_props)
                _tag_redraw(context)
                return {"RUNNING_MODAL"}

            if first_pick:
                scene_props.fit_segments_u = DEFAULT_SEG_U
                scene_props.fit_segments_v = DEFAULT_SEG_V
            if self._rebuild_preview(context):
                self.report(
                    {"INFO"},
                    f"拟合域：领域 {self._target_label()}，"
                    "继续点击编号可增删，滚轮/Page 调控制点",
                )
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


__all__ = (
    "ARE_OT_fit_region",
    "ARE_OT_confirm_fit_region",
    "FIT_COLLECTION_NAME",
)
