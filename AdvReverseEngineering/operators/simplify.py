# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""视图简化：隐藏原始备份 + Decimate 工作副本 + 防抖重建。"""

from __future__ import annotations

import time

import bpy

from ..registration import SCENE_PROP_NAME


BACKUP_COLLECTION_NAME = "ARE_原始备份"
BACKUP_SUFFIX = ".are_pristine"
ROLE_ATTR = "are_simplify_role"
SOURCE_NAME_ATTR = "are_simplify_source_name"
DECIMATE_MODIFIER_NAME = "ARE_Decimate"

# 防抖计时器状态（模块级，插件注销时清理）
_REBUILD_TIMER_REGISTERED = False
_PENDING_REBUILD_AT = 0.0
_DEBOUNCE_SECONDS = 0.5


def cancel_simplify_timers() -> None:
    """注销简化防抖计时器。"""
    global _REBUILD_TIMER_REGISTERED, _PENDING_REBUILD_AT
    if _REBUILD_TIMER_REGISTERED:
        try:
            bpy.app.timers.unregister(_rebuild_timer_callback)
        except Exception:
            pass
    _REBUILD_TIMER_REGISTERED = False
    _PENDING_REBUILD_AT = 0.0


def schedule_simplify_rebuild(context: bpy.types.Context) -> None:
    """
    百分比变化后约 0.5 秒再建工作副本。

    首次（未进入会话）也会在防抖后自动进入简化会话。
    100% 且尚未进入会话时不创建备份，避免误触。
    """
    global _REBUILD_TIMER_REGISTERED, _PENDING_REBUILD_AT

    scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
    if scene_props is None:
        return

    percent = float(scene_props.viewport_simplify_percent)
    if not scene_props.simplify_active and percent >= 99.9:
        scene_props.simplify_rebuild_pending = False
        if not scene_props.simplify_status:
            scene_props.simplify_status = ""
        return

    scene_props.simplify_rebuild_pending = True
    scene_props.simplify_status = "等待重建预览…"
    _PENDING_REBUILD_AT = time.monotonic() + _DEBOUNCE_SECONDS

    if not _REBUILD_TIMER_REGISTERED:
        bpy.app.timers.register(
            _rebuild_timer_callback,
            first_interval=_DEBOUNCE_SECONDS,
        )
        _REBUILD_TIMER_REGISTERED = True


def _rebuild_timer_callback() -> float | None:
    """防抖到期后执行一次重建；未到期则继续等待。"""
    global _REBUILD_TIMER_REGISTERED

    remaining = _PENDING_REBUILD_AT - time.monotonic()
    if remaining > 0.05:
        return remaining

    _REBUILD_TIMER_REGISTERED = False
    try:
        context = bpy.context
        if context is None or context.scene is None:
            return None
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is None:
            return None
        percent = float(scene_props.viewport_simplify_percent)
        if (
            scene_props.simplify_active
            and abs(percent - float(scene_props.simplify_applied_percent))
            < 1e-6
        ):
            scene_props.simplify_rebuild_pending = False
            return None
        rebuild_simplify_preview(context, percent)
    except Exception as exc:
        print(f"AdvReverseEngineering: 简化重建失败: {exc}")
    return None


def _triangle_count(mesh: bpy.types.Mesh) -> int:
    """估算三角面数（多边形按三角扇展开）。"""
    if mesh is None or len(mesh.polygons) == 0:
        return 0
    totals = [0] * len(mesh.polygons)
    mesh.polygons.foreach_get("loop_total", totals)
    return int(sum(max(value - 2, 0) for value in totals))


def _ensure_backup_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    """获取或创建隐藏的原始备份集合。"""
    collection = bpy.data.collections.get(BACKUP_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(BACKUP_COLLECTION_NAME)
        scene.collection.children.link(collection)
    collection.hide_viewport = True
    collection.hide_render = True
    return collection


def _unlink_from_all_collections(obj: bpy.types.Object) -> list[str]:
    """从所有集合断开链接，返回原集合名称列表。"""
    names: list[str] = []
    for collection in list(obj.users_collection):
        names.append(collection.name)
        collection.objects.unlink(obj)
    return names


def _link_to_collections(
    obj: bpy.types.Object,
    collection_names: list[str],
    fallback: bpy.types.Collection,
) -> None:
    """按名称把对象链回集合；名称失效则落到 fallback。"""
    linked = False
    for name in collection_names:
        collection = bpy.data.collections.get(name)
        if collection is None:
            continue
        if obj.name not in collection.objects:
            collection.objects.link(obj)
        linked = True
    if not linked and obj.name not in fallback.objects:
        fallback.objects.link(obj)


def _clear_downstream_results(
    context: bpy.types.Context,
    obj: bpy.types.Object | None,
) -> None:
    """简化重建后清理底面/领域结果，避免陈旧面索引。"""
    from ..ui.overlay import (
        clear_bottom_face_highlight,
        clear_region_highlight,
    )
    from .regions import (
        REGION_COLORS_ATTR,
        REGION_VERSION_ATTR,
        clear_region_ids,
    )

    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    if obj is not None and obj.type == "MESH":
        clear_region_ids(obj.data)
        if REGION_VERSION_ATTR in obj:
            del obj[REGION_VERSION_ATTR]
        if REGION_COLORS_ATTR in obj:
            del obj[REGION_COLORS_ATTR]
        if "are_bottom_faces" in obj:
            del obj["are_bottom_faces"]

    if scene_props.region_object == obj or obj is None:
        scene_props.region_object = None
        scene_props.region_count = 0
        scene_props.region_ignored_face_count = 0
        scene_props.region_ignored_region_count = 0
        scene_props.region_status = ""
        scene_props.region_status_detail = "简化后请重新识别领域"
        scene_props.region_version = int(scene_props.region_version) + 1

    if scene_props.highlight_object == obj or obj is None:
        clear_bottom_face_highlight(context)
    clear_region_highlight(context, obj)


def _delete_object_and_orphan_mesh(obj: bpy.types.Object) -> None:
    """删除对象；若 mesh 无其他用户则一并删除。"""
    mesh = obj.data if obj.type == "MESH" else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def _apply_decimate(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    ratio: float,
) -> None:
    """添加并应用 Decimate 修改器。"""
    # 清理旧修饰器，避免残留。
    for modifier in list(obj.modifiers):
        if modifier.name.startswith("ARE_Decimate"):
            obj.modifiers.remove(modifier)

    clamped = max(0.01, min(1.0, float(ratio)))
    if clamped >= 0.999:
        return

    modifier = obj.modifiers.new(DECIMATE_MODIFIER_NAME, "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = clamped

    view_layer = context.view_layer
    with context.temp_override(
        object=obj,
        active_object=obj,
        selected_objects=[obj],
        selected_editable_objects=[obj],
        view_layer=view_layer,
    ):
        bpy.ops.object.modifier_apply(modifier=DECIMATE_MODIFIER_NAME)


def _create_working_from_backup(
    context: bpy.types.Context,
    backup: bpy.types.Object,
    display_name: str,
    percent: float,
) -> bpy.types.Object:
    """从备份复制独立 mesh 的工作副本并应用简化。"""
    scene = context.scene
    mesh_copy = backup.data.copy()
    mesh_copy.name = f"{display_name}_simplified"
    working = bpy.data.objects.new(display_name, mesh_copy)
    working.matrix_world = backup.matrix_world.copy()
    working.scale = backup.scale.copy()
    working.rotation_euler = backup.rotation_euler.copy()
    working.location = backup.location.copy()
    working[ROLE_ATTR] = "working"
    working[SOURCE_NAME_ATTR] = display_name

    # 复用摆正基准矩阵（若备份上有）。
    if "are_orientation_base_matrix" in backup:
        working["are_orientation_base_matrix"] = list(
            backup["are_orientation_base_matrix"]
        )

    # 链接到备份原先所在集合；若无记录则挂到场景根。
    original_collections = list(backup.get("are_simplify_home_collections", []))
    if isinstance(original_collections, str):
        original_collections = [original_collections]
    _link_to_collections(
        working,
        [str(name) for name in original_collections],
        scene.collection,
    )

    working.hide_set(False)
    working.hide_viewport = False
    working.hide_render = False

    for other in context.view_layer.objects:
        other.select_set(False)
    working.select_set(True)
    context.view_layer.objects.active = working

    _apply_decimate(context, working, percent / 100.0)
    return working


def enter_or_rebuild_simplify(
    context: bpy.types.Context,
    percent: float,
) -> bpy.types.Object:
    """
    进入简化会话或从备份重建工作副本。

    返回当前工作副本。
    """
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    active = context.active_object

    if scene_props.simplify_active and scene_props.simplify_backup is not None:
        backup = scene_props.simplify_backup
        display_name = scene_props.simplify_source_name or backup.name(
            SOURCE_NAME_ATTR,
            backup.name.replace(BACKUP_SUFFIX, ""),
        )
        old_working = scene_props.simplify_working
        if old_working is not None:
            _clear_downstream_results(context, old_working)
            _delete_object_and_orphan_mesh(old_working)

        working = _create_working_from_backup(
            context,
            backup,
            display_name,
            percent,
        )
    else:
        if active is None or active.type != "MESH":
            raise ValueError("请选中网格对象后再简化")
        if active.mode != "OBJECT":
            raise ValueError("请先退出编辑模式再简化")
        if active.get(ROLE_ATTR) == "backup":
            raise ValueError("不能直接简化备份对象，请选中工作副本")

        source = active
        display_name = source.name
        home_collections = [col.name for col in source.users_collection]
        original_faces = _triangle_count(source.data)

        backup_collection = _ensure_backup_collection(context.scene)
        _unlink_from_all_collections(source)
        if source.name not in backup_collection.objects:
            backup_collection.objects.link(source)

        source.name = f"{display_name}{BACKUP_SUFFIX}"
        source[ROLE_ATTR] = "backup"
        source[SOURCE_NAME_ATTR] = display_name
        source["are_simplify_home_collections"] = home_collections
        source.hide_set(True)
        source.hide_viewport = True
        source.hide_render = True
        source.select_set(False)

        scene_props.simplify_backup = source
        scene_props.simplify_source_name = display_name
        scene_props.simplify_original_faces = original_faces
        scene_props.simplify_active = True

        working = _create_working_from_backup(
            context,
            source,
            display_name,
            percent,
        )

    scene_props.simplify_working = working
    scene_props.simplify_applied_percent = float(percent)
    scene_props.simplify_current_faces = _triangle_count(working.data)
    scene_props.simplify_rebuild_pending = False
    scene_props.simplify_status = (
        f"预览 {percent:.1f}% · "
        f"{scene_props.simplify_original_faces} → "
        f"{scene_props.simplify_current_faces} 面"
    )

    # 摆正对象名跟随工作副本。
    scene_props.orientation_object_name = working.name_full
    scene_props.region_status_detail = "简化后请重新识别领域"
    return working


def rebuild_simplify_preview(
    context: bpy.types.Context,
    percent: float,
) -> bpy.types.Object:
    """供计时器与算子调用的重建入口。"""
    return enter_or_rebuild_simplify(context, percent)


def apply_simplify_session(context: bpy.types.Context) -> None:
    """确认简化：删除原始备份，保留工作副本。"""
    scene_props = getattr(context.scene, SCENE_PROP_NAME)
    if not scene_props.simplify_active:
        raise ValueError("当前没有可应用的简化会话")

    working = scene_props.simplify_working
    backup = scene_props.simplify_backup
    if working is None:
        raise ValueError("找不到简化工作副本")

    if backup is not None:
        _delete_object_and_orphan_mesh(backup)

    if ROLE_ATTR in working:
        del working[ROLE_ATTR]
    if SOURCE_NAME_ATTR in working:
        del working[SOURCE_NAME_ATTR]

    # 若备份集合已空，可删除集合。
    collection = bpy.data.collections.get(BACKUP_COLLECTION_NAME)
    if collection is not None and len(collection.objects) == 0:
        bpy.data.collections.remove(collection)

    scene_props.simplify_active = False
    scene_props.simplify_backup = None
    scene_props.simplify_working = working
    scene_props.simplify_rebuild_pending = False
    scene_props.simplify_status = (
        f"已应用 {scene_props.simplify_applied_percent:.1f}% · "
        f"{scene_props.simplify_current_faces} 面"
    )

    working.select_set(True)
    context.view_layer.objects.active = working


class ARE_OT_simplify_apply(bpy.types.Operator):
    """应用当前视图简化，删除原始备份。"""

    bl_idname = "are.simplify_apply"
    bl_label = "应用"
    bl_description = "确认简化效果并删除隐藏的原始备份网格"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        return (
            scene_props is not None
            and scene_props.simplify_active
            and scene_props.simplify_working is not None
            and not scene_props.simplify_rebuild_pending
        )

    def execute(self, context: bpy.types.Context):
        try:
            apply_simplify_session(context)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"应用简化失败: {exc}")
            return {"CANCELLED"}

        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        self.report({"INFO"}, scene_props.simplify_status)
        return {"FINISHED"}


class ARE_OT_simplify_rebuild(bpy.types.Operator):
    """立即按当前百分比重建简化预览（供测试与强制刷新）。"""

    bl_idname = "are.simplify_rebuild"
    bl_label = "重建简化预览"
    bl_description = "立即按当前视图简化百分比从原始备份重建工作副本"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME, None)
        if scene_props is not None and scene_props.simplify_active:
            return scene_props.simplify_backup is not None
        return obj is not None and obj.type == "MESH" and obj.mode == "OBJECT"

    def execute(self, context: bpy.types.Context):
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        try:
            rebuild_simplify_preview(
                context,
                float(scene_props.viewport_simplify_percent),
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"简化失败: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"}, scene_props.simplify_status)
        return {"FINISHED"}


__all__ = (
    "ARE_OT_simplify_apply",
    "ARE_OT_simplify_rebuild",
    "schedule_simplify_rebuild",
    "cancel_simplify_timers",
    "rebuild_simplify_preview",
    "apply_simplify_session",
)
