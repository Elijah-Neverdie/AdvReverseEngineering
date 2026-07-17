# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""领域自动识别与清除操作符。"""

from __future__ import annotations

import bpy
import numpy as np

from ..algorithms.regions import segment_regions_by_normal
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import (
    clear_region_highlight,
    set_region_highlight,
)
from ..utils.mesh import extract_face_topology, extract_mesh_data
from ..utils.progress import progress_scope


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
                # UI 中最小面积占比以百分比存储（0.1 表示 0.1%）。
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
                # 扁平 RGBA，供覆盖层稳定复现颜色。
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
                    scene_props.region_status_detail = (
                        "未启用忽略离散面"
                    )

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
            scene_props.region_version = (
                int(scene_props.region_version) + 1
            )

        clear_region_highlight(context, obj)
        self.report({"INFO"}, "已清除领域标记")
        return {"FINISHED"}


# 供测试与外部模块引用
__all__ = (
    "ARE_OT_segment_regions",
    "ARE_OT_clear_regions",
    "REGION_ID_ATTR",
    "REGION_VERSION_ATTR",
    "REGION_COLORS_ATTR",
    "write_region_ids",
    "clear_region_ids",
    "read_region_ids",
)
