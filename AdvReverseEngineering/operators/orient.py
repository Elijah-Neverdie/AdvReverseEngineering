# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""自动摆正操作符。"""

from __future__ import annotations

import bpy
from mathutils import Matrix

from ..algorithms.bottom_faces import detect_bottom_face_indices
from ..algorithms.orientation import (
    ORIENTATION_STRATEGIES,
    compute_up_axis,
    get_strategy,
)
from ..algorithms.selected_plane import orient_from_selected_plane
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import set_bottom_face_highlight
from ..utils.math import apply_rotation_to_object, set_object_origin_world
from ..utils.mesh import extract_mesh_data, extract_selected_world_points
from ..utils.progress import progress_scope


BASE_MATRIX_KEY = "are_orientation_base_matrix"
SEQUENCE_ICONS = ("1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣")


def _matrix_to_flat(matrix: Matrix) -> list[float]:
    """将 4x4 Matrix 展平，便于保存到对象自定义属性。"""
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _flat_to_matrix(values) -> Matrix:
    """从对象自定义属性恢复 4x4 Matrix。"""
    flat = list(values)
    if len(flat) != 16:
        raise ValueError("已保存的基准姿态数据无效")
    return Matrix((
        flat[0:4],
        flat[4:8],
        flat[8:12],
        flat[12:16],
    ))


def _restore_or_store_baseline(
    obj: bpy.types.Object,
    scene_props,
) -> None:
    """切换方案时始终从同一基准姿态计算，避免旋转逐次叠加。"""
    is_new_object = scene_props.orientation_object_name != obj.name_full
    if is_new_object or BASE_MATRIX_KEY not in obj:
        obj[BASE_MATRIX_KEY] = _matrix_to_flat(obj.matrix_world)
        scene_props.orientation_object_name = obj.name_full
        scene_props.orientation_method_index = 0
        return
    obj.matrix_world = _flat_to_matrix(obj[BASE_MATRIX_KEY])


class ARE_OT_auto_orient(bpy.types.Operator):
    """
    一键自动摆正，重复点击循环切换摆正策略。

    策略顺序:
        PCA → RANSAC → 法线聚类 → OBB → 组合流程
    """

    bl_idname = "are.auto_orient"
    bl_label = "自动摆正"
    bl_description = "一键摆正物体；再次点击将切换下一种摆正方法"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        scene_props = getattr(context.scene, SCENE_PROP_NAME)
        edit_mode = obj.mode == "EDIT"
        bottom_faces: list[int] = []

        try:
            with progress_scope(context, "自动摆正", 6) as step:
                step("读取编辑模式选区")
                selected_points = extract_selected_world_points(obj)

                if not edit_mode:
                    _restore_or_store_baseline(obj, scene_props)

                step("读取网格数据")
                mesh_data = extract_mesh_data(obj)

                if edit_mode:
                    if len(selected_points) < 3:
                        raise ValueError(
                            "编辑模式下请至少选择 3 个不共线的点，"
                            "也可以选择面或边"
                        )
                    selected_result = orient_from_selected_plane(
                        mesh_data,
                        selected_points,
                    )
                    rotation = selected_result.rotation
                    pivot = selected_result.pivot
                    method_label = selected_result.method_label
                    next_label = "重新按当前选区判断"
                    sequence_icon = "🎯"
                    step(f"计算: {method_label}")
                else:
                    strategy_index = (
                        scene_props.orientation_method_index
                        % len(ORIENTATION_STRATEGIES)
                    )
                    strategy = get_strategy(strategy_index)
                    next_strategy = get_strategy(strategy_index + 1)
                    settings = {
                        "use_pca": scene_props.use_pca,
                        "detect_largest_plane": (
                            scene_props.detect_largest_plane
                        ),
                        "normal_clustering": scene_props.normal_clustering,
                        "obb_refinement": scene_props.obb_refinement,
                    }
                    step(f"计算: {strategy.label}")
                    rotation = strategy.compute(mesh_data, settings)
                    pivot = mesh_data["centroid"]
                    method_label = strategy.label
                    next_label = next_strategy.label
                    sequence_icon = SEQUENCE_ICONS[strategy_index]

                step("应用旋转")
                apply_rotation_to_object(
                    obj,
                    rotation,
                    pivot,
                )
                if edit_mode:
                    set_object_origin_world(obj, pivot)

                step("检测底面")
                mesh_data = extract_mesh_data(obj)
                bottom_faces = detect_bottom_face_indices(
                    mesh_data,
                )
                set_bottom_face_highlight(context, obj, bottom_faces)

                step("更新状态")
                up_axis = compute_up_axis(rotation)
                scene_props.estimated_up_axis = tuple(up_axis.tolist())
                scene_props.last_orientation_method = method_label
                scene_props.next_orientation_method = next_label
                scene_props.orientation_status = (
                    f"{sequence_icon} 已使用「{method_label}」摆正"
                )
                scene_props.orientation_status_detail = (
                    f"底面 {len(bottom_faces)} 个面已紫色高亮"
                )
                scene_props.orientation_status_next = (
                    f"再次点击将切换为「{next_label}」"
                )
                if not edit_mode:
                    scene_props.orientation_method_index = (
                        (strategy_index + 1)
                        % len(ORIENTATION_STRATEGIES)
                    )

        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"摆正失败: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            (
                f"{scene_props.orientation_status}，"
                f"{scene_props.orientation_status_detail}；"
                f"{scene_props.orientation_status_next}"
            ),
        )
        return {"FINISHED"}
