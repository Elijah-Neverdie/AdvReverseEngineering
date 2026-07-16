# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""自动摆正操作符。"""

from __future__ import annotations

import bpy

from ..algorithms.bottom_faces import detect_bottom_face_indices
from ..algorithms.orientation import (
    ORIENTATION_STRATEGIES,
    compute_up_axis,
    get_strategy,
)
from ..registration import SCENE_PROP_NAME
from ..ui.overlay import set_bottom_face_highlight
from ..utils.math import apply_rotation_to_object
from ..utils.mesh import extract_mesh_data
from ..utils.progress import progress_scope


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

        strategy_index = scene_props.orientation_method_index % len(
            ORIENTATION_STRATEGIES
        )
        strategy = get_strategy(strategy_index)
        next_strategy = get_strategy(strategy_index + 1)

        settings = {
            "use_pca": scene_props.use_pca,
            "detect_largest_plane": scene_props.detect_largest_plane,
            "normal_clustering": scene_props.normal_clustering,
            "obb_refinement": scene_props.obb_refinement,
        }

        try:
            with progress_scope(context, "自动摆正", 5) as step:
                step("读取网格数据")
                mesh_data = extract_mesh_data(obj)

                step(f"计算: {strategy.label}")
                rotation = strategy.compute(mesh_data, settings)

                step("应用旋转")
                apply_rotation_to_object(
                    obj,
                    rotation,
                    mesh_data["centroid"],
                )

                step("检测底面")
                mesh_data = extract_mesh_data(obj)
                bottom_faces = detect_bottom_face_indices(
                    mesh_data,
                    obj.data.polygons,
                )
                set_bottom_face_highlight(context, obj, bottom_faces)

                step("更新状态")
                up_axis = compute_up_axis(rotation)
                scene_props.estimated_up_axis = tuple(up_axis.tolist())
                scene_props.last_orientation_method = strategy.label
                scene_props.next_orientation_method = next_strategy.label
                scene_props.orientation_method_index = (
                    (strategy_index + 1) % len(ORIENTATION_STRATEGIES)
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
                f"已使用「{strategy.label}」摆正，"
                f"底面 {len(bottom_faces)} 个面已紫色高亮；"
                f"再次点击将切换为「{next_strategy.label}」"
            ),
        )
        return {"FINISHED"}
