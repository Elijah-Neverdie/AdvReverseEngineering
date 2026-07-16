# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""落地操作符（占位，第六步完整实现）。"""

from __future__ import annotations

import bpy


class ARE_OT_move_to_ground(bpy.types.Operator):
    """将模型最低点移至地面。"""

    bl_idname = "are.move_to_ground"
    bl_label = "移至地面"
    bl_description = "最低点对齐 Z=0，支持偏移"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context: bpy.types.Context):
        self.report({"INFO"}, "移至地面功能将在后续步骤实现")
        return {"FINISHED"}
