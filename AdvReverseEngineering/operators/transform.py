# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""变换应用操作符（占位，第六步完整实现）。"""

from __future__ import annotations

import bpy


class ARE_OT_apply_transform(bpy.types.Operator):
    """应用对象变换到网格数据。"""

    bl_idname = "are.apply_transform"
    bl_label = "应用变换"
    bl_description = "应用旋转 / 缩放 / 位移到网格"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context: bpy.types.Context):
        self.report({"INFO"}, "应用变换功能将在后续步骤实现")
        return {"FINISHED"}
