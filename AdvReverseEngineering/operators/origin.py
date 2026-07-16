# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""原点设置操作符（占位，第六步完整实现）。"""

from __future__ import annotations

import bpy


class ARE_OT_set_origin(bpy.types.Operator):
    """设置对象原点。"""

    bl_idname = "are.set_origin"
    bl_label = "设置原点"
    bl_description = "按选定模式设置对象原点"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context: bpy.types.Context):
        self.report({"INFO"}, "设置原点功能将在后续步骤实现")
        return {"FINISHED"}
