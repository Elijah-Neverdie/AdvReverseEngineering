# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""
AdvReverseEngineering — 摄影测量 / 激光扫描模型逆向工程预处理插件。

功能概览：
    - 网格分析（Mesh Analysis）
    - 网格清理（Mesh Cleanup）
    - 自动摆正（Auto Orientation）
    - 原点设置（Origin）
    - 落地（Ground）
    - 变换应用（Transform）

UI 位置：View3D > Sidebar (N) > 逆向工具
"""

from __future__ import annotations

bl_info = {
    "name": "AdvReverseEngineering",
    "author": "AdvReverseEngineering Team",
    "version": (0, 6, 5),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > 逆向工具",
    "description": (
        "Photogrammetry / 3D scan reverse engineering preprocessing: "
        "auto orient, origin, ground, mesh cleanup for UE5, CAD, 3D print"
    ),
    "warning": "",
    "doc_url": "",
    "tracker_url": "",
    "category": "Mesh",
}


def register() -> None:
    """插件注册入口，委托 registration 模块统一注册。"""
    from . import registration

    registration.register()


def unregister() -> None:
    """插件注销入口，委托 registration 模块统一注销。"""
    from . import registration

    registration.unregister()


if __name__ == "__main__":
    register()
