# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""
AdvReverseEngineering 插件注册中心。

负责按固定顺序注册 / 注销 Blender 类型：
    1. PropertyGroup（场景属性）
    2. Operator（操作符）
    3. Panel（侧边栏面板）

设计原则：
    - UI、Operator 不直接耦合算法模块
    - 各子包通过 classes 元组暴露可注册类型
    - 便于未来扩展 ICP、Primitive Fitting 等新模块
"""

from __future__ import annotations

import bpy

# ---------------------------------------------------------------------------
# 侧边栏 Tab 名称（View3D > Sidebar > 逆向工具）
# ---------------------------------------------------------------------------
TAB_CATEGORY = "逆向工具"

# Scene 上挂载 PropertyGroup 的属性名（第五步填充具体 PropertyGroup）
SCENE_PROP_NAME = "adv_reverse_engineering"


def _get_property_classes() -> list[type]:
    """
    收集 PropertyGroup 类列表。

    第五步将在 properties 模块中实现具体属性组。
    """
    try:
        from . import properties

        return list(properties.classes)
    except ImportError:
        return []


def _get_operator_classes() -> list[type]:
    """收集 Operator 类列表（第六步填充）。"""
    from .operators import classes

    return list(classes)


def _get_panel_classes() -> list[type]:
    """收集 Panel 类列表（第四步填充）。"""
    from .ui import classes

    return list(classes)


def _register_classes(class_list: list[type], label: str) -> None:
    """批量注册 Blender 类型，失败时抛出明确错误。"""
    for cls in class_list:
        try:
            bpy.utils.register_class(cls)
        except Exception as exc:
            raise RuntimeError(
                f"AdvReverseEngineering: 注册 {label} '{cls.__name__}' 失败"
            ) from exc


def _unregister_classes(class_list: list[type], label: str) -> None:
    """按相反顺序批量注销 Blender 类型。"""
    for cls in reversed(class_list):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            # 插件重载时可能已被注销，忽略
            pass
        except Exception as exc:
            raise RuntimeError(
                f"AdvReverseEngineering: 注销 {label} '{cls.__name__}' 失败"
            ) from exc


def _register_scene_properties(property_classes: list[type]) -> None:
    """将主 PropertyGroup 挂载到 bpy.types.Scene。"""
    if not property_classes:
        return

    # 约定：列表中第一个 PropertyGroup 为场景根属性组
    root_prop = property_classes[0]
    setattr(
        bpy.types.Scene,
        SCENE_PROP_NAME,
        bpy.props.PointerProperty(type=root_prop),
    )


def _unregister_scene_properties() -> None:
    """从 Scene 移除插件属性指针。"""
    if hasattr(bpy.types.Scene, SCENE_PROP_NAME):
        delattr(bpy.types.Scene, SCENE_PROP_NAME)


def register() -> None:
    """注册插件全部 Blender 类型。"""
    from .ui.overlay import register_draw_handler

    property_classes = _get_property_classes()
    operator_classes = _get_operator_classes()
    panel_classes = _get_panel_classes()

    # PropertyGroup 必须最先注册（Operator / Panel 可能依赖属性）
    _register_classes(property_classes, "PropertyGroup")
    _register_scene_properties(property_classes)

    _register_classes(operator_classes, "Operator")
    _register_classes(panel_classes, "Panel")

    try:
        register_draw_handler()
    except Exception as exc:
        print(f"AdvReverseEngineering: 视口高亮注册失败（插件其余功能仍可用）: {exc}")


def unregister() -> None:
    """按相反顺序注销插件全部 Blender 类型。"""
    from .ui.overlay import unregister_draw_handler

    property_classes = _get_property_classes()
    operator_classes = _get_operator_classes()
    panel_classes = _get_panel_classes()

    unregister_draw_handler()
    _unregister_classes(panel_classes, "Panel")
    _unregister_classes(operator_classes, "Operator")

    _unregister_scene_properties()
    _unregister_classes(property_classes, "PropertyGroup")
