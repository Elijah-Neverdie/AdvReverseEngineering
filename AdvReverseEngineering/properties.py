# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""AdvReverseEngineering 插件属性组定义。"""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


class ARE_SceneProperties(bpy.types.PropertyGroup):
    """场景级插件属性，供 UI 面板与 Operator 读写。"""

    # ------------------------------------------------------------------
    # 网格分析结果（由 analyze operator 写入）
    # ------------------------------------------------------------------
    has_analysis: BoolProperty(
        name="已分析",
        description="是否已完成网格分析",
        default=False,
    )
    vertex_count: IntProperty(name="顶点数", default=0)
    edge_count: IntProperty(name="边数", default=0)
    face_count: IntProperty(name="面数", default=0)
    bbox_min: FloatVectorProperty(
        name="包围盒最小点",
        size=3,
        default=(0.0, 0.0, 0.0),
    )
    bbox_max: FloatVectorProperty(
        name="包围盒最大点",
        size=3,
        default=(0.0, 0.0, 0.0),
    )
    dimension: FloatVectorProperty(
        name="尺寸",
        size=3,
        default=(0.0, 0.0, 0.0),
    )
    center: FloatVectorProperty(
        name="中心",
        size=3,
        default=(0.0, 0.0, 0.0),
    )
    estimated_up_axis: FloatVectorProperty(
        name="估计上方向",
        size=3,
        default=(0.0, 0.0, 1.0),
    )

    # ------------------------------------------------------------------
    # 网格清理
    # ------------------------------------------------------------------
    remove_outlier_vertices: BoolProperty(
        name="删除离群顶点",
        description="移除距离过远的孤立顶点",
        default=True,
    )
    merge_duplicate_vertices: BoolProperty(
        name="合并重复顶点",
        description="合并位置重合的顶点",
        default=True,
    )
    recalculate_normals: BoolProperty(
        name="重新计算法线",
        description="按面方向重算法线",
        default=True,
    )
    fill_small_holes: BoolProperty(
        name="填充小孔洞",
        description="自动填补面积较小的孔洞",
        default=False,
    )
    voxel_size: FloatProperty(
        name="体素尺寸",
        description="体素采样网格大小",
        default=0.001,
        min=0.0001,
        max=10.0,
        precision=4,
        step=0.01,
        unit="LENGTH",
    )

    # ------------------------------------------------------------------
    # 自动摆正
    # ------------------------------------------------------------------
    use_pca: BoolProperty(
        name="使用 PCA",
        description="主成分分析粗定位",
        default=True,
    )
    detect_largest_plane: BoolProperty(
        name="检测最大平面 (RANSAC)",
        description="RANSAC 检测最大平面作为底面",
        default=True,
    )
    normal_clustering: BoolProperty(
        name="法线聚类",
        description="按法线方向聚类修正姿态",
        default=True,
    )
    obb_refinement: BoolProperty(
        name="OBB 精修",
        description="有向包围盒 ±15° 精修",
        default=True,
    )
    orientation_method_index: IntProperty(
        name="摆正方法索引",
        description="下次摆正将使用的策略索引（内部循环）",
        default=0,
        min=0,
    )
    last_orientation_method: StringProperty(
        name="上次摆正方法",
        description="最近一次使用的摆正策略",
        default="",
    )
    next_orientation_method: StringProperty(
        name="下次摆正方法",
        description="下次点击将使用的摆正策略",
        default="PCA 主方向",
    )
    orientation_status: StringProperty(
        name="摆正状态",
        description="最近一次使用的方案序号与名称",
        default="",
    )
    orientation_status_detail: StringProperty(
        name="底面高亮状态",
        description="最近一次识别并高亮的底面数量",
        default="",
    )
    orientation_status_next: StringProperty(
        name="下一摆正方案",
        description="再次点击时将使用的方案",
        default="",
    )
    orientation_object_name: StringProperty(
        name="摆正对象",
        description="当前循环方案所对应的对象名称",
        default="",
    )
    highlight_object: PointerProperty(
        name="高亮对象",
        description="底面高亮显示的目标对象",
        type=bpy.types.Object,
    )
    show_bottom_highlight: BoolProperty(
        name="显示底面",
        description="在 3D 视口中以紫色显示当前选中物体的检测底面",
        default=True,
    )

    # ------------------------------------------------------------------
    # 原点设置
    # ------------------------------------------------------------------
    origin_mode: EnumProperty(
        name="原点模式",
        description="原点设置方式",
        items=(
            ("GEOMETRY", "几何中心", "使用网格几何中心"),
            ("BBOX", "包围盒中心", "使用世界空间包围盒中心"),
            ("BOTTOM", "底面中心", "使用底面中心点"),
        ),
        default="BOTTOM",
    )

    # ------------------------------------------------------------------
    # 落地
    # ------------------------------------------------------------------
    ground_offset: FloatProperty(
        name="落地偏移",
        description="最低点对齐 Z=0 后的额外偏移",
        default=0.0,
        unit="LENGTH",
    )

    # ------------------------------------------------------------------
    # 变换应用
    # ------------------------------------------------------------------
    apply_rotation: BoolProperty(
        name="应用旋转",
        description="将旋转变换烘焙到网格数据",
        default=True,
    )
    apply_scale: BoolProperty(
        name="应用缩放",
        description="将缩放变换烘焙到网格数据",
        default=False,
    )
    apply_location: BoolProperty(
        name="应用位移",
        description="将位移变换烘焙到网格数据",
        default=False,
    )


classes = (ARE_SceneProperties,)
