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


def _on_viewport_simplify_percent_update(self, context) -> None:
    """百分比变化时调度防抖重建，避免拖动滑块连续重算。"""
    from .operators.simplify import schedule_simplify_rebuild

    schedule_simplify_rebuild(context)


def _on_show_region_highlight_update(self, context) -> None:
    """显示领域开关变化时同步编号标签覆盖层。"""

    def _deferred() -> float | None:
        try:
            from .operators.regions import sync_region_label_overlay

            sync_region_label_overlay(bpy.context)
        except Exception:
            pass
        return None

    try:
        bpy.app.timers.register(_deferred, first_interval=0.0)
    except Exception:
        _deferred()


def _on_fit_segments_update(self, context) -> None:
    """拟合模态中侧栏段数变化时重建预览。"""
    if not getattr(self, "fit_mode_active", False):
        return
    if getattr(self, "fit_phase", "") != "PREVIEW":
        return
    try:
        from .operators.region_fit import _get_active_fit_op

        op = _get_active_fit_op()
        if op is not None and not getattr(op, "_updating_segments", False):
            op._rebuild_preview(context)
            return
    except Exception:
        pass
    # 模态实例不可用时，置位由 TIMER 消费
    self.fit_stage_rebuild_requested = True


def _on_fit_stage_params_update(self, context) -> None:
    """分步参数变化时按当前阶段实时重建曲线预览。"""
    if not getattr(self, "fit_mode_active", False):
        return
    phase = str(getattr(self, "fit_phase", "") or "")
    if phase not in {"ISLANDS", "STITCH", "BRIDGE"}:
        return
    try:
        from .operators.region_fit import _get_active_fit_op

        op = _get_active_fit_op()
        if op is not None and not getattr(op, "_updating_stage_params", False):
            op._rebuild_stage_preview(context)
            return
    except Exception:
        pass
    self.fit_stage_rebuild_requested = True


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
        description="兼容旧属性；底面已并入“显示领域”统一控制",
        default=True,
    )

    # ------------------------------------------------------------------
    # 领域分割（Geomagic Design X 风格 Region）
    # ------------------------------------------------------------------
    region_wireframe_threshold: FloatProperty(
        name="线框阈值",
        description=(
            "对齐视图叠加层「几何数据 → 线框」滑条："
            "越低只把更硬的橘色边当作领域边界（建议 0.1）"
        ),
        default=0.1,
        min=0.0,
        max=1.0,
        soft_min=0.05,
        soft_max=0.5,
        precision=3,
        step=1,
    )
    region_normal_threshold: FloatProperty(
        name="法线阈值（兼容）",
        description="旧版角度阈值，仅内部兼容保留",
        default=15.0,
        min=0.1,
        max=90.0,
        soft_min=1.0,
        soft_max=45.0,
        precision=1,
    )
    region_smooth_iterations: IntProperty(
        name="法线平滑",
        description=(
            "边保护法线平滑迭代次数；细碎扫描网格建议 1~3，"
            "可抑制噪声让硬边更完整，0 表示不平滑"
        ),
        default=2,
        min=0,
        max=10,
        soft_max=5,
    )
    region_ignore_discrete: BoolProperty(
        name="合并碎屑领域",
        description="将面积过小的碎屑并入相邻大领域，保持连续而不标成忽略",
        default=True,
    )
    region_min_area_ratio: FloatProperty(
        name="碎屑面积占比",
        description="领域面积低于网格总面积该比例时并入邻居（百分比）",
        default=0.1,
        min=0.001,
        max=50.0,
        soft_min=0.01,
        soft_max=5.0,
        precision=3,
        subtype="PERCENTAGE",
    )
    show_region_highlight: BoolProperty(
        name="显示领域",
        description="在 3D 视口中显示自动识别的领域颜色，以及作为前置固定领域的紫色底面",
        default=True,
        update=_on_show_region_highlight_update,
    )
    label_hover_id: IntProperty(
        name="标签悬停领域",
        description="空闲状态下鼠标悬停的领域编号",
        default=-1,
        min=-1,
    )
    region_object: PointerProperty(
        name="领域对象",
        description="当前领域分割结果所属对象",
        type=bpy.types.Object,
    )
    region_count: IntProperty(
        name="领域数量",
        description="最近一次识别到的有效领域数量",
        default=0,
        min=0,
    )
    region_ignored_face_count: IntProperty(
        name="忽略面数",
        description="因面积过小被忽略的面数量",
        default=0,
        min=0,
    )
    region_ignored_region_count: IntProperty(
        name="忽略领域数",
        description="因面积过小被忽略的领域数量",
        default=0,
        min=0,
    )
    region_version: IntProperty(
        name="领域结果版本",
        description="每次识别递增，用于覆盖层缓存失效",
        default=0,
        min=0,
    )
    region_status: StringProperty(
        name="领域状态",
        description="最近一次领域识别的摘要",
        default="",
    )
    region_status_detail: StringProperty(
        name="领域状态详情",
        description="忽略离散面等附加说明",
        default="",
    )

    # ------------------------------------------------------------------
    # 简化（视图 Decimate 预览 + 原始备份）
    # ------------------------------------------------------------------
    show_simplify_section: BoolProperty(
        name="简化",
        description="展开/收起简化卷展栏",
        default=True,
    )
    show_region_section: BoolProperty(
        name="领域",
        description="展开/收起领域卷展栏",
        default=True,
    )
    viewport_simplify_percent: FloatProperty(
        name="视图简化",
        description="保留三角面比例（百分比）；停止拖动约 0.5 秒后自动重建预览",
        default=100.0,
        min=1.0,
        max=100.0,
        soft_min=5.0,
        soft_max=100.0,
        precision=1,
        subtype="PERCENTAGE",
        update=_on_viewport_simplify_percent_update,
    )
    simplify_active: BoolProperty(
        name="简化会话中",
        description="是否已创建隐藏原始备份与工作副本",
        default=False,
    )
    simplify_backup: PointerProperty(
        name="简化备份",
        description="隐藏的原始网格备份对象",
        type=bpy.types.Object,
    )
    simplify_working: PointerProperty(
        name="简化工作副本",
        description="当前用于显示与后续制作的简化对象",
        type=bpy.types.Object,
    )
    simplify_source_name: StringProperty(
        name="简化源名称",
        description="进入简化会话前的对象显示名称",
        default="",
    )
    simplify_applied_percent: FloatProperty(
        name="已应用简化百分比",
        description="当前工作副本实际使用的保留比例",
        default=100.0,
        min=1.0,
        max=100.0,
    )
    simplify_original_faces: IntProperty(
        name="原始面数",
        description="备份网格的三角面数",
        default=0,
        min=0,
    )
    simplify_current_faces: IntProperty(
        name="当前面数",
        description="工作副本的三角面数",
        default=0,
        min=0,
    )
    simplify_status: StringProperty(
        name="简化状态",
        description="简化会话摘要",
        default="",
    )
    simplify_rebuild_pending: BoolProperty(
        name="等待重建",
        description="百分比已变，等待防抖计时器重建",
        default=False,
    )

    # ------------------------------------------------------------------
    # 领域合并模态
    # ------------------------------------------------------------------
    merge_mode_active: BoolProperty(
        name="合并模式",
        description="是否正在进行领域合并",
        default=False,
    )
    merge_anchor_id: IntProperty(
        name="合并锚点",
        description="当前锚点领域编号，-1 表示未选择",
        default=-1,
        min=-1,
    )
    merge_confirm_requested: BoolProperty(
        name="请求确认合并",
        description="面板确认按钮通知模态算子提交",
        default=False,
    )
    merge_hover_id: IntProperty(
        name="悬停领域",
        description="鼠标悬停的领域编号",
        default=-1,
        min=-1,
    )
    merge_status: StringProperty(
        name="合并状态",
        description="合并模式提示文案",
        default="",
    )
    show_merge_help: BoolProperty(
        name="合并说明",
        description="展开/收起合并模式操作说明",
        default=False,
    )

    # ------------------------------------------------------------------
    # 领域移除模态
    # ------------------------------------------------------------------
    remove_mode_active: BoolProperty(
        name="移除模式",
        description="是否正在进行领域移除",
        default=False,
    )
    remove_confirm_requested: BoolProperty(
        name="请求确认移除",
        description="面板确认按钮通知模态算子提交",
        default=False,
    )
    remove_hover_id: IntProperty(
        name="移除悬停领域",
        description="移除模式下鼠标悬停的领域编号",
        default=-1,
        min=-1,
    )
    remove_status: StringProperty(
        name="移除状态",
        description="移除模式提示文案",
        default="",
    )
    show_remove_help: BoolProperty(
        name="移除说明",
        description="展开/收起移除模式操作说明",
        default=False,
    )

    # ------------------------------------------------------------------
    # 领域拆分模态
    # ------------------------------------------------------------------
    split_mode_active: BoolProperty(
        name="拆分模式",
        description="是否正在进行领域智能拆分",
        default=False,
    )
    split_confirm_requested: BoolProperty(
        name="请求确认拆分",
        description="面板确认按钮通知模态算子提交拆分",
        default=False,
    )
    split_status: StringProperty(
        name="拆分状态",
        description="拆分模式提示文案",
        default="",
    )
    split_target_id: IntProperty(
        name="拆分目标领域",
        description="当前选中的待拆分领域编号，-1 表示未选择",
        default=-1,
        min=-1,
    )
    split_hover_id: IntProperty(
        name="拆分悬停领域",
        description="拆分模式下鼠标悬停的领域编号",
        default=-1,
        min=-1,
    )
    split_phase: StringProperty(
        name="拆分阶段",
        description="SELECT / EDGE / PREVIEW / IDLE",
        default="IDLE",
    )
    split_brush_radius: FloatProperty(
        name="拆分笔刷半径",
        description="旧版圆形笔刷半径（兼容保留）",
        default=40.0,
        min=8.0,
        max=200.0,
    )
    split_hard_threshold: FloatProperty(
        name="线框阈值",
        description=(
            "与「识别领域」相同的 Blender 线框阈值；Ctrl+滚轮调节。"
            "越大越敏感，会显示曲面上更缓的折棱作为候选硬边"
        ),
        default=0.25,
        min=0.02,
        max=1.0,
        soft_min=0.05,
        soft_max=0.8,
        precision=2,
        step=5,
    )
    show_split_help: BoolProperty(
        name="拆分说明",
        description="展开/收起拆分模式操作说明",
        default=False,
    )

    # ------------------------------------------------------------------
    # 领域拟合模态
    # ------------------------------------------------------------------
    region_fit_triangle_ratio: FloatProperty(
        name="三边判定阈值",
        description=(
            "最短边长度低于第二短边该比例时视为三边曲面（百分比）；"
            "扁长条带的两条短边长度接近，不会被误判为三边"
        ),
        default=15.0,
        min=1.0,
        max=50.0,
        soft_min=5.0,
        soft_max=30.0,
        precision=1,
        subtype="PERCENTAGE",
    )
    fit_mode_active: BoolProperty(
        name="拟合模式",
        description="是否正在进行领域曲面拟合",
        default=False,
    )
    fit_confirm_requested: BoolProperty(
        name="请求确认拟合",
        description="面板确认按钮通知模态算子提交拟合",
        default=False,
    )
    fit_advance_requested: BoolProperty(
        name="请求拟合下一步",
        description="面板「下一步」通知模态算子前进",
        default=False,
    )
    fit_retreat_requested: BoolProperty(
        name="请求拟合上一步",
        description="面板「上一步」通知模态算子回退",
        default=False,
    )
    fit_stage_rebuild_requested: BoolProperty(
        name="请求重建拟合阶段预览",
        description="侧栏参数变化后由模态 TIMER 重建预览",
        default=False,
    )
    fit_preview_requested: BoolProperty(
        name="请求拟合成面",
        description="面板「拟合成面」通知模态进入曲面预览",
        default=False,
    )
    fit_target_id: IntProperty(
        name="拟合目标领域",
        description="当前选中的待拟合领域编号，-1 表示未选择",
        default=-1,
        min=-1,
    )
    fit_hover_id: IntProperty(
        name="拟合悬停领域",
        description="拟合模式下鼠标悬停的领域编号",
        default=-1,
        min=-1,
    )
    fit_phase: StringProperty(
        name="拟合阶段",
        description="SELECT / ISLANDS / STITCH / BRIDGE / PREVIEW / IDLE",
        default="IDLE",
    )
    fit_island_min_perimeter: FloatProperty(
        name="碎岛周长阈值",
        description="相对最长孤岛周长的比例，更短的碎环不参与第 1 步外围曲线",
        default=12.0,
        min=1.0,
        max=50.0,
        soft_min=5.0,
        soft_max=30.0,
        precision=1,
        subtype="PERCENTAGE",
        update=_on_fit_stage_params_update,
    )
    fit_stitch_gap: FloatProperty(
        name="缝合间隙",
        description=(
            "相对领域包围盒对角线的比例；邻近折角连线平均距离小于该阈值时，"
            "把折角之间的相向边链当作缝合曲线删除并合并外轮廓"
        ),
        default=8.0,
        min=0.5,
        max=40.0,
        soft_min=2.0,
        soft_max=20.0,
        precision=1,
        subtype="PERCENTAGE",
        update=_on_fit_stage_params_update,
    )
    fit_bridge_gap: FloatProperty(
        name="桥接间隙",
        description="相对领域包围盒对角线的比例；第 3 步把更远孤岛纳入桥接包络",
        default=25.0,
        min=1.0,
        max=80.0,
        soft_min=8.0,
        soft_max=50.0,
        precision=1,
        subtype="PERCENTAGE",
        update=_on_fit_stage_params_update,
    )
    fit_bridge_enabled: BoolProperty(
        name="启用远岛桥接",
        description="第 3 步是否桥接距离更远的同领域孤岛",
        default=True,
        update=_on_fit_stage_params_update,
    )
    fit_segments_u: IntProperty(
        name="U 向段数",
        description="四边面对边 U 向分段，或三边面底边分段",
        default=4,
        min=1,
        max=64,
        update=_on_fit_segments_update,
    )
    fit_segments_v: IntProperty(
        name="V 向段数",
        description="四边面对边 V 向分段，或三边面两条长边分段",
        default=4,
        min=1,
        max=64,
        update=_on_fit_segments_update,
    )
    fit_topology: StringProperty(
        name="拟合拓扑",
        description="TRI 或 QUAD",
        default="",
    )
    fit_status: StringProperty(
        name="拟合状态",
        description="拟合模式提示文案",
        default="",
    )
    fit_status_detail: StringProperty(
        name="拟合状态详情",
        description="拟合附加说明",
        default="",
    )
    show_fit_help: BoolProperty(
        name="拟合说明",
        description="展开/收起拟合模式操作说明",
        default=False,
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
