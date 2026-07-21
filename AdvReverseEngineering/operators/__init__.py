"""AdvReverseEngineering 操作符子包。"""

from .orient import ARE_OT_auto_orient
from .curve_edit import (
    ARE_OT_confirm_fit_bezier_curve,
    ARE_OT_confirm_split_fit_curve,
    ARE_OT_fit_bezier_curve,
    ARE_OT_split_fit_curve,
)
from .region_fit import (
    ARE_OT_build_fit_surface,
    ARE_OT_confirm_fit_region,
    ARE_OT_fit_region,
    ARE_OT_fit_step_back,
    ARE_OT_fit_step_next,
)
from .regions import (
    ARE_OT_clear_regions,
    ARE_OT_confirm_merge_regions,
    ARE_OT_confirm_remove_regions,
    ARE_OT_confirm_split_regions,
    ARE_OT_merge_regions,
    ARE_OT_remove_regions,
    ARE_OT_segment_regions,
    ARE_OT_split_regions,
    ARE_OT_update_label_hover,
)
from .simplify import ARE_OT_simplify_apply, ARE_OT_simplify_rebuild
from .update import (
    ARE_AddonPreferences,
    ARE_OT_check_github_update,
    ARE_OT_update_from_github,
)

classes = (
    ARE_AddonPreferences,
    ARE_OT_auto_orient,
    ARE_OT_simplify_rebuild,
    ARE_OT_simplify_apply,
    ARE_OT_segment_regions,
    ARE_OT_clear_regions,
    ARE_OT_update_label_hover,
    ARE_OT_merge_regions,
    ARE_OT_confirm_merge_regions,
    ARE_OT_remove_regions,
    ARE_OT_confirm_remove_regions,
    ARE_OT_split_regions,
    ARE_OT_confirm_split_regions,
    ARE_OT_fit_region,
    ARE_OT_fit_step_next,
    ARE_OT_fit_step_back,
    ARE_OT_build_fit_surface,
    ARE_OT_confirm_fit_region,
    ARE_OT_split_fit_curve,
    ARE_OT_confirm_split_fit_curve,
    ARE_OT_fit_bezier_curve,
    ARE_OT_confirm_fit_bezier_curve,
    ARE_OT_check_github_update,
    ARE_OT_update_from_github,
)

__all__ = ("classes",)
