"""AdvReverseEngineering 操作符子包。"""

from .orient import ARE_OT_auto_orient
from .regions import (
    ARE_OT_clear_regions,
    ARE_OT_confirm_merge_regions,
    ARE_OT_merge_regions,
    ARE_OT_segment_regions,
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
    ARE_OT_merge_regions,
    ARE_OT_confirm_merge_regions,
    ARE_OT_check_github_update,
    ARE_OT_update_from_github,
)

__all__ = ("classes",)
