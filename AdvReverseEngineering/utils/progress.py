# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""进度条与状态栏工具。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Generator

import bpy


@contextmanager
def progress_scope(
    context: bpy.types.Context,
    title: str,
    steps: int,
) -> Generator[Callable[[str], None], None, None]:
    """
    进度条上下文管理器。

    用法:
        with progress_scope(context, "摆正中", 4) as step:
            step("读取网格")
            ...
    """
    wm = context.window_manager
    wm.progress_begin(0, steps)

    def advance(message: str) -> None:
        index = advance.index
        advance.index += 1
        wm.progress_update(index)
        context.workspace.status_text_set(f"{title}: {message}")

    advance.index = 0

    try:
        yield advance
    finally:
        wm.progress_end()
        context.workspace.status_text_set(None)
