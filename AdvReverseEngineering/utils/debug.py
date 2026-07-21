# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""轻量文件调试日志（用于定位 Blender 卡死）。"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path


_LOG_NAME = "AdvReverseEngineering_debug.log"
_enabled = True
_last_path: Path | None = None


def debug_log_path() -> Path:
    """日志文件路径：%TEMP%/AdvReverseEngineering_debug.log"""
    global _last_path
    path = Path(os.environ.get("TEMP", os.environ.get("TMP", "."))) / _LOG_NAME
    _last_path = path
    return path


def are_debug(message: str, *, exc: BaseException | None = None) -> None:
    """追加一行时间戳日志；失败时静默（不能反卡死调试本身）。"""
    if not _enabled:
        return
    try:
        path = debug_log_path()
        stamp = time.strftime("%H:%M:%S")
        ms = int((time.time() % 1) * 1000)
        line = f"[{stamp}.{ms:03d}] {message}\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            if exc is not None:
                handle.write("".join(traceback.format_exception(exc)))
                handle.write("\n")
    except Exception:
        pass


def are_debug_clear() -> None:
    """清空日志（进入合并等关键操作前调用）。"""
    try:
        path = debug_log_path()
        path.write_text(
            f"# AdvReverseEngineering debug log\n# path={path}\n",
            encoding="utf-8",
        )
    except Exception:
        pass
