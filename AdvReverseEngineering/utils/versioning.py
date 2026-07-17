# SPDX-License-Identifier: GPL-3.0-or-later

"""插件版本解析与比较工具。"""

from __future__ import annotations

import re


VERSION_PATTERN = re.compile(
    r"[\"']version[\"']\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)


def parse_bl_info_version(source: str) -> tuple[int, int, int]:
    """从远端 __init__.py 中解析 bl_info.version。"""
    match = VERSION_PATTERN.search(source)
    if match is None:
        raise ValueError("远端代码中未找到有效的 bl_info.version")
    return tuple(int(value) for value in match.groups())


def format_version(version: tuple[int, ...]) -> str:
    """将版本元组格式化为点分版本号。"""
    return ".".join(str(value) for value in version)
