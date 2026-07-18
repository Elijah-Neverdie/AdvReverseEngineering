# SPDX-License-Identifier: GPL-3.0-or-later

"""插件版本解析与比较工具。"""

from __future__ import annotations

import re
from pathlib import Path


VERSION_PATTERN = re.compile(
    r"[\"']version[\"']\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)


def parse_bl_info_version(source: str) -> tuple[int, int, int]:
    """从源码文本中解析 bl_info.version。"""
    match = VERSION_PATTERN.search(source)
    if match is None:
        raise ValueError("代码中未找到有效的 bl_info.version")
    return tuple(int(value) for value in match.groups())


def format_version(version: tuple[int, ...]) -> str:
    """将版本元组格式化为点分版本号。"""
    return ".".join(str(value) for value in version)


def read_version_file(path: Path) -> tuple[int, int, int]:
    """从 __init__.py 文件读取 bl_info.version。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_bl_info_version(text)


def compare_versions(
    local: tuple[int, ...],
    remote: tuple[int, ...],
) -> str:
    """
    比较本地与远端版本。

    返回:
      - "AVAILABLE": 远端更新，可升级
      - "CURRENT": 版本相同
      - "AHEAD": 本地新于远端（强行更新会降级）
    """
    local_t = tuple(int(v) for v in local)
    remote_t = tuple(int(v) for v in remote)
    if remote_t > local_t:
        return "AVAILABLE"
    if remote_t < local_t:
        return "AHEAD"
    return "CURRENT"
