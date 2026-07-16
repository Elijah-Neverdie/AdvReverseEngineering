# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""从 GitHub 下载并更新插件。"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import bpy


ADDON_MODULE = "AdvReverseEngineering"


def _addon_root() -> Path:
    """当前插件根目录（含 __init__.py）。"""
    return Path(__file__).resolve().parent.parent


def _prefs(context: bpy.types.Context):
    """读取插件偏好设置。"""
    return context.preferences.addons[ADDON_MODULE].preferences


def _download_zip(url: str, timeout: float = 60.0) -> bytes:
    """下载 ZIP 字节流。"""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AdvReverseEngineering-Updater/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _find_addon_in_extract(extract_dir: Path) -> Path | None:
    """
    在解压目录中定位 AdvReverseEngineering 包根。

    兼容:
      - AdvReverseEngineering-main/__init__.py（仓库根即插件）
      - AdvReverseEngineering-main/AdvReverseEngineering/__init__.py
    """
    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    search_roots = children if children else [extract_dir]

    for root in search_roots:
        if (root / "__init__.py").exists():
            text = (root / "__init__.py").read_text(
                encoding="utf-8",
                errors="ignore",
            )
            if "bl_info" in text:
                return root

        nested = root / "AdvReverseEngineering"
        if (nested / "__init__.py").exists():
            return nested

    for path in extract_dir.rglob("__init__.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "bl_info" in text and "AdvReverseEngineering" in text:
            return path.parent

    return None


def _copy_tree(src: Path, dst: Path) -> None:
    """将 src 覆盖复制到 dst，跳过 .git / __pycache__。"""
    skip_names = {".git", "__pycache__", ".gitignore"}

    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in skip_names]
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.endswith(".pyc"):
                continue
            shutil.copy2(Path(root) / name, target_dir / name)


class ARE_OT_update_from_github(bpy.types.Operator):
    """从 GitHub 拉取最新代码并覆盖本地插件。"""

    bl_idname = "are.update_from_github"
    bl_label = "从 GitHub 更新"
    bl_description = "下载 GitHub 仓库 ZIP 并覆盖当前插件"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        try:
            prefs = _prefs(context)
        except (KeyError, AttributeError):
            self.report({"ERROR"}, "无法读取插件偏好设置")
            return {"CANCELLED"}

        owner = (prefs.github_owner or "").strip()
        repo = (prefs.github_repo or "").strip()
        branch = (prefs.github_branch or "main").strip() or "main"

        if not owner or not repo:
            self.report({"ERROR"}, "请先填写 GitHub 用户名与仓库名")
            return {"CANCELLED"}

        zip_url = (
            f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        )
        addon_root = _addon_root()

        self.report({"INFO"}, f"正在下载 {owner}/{repo}@{branch} …")

        try:
            data = _download_zip(zip_url)
        except urllib.error.HTTPError as exc:
            self.report(
                {"ERROR"},
                (
                    f"下载失败 HTTP {exc.code}。"
                    "请确认仓库为公开，且用户名/仓库/分支正确"
                ),
            )
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"下载失败: {exc}")
            return {"CANCELLED"}

        temp_dir = Path(tempfile.mkdtemp(prefix="are_update_"))
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(temp_dir)

            source = _find_addon_in_extract(temp_dir)
            if source is None:
                self.report({"ERROR"}, "ZIP 中未找到 AdvReverseEngineering 插件")
                return {"CANCELLED"}

            _copy_tree(source, addon_root)

        except Exception as exc:
            self.report({"ERROR"}, f"更新失败: {exc}")
            return {"CANCELLED"}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.report(
            {"INFO"},
            "已从 GitHub 更新，请重启 Blender 或重新启用插件",
        )
        return {"FINISHED"}


class ARE_AddonPreferences(bpy.types.AddonPreferences):
    """插件偏好：GitHub 同步配置。"""

    bl_idname = ADDON_MODULE

    github_owner: bpy.props.StringProperty(
        name="GitHub 用户名",
        description="例如 your-name",
        default="",
    )
    github_repo: bpy.props.StringProperty(
        name="仓库名",
        description="例如 AdvReverseEngineering",
        default="AdvReverseEngineering",
    )
    github_branch: bpy.props.StringProperty(
        name="分支",
        description="通常为 main 或 master",
        default="main",
    )

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.label(
            text="多电脑同步：代码推送到 GitHub 后，在此填写仓库信息并更新",
        )
        layout.prop(self, "github_owner")
        layout.prop(self, "github_repo")
        layout.prop(self, "github_branch")
        layout.operator("are.update_from_github", icon="FILE_REFRESH")


classes = (
    ARE_AddonPreferences,
    ARE_OT_update_from_github,
)
