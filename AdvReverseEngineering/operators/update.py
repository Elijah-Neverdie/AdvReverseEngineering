# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""GitHub 版本自检与插件更新。"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import bpy

from .. import bl_info
from ..utils.versioning import format_version, parse_bl_info_version


ADDON_MODULE = "AdvReverseEngineering"
DEFAULT_GITHUB_OWNER = "Elijah-Neverdie"
DEFAULT_GITHUB_REPO = "AdvReverseEngineering"
DEFAULT_GITHUB_BRANCH = "main"
CURRENT_VERSION = tuple(bl_info["version"])

_CHECK_LOCK = threading.Lock()
_CHECK_RESULT: dict | None = None
_CHECK_THREAD: threading.Thread | None = None


def _addon_root() -> Path:
    """当前插件根目录（含 __init__.py）。"""
    return Path(__file__).resolve().parent.parent


def _prefs(context: bpy.types.Context):
    """读取插件偏好设置。"""
    return context.preferences.addons[ADDON_MODULE].preferences


def _global_prefs():
    """从当前 Blender 上下文读取插件偏好。"""
    addon = bpy.context.preferences.addons.get(ADDON_MODULE)
    return addon.preferences if addon is not None else None


def _normalize_repository_preferences(prefs) -> None:
    """迁移早期错误用户名，并补齐官方仓库默认值。"""
    owner = (prefs.github_owner or "").strip()
    if not owner or owner == "Elijah_Neverdie":
        prefs.github_owner = DEFAULT_GITHUB_OWNER
    if not (prefs.github_repo or "").strip():
        prefs.github_repo = DEFAULT_GITHUB_REPO
    if not (prefs.github_branch or "").strip():
        prefs.github_branch = DEFAULT_GITHUB_BRANCH


def _repository_values(prefs) -> tuple[str, str, str]:
    """返回清理后的 GitHub owner/repo/branch。"""
    _normalize_repository_preferences(prefs)
    return (
        prefs.github_owner.strip(),
        prefs.github_repo.strip(),
        prefs.github_branch.strip(),
    )


def _download_zip(url: str, timeout: float = 60.0) -> bytes:
    """下载 ZIP 字节流。"""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AdvReverseEngineering-Updater/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _remote_init_url(owner: str, repo: str, branch: str) -> str:
    """构建仓库内插件 __init__.py 的 raw URL。"""
    return (
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/"
        "AdvReverseEngineering/__init__.py"
    )


def _fetch_remote_version(
    owner: str,
    repo: str,
    branch: str,
    timeout: float = 12.0,
) -> tuple[int, int, int]:
    """下载远端 __init__.py 并解析版本。"""
    request = urllib.request.Request(
        _remote_init_url(owner, repo, branch),
        headers={"User-Agent": "AdvReverseEngineering-VersionCheck/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        source = response.read().decode("utf-8", errors="replace")
    return parse_bl_info_version(source)


def _check_worker(owner: str, repo: str, branch: str) -> None:
    """后台线程执行网络请求，不访问 Blender RNA。"""
    global _CHECK_RESULT
    try:
        version = _fetch_remote_version(owner, repo, branch)
        result = {"version": version, "error": ""}
    except Exception as exc:
        result = {"version": None, "error": str(exc)}

    with _CHECK_LOCK:
        _CHECK_RESULT = result


def _tag_preferences_redraw() -> None:
    """版本状态变化后刷新界面。"""
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return
    for window in window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def _poll_check_result() -> float | None:
    """主线程轮询后台结果并安全写入 AddonPreferences。"""
    global _CHECK_RESULT, _CHECK_THREAD

    with _CHECK_LOCK:
        result = _CHECK_RESULT
        if result is not None:
            _CHECK_RESULT = None

    if result is None:
        if _CHECK_THREAD is not None and _CHECK_THREAD.is_alive():
            return 0.25
        return None

    prefs = _global_prefs()
    if prefs is None:
        return None

    error = result["error"]
    if error:
        prefs.update_check_state = "ERROR"
        prefs.update_available = False
        prefs.update_check_message = f"检查失败: {error}"
    else:
        remote_version = tuple(result["version"])
        prefs.latest_version = format_version(remote_version)
        prefs.update_available = remote_version > CURRENT_VERSION
        if prefs.update_available:
            prefs.update_check_state = "AVAILABLE"
            prefs.update_check_message = (
                f"发现新版本 v{prefs.latest_version}"
            )
        else:
            prefs.update_check_state = "CURRENT"
            prefs.update_check_message = "当前为最新版"

    _CHECK_THREAD = None
    _tag_preferences_redraw()
    return None


def start_update_check() -> bool:
    """启动一次非阻塞版本检查；正在检查时不重复创建线程。"""
    global _CHECK_RESULT, _CHECK_THREAD

    prefs = _global_prefs()
    if prefs is None:
        return False
    if _CHECK_THREAD is not None and _CHECK_THREAD.is_alive():
        return False

    owner, repo, branch = _repository_values(prefs)
    prefs.update_check_state = "CHECKING"
    prefs.update_available = False
    prefs.update_check_message = "正在检查 GitHub 版本…"

    with _CHECK_LOCK:
        _CHECK_RESULT = None
    _CHECK_THREAD = threading.Thread(
        target=_check_worker,
        args=(owner, repo, branch),
        name="ARE-GitHub-VersionCheck",
        daemon=True,
    )
    _CHECK_THREAD.start()

    if not bpy.app.timers.is_registered(_poll_check_result):
        bpy.app.timers.register(
            _poll_check_result,
            first_interval=0.25,
        )
    return True


def _startup_update_check() -> None:
    """插件注册完成后延迟启动版本自检。"""
    start_update_check()
    return None


def schedule_startup_update_check() -> None:
    """安排启动自检，避免阻塞插件注册阶段。"""
    if not bpy.app.timers.is_registered(_startup_update_check):
        bpy.app.timers.register(
            _startup_update_check,
            first_interval=1.0,
        )


def cancel_update_check_timers() -> None:
    """插件注销时移除尚未执行的计时器。"""
    for callback in (_startup_update_check, _poll_check_result):
        if bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)


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

        if not prefs.update_available:
            self.report({"INFO"}, "当前没有可用更新")
            return {"CANCELLED"}

        owner, repo, branch = _repository_values(prefs)

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

        prefs.update_available = False
        prefs.update_check_state = "UPDATED"
        prefs.update_check_message = (
            f"已更新到 v{prefs.latest_version}，请重启 Blender"
        )
        self.report(
            {"INFO"},
            "已从 GitHub 更新，请重启 Blender 或重新启用插件",
        )
        return {"FINISHED"}


class ARE_OT_check_github_update(bpy.types.Operator):
    """手动重新检查 GitHub 版本。"""

    bl_idname = "are.check_github_update"
    bl_label = "重新检查更新"
    bl_description = "立即检查 GitHub 上是否有新版本"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        if start_update_check():
            self.report({"INFO"}, "正在后台检查 GitHub 版本")
        else:
            self.report({"INFO"}, "版本检查已在进行中")
        return {"FINISHED"}


class ARE_AddonPreferences(bpy.types.AddonPreferences):
    """插件偏好：GitHub 同步配置。"""

    bl_idname = ADDON_MODULE

    github_owner: bpy.props.StringProperty(
        name="GitHub 用户名",
        description="例如 your-name",
        default=DEFAULT_GITHUB_OWNER,
    )
    github_repo: bpy.props.StringProperty(
        name="仓库名",
        description="例如 AdvReverseEngineering",
        default=DEFAULT_GITHUB_REPO,
    )
    github_branch: bpy.props.StringProperty(
        name="分支",
        description="通常为 main 或 master",
        default=DEFAULT_GITHUB_BRANCH,
    )
    show_github_sync: bpy.props.BoolProperty(
        name="GitHub 同步",
        description="展开或收起 GitHub 同步设置",
        default=False,
    )
    update_check_state: bpy.props.StringProperty(
        name="更新检查状态",
        default="NOT_CHECKED",
    )
    latest_version: bpy.props.StringProperty(
        name="最新版本",
        default="",
    )
    update_available: bpy.props.BoolProperty(
        name="有可用更新",
        default=False,
    )
    update_check_message: bpy.props.StringProperty(
        name="更新提示",
        default="尚未检查更新",
    )

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.label(
            text="多电脑同步：代码推送到 GitHub 后，在此填写仓库信息并更新",
        )
        layout.prop(self, "github_owner")
        layout.prop(self, "github_repo")
        layout.prop(self, "github_branch")
        layout.label(text=self.update_check_message)
        layout.operator("are.check_github_update", icon="FILE_REFRESH")


classes = (
    ARE_AddonPreferences,
    ARE_OT_check_github_update,
    ARE_OT_update_from_github,
)
