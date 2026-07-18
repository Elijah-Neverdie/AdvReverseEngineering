# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 AdvReverseEngineering Contributors

"""GitHub 版本自检与插件更新。"""

from __future__ import annotations

import io
import json
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
from ..utils.versioning import (
    compare_versions,
    format_version,
    parse_bl_info_version,
    read_version_file,
)


ADDON_MODULE = "AdvReverseEngineering"
DEFAULT_GITHUB_OWNER = "Elijah-Neverdie"
DEFAULT_GITHUB_REPO = "AdvReverseEngineering"
DEFAULT_GITHUB_BRANCH = "main"

_CHECK_LOCK = threading.Lock()
_CHECK_RESULT: dict | None = None
_CHECK_THREAD: threading.Thread | None = None
_SKIP_BRANCH_UPDATE = False

# EnumProperty items 回调返回的字符串必须被 Python 长期持有，否则可能崩溃
_BRANCH_ITEM_CACHE: list[tuple[str, str, str]] = []
_BRANCH_ENUM_ITEMS: list[tuple[str, str, str]] = [
    (DEFAULT_GITHUB_BRANCH, DEFAULT_GITHUB_BRANCH, "默认分支"),
]
_BRANCH_ITEM_CACHE.extend(_BRANCH_ENUM_ITEMS)


def _addon_root() -> Path:
    """当前插件根目录（含 __init__.py）。"""
    return Path(__file__).resolve().parent.parent


def get_installed_version() -> tuple[int, int, int]:
    """
    读取已安装插件磁盘上的版本。

    优先读 __init__.py，避免「文件已更新但进程内 bl_info 仍是旧值」
    导致检查结果与重启后显示不一致。
    """
    init_path = _addon_root() / "__init__.py"
    try:
        return read_version_file(init_path)
    except Exception:
        return tuple(int(v) for v in bl_info["version"])


def get_display_version_text() -> str:
    """侧栏标题用的当前安装版本号。"""
    return format_version(get_installed_version())


def _prefs(context: bpy.types.Context):
    """读取插件偏好设置。"""
    return context.preferences.addons[ADDON_MODULE].preferences


def _global_prefs():
    """从当前 Blender 上下文读取插件偏好。"""
    addon = bpy.context.preferences.addons.get(ADDON_MODULE)
    return addon.preferences if addon is not None else None


def _repository_values(prefs) -> tuple[str, str, str]:
    """返回固定仓库与当前所选分支。"""
    branch = str(getattr(prefs, "github_branch", "") or "").strip()
    if not branch:
        branch = DEFAULT_GITHUB_BRANCH
    return DEFAULT_GITHUB_OWNER, DEFAULT_GITHUB_REPO, branch


def _order_branch_names(names: list[str], default_branch: str = "") -> list[str]:
    """默认分支优先，其余按名称排序。"""
    unique = []
    seen = set()
    for name in names:
        text = str(name).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)

    ordered: list[str] = []
    for preferred in (default_branch, "main", "master"):
        if preferred and preferred in seen and preferred not in ordered:
            ordered.append(preferred)
    for name in sorted(unique):
        if name not in ordered:
            ordered.append(name)
    if not ordered:
        ordered = [DEFAULT_GITHUB_BRANCH]
    return ordered


def _set_branch_enum_items(names: list[str], default_branch: str = "") -> None:
    """更新分支下拉选项，并保留字符串引用。"""
    global _BRANCH_ENUM_ITEMS
    ordered = _order_branch_names(names, default_branch=default_branch)
    items = [(name, name, "") for name in ordered]
    _BRANCH_ITEM_CACHE.extend(items)
    _BRANCH_ENUM_ITEMS = items


def _github_branch_items(self, context):
    """EnumProperty items 回调。"""
    return _BRANCH_ENUM_ITEMS


def _ensure_branch_selection(prefs) -> None:
    """若当前分支不在列表中，回退到列表第一项。"""
    global _SKIP_BRANCH_UPDATE
    identifiers = {item[0] for item in _BRANCH_ENUM_ITEMS}
    current = str(getattr(prefs, "github_branch", "") or "")
    if current in identifiers:
        return
    fallback = _BRANCH_ENUM_ITEMS[0][0]
    _SKIP_BRANCH_UPDATE = True
    try:
        prefs.github_branch = fallback
    finally:
        _SKIP_BRANCH_UPDATE = False


def _on_github_branch_update(self, context) -> None:
    """切换分支后重新检查该分支版本。"""
    if _SKIP_BRANCH_UPDATE:
        return
    start_update_check()


def reset_update_check_ui() -> None:
    """
    清空偏好里持久化的旧检查结果。

    Blender 会保存 AddonPreferences；若上次留下 update_available=True，
    启动瞬间可能误显示「可更新」，点下去会把较新的本地代码降级。
    """
    prefs = _global_prefs()
    if prefs is None:
        return
    prefs.update_check_state = "NOT_CHECKED"
    prefs.update_available = False
    prefs.latest_version = ""
    prefs.update_check_message = "尚未检查更新"
    prefs.installed_version = format_version(get_installed_version())


def _download_zip(url: str, timeout: float = 60.0) -> bytes:
    """下载 ZIP 字节流。"""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AdvReverseEngineering-Updater/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _http_get_json(url: str, timeout: float = 12.0):
    """GET JSON（GitHub API）。"""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AdvReverseEngineering-VersionCheck/1.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _fetch_github_branches(
    owner: str,
    repo: str,
    timeout: float = 12.0,
) -> tuple[list[str], str]:
    """拉取仓库全部分支名与默认分支。"""
    default_branch = DEFAULT_GITHUB_BRANCH
    try:
        repo_info = _http_get_json(
            f"https://api.github.com/repos/{owner}/{repo}",
            timeout=timeout,
        )
        if isinstance(repo_info, dict):
            default_branch = str(
                repo_info.get("default_branch") or DEFAULT_GITHUB_BRANCH
            )
    except Exception:
        pass

    names: list[str] = []
    page = 1
    while page <= 10:
        payload = _http_get_json(
            (
                f"https://api.github.com/repos/{owner}/{repo}/branches"
                f"?per_page=100&page={page}"
            ),
            timeout=timeout,
        )
        if not isinstance(payload, list) or not payload:
            break
        for item in payload:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
        if len(payload) < 100:
            break
        page += 1

    return _order_branch_names(names, default_branch=default_branch), default_branch


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
    """后台线程：拉取分支列表 + 当前分支版本。"""
    global _CHECK_RESULT
    branches: list[str] = []
    default_branch = DEFAULT_GITHUB_BRANCH
    try:
        branches, default_branch = _fetch_github_branches(owner, repo)
    except Exception:
        branches = [DEFAULT_GITHUB_BRANCH, branch]
        default_branch = DEFAULT_GITHUB_BRANCH

    try:
        version = _fetch_remote_version(owner, repo, branch)
        result = {
            "version": version,
            "branches": branches,
            "default_branch": default_branch,
            "error": "",
        }
    except Exception as exc:
        result = {
            "version": None,
            "branches": branches,
            "default_branch": default_branch,
            "error": str(exc),
        }

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


def _apply_check_result(prefs, remote_version: tuple[int, int, int]) -> None:
    """根据磁盘安装版本与远端版本写入偏好状态。"""
    installed = get_installed_version()
    prefs.installed_version = format_version(installed)
    prefs.latest_version = format_version(remote_version)
    status = compare_versions(installed, remote_version)
    prefs.update_available = status == "AVAILABLE"

    if status == "AVAILABLE":
        prefs.update_check_state = "AVAILABLE"
        prefs.update_check_message = (
            f"发现新版本 v{prefs.latest_version}"
            f"（当前安装 v{prefs.installed_version}）"
        )
    elif status == "AHEAD":
        prefs.update_check_state = "AHEAD"
        prefs.update_check_message = (
            f"本地 v{prefs.installed_version} 新于服务端 "
            f"v{prefs.latest_version}，跳过更新以免降级"
        )
    else:
        prefs.update_check_state = "CURRENT"
        prefs.update_check_message = (
            f"当前为最新版 v{prefs.installed_version}"
        )


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

    branches = result.get("branches") or []
    if branches:
        _set_branch_enum_items(
            list(branches),
            default_branch=str(result.get("default_branch") or ""),
        )
        _ensure_branch_selection(prefs)

    error = result["error"]
    if error:
        prefs.update_check_state = "ERROR"
        prefs.update_available = False
        prefs.update_check_message = f"检查失败: {error}"
        prefs.installed_version = format_version(get_installed_version())
    else:
        _apply_check_result(prefs, tuple(result["version"]))

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
    prefs.installed_version = format_version(get_installed_version())
    prefs.update_check_message = (
        f"正在检查 GitHub 版本…（本地 v{prefs.installed_version}）"
    )

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
    reset_update_check_ui()
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

        owner, repo, branch = _repository_values(prefs)
        installed = get_installed_version()

        # 更新前重新拉取远端版本，防止偏好里残留的「可更新」造成降级
        try:
            remote_version = _fetch_remote_version(owner, repo, branch)
        except Exception as exc:
            self.report({"ERROR"}, f"无法确认远端版本: {exc}")
            return {"CANCELLED"}

        status = compare_versions(installed, remote_version)
        _apply_check_result(prefs, remote_version)
        if status != "AVAILABLE":
            if status == "AHEAD":
                self.report(
                    {"WARNING"},
                    (
                        f"本地 v{format_version(installed)} 新于服务端 "
                        f"v{format_version(remote_version)}，已取消以免降级"
                    ),
                )
            else:
                self.report({"INFO"}, "当前没有可用更新")
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
                f"下载失败 HTTP {exc.code}。请确认仓库为公开且分支存在",
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

            # 解压包版本再确认一次
            try:
                zip_version = read_version_file(source / "__init__.py")
            except Exception:
                zip_version = remote_version
            if compare_versions(installed, zip_version) != "AVAILABLE":
                self.report(
                    {"WARNING"},
                    (
                        f"下载包 v{format_version(zip_version)} "
                        f"不高于本地 v{format_version(installed)}，已取消"
                    ),
                )
                return {"CANCELLED"}

            _copy_tree(source, addon_root)

        except Exception as exc:
            self.report({"ERROR"}, f"更新失败: {exc}")
            return {"CANCELLED"}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        prefs.update_available = False
        prefs.update_check_state = "UPDATED"
        prefs.installed_version = format_version(get_installed_version())
        prefs.latest_version = format_version(remote_version)
        prefs.update_check_message = (
            f"已更新到 v{prefs.installed_version}，请重启 Blender"
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

    github_branch: bpy.props.EnumProperty(
        name="分支",
        description="从官方仓库自动获取的分支列表",
        items=_github_branch_items,
        # items 为回调时，default 只能是整数下标（0 = 列表第一项）
        default=0,
        update=_on_github_branch_update,
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
    installed_version: bpy.props.StringProperty(
        name="已安装版本",
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
            text=(
                f"官方仓库: {DEFAULT_GITHUB_OWNER}/{DEFAULT_GITHUB_REPO}"
            ),
        )
        layout.prop(self, "github_branch", text="分支")
        layout.label(text=self.update_check_message)
        layout.operator("are.check_github_update", icon="FILE_REFRESH")


classes = (
    ARE_AddonPreferences,
    ARE_OT_check_github_update,
    ARE_OT_update_from_github,
)
