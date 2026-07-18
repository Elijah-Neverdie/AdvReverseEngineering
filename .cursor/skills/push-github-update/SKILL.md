---
name: push-github-update
description: >-
  Commit and push AdvReverseEngineering changes to GitHub after each code
  iteration so the remote version stays in sync with local. Use when finishing a
  feature or bugfix, bumping bl_info version, syncing the Blender addon, or
  whenever local commits/changes are ahead of origin. Also use when the user asks
  to upload, push, 上传, or 推送代码.
---

# Push GitHub Update

在 **AdvReverseEngineering** 仓库完成一轮有意义的代码改动后，必须执行本流程，避免 GitHub 版本落后于本地（更新检查会显示「服务端版本更低」）。

## When

在以下时机 **主动执行**（无需用户再说「推送」）：

- 功能/缺陷修复已完成，准备向用户汇报结果前
- 修改了 `bl_info["version"]` 或插件安装目录同步逻辑后
- 本地存在未提交改动，或 `git status` 显示 ahead of `origin`

跳过时机：用户明确说不要推送；仅只读排查；改动只有本地诊断脚本且用户未要求入库。

## Steps

按顺序执行；PowerShell 下用多条命令，不要用 `&&` 链。

### 1. 检查状态

并行运行：

```powershell
git status -sb
git diff
git diff --staged
git log -5 --oneline
git rev-parse --abbrev-ref HEAD
```

确认当前分支（通常 `main`）且跟踪 `origin`。

### 2. 版本号

若本轮有面向用户的功能/修复，确保 `AdvReverseEngineering/__init__.py` 中 `bl_info["version"]` 已递增（patch +1）。纯文档/skill/规则改动可不涨版本。

### 3. 暂存与提交

- 只暂存本轮相关源码与测试；排除密钥、`.env`、大型无关产物
- `tools/_diag_*.py` 等临时诊断脚本默认不提交，除非用户要求或已成为常用工具
- 提交信息用 1–2 句说明 **why**，英文或中文与仓库近期风格一致

```powershell
git add <paths...>
git commit -m @"
简要说明本轮为何改动。

"@
```

若无改动可提交且仅有 unpushed commits，跳到推送。

遵循仓库既有 commit 规则：不改 git config；不用 `--no-verify`；不 amend 他人/已推送提交。

若 `git commit` 报 unknown identity，**不要**运行 `git config`；改为仅在该次命令设置环境变量（与近期提交作者一致）：

```powershell
$env:GIT_AUTHOR_NAME = "AdvReverseEngineering"
$env:GIT_AUTHOR_EMAIL = "advreverseengineering@users.noreply.github.com"
$env:GIT_COMMITTER_NAME = $env:GIT_AUTHOR_NAME
$env:GIT_COMMITTER_EMAIL = $env:GIT_AUTHOR_EMAIL
```

推送若遇 `schannel` SSL 失败，可重试一次，或改用：

```powershell
git -c http.version=HTTP/1.1 push -u origin HEAD
```

### 4. 推送

```powershell
git push -u origin HEAD
```

失败时根据报错处理（认证、SSL、非 fast-forward）。需要 `required_permissions: ["all"]` 或网络权限时照常申请。

### 5. 同步 Blender 插件目录（本机）

推送成功后，将插件包同步到本机安装副本：

```powershell
$src = "D:\Code\AdvReverseEngineering\AdvReverseEngineering"
foreach ($d in @(
  "$env:APPDATA\Blender Foundation\Blender\5.1\scripts\addons\AdvReverseEngineering",
  "$env:APPDATA\Blender Foundation\Blender\4.2\scripts\addons\AdvReverseEngineering"
)) {
  if (Test-Path (Split-Path $d -Parent)) {
    robocopy $src $d /MIR /XD __pycache__ .git /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
  }
}
```

### 6. 向用户汇报

简短说明：提交哈希、新版本号、是否已推到 `origin`、插件目录是否已同步。提醒如需侧栏立即反映新版本可重启 Blender。

## Safety

- 永不 `push --force` 到 `main`/`master`，除非用户明确要求
- 永不提交含密钥的文件
- 远程低于本地时，优先完成本流程，而不是让用户从 GitHub「更新」造成降级
