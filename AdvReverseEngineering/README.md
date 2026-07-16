# AdvReverseEngineering

Blender 4.x 逆向工程预处理插件（摄影测量 / 激光扫描模型自动摆正等）。

## 安装

将 `AdvReverseEngineering` 文件夹放到：

```
%APPDATA%\Blender Foundation\Blender\4.2\scripts\addons\
```

在 Blender：`编辑 → 偏好设置 → 插件` 中启用 **AdvReverseEngineering**。

侧边栏（`N`）→ **逆向工具**。

## 多电脑同步（GitHub）

### 1. 首次推送到 GitHub（开发机）

1. 在 [GitHub](https://github.com/new) 新建公开仓库，名称建议：`AdvReverseEngineering`
2. 仓库结构任选其一：
   - **推荐**：仓库根目录就是插件（根目录有 `__init__.py`）
   - 或：仓库内有 `AdvReverseEngineering/` 子目录
3. 本机推送示例：

```bat
cd /d d:\Code\cursor_projects\reverse_engineering
git remote add origin https://github.com/你的用户名/AdvReverseEngineering.git
git push -u origin main
```

若仓库根需要是插件本身：

```bat
cd /d d:\Code\cursor_projects\reverse_engineering\AdvReverseEngineering
git init -b main
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的用户名/AdvReverseEngineering.git
git push -u origin main
```

### 2. 其他电脑安装

1. 下载仓库 ZIP，解压到 `scripts\addons\AdvReverseEngineering`
2. 或在 Blender 启用插件后，在侧边栏填写 GitHub 用户名/仓库名，点 **从 GitHub 更新**

### 3. 更新

任意电脑修改并 `git push` 后，其他电脑在侧边栏点 **从 GitHub 更新**，然后重启 Blender。

也可在：`偏好设置 → 插件 → AdvReverseEngineering` 中配置并更新。

> 注意：当前更新器通过公开仓库 ZIP 下载，**仓库需为 Public**。

## 功能

- 自动摆正（多种策略循环切换）
- 底面紫色高亮
- 从 GitHub 一键更新
