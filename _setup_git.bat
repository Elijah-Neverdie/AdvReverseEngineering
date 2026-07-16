@echo off
setlocal
set LOG=d:\Code\cursor_projects\reverse_engineering\_git_setup_log.txt
echo ==== START %2026/07/16 ÖÜËÄ% %20:50:02.92% ====>%%LOG%%
echo. >>%%LOG%%
echo ==== 1. ROBOCOPY ====>>%%LOG%%
robocopy "C:\Users\A\AppData\Roaming\Blender Foundation\Blender\4.2\scripts\addons\AdvReverseEngineering" "d:\Code\cursor_projects\reverse_engineering\AdvReverseEngineering" /E /NFL /NDL /NJH /NJS /nc /ns /np >>%%LOG%% 2>&1
set RC=%0%
echo ROBOCOPY_EXIT=%%RC%%>>%%LOG%%
echo. >>%%LOG%%
echo ==== 2. CREATE .gitignore ====>>%%LOG%%
(echo __pycache__/& echo *.py[cod]& echo *$py.class& echo *.blend1& echo *.blend2& echo .DS_Store& echo Thumbs.db& echo .idea/& echo .vscode/& echo *.log& echo _git_setup_log.txt)>d:\Code\cursor_projects\reverse_engineering\.gitignore
echo GITIGNORE_EXIT=%0%>>%%LOG%%
type d:\Code\cursor_projects\reverse_engineering\.gitignore >>%%LOG%%
echo. >>%%LOG%%
cd /d d:\Code\cursor_projects\reverse_engineering
echo ==== 3. GIT INIT ====>>%%LOG%%
"C:\Program Files\Git\cmd\git.exe" init -b main >>%%LOG%% 2>&1
echo GIT_INIT_EXIT=%0%>>%%LOG%%
echo ==== 4. GIT ADD ====>>%%LOG%%
"C:\Program Files\Git\cmd\git.exe" add . >>%%LOG%% 2>&1
echo GIT_ADD_EXIT=%0%>>%%LOG%%
echo ==== 5. GIT STATUS ====>>%%LOG%%
"C:\Program Files\Git\cmd\git.exe" status >>%%LOG%% 2>&1
echo GIT_STATUS_EXIT=%0%>>%%LOG%%
echo ==== 6. GIT COMMIT ====>>%%LOG%%
"C:\Program Files\Git\cmd\git.exe" commit -m "Initial commit: AdvReverseEngineering Blender 4.x addon" >>%%LOG%% 2>&1
echo GIT_COMMIT_EXIT=%0%>>%%LOG%%
echo ==== END ====>>%%LOG%%
