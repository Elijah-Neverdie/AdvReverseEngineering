@echo off
set LOG=d:\Code\cursor_projects\reverse_engineering\_git_setup_log.txt
echo ===== %DATE% %TIME% START =====>>"%LOG%"
echo [1] Robocopy AdvReverseEngineering>>"%LOG%"
robocopy "C:\Users\A\AppData\Roaming\Blender Foundation\Blender\4.2\scripts\addons\AdvReverseEngineering" "d:\Code\cursor_projects\reverse_engineering\AdvReverseEngineering" /E >>"%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo ROBOCOPY_EXIT=%RC%>>"%LOG%"
echo ROBOCOPY_EXIT=%RC%
echo [2] Copy README.md>>"%LOG%"
if exist "d:\Code\cursor_projects\reverse_engineering\AdvReverseEngineering\README.md" (
  copy /Y "d:\Code\cursor_projects\reverse_engineering\AdvReverseEngineering\README.md" "d:\Code\cursor_projects\reverse_engineering\README.md" >>"%LOG%" 2>&1
  echo README_COPY_EXIT=%ERRORLEVEL%>>"%LOG%"
  echo README_COPY_EXIT=%ERRORLEVEL%
) else (
  echo README.md not found in addon>>"%LOG%"
  echo README.md not found in addon
)
cd /d "d:\Code\cursor_projects\reverse_engineering"
echo [3] git add and commit>>"%LOG%"
set GIT_AUTHOR_NAME=AdvReverseEngineering
set GIT_AUTHOR_EMAIL=advreverseengineering@users.noreply.github.com
set GIT_COMMITTER_NAME=AdvReverseEngineering
set GIT_COMMITTER_EMAIL=advreverseengineering@users.noreply.github.com
"C:\Program Files\Git\cmd\git.exe" add -A >>"%LOG%" 2>&1
echo GIT_ADD_EXIT=%ERRORLEVEL%>>"%LOG%"
echo GIT_ADD_EXIT=%ERRORLEVEL%
"C:\Program Files\Git\cmd\git.exe" commit -m "Add GitHub sync updater and Chinese UI auto-orient" >>"%LOG%" 2>&1
set CE=%ERRORLEVEL%
echo GIT_COMMIT_EXIT=%CE%>>"%LOG%"
echo GIT_COMMIT_EXIT=%CE%
echo [4] git status and log>>"%LOG%"
"C:\Program Files\Git\cmd\git.exe" status >>"%LOG%" 2>&1
"C:\Program Files\Git\cmd\git.exe" log -1 --oneline >>"%LOG%" 2>&1
"C:\Program Files\Git\cmd\git.exe" status
"C:\Program Files\Git\cmd\git.exe" log -1 --oneline
echo ===== DONE =====>>"%LOG%"
