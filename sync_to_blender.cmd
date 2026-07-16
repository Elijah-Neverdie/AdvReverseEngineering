@echo off
setlocal

set "SRC=%~dp0AdvReverseEngineering"
set "DST=%APPDATA%\Blender Foundation\Blender\4.2\scripts\addons\AdvReverseEngineering"

if not exist "%DST%" mkdir "%DST%"

robocopy "%SRC%" "%DST%" /MIR /NFL /NDL /NJH /NJS /nc /ns /np
if %ERRORLEVEL% GEQ 8 (
    echo [ERROR] Sync failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo [OK] Synced to: %DST%
exit /b 0
