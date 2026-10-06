@echo off
setlocal
set "APP_ROOT=%~dp0"
set "APP_ROOT_NO_SLASH=%APP_ROOT:~0,-1%"
set "PYTHON=%APP_ROOT_NO_SLASH%\.venv\Scripts\python.exe"
set "PALMISTRY_URL=http://127.0.0.1:8501/keypoints"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%APP_ROOT_NO_SLASH%\open_palmistry_desktop.ps1" -AppRoot "%APP_ROOT_NO_SLASH%" -PythonPath "%PYTHON%" -Url "%PALMISTRY_URL%"
endlocal
