@echo off
REM ML Toolbox 一键启动（看门人两模式：本地 GUI / AI 模式）
REM 双击本文件即可；等价于在本目录运行 python launcher.py
chcp 65001 >nul 2>&1
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo [错误] PATH 里找不到 python。请先安装 Python 3.10 并勾选"Add to PATH"。
  pause
  exit /b 1
)

python launcher.py
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo.
  echo [提示] 看门人以退出码 %RC% 结束。上面（以及本窗口）的日志是排错依据；
  echo        常见原因见 docs\交付说明.md 第 6 节"故障排查"。
  pause
)
exit /b %RC%
