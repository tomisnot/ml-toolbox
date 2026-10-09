@echo off
REM ML Toolbox 一键启动：起**整站**（本软件 + harness + 进程编排，全归 mecha 的统一启动器）。
REM ⚠ 本项目**不再自己编排进程**（旧 launcher.py 已删）⇒ 入口就是框架那一条。
REM   活知识（旧启动器踩过的坑与实测常量）：docs\旧启动器-活知识存档.md
chcp 65001 >nul 2>&1
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo [错误] PATH 里找不到 python。请先安装 Python 3.10 并勾选"Add to PATH"。
  pause
  exit /b 1
)

REM ⚠ cwd 必须是项目仓（声明发现点 `ml_mecha/mecha_site.py` 在仓内），
REM   并用 `--root` 显式告诉框架去哪儿找声明（不靠 cwd 的隐式约定）。
python -m mecha --all --root "%~dp0"
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo.
  echo [提示] 启动器以退出码 %RC% 结束。上面（以及本窗口）的日志是排错依据；
  echo        常见原因见 docs\交付说明.md 第 6 节"故障排查"。
  pause
)
exit /b %RC%
