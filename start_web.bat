@echo off
REM 启动 Web 界面

echo ========================================
echo 启动 SmartDoc Web 界面
echo ========================================
echo.

REM 激活虚拟环境
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo 警告：未找到虚拟环境
    echo 请先运行 setup_windows.bat 进行部署
    pause
    exit /b 1
)

REM 进入 Web 目录
cd web

REM 启动开发服务器
echo.
echo Web 界面启动中...
echo 访问地址：http://localhost:5000
echo.
echo 按 Ctrl+C 停止服务
echo.

pnpm dev

pause
