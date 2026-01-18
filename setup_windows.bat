@echo off
REM SmartDoc 项目本地部署脚本 (Windows)

echo ========================================
echo SmartDoc 本地部署向导
echo ========================================
echo.

REM 检查 Python
echo [1/6] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
python --version

REM 检查 Node.js
echo.
echo [2/6] 检查 Node.js 环境...
node --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到 Node.js，请先安装 Node.js 18+
    pause
    exit /b 1
)
node --version

REM 创建虚拟环境
echo.
echo [3/6] 创建 Python 虚拟环境...
if not exist "venv" (
    python -m venv venv
    echo 虚拟环境创建成功
) else (
    echo 虚拟环境已存在
)

REM 激活虚拟环境
echo.
echo [4/6] 激活虚拟环境并安装依赖...
call venv\Scripts\activate.bat

REM 安装 Python 依赖
echo 安装 Python 依赖中...
pip install -r requirements.txt

REM 检查 pnpm
echo.
echo [5/6] 检查 pnpm...
pnpm --version >nul 2>&1
if errorlevel 1 (
    echo pnpm 未安装，正在安装...
    npm install -g pnpm
)

REM 安装 Web 依赖
echo 安装 Web 依赖中...
cd web
pnpm install
cd ..

REM 检查数据集
echo.
echo [6/6] 检查数据集...
if exist "C:\Users\dengyu\scikit_learn_data\smartdoc15-ch1" (
    echo 数据集已找到
) else (
    echo 警告：未找到数据集
    echo 请将 SmartDoc 2015 数据集解压到：
    echo C:\Users\dengyu\scikit_learn_data\smartdoc15-ch1
)

echo.
echo ========================================
echo 部署完成！
echo ========================================
echo.
echo 下一步操作：
echo.
echo 1. 准备训练数据：
echo    python scripts/prepare_data.py
echo.
echo 2. 训练模型（可选）：
echo    python scripts/train.py --config configs/train.yaml
echo.
echo 3. 启动 Web 界面：
echo    cd web && pnpm dev
echo.
echo 或者直接运行：
echo    start_web.bat
echo.

pause
