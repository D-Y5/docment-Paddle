@echo off
chcp 65001 >nul
echo ==========================================
echo   SmartDoc 文档矫正系统
echo   基于深度学习的文档四边界定位与透视矫正
echo ==========================================
echo.

REM 检查Python环境
echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)
python --version

REM 检查依赖
echo.
echo 检查Python依赖...
python -c "import torch; import cv2; import numpy; print('✓ 所有依赖已安装')" 2>nul
if errorlevel 1 (
    echo 错误: 缺少Python依赖，请运行: pip install -r requirements.txt
    pause
    exit /b 1
)

REM 检查数据集
echo.
echo 检查数据集...
if not exist "C:\Users\dengyu\smartdoc15_train_samples" (
    echo 警告: 未找到训练样本，请运行: python scripts\prepare_data.py
) else (
    echo ✓ 训练样本已存在
)

REM 菜单
echo.
echo 请选择操作:
echo 1. 生成训练样本
echo 2. 训练模型
echo 3. 推理测试
echo 4. 运行评测
echo 5. 启动Web界面
echo 6. 退出
echo.
set /p choice="请输入选项 (1-6): "

if "%choice%"=="1" (
    echo.
    echo 生成训练样本...
    python scripts\prepare_data.py --verify --num-verify 10
) else if "%choice%"=="2" (
    echo.
    echo 训练模型...
    python scripts\train.py --config configs\train.yaml
) else if "%choice%"=="3" (
    echo.
    set /p image_path="输入图像路径: "
    set /p output_path="输出图像路径: "
    python scripts\inference.py --checkpoint outputs\docaligner_20240115\checkpoint_best.pth --input "%image_path%" --output "%output_path%" --save-visualization
) else if "%choice%"=="4" (
    echo.
    python scripts\evaluate.py --checkpoint outputs\docaligner_20240115\checkpoint_best.pth --data-root C:\Users\dengyu\smartdoc15_train_samples
) else if "%choice%"=="5" (
    echo.
    echo 启动Web界面...
    cd web
    pnpm dev
) else if "%choice%"=="6" (
    echo 退出
    exit /b 0
) else (
    echo 无效选项
    pause
    exit /b 1
)

echo.
echo 操作完成！
pause
