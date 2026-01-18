#!/bin/bash

# SmartDoc 文档矫正系统 - 快速启动脚本

echo "=========================================="
echo "  SmartDoc 文档矫正系统"
echo "  基于深度学习的文档四边界定位与透视矫正"
echo "=========================================="
echo ""

# 检查Python环境
echo "检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi
echo "Python版本: $(python --version)"

# 检查依赖
echo ""
echo "检查Python依赖..."
python -c "import torch; import cv2; import numpy; print('✓ 所有依赖已安装')" 2>/dev/null || {
    echo "错误: 缺少Python依赖，请运行: pip install -r requirements.txt"
    exit 1
}

# 检查数据集
echo ""
echo "检查数据集..."
if [ ! -d "C:/Users/dengyu/smartdoc15_train_samples" ]; then
    echo "警告: 未找到训练样本，请运行: python scripts/prepare_data.py"
else
    echo "✓ 训练样本已存在"
fi

# 菜单
echo ""
echo "请选择操作:"
echo "1. 生成训练样本"
echo "2. 训练模型"
echo "3. 推理测试"
echo "4. 运行评测"
echo "5. 启动Web界面"
echo "6. 退出"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "生成训练样本..."
        python scripts/prepare_data.py --verify --num-verify 10
        ;;
    2)
        echo ""
        echo "训练模型..."
        python scripts/train.py --config configs/train.yaml
        ;;
    3)
        echo ""
        read -p "输入图像路径: " image_path
        read -p "输出图像路径: " output_path
        python scripts/inference.py \
            --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
            --input "$image_path" \
            --output "$output_path" \
            --save-visualization
        ;;
    4)
        echo ""
        python scripts/evaluate.py \
            --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
            --data-root C:/Users/dengyu/smartdoc15_train_samples
        ;;
    5)
        echo ""
        echo "启动Web界面..."
        cd web
        pnpm dev
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "操作完成！"
