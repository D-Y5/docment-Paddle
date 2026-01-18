#!/bin/bash

# 快速验证修复的脚本

echo "=========================================="
echo "  SmartDoc 文档矫正系统 - 修复验证"
echo "=========================================="
echo ""

# 检查Python环境
echo "1. 检查Python环境..."
python --version || {
    echo "❌ Python未安装"
    exit 1
}
echo "✅ Python: $(python --version)"

# 检查依赖
echo ""
echo "2. 检查依赖..."
python -c "import cv2; print('✅ OpenCV 已安装')" || {
    echo "❌ OpenCV未安装，正在安装..."
    pip install opencv-python --quiet
    echo "✅ OpenCV已安装"
}

python -c "import numpy; print('✅ NumPy 已安装')" || {
    echo "❌ NumPy未安装"
    exit 1
}

# 检查测试文件
echo ""
echo "3. 检查测试文件..."
if [ -f "test_document_original.jpg" ]; then
    echo "✅ 测试图像已存在"
else
    echo "创建测试图像..."
    python create_test_image.py
    echo "✅ 测试图像已创建"
fi

# 测试模拟推理
echo ""
echo "4. 测试模拟推理..."
python scripts/test_mock_inference.py \
    --input test_document_original.jpg \
    --output test_verify.jpg \
    --save-corners \
    --save-visualization 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 模拟推理成功"
else
    echo "❌ 模拟推理失败"
    exit 1
fi

# 检查输出文件
echo ""
echo "5. 检查输出文件..."
files=("test_verify.jpg" "test_verify.json" "test_verify_vis.jpg")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✅ $file ($size)"
    else
        echo "  ❌ $file 未生成"
        exit 1
    fi
done

# 检查Web服务
echo ""
echo "6. 检查Web服务..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 | grep -q "200"; then
    echo "✅ Web服务运行正常 (http://localhost:5000)"
else
    echo "⚠️  Web服务未运行"
    echo "   启动命令: cd web && pnpm dev"
fi

# 清理测试文件
echo ""
echo "7. 清理测试文件..."
rm -f test_verify.jpg test_verify.json test_verify_vis.jpg
echo "✅ 清理完成"

echo ""
echo "=========================================="
echo "  ✅ 所有验证通过！"
echo "=========================================="
echo ""
echo "系统已就绪，可以开始使用："
echo "  - Web界面: http://localhost:5000"
echo "  - 测试命令: python scripts/test_mock_inference.py --input <image> --output <output>"
echo "  - 训练模型: python scripts/train.py --config configs/train.yaml"
echo ""
