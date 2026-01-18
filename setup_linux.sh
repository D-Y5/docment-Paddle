#!/bin/bash
# SmartDoc 项目本地部署脚本 (Linux/macOS)

echo "========================================"
echo "SmartDoc 本地部署向导"
echo "========================================"
echo ""

# 检查 Python
echo "[1/6] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 Python3，请先安装 Python 3.10+"
    exit 1
fi
python3 --version

# 检查 Node.js
echo ""
echo "[2/6] 检查 Node.js 环境..."
if ! command -v node &> /dev/null; then
    echo "错误：未找到 Node.js，请先安装 Node.js 18+"
    exit 1
fi
node --version

# 创建虚拟环境
echo ""
echo "[3/6] 创建 Python 虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "虚拟环境创建成功"
else
    echo "虚拟环境已存在"
fi

# 激活虚拟环境
echo ""
echo "[4/6] 激活虚拟环境并安装依赖..."
source venv/bin/activate

# 安装 Python 依赖
echo "安装 Python 依赖中..."
pip install -r requirements.txt

# 检查 pnpm
echo ""
echo "[5/6] 检查 pnpm..."
if ! command -v pnpm &> /dev/null; then
    echo "pnpm 未安装，正在安装..."
    npm install -g pnpm
fi

# 安装 Web 依赖
echo "安装 Web 依赖中..."
cd web
pnpm install
cd ..

# 检查数据集
echo ""
echo "[6/6] 检查数据集..."
DATA_PATH="$HOME/scikit_learn_data/smartdoc15-ch1"
if [ -d "$DATA_PATH" ]; then
    echo "数据集已找到：$DATA_PATH"
else
    echo "警告：未找到数据集"
    echo "请将 SmartDoc 2015 数据集解压到："
    echo "$DATA_PATH"
fi

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""
echo "下一步操作："
echo ""
echo "1. 准备训练数据："
echo "   python scripts/prepare_data.py"
echo ""
echo "2. 训练模型（可选）："
echo "   python scripts/train.py --config configs/train.yaml"
echo ""
echo "3. 启动 Web 界面："
echo "   cd web && pnpm dev"
echo ""
echo "或者直接运行："
echo "   ./start_web.sh"
echo ""
