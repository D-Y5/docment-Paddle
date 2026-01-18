#!/bin/bash
# 启动 Web 界面

echo "========================================"
echo "启动 SmartDoc Web 界面"
echo "========================================"
echo ""

# 激活虚拟环境
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "警告：未找到虚拟环境"
    echo "请先运行 ./setup_linux.sh 进行部署"
    exit 1
fi

# 进入 Web 目录
cd web

# 启动开发服务器
echo ""
echo "Web 界面启动中..."
echo "访问地址：http://localhost:5000"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

pnpm dev
