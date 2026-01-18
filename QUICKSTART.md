# 快速入门指南

## 5分钟快速开始

### 步骤1: 安装依赖

```bash
pip install -r requirements.txt
cd web && pnpm install && cd ..
```

### 步骤2: 生成训练样本

编辑`scripts/prepare_data.py`，修改数据集路径：

```python
DATA_ROOT = "C:/Users/dengyu/smartdoc15-ch1-pywrapper"
OUTPUT_ROOT = "C:/Users/dengyu/smartdoc15_train_samples"
AUTO_DOWNLOAD = False
```

运行：

```bash
python scripts/prepare_data.py
```

### 步骤3: 训练模型

```bash
python scripts/train.py --config configs/train.yaml
```

### 步骤4: 测试推理

```bash
python scripts/inference.py \
    --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
    --input test.jpg \
    --output corrected.jpg
```

### 步骤5: 启动Web界面

```bash
cd web
pnpm dev
```

访问: http://localhost:5000

## 使用快速启动脚本

### Windows

双击运行: `quick_start.bat`

### Linux/Mac

```bash
chmod +x quick_start.sh
./quick_start.sh
```

## 常用命令

```bash
# 数据准备
python scripts/prepare_data.py --verify --num-verify 10

# 训练模型
python scripts/train.py --config configs/train.yaml

# 推理
python scripts/inference.py --checkpoint <path> --input <image> --output <output>

# 评测
python scripts/evaluate.py --checkpoint <path> --data-root <path>

# Web界面
cd web && pnpm dev

# 监控训练
tensorboard --logdir outputs/docaligner_20240115/logs
```

## 下一步

查看详细文档：
- [USAGE.md](USAGE.md) - 完整使用指南
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 项目总结
- [README.md](README.md) - 项目说明

需要帮助？
- 查看各脚本中的注释
- 检查日志文件
- 提交Issue
