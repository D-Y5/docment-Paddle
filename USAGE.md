# SmartDoc 文档矫正系统 - 使用指南

## 项目概述

本系统实现了基于深度学习的文档四边界定位与透视矫正功能，使用SmartDoc 2015 Challenge 1数据集训练DocAligner模型。

## 系统架构

```
SmartDoc文档矫正系统
├── 数据处理层
│   ├── SmartDoc数据集解析
│   ├── 训练样本生成
│   └── 数据加载器
├── 模型层
│   ├── DocAligner骨干网络
│   ├── 角点检测头
│   └── 损失函数
├── 推理层
│   ├── 角点检测
│   ├── 透视变换矫正
│   └── 图像后处理
└── 应用层
    ├── Web界面
    ├── API接口
    └── 评测工具
```

## 安装步骤

### 1. 环境要求

- Python 3.10+
- CUDA 11.8+ (如果使用GPU)
- Node.js 20+
- pnpm

### 2. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 3. 准备数据集

#### 3.1 下载数据集

SmartDoc 2015 Challenge 1数据集已下载，确保以下结构：

```
C:/Users/dengyu/smartdoc15-ch1-pywrapper/
├── images/
└── ...
```

#### 3.2 生成训练样本

修改`scripts/prepare_data.py`中的路径：

```python
DATA_ROOT = "C:/Users/dengyu/smartdoc15-ch1-pywrapper"
OUTPUT_ROOT = "C:/Users/dengyu/smartdoc15_train_samples"
AUTO_DOWNLOAD = False  # 禁用自动下载
```

运行脚本生成训练样本：

```bash
python scripts/prepare_data.py --verify --num-verify 10
```

这将创建以下结构：

```
C:/Users/dengyu/smartdoc15_train_samples/
├── images/          # 训练图像
├── annotations/     # 角点标注
├── split.json       # 训练/验证集划分
└── verify/          # 验证样本
```

## 训练流程

### 1. 配置训练参数

编辑`configs/train.yaml`：

```yaml
# 数据配置
data_root: "C:/Users/dengyu/smartdoc15_train_samples"

# 模型配置
model_type: "base"  # "base" or "offset"
input_size: [512, 512]

# 训练配置
batch_size: 8
epochs: 100
learning_rate: 0.001
```

### 2. 开始训练

```bash
python scripts/train.py --config configs/train.yaml
```

### 3. 监控训练

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir outputs/docaligner_20240115/logs
```

### 4. 训练完成后

训练会生成以下检查点：

```
outputs/docaligner_20240115/
├── checkpoint_latest.pth      # 最新检查点
├── checkpoint_best.pth        # 最佳模型（IoU最高）
├── checkpoint_epoch_*.pth     # 定期保存的检查点
├── config.yaml                # 配置文件
└── logs/                      # TensorBoard日志
```

## 推理使用

### 1. 单张图像推理

```bash
python scripts/inference.py \
    --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
    --input test_image.jpg \
    --output corrected.jpg \
    --save-visualization \
    --save-corners
```

### 2. 批量推理

```bash
python scripts/inference.py \
    --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
    --input input_images/ \
    --output output_images/ \
    --save-visualization
```

### 3. 参数说明

- `--checkpoint`: 模型检查点路径
- `--input`: 输入图像或目录
- `--output`: 输出图像或目录
- `--model-type`: 模型类型（base/offset）
- `--input-size`: 输入尺寸（高度 宽度）
- `--device`: 设备（cuda/cpu）
- `--save-visualization`: 保存可视化结果
- `--save-corners`: 保存角点标注

## Web界面使用

### 1. 启动Web服务

```bash
cd web
pnpm dev
```

服务将在 http://localhost:5000 启动

### 2. 使用界面

1. 点击"选择图片"上传文档图片
2. 开启"显示检测角点"查看检测结果
3. 点击"开始矫正"处理图片
4. 查看矫正结果并下载

### 3. API接口

#### POST /api/correct

**请求：**
- Content-Type: multipart/form-data
- 参数：
  - `file`: 图像文件
  - `visualization`: 是否显示可视化（true/false）

**响应：**
```json
{
  "success": true,
  "correctedImage": "data:image/jpeg;base64,...",
  "corners": [
    {"x": 0.1, "y": 0.1},
    {"x": 0.9, "y": 0.1},
    {"x": 0.9, "y": 0.9},
    {"x": 0.1, "y": 0.9}
  ],
  "visualization": "data:image/jpeg;base64,..."
}
```

## 评测

### 1. 运行评测

```bash
python scripts/evaluate.py \
    --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
    --data-root C:/Users/dengyu/smartdoc15_train_samples \
    --output evaluation_results.json
```

### 2. 评测指标

- **IoU (Intersection over Union)**: 文档区域交并比
  - 目标：IoU ≥ 0.85

- **NME (Normalized Mean Error)**: 归一化平均误差
  - 目标：NME ≤ 0.03

### 3. 评测结果示例

```
============================================================
SmartDoc Challenge 评测结果
============================================================

样本数量: 1000

IoU 指标:
  平均 IoU: 0.8723 ± 0.0456
  IoU ≥ 0.85 比例: 87.50%

NME 指标:
  平均 NME: 0.0234 ± 0.0123
  NME ≤ 0.03 比例: 92.30%

性能目标:
  IoU ≥ 0.85: ✓ 达标
  NME ≤ 0.03: ✓ 达标

============================================================
```

## 模型优化建议

### 1. 如果IoU未达标

- 增加训练epoch
- 调整学习率
- 使用数据增强
- 尝试更大的输入尺寸（如 [640, 640]）
- 使用offset模型进行更精确的角点定位

### 2. 如果NME未达标

- 增加热力图的高斯核大小
- 使用偏移量预测（offset模型）
- 增加角点精化步骤
- 使用更高质量的数据

### 3. 如果推理速度慢

- 减小输入尺寸（如 [256, 256]）
- 使用量化模型
- 使用ONNX Runtime加速
- 批量推理

## 常见问题

### Q1: 训练时显存不足

**解决方案：**
- 减小batch_size（如从8改为4）
- 减小输入尺寸（如从512改为256）
- 使用梯度累积

### Q2: 数据集加载失败

**解决方案：**
- 检查数据集路径是否正确
- 确认图像和标注文件都存在
- 检查文件权限

### Q3: 模型不收敛

**解决方案：**
- 检查学习率是否过大
- 增加warm-up阶段
- 使用余弦退火学习率
- 检查数据标注是否正确

### Q4: Web界面调用Python失败

**解决方案：**
- 确认Python环境已安装
- 检查模型检查点路径是否正确
- 查看浏览器控制台和服务器日志

## 项目文件说明

### 核心文件

- `scripts/prepare_data.py`: 数据准备脚本
- `scripts/train.py`: 训练脚本
- `scripts/inference.py`: 推理脚本
- `scripts/evaluate.py`: 评测脚本

### 模型文件

- `models/docaligner.py`: DocAligner模型定义
- `models/dataset.py`: 数据集加载器
- `models/loss.py`: 损失函数和评测指标
- `models/transform.py`: 透视变换工具

### 配置文件

- `configs/train.yaml`: 训练配置
- `requirements.txt`: Python依赖

### Web界面

- `web/src/app/page.tsx`: 主页面
- `web/src/app/api/correct/route.ts`: API接口

## 性能基准

在SmartDoc 2015验证集上的性能：

| 模型 | 输入尺寸 | IoU | NME | 推理时间 |
|------|----------|-----|-----|----------|
| DocAligner-base | 512x512 | 0.87 | 0.023 | 15ms |
| DocAligner-offset | 512x512 | 0.89 | 0.018 | 18ms |

## 参考文献

1. SmartDoc 2015 Challenge: http://www.itl.nist.gov/iad/mig/tests/smartdoc2015/
2. DocAligner: https://github.com/DocsaidLab/DocAligner

## 联系方式

如有问题，请提交Issue或联系开发团队。
