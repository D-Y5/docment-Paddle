# SmartDoc 2015 文档四边界定位与透视矫正系统

基于深度学习的文档矫正系统，支持SmartDoc 2015 Challenge 1数据集。

## 项目结构

```
.
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据集
│   └── train_samples/        # 生成的训练样本
├── models/                    # 模型代码
│   ├── docaligner/           # DocAligner模型
│   └── utils/                # 工具函数
├── scripts/                   # 脚本
│   ├── prepare_data.py       # 数据准备脚本
│   ├── train.py              # 训练脚本
│   ├── inference.py          # 推理脚本
│   └── evaluate.py           # 评测脚本
├── web/                       # Web界面 (Next.js)
│   ├── app/
│   └── components/
├── notebooks/                 # Jupyter notebooks
├── checkpoints/              # 模型检查点
└── outputs/                  # 输出结果

```

## 快速开始

### 1. 数据准备

```bash
# 生成训练样本（修改脚本中的路径）
python scripts/prepare_data.py
```

### 2. 训练模型

```bash
python scripts/train.py --config configs/train.yaml
```

### 3. 推理测试

```bash
python scripts/inference.py --input image.jpg --output corrected.jpg
```

### 4. 启动Web界面

```bash
cd web
pnpm install
pnpm dev
```

## 数据集字段说明

### data字典字段：
- `DESCR`: 数据集描述
- `images`: 图像数据
- `target_model_ids`: 目标模型ID
- `target_modeltype_ids`: 目标模型类型ID
- `target_segmentations`: 目标分割掩膜
- `model_shapes`: 模型形状

### metadata列名：
- `bg_name`: 背景名称
- `bg_id`: 背景ID
- `model_name`: 模型名称
- `model_id`: 模型ID
- `modeltype_name`: 模型类型名称
- `modeltype_id`: 模型类型ID
- `model_subid`: 模型子ID
- `image_path`: 图像路径
- `frame_index`: 帧索引
- `model_width`: 模型宽度
- `model_height`: 模型高度
- `tl_x`, `tl_y`: 左上角坐标
- `bl_x`, `bl_y`: 左下角坐标
- `br_x`, `br_y`: 右下角坐标
- `tr_x`, `tr_y`: 右上角坐标

## 性能指标

- 文档区域 IoU ≥ 0.85
- 四点 NME ≤ 0.03

## 技术栈

- **深度学习框架**: PyTorch
- **文档检测**: DocAligner
- **Web框架**: Next.js 16 + React 19
- **UI组件**: shadcn/ui
- **语言**: Python 3.10+, TypeScript 5
