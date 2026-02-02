# 文档四边界定位与透视矫正系统

本系统基于深度学习实现文档四边界定位与透视矫正，支持从拍摄的文档图像中自动检测四角点并生成平展的文档图像。

## 系统架构

```
smartdoc15-ch1/
├── frames/            # 视频帧数据
├── models/            # 静态文档图像
├── work/
│   ├── train_samples/  # 训练样本
│   ├── val_samples/    # 验证样本
│   └── models/         # 训练模型
├── generate_samples.py  # 训练样本生成脚本
├── model.py            # 模型定义
├── train.py            # 训练脚本
├── train.yaml          # 训练配置文件
├── align.py            # 文档对齐脚本
└── evaluate.py         # 评测脚本
```

## 环境配置

本系统使用飞桨（PaddlePaddle）框架实现，推荐在飞桨平台训练。环境配置要求：

- GPU: Tesla V100 (16GB Video Memory)
- CPU: 2 Cores
- RAM: 16GB
- Disk: 100GB
- PaddlePaddle: 2.4.0+

### 安装依赖

使用 requirements.txt 安装所需依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

1. **数据集下载**
   - 数据集已下载，位于 `smartdoc15-ch1/` 目录下
   - 包含 `frames/`（视频帧）和 `models/`（静态图像）两个目录

2. **生成训练样本（包含静态和动态数据）**

   ```bash
   # 处理静态文档图像（models目录）
   python generate_samples.py --data_root models --output_root work/train_samples

   # 处理动态视频帧（frames目录）
   python generate_samples.py --data_root frames --output_root work/train_samples
   ```

3. **生成验证样本（包含静态和动态数据）**

   ```bash
   # 处理静态文档图像（models目录）
   python generate_samples.py --data_root models --output_root work/val_samples

   # 处理动态视频帧（frames目录）
   python generate_samples.py --data_root frames --output_root work/val_samples
   ```

## 模型训练

1. **修改配置文件**
   编辑 `train.yaml` 文件，设置训练参数：

   ```yaml
   # 数据集配置
   dataset:
     train_root: "work/train_samples"
     val_root: "work/val_samples"
     image_size: [640, 640]
     batch_size: 8

   # 训练配置
   train:
     epochs: 100
     learning_rate: 0.001
     save_dir: "work/models"
   ```

2. **启动训练**
   ```bash
   python train.py --config train.yaml
   ```

## 文档对齐

使用训练好的模型对齐文档：

```bash
python align.py --model work/model_epoch_100.pdparams --input input_image.jpg --output output_aligned.jpg --format jpeg
```

参数说明：
- `--model`: 训练好的模型路径
- `--input`: 输入图像路径
- `--output`: 输出对齐后图像路径
- `--format`: 输出格式（jpeg 或 pdf）

## 模型评测

评估模型性能：

```bash
python evaluate.py --model work/model_epoch_100.pdparams --test_root work/val_samples
```

评测指标：

- **文档区域 IoU**: 目标 ≥ 0.85
- **四点 NME**: 目标 ≤ 0.03

## 模型架构

本系统使用轻量级 UNet 架构实现文档四角点热力图回归：

1. **编码器**：4 个卷积块，逐步下采样提取特征
2. **解码器**：4 个反卷积块，逐步上采样恢复空间分辨率
3. **输出层**：4 通道热力图，对应文档四角点

## 性能优化

1. **数据增强**：训练时使用亮度、对比度、饱和度、旋转等数据增强
2. **学习率调度**：可在训练脚本中添加学习率调度器
3. **模型量化**：训练完成后可使用飞桨的模型量化工具优化推理速度

## 常见问题

1. **模型预测不准确**
   - 检查训练数据是否足够
   - 增加训练轮数
   - 调整学习率和 batch size

2. **透视变换失败**
   - 确保四角点排序正确
   - 检查输入图像质量

3. **内存不足**
   - 减小 batch size
   - 降低输入图像尺寸

## 后续工作

1. **Web 界面**：实现基于 Flask 或 Streamlit 的 Web 界面
2. **实时处理**：优化模型以支持实时视频流处理
3. **多语言支持**：扩展模型以支持多语言文档
4. **文档内容识别**：集成 OCR 功能，实现文档内容识别
