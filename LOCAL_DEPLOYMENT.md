# 本地部署指南

本指南帮助你在本地环境部署 SmartDoc 文档矫正系统。

## 系统要求

### 硬件要求
- **CPU**: 4核心以上
- **内存**: 8GB 以上（建议16GB）
- **GPU**: NVIDIA GPU（可选，用于加速训练）
  - CUDA 11.8+
  - 显存 ≥ 8GB

### 软件要求
- **操作系统**: Windows 10/11, macOS, Linux (Ubuntu 20.04+)
- **Python**: 3.10 或更高版本
- **Node.js**: 18+ (用于Web界面)
- **包管理器**: pnpm (推荐)

---

## 部署步骤

### 第一步：克隆项目

```bash
git clone <repository-url>
cd <project-directory>
```

### 第二步：安装 Python 依赖

#### 2.1 创建虚拟环境（推荐）

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2.2 安装 Python 包

```bash
pip install -r requirements.txt
```

**重要提示：**
- 如果你没有 NVIDIA GPU，PyTorch 会自动安装 CPU 版本
- 如果需要 GPU 支持，请访问 [PyTorch官网](https://pytorch.org/) 选择适合你的 CUDA 版本

### 第三步：准备数据集

#### 3.1 下载 SmartDoc 2015 数据集

方法1：使用 sklearn 自动下载
```bash
# 确保数据集会下载到正确路径
python -c "from sklearn.datasets import fetch_mldata; fetch_mldata('SmartDoc15-Ch1', data_home='./data')"
```

方法2：手动下载并解压
- 下载地址：[SmartDoc 2015 Challenge 1](https://github.com/CanPacis/smartdoc15-ch1-pywrapper)
- 解压到：`C:/Users/dengyu/scikit_learn_data/smartdoc15-ch1` (Windows)
- 或：`~/scikit_learn_data/smartdoc15-ch1` (macOS/Linux)

#### 3.2 生成训练样本

```bash
python scripts/prepare_data.py
```

这将生成：
- 训练图像：`C:/Users/dengyu/smartdoc15_train_samples/images/`
- 标注文件：`C:/Users/dengyu/smartdoc15_train_samples/annotations/`

### 第四步：训练模型（可选）

如果你已经有训练好的模型，可以跳过此步骤。

```bash
python scripts/train.py --config configs/train.yaml
```

训练参数：
- **Epochs**: 100
- **Batch Size**: 8
- **学习率**: 0.001
- **输出目录**: `outputs/docaligner_20240115/`

训练过程中会每 10 个 epoch 保存一次检查点。

### 第五步：部署 Web 界面

#### 5.1 进入 Web 目录

```bash
cd web
```

#### 5.2 安装 Node.js 依赖

```bash
# 安装 pnpm（如果尚未安装）
npm install -g pnpm

# 安装项目依赖
pnpm install
```

#### 5.3 启动开发服务器

```bash
pnpm dev
```

默认访问地址：`http://localhost:5000`

#### 5.4 构建生产版本（可选）

```bash
# 构建生产版本
pnpm build

# 启动生产服务器
pnpm start
```

---

## 使用方法

### 1. 通过 Web 界面使用

1. 打开浏览器访问 `http://localhost:5000`
2. 上传待矫正的文档图片
3. 点击"开始矫正"按钮
4. 系统会自动检测文档四边界并进行透视矫正
5. 可以下载矫正后的结果（支持 JPEG/PDF 格式）

### 2. 通过 Python 脚本使用

**单张图片推理：**
```bash
python scripts/inference.py --input test.jpg --output corrected.jpg
```

**批量推理：**
```bash
python scripts/inference.py --input-dir ./test_images --output-dir ./results
```

### 3. 模型评测

```bash
python scripts/evaluate.py
```

---

## 目录结构

```
smartdoc-alignment/
├── data/                          # 数据目录
│   ├── raw/                      # 原始数据集
│   └── train_samples/            # 训练样本
├── models/                        # 模型代码
│   ├── docaligner.py             # DocAligner 模型
│   ├── loss.py                   # 损失函数
│   ├── transform.py              # 数据变换
│   └── dataset.py                # 数据集类
├── scripts/                       # 脚本
│   ├── prepare_data.py           # 数据准备
│   ├── train.py                  # 训练脚本
│   ├── inference.py              # 推理脚本
│   └── evaluate.py               # 评测脚本
├── configs/                       # 配置文件
│   └── train.yaml                # 训练配置
├── web/                          # Web 界面
│   ├── src/app/                   # Next.js App Router
│   ├── src/components/           # React 组件
│   └── package.json              # Node.js 依赖
├── checkpoints/                   # 模型检查点
├── outputs/                       # 训练输出
└── requirements.txt               # Python 依赖
```

---

## 常见问题

### Q1: PyTorch 安装失败怎么办？

**A:** 
- Windows: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- macOS (M1/M2): `pip install torch torchvision`
- Linux: `pip install torch torchvision`

参考：[PyTorch 安装指南](https://pytorch.org/get-started/locally/)

### Q2: 没有GPU可以训练吗？

**A:** 可以，但速度会很慢。建议：
1. 减小 batch_size（改为 2 或 4）
2. 减少输入图像尺寸
3. 使用云端 GPU（如 Google Colab, Kaggle）

### Q3: 数据集下载失败怎么办？

**A:**
1. 手动从 GitHub 下载：[smartdoc15-ch1-pywrapper](https://github.com/CanPacis/smartdoc15-ch1-pywrapper)
2. 解压到 `scikit_learn_data/smartdoc15-ch1` 目录
3. 确保目录结构正确

### Q4: Web 界面启动失败？

**A:**
1. 检查 Node.js 版本：`node --version`（需要 >= 18）
2. 清除缓存：`rm -rf node_modules && pnpm install`
3. 检查端口 5000 是否被占用

### Q5: 推理速度慢怎么办？

**A:**
1. 使用 GPU 加速
2. 减小输入图像尺寸
3. 批量处理图片

### Q6: 如何导出为其他格式？

**A:** 修改 `scripts/inference.py` 或 Web 界面的导出逻辑：
- JPEG: 使用 OpenCV 的 `cv2.imwrite`
- PDF: 使用 `img2pdf` 库
- PNG: 使用 `cv2.imwrite` 或 PIL

---

## 性能优化建议

### 1. 训练优化
- 使用混合精度训练（`torch.cuda.amp`）
- 增加数据加载线程数（`num_workers`）
- 使用 SSD 存储数据集

### 2. 推理优化
- 使用 ONNX 格式导出模型
- 使用 TensorRT 加速（NVIDIA GPU）
- 批量推理

### 3. 部署优化
- 使用 Docker 容器化
- 使用 Nginx 反向代理
- 启用 CDN 加速静态资源

---

## 离线部署

如果需要离线部署：

### 1. 预下载依赖

```bash
# Python 依赖
pip download -r requirements.txt -d ./python_packages

# Node.js 依赖
pnpm offline-generate
```

### 2. 使用本地安装

```bash
# 安装 Python 包（离线）
pip install --no-index --find-links=./python_packages -r requirements.txt

# 安装 Node.js 包（离线）
pnpm install --offline
```

---

## 技术支持

如遇到问题，请：
1. 查看日志文件：`outputs/docaligner_20240115/logs/`
2. 检查错误信息并搜索解决方案
3. 提交 Issue 到项目仓库

---

## 许可证

本项目遵循相关开源许可证。SmartDoc 2015 数据集请遵循其使用条款。
