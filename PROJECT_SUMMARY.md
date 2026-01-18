# SmartDoc 文档矫正系统 - 项目总结

## 项目概述

本项目实现了一个完整的基于深度学习的文档四边界定位与透视矫正系统，使用SmartDoc 2015 Challenge 1数据集训练DocAligner模型。

## 已完成的功能

### 1. 数据处理模块 ✓

**文件：** `scripts/prepare_data.py`

- 解析SmartDoc 2015数据集
- 生成训练样本（图像 + 四点标注）
- 支持从分割掩膜中提取角点
- 自动划分训练集/验证集
- 样本验证功能

**数据字段支持：**
- `DESCR`, `images`, `target_model_ids`, `target_modeltype_ids`, `target_segmentations`, `model_shapes`
- `bg_name`, `bg_id`, `model_name`, `model_id`, `modeltype_name`, `modeltype_id`, `model_subid`
- `image_path`, `frame_index`, `model_width`, `model_height`
- `tl_x`, `tl_y`, `tr_x`, `tr_y`, `br_x`, `br_y`, `bl_x`, `bl_y`

### 2. 模型训练模块 ✓

**文件：**
- `models/docaligner.py` - DocAligner模型定义
- `models/dataset.py` - 数据集加载器
- `models/loss.py` - 损失函数和评测指标
- `scripts/train.py` - 训练脚本
- `configs/train.yaml` - 训练配置

**模型特性：**
- 基于热力图的角点检测
- 支持base和offset两种模型
- Hourglass沙漏模块用于多尺度特征融合
- Focal Loss + MSE Loss混合损失
- 支持IoU和NME评测指标

### 3. 透视变换矫正模块 ✓

**文件：** `models/transform.py`

- 自动角点排序（左上、右上、右下、左下）
- 智能文档尺寸估计（保持宽高比）
- 透视变换矩阵计算
- 边缘填充与裁剪

### 4. 推理模块 ✓

**文件：** `scripts/inference.py`

- 单张图像推理
- 批量图像推理
- 支持可视化输出
- 角点标注保存
- 命令行参数配置

### 5. Web界面 ✓

**文件：**
- `web/src/app/page.tsx` - 主页面
- `web/src/app/api/correct/route.ts` - API接口

**功能：**
- 图片上传与预览
- 一键文档矫正
- 角点可视化
- 矫正结果下载
- 响应式设计

**API接口：**
```
POST /api/correct
- 接收：图像文件 + 参数
- 返回：base64编码的矫正图像 + 角点坐标
```

### 6. 评测模块 ✓

**文件：** `scripts/evaluate.py`

- SmartDoc Challenge标准评测
- IoU（交并比）计算
- NME（归一化平均误差）计算
- 批量评测支持
- 结果可视化与导出

**性能指标：**
- 文档区域 IoU ≥ 0.85
- 四点 NME ≤ 0.03

### 7. 工具脚本 ✓

- `quick_start.sh` - Linux/Mac快速启动脚本
- `quick_start.bat` - Windows快速启动脚本
- `requirements.txt` - Python依赖列表
- `USAGE.md` - 详细使用指南

## 项目结构

```
smartdoc-doc-alignment/
├── scripts/                    # 脚本目录
│   ├── prepare_data.py        # 数据准备
│   ├── train.py               # 训练脚本
│   ├── inference.py           # 推理脚本
│   └── evaluate.py            # 评测脚本
├── models/                     # 模型目录
│   ├── docaligner.py          # DocAligner模型
│   ├── dataset.py             # 数据集加载
│   ├── loss.py                # 损失函数
│   └── transform.py           # 透视变换
├── configs/                    # 配置目录
│   └── train.yaml             # 训练配置
├── web/                        # Web界面
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx       # 主页面
│   │   │   └── api/
│   │   │       └── correct/
│   │   │           └── route.ts # API接口
│   │   └── components/
│   │       └── ui/            # UI组件
│   └── package.json
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   └── train_samples/         # 训练样本
├── outputs/                    # 输出目录
│   └── docaligner_*/
│       ├── checkpoint_best.pth
│       └── logs/
├── requirements.txt            # Python依赖
├── USAGE.md                    # 使用指南
├── README.md                   # 项目说明
├── quick_start.sh             # 快速启动(Linux)
└── quick_start.bat            # 快速启动(Windows)
```

## 技术栈

### 后端（Python）
- **深度学习框架：** PyTorch 2.0+
- **计算机视觉：** OpenCV, Pillow
- **数据处理：** NumPy, scikit-learn
- **可视化：** TensorBoard, matplotlib
- **几何计算：** Shapely（IoU计算）

### 前端（Web）
- **框架：** Next.js 16 + React 19
- **语言：** TypeScript 5
- **UI组件：** shadcn/ui
- **样式：** Tailwind CSS 4
- **图标：** Lucide React

## 核心算法

### 1. 角点检测算法

```
输入图像 → 骨干网络特征提取 → Hourglass多尺度融合 → 热力图生成 → 角点提取
```

**关键技术：**
- Hourglass模块用于捕获多尺度特征
- Focal Loss处理类别不平衡
- 高斯热力图编码角点位置

### 2. 角点提取算法

```
热力图 → 寻找最大值位置 → 子像素精化 → 角点排序
```

**关键技术：**
- 最大响应点定位
- 角点排序（左上→右上→右下→左下）

### 3. 透视变换算法

```
四个角点 → 计算透视变换矩阵 → 文档尺寸估计 → 执行变换
```

**关键技术：**
- OpenCV getPerspectiveTransform
- 智能宽高比保持
- 边缘填充

## 使用流程

### 1. 数据准备

```bash
# 修改路径后运行
python scripts/prepare_data.py --verify
```

### 2. 模型训练

```bash
python scripts/train.py --config configs/train.yaml
```

监控训练：
```bash
tensorboard --logdir outputs/docaligner_20240115/logs
```

### 3. 推理测试

```bash
# 单张图像
python scripts/inference.py \
    --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
    --input test.jpg \
    --output corrected.jpg

# 批量处理
python scripts/inference.py \
    --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
    --input input_dir/ \
    --output output_dir/
```

### 4. Web界面使用

```bash
cd web
pnpm dev
# 访问 http://localhost:5000
```

### 5. 性能评测

```bash
python scripts/evaluate.py \
    --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
    --data-root C:/Users/dengyu/smartdoc15_train_samples
```

## 性能目标与评估

### 目标指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| IoU | ≥ 0.85 | 文档区域交并比 |
| NME | ≤ 0.03 | 归一化平均误差 |

### 评测方法

1. **IoU计算**
   - 使用Shapely库计算多边形交并比
   - 支持任意四边形

2. **NME计算**
   - 计算角点与真实标注的欧氏距离
   - 使用对角线长度归一化

3. **评测流程**
   - 在完整验证集上运行推理
   - 计算平均IoU和NME
   - 统计达标率

## 扩展功能建议

### 短期优化

1. **角点微调**
   - 添加手动调整角点功能
   - 支持拖拽调整

2. **批量导出**
   - 支持导出为PDF
   - 批量处理优化

3. **模型优化**
   - 模型量化
   - ONNX导出

### 长期规划

1. **视频处理**
   - 支持视频帧提取
   - 实时视频矫正

2. **移动端部署**
   - 转换为TensorFlow Lite
   - 移动端App开发

3. **API服务化**
   - RESTful API
   - Docker容器化部署

## 常见问题解决

### Q1: 数据集路径问题

修改`scripts/prepare_data.py`中的路径：
```python
DATA_ROOT = "C:/Users/dengyu/smartdoc15-ch1-pywrapper"
OUTPUT_ROOT = "C:/Users/dengyu/smartdoc15_train_samples"
```

### Q2: 显存不足

减小batch_size或输入尺寸：
```yaml
batch_size: 4
input_size: [256, 256]
```

### Q3: Web界面无法调用Python

检查：
1. Python环境是否正确安装
2. 模型检查点路径是否正确
3. 查看浏览器控制台错误

### Q4: IoU未达标

尝试：
1. 增加训练epoch
2. 调整学习率
3. 使用offset模型
4. 增加数据增强

## 项目亮点

1. **完整的端到端解决方案**
   - 从数据准备到模型训练再到推理部署
   - 包含完整的评测体系

2. **用户友好的Web界面**
   - 直观的操作界面
   - 实时预览与下载
   - 响应式设计

3. **模块化设计**
   - 代码结构清晰
   - 易于扩展和维护
   - 详细的注释和文档

4. **性能优化**
   - 批量推理支持
   - GPU加速
   - 高效的数据加载

5. **评测标准**
   - 遵循SmartDoc Challenge标准
   - 多维度性能指标
   - 可视化结果展示

## 参考资源

1. **SmartDoc 2015 Challenge**
   - 官网: http://www.itl.nist.gov/iad/mig/tests/smartdoc2015/
   - 数据集下载与说明

2. **DocAligner**
   - GitHub: https://github.com/DocsaidLab/DocAligner
   - 论文与模型架构

3. **技术博客与教程**
   - PyTorch官方文档
   - OpenCV文档
   - Next.js文档

## 致谢

感谢以下开源项目和社区的支持：
- SmartDoc 2015 Challenge组织者
- DocAligner作者
- PyTorch社区
- shadcn/ui组件库

## 许可证

本项目遵循MIT许可证。

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。

---

**项目状态：** 已完成 ✓
**最后更新：** 2024-01-15
**版本：** 1.0.0
