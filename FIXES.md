# 修复说明

## 已修复的问题

### 1. "argument of type 'NoneType' is not iterable" 错误

**原因：**
- 图像读取失败时返回None，后续代码尝试对None进行操作
- 模型检查点文件不存在时未进行验证
- JSON数据中可能缺少预期的字段

**修复内容：**

#### a) scripts/inference.py
- 添加图像读取失败的检查
- 添加模型检查点文件存在性验证
- 改进错误处理

```python
# 修复前
image = cv2.imread(input_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 修复后
image = cv2.imread(input_path)
if image is None:
    raise ValueError(f"Failed to read image from {input_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

#### b) web/src/app/api/correct/route.ts
- 添加JSON数据的安全检查
- 改进API错误响应处理
- 修正路径计算问题

```typescript
// 修复前
corners = cornersJson.corners.map((c: number[]) => ({...}));

// 修复后
if (cornersJson && cornersJson.corners && Array.isArray(cornersJson.corners)) {
  corners = cornersJson.corners.map((c: number[]) => ({...}));
}
```

#### c) web/src/app/page.tsx
- 添加API响应验证
- 改进错误显示
- 添加模拟推理提示

### 2. 新增功能

#### 模拟推理脚本

创建了 `scripts/test_mock_inference.py`，用于在模型未训练时测试Web界面：
- 自动检测文档边界（模拟）
- 简单裁剪矫正
- 生成角点标注
- 生成可视化结果

#### 自动模式切换

API现在会自动检测是否有训练好的模型：
- 有模型 → 使用真实推理
- 无模型 → 使用模拟推理（演示模式）

### 3. 测试验证

已创建测试图像并验证：
- 原始测试文档：test_document_original.jpg
- 矫正结果：test_document_corrected.jpg
- 角点标注：test_document_corrected.json
- 可视化：test_document_corrected_vis.jpg

## 使用方法

### 方式1：使用真实模型（需要先训练）

```bash
# 1. 准备数据
python scripts/prepare_data.py

# 2. 训练模型
python scripts/train.py --config configs/train.yaml

# 3. 使用Web界面
cd web && pnpm dev
```

### 方式2：使用模拟推理（演示模式）

```bash
# 直接启动Web界面（会自动使用模拟推理）
cd web && pnpm dev
```

Web界面会自动检测并选择：
- ✓ 如果存在 `outputs/docaligner_20240115/checkpoint_best.pth` → 真实推理
- ℹ️ 如果不存在 → 模拟推理（显示提示）

### 方式3：命令行测试

```bash
# 使用模拟推理
python scripts/test_mock_inference.py \
    --input test_document_original.jpg \
    --output corrected.jpg \
    --save-visualization \
    --save-corners

# 使用真实推理（需要先训练模型）
python scripts/inference.py \
    --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
    --input test_document_original.jpg \
    --output corrected.jpg \
    --save-visualization \
    --save-corners
```

## Web界面状态

✅ **服务运行中**: http://localhost:5000

### 功能特性

- ✅ 图片上传与预览
- ✅ 实时文档矫正（自动选择模式）
- ✅ 角点可视化
- ✅ 矫正结果下载
- ✅ 错误处理与提示
- ✅ 模拟推理支持

### 界面元素

1. **上传图片**: 点击按钮选择文档图片
2. **显示检测角点**: 切换开关控制是否显示角点
3. **开始矫正**: 处理图片
4. **矫正结果**: 显示矫正后的文档
5. **下载**: 下载矫正后的图片

## 错误处理

### 常见错误及解决方案

#### 1. "Failed to read image"
**原因**: 图像文件损坏或格式不支持
**解决**: 尝试其他图像文件

#### 2. "Checkpoint file not found"
**原因**: 模型未训练
**解决**: 系统会自动使用模拟推理，或先训练模型

#### 3. "Processing failed"
**原因**: Python脚本执行失败
**解决**: 查看服务器日志或使用模拟推理模式

## 依赖安装

### 系统依赖（已安装）

```bash
apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
```

### Python依赖

```bash
pip install -r requirements.txt
```

关键依赖：
- opencv-python >= 4.8.0
- torch >= 2.0.0
- numpy >= 1.24.0
- Pillow >= 10.0.0

## 下一步

### 1. 快速测试
```bash
# 访问Web界面
# 上传 test_document_original.jpg
# 查看矫正结果
```

### 2. 训练真实模型
```bash
# 修改数据路径
python scripts/prepare_data.py

# 开始训练
python scripts/train.py --config configs/train.yaml

# 监控训练
tensorboard --logdir outputs/docaligner_20240115/logs
```

### 3. 评测性能
```bash
python scripts/evaluate.py \
    --checkpoint outputs/docaligner_20240115/checkpoint_best.pth \
    --data-root C:/Users/dengyu/smartdoc15_train_samples
```

## 技术细节

### 修复的文件

1. **scripts/inference.py**
   - 行 252-257: 添加图像读取验证
   - 行 48: 添加检查点验证

2. **web/src/app/api/correct/route.ts**
   - 行 99-108: 添加JSON数据安全检查
   - 行 82-91: 自动模式切换
   - 行 93: 设置工作目录

3. **web/src/app/page.tsx**
   - 行 19-23: 添加isMock字段
   - 行 165-169: 改进API响应处理
   - 行 185-194: 添加错误提示
   - 行 204-208: 添加使用提示
   - 行 252-258: 添加模拟推理提示

### 新增文件

1. **scripts/test_mock_inference.py**: 模拟推理脚本
2. **create_test_image.py**: 测试图像生成脚本

## 性能优化

- ✅ 自动检测模型，避免不必要的加载尝试
- ✅ 改进错误处理，提供清晰的错误消息
- ✅ 模拟推理用于快速演示
- ✅ 支持批量处理

## 支持的图像格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

## 联系支持

如有问题，请：
1. 查看浏览器控制台错误
2. 查看服务器日志
3. 检查Python依赖是否完整
4. 尝试使用模拟推理模式

---

**最后更新**: 2024-01-15
**状态**: ✅ 已修复并可正常运行
