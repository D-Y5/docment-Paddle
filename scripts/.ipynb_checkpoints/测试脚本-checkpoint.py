import paddle
import numpy as np
from models.docaligner import create_docaligner_model

# 创建模型
model = create_docaligner_model(model_type="base", num_corners=4, pretrained=False)
model.eval()

# 随机输入
x = paddle.randn(2, 3, 512, 512)
with paddle.no_grad():
    out = model(x)

# 打印数值
print(f"输出最大值: {out['heatmap'].max().item():.6f}")
print(f"输出最小值: {out['heatmap'].min().item():.6f}")
print(f"输出均值: {out['heatmap'].mean().item():.6f}")

# 检查角点提取
pred_corners = model._extract_corners_from_heatmap(out["heatmap"])
print(f"预测角点:\n{pred_corners}")