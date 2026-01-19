import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Dict

class DocAligner(nn.Layer):
    """
    修改后的DocAligner模型：
    - 直接回归四个角点坐标(8个值)
    - 添加几何约束损失
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2D(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2),
            nn.Conv2D(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2),
            nn.Conv2D(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2),
            nn.Conv2D(256, 512, 3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512*8*8, 1024),  # 假设输入512x512，经过3次下采样后为64x64
            nn.ReLU(),
            nn.Linear(1024, 8)  # 输出8个值(4个角点x,y坐标)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = paddle.flatten(features, 1)
        corners = self.fc(features)
        return {
            "corners": corners.reshape(-1, 4, 2),  # [B,4,2]
            "features": features
        }

def geometric_constraint_loss(pred_corners):
    """
    几何约束损失：
    1. 保证四边形凸性
    2. 惩罚自交四边形
    """
    # 计算四条边向量
    edge1 = pred_corners[:, 1] - pred_corners[:, 0]
    edge2 = pred_corners[:, 2] - pred_corners[:, 1] 
    edge3 = pred_corners[:, 3] - pred_corners[:, 2]
    edge4 = pred_corners[:, 0] - pred_corners[:, 3]
    
    # 计算相邻边叉积判断凸性
    cross1 = paddle.cross(edge1, edge2)
    cross2 = paddle.cross(edge2, edge3)
    cross3 = paddle.cross(edge3, edge4)
    cross4 = paddle.cross(edge4, edge1)
    
    # 凸性损失(所有叉积应同号)
    convex_loss = F.relu(-cross1*cross2) + F.relu(-cross2*cross3) + F.relu(-cross3*cross4)
    
    return convex_loss.mean()

class DocAlignerLoss(nn.Layer):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha  # 几何约束权重
        
    def forward(self, pred, target):
        # 坐标回归损失
        coord_loss = F.mse_loss(pred["corners"], target["corners"])
        
        # 几何约束损失
        geo_loss = geometric_constraint_loss(pred["corners"])
        
        return coord_loss + self.alpha * geo_loss
