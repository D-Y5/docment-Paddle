import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import cv2

class LightweightUNet(nn.Layer):
    """轻量级UNet结构用于文档区域分割"""
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(LightweightUNet, self).__init__()
        
        features = init_features
        self.encoder1 = LightweightUNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        self.encoder2 = LightweightUNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        self.encoder3 = LightweightUNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        self.encoder4 = LightweightUNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        self.bottleneck = LightweightUNet._block(features * 8, features * 16, name="bottleneck")
        
        self.upconv4 = nn.Conv2DTranspose(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = LightweightUNet._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.Conv2DTranspose(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = LightweightUNet._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.Conv2DTranspose(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = LightweightUNet._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.Conv2DTranspose(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = LightweightUNet._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv2D(in_channels=features, out_channels=out_channels, kernel_size=1)
    
    @staticmethod
    def _block(in_channels, features, name):
        """基本卷积块"""
        return nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(features),
            nn.ReLU(),
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(features),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """前向传播"""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = paddle.concat([dec4, enc4], axis=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = paddle.concat([dec3, enc3], axis=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = paddle.concat([dec2, enc2], axis=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = paddle.concat([dec1, enc1], axis=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)

class DocDetector(nn.Layer):
    """文档边界检测器"""
    
    def __init__(self, backbone='unet', input_size=(640, 640)):
        super(DocDetector, self).__init__()
        self.input_size = input_size
        
        if backbone == 'unet':
            self.backbone = LightweightUNet()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)
    
    def predict_corners(self, x):
        """预测四角点坐标"""
        batch_size = x.shape[0]
        all_corners = []
        
        # 获取分割结果
        outputs = self.forward(x)
        # 应用sigmoid激活函数
        masks = F.sigmoid(outputs)
        
        for b in range(batch_size):
            # 将掩码转换为numpy数组
            mask = masks[b, 0].numpy()
            # 阈值处理得到二值掩码
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # 提取四角点
            corners = self._extract_corners(binary_mask)
            all_corners.extend(corners)
        
        return all_corners
    
    def _extract_corners(self, binary_mask):
        """从二值掩码中提取四角点"""
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果没有找到轮廓，返回默认四角点
            h, w = binary_mask.shape
            margin = int(min(h, w) * 0.1)
            return [[margin, margin], [w - margin, margin], [w - margin, h - margin], [margin, h - margin]]
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 近似多边形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 如果近似结果是四边形，直接使用
        if len(approx) == 4:
            corners = [list(pt[0]) for pt in approx]
            # 排序四角点（左上、右上、右下、左下）
            corners = self._sort_corners(corners)
            return corners
        
        # 否则使用最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        corners = [list(pt) for pt in box]
        # 排序四角点
        corners = self._sort_corners(corners)
        return corners
    
    def _sort_corners(self, corners):
        """排序四角点为左上、右上、右下、左下"""
        # 计算中心点
        center = np.mean(corners, axis=0)
        
        # 按角度排序
        sorted_corners = []
        for corner in corners:
            angle = np.arctan2(corner[1] - center[1], corner[0] - center[0])
            sorted_corners.append((corner, angle))
        
        sorted_corners.sort(key=lambda x: x[1])
        corners = [list(corner[0]) for corner in sorted_corners]
        
        # 调整顺序为左上、右上、右下、左下
        # 找到最左上角的点
        top_left_idx = np.argmin([corner[0] + corner[1] for corner in corners])
        # 重新排列
        sorted_corners = corners[top_left_idx:] + corners[:top_left_idx]
        
        return sorted_corners
