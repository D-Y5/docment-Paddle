import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class LightweightUNet(nn.Layer):
    """轻量级UNet结构用于文档四角点热力图回归"""
    
    def __init__(self, in_channels=3, out_channels=4, init_features=32):
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
        heatmaps = self.forward(x)
        corners = []
        
        # 从每个热力图中提取峰值点
        for i in range(4):
            heatmap = heatmaps[:, i, :, :]
            # 使用最大值池化找到峰值点
            max_pool = F.max_pool2D(heatmap.unsqueeze(1), kernel_size=3, stride=1, padding=1)
            peak_mask = paddle.equal(heatmap.unsqueeze(1), max_pool).cast('float32')
            peak_heatmap = heatmap.unsqueeze(1) * peak_mask
            
            # 计算峰值点坐标
            batch_size = x.shape[0]
            for b in range(batch_size):
                # 找到最大值的位置
                max_val = paddle.max(peak_heatmap[b])
                if max_val > 0:
                    # 获取坐标
                    coords = paddle.where(peak_heatmap[b] == max_val)
                    y, x_coord = coords[1][0].item(), coords[2][0].item()
                    # 归一化到输入图像尺寸
                    y = y / heatmap.shape[1] * self.input_size[0]
                    x_coord = x_coord / heatmap.shape[2] * self.input_size[1]
                    corners.append([x_coord, y])
                else:
                    # 如果没有找到峰值点，返回默认值
                    corners.append([0, 0])
        
        return corners
