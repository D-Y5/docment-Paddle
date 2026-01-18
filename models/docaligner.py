import paddle

"""
DocAligner: 基于热力图的文档四点检测模型
参考: https://github.com/DocsaidLab/DocAligner
"""
from typing import Dict, Tuple


class HourglassBlock(paddle.nn.Layer):
    """沙漏模块 - 用于多尺度特征融合"""

    def __init__(self, in_channels: int, out_channels: int):
        super(HourglassBlock, self).__init__()
        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            paddle.nn.BatchNorm2D(num_features=out_channels),
            paddle.nn.ReLU(),
        )
        self.down1 = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            paddle.nn.BatchNorm2D(num_features=out_channels),
            paddle.nn.ReLU(),
        )
        self.down2 = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            paddle.nn.BatchNorm2D(num_features=out_channels),
            paddle.nn.ReLU(),
        )
        self.up1 = paddle.nn.Sequential(
            paddle.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            paddle.nn.BatchNorm2D(num_features=out_channels),
            paddle.nn.ReLU(),
        )
        self.up2 = paddle.nn.Sequential(
            paddle.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            paddle.nn.BatchNorm2D(num_features=out_channels),
            paddle.nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = paddle.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x += x2
        x = self.up2(x)
        x = paddle.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x += x1
        return x


class DocAlignerBackbone(paddle.nn.Layer):
    """DocAligner骨干网络"""

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super(DocAlignerBackbone, self).__init__()
        self.stem = paddle.nn.Sequential(
            paddle.nn.Conv2d(
                in_channels, out_channels // 2, 7, stride=2, padding=3, bias=False
            ),
            paddle.nn.BatchNorm2D(num_features=out_channels // 2),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(out_channels // 2, out_channels, 3)
        self.layer2 = self._make_layer(out_channels, out_channels * 2, 4, stride=2)
        self.layer3 = self._make_layer(out_channels * 2, out_channels * 4, 6, stride=2)
        self.layer4 = self._make_layer(out_channels * 4, out_channels * 8, 3, stride=2)
        self.hourglass = HourglassBlock(out_channels * 8, out_channels * 8)

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int = 1
    ) -> paddle.nn.Sequential:
        """创建残差层"""
        layers = []
        layers.append(
            paddle.nn.Sequential(
                paddle.nn.Conv2d(
                    in_channels, out_channels, 3, stride, padding=1, bias=False
                ),
                paddle.nn.BatchNorm2D(num_features=out_channels),
                paddle.nn.ReLU(),
            )
        )
        for _ in range(1, blocks):
            layers.append(
                paddle.nn.Sequential(
                    paddle.nn.Conv2d(
                        out_channels, out_channels, 3, padding=1, bias=False
                    ),
                    paddle.nn.BatchNorm2D(num_features=out_channels),
                    paddle.nn.ReLU(),
                )
            )
        return paddle.nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.hourglass(x)
        return x


class CornerDetectionHead(paddle.nn.Layer):
    """角点检测头"""

    def __init__(self, in_channels: int, num_corners: int = 4):
        super(CornerDetectionHead, self).__init__()
        self.num_corners = num_corners
        self.upsample1 = paddle.nn.Sequential(
            paddle.nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            paddle.nn.BatchNorm2D(num_features=in_channels // 2),
            paddle.nn.ReLU(),
        )
        self.upsample2 = paddle.nn.Sequential(
            paddle.nn.Conv2d(
                in_channels // 2, in_channels // 4, 3, padding=1, bias=False
            ),
            paddle.nn.BatchNorm2D(num_features=in_channels // 4),
            paddle.nn.ReLU(),
        )
        self.heatmap = paddle.nn.Conv2d(in_channels // 4, num_corners, 1)

    def forward(self, x):
        x = self.upsample1(x)
        x = paddle.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = self.upsample2(x)
        x = paddle.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        heatmap = self.heatmap(x)
        return heatmap


class DocAligner(paddle.nn.Layer):
    """
    DocAligner: 文档四点检测模型

    输入: 图像 (B, 3, H, W)
    输出: 热力图 (B, 4, H/4, W/4)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_corners: int = 4,
        backbone_channels: int = 64,
        pretrained: bool = False,
    ):
        super(DocAligner, self).__init__()
        self.num_corners = num_corners
        self.backbone = DocAlignerBackbone(in_channels, backbone_channels)
        self.head = CornerDetectionHead(backbone_channels * 8, num_corners)
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, paddle.nn.Conv2d):
                paddle.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    paddle.nn.init.constant_(m.bias, 0)
            elif isinstance(m, paddle.nn.BatchNorm2D):
                paddle.nn.init.constant_(m.weight, 1)
                paddle.nn.init.constant_(m.bias, 0)

    def forward(self, x: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            output: 包含热力图的字典
                - heatmap: 角点热力图 (B, 4, H/4, W/4)
        """
        features = self.backbone(x)
        heatmap = self.head(features)
        return {"heatmap": heatmap}


class DocAlignerWithOffset(DocAligner):
    """
    DocAligner with Offset Prediction
    除了热力图，还预测角点的偏移量
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_corners: int = 4,
        backbone_channels: int = 64,
        pretrained: bool = False,
    ):
        super().__init__(in_channels, num_corners, backbone_channels, pretrained)
        self.offset_head = paddle.nn.Sequential(
            paddle.nn.Conv2d(
                backbone_channels * 8, backbone_channels // 2, 3, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2d(backbone_channels // 2, num_corners * 2, 1),
        )

    def forward(self, x: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        """
        前向传播

        Returns:
            output: 包含热力图和偏移量的字典
                - heatmap: 角点热力图 (B, 4, H/4, W/4)
                - offset: 角点偏移量 (B, 8, H/4, W/4)
        """
        features = self.backbone(x)
        heatmap = self.head(features)
        offset = self.offset_head(features)
        return {"heatmap": heatmap, "offset": offset}


def create_docaligner_model(
    model_type: str = "base", num_corners: int = 4, pretrained: bool = False
) -> paddle.nn.Layer:
    """
    创建DocAligner模型

    Args:
        model_type: 模型类型 ("base" or "offset")
        num_corners: 角点数量
        pretrained: 是否使用预训练权重

    Returns:
        model: DocAligner模型
    """
    if model_type == "base":
        model = DocAligner(num_corners=num_corners, pretrained=pretrained)
    elif model_type == "offset":
        model = DocAlignerWithOffset(num_corners=num_corners, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


if __name__ == "__main__":
    device = paddle.device("cuda" if paddle.cuda.is_available() else "cpu")
    model = create_docaligner_model(model_type="base")
    model = model.to(device)
    x = paddle.randn(2, 3, 512, 512).to(device)
    with paddle.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Heatmap shape: {output['heatmap'].shape}")
    total_params = sum(p.size for p in model.parameters())
    trainable_params = sum(p.size for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
