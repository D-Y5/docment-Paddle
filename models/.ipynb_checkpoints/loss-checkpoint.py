import paddle
import paddle.nn.functional as F

"""
损失函数：用于训练DocAligner模型
"""
from typing import Dict, Optional


class FocalLoss(paddle.nn.Layer):
    """Focal Loss for heatmap regression"""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        """
        Args:
            alpha: 调节难易样本权重
            beta: 调节正负样本权重
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        pred: paddle.Tensor,
        target: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Args:
            pred: 预测热力图 (B, K, H, W)
            target: 目标热力图 (B, K, H, W)
            mask: 掩膜 (B, K, H, W), 用于标记有效区域

        Returns:
            loss: Focal loss
        """
        pred = F.sigmoid(pred)  # 核心修复：约束值域
        pos_mask = target > 0
        neg_mask = target <= 0
        loss = (
            -self.alpha
            * (target - pred).abs() ** self.beta
            * (
                pos_mask.float() * paddle.log(pred.clip(min=1e-07))
                + neg_mask.float() * paddle.log((1 - pred).clip(min=1e-07))
            )
        )
        if mask is not None:
            loss = loss * mask
        return loss.mean()


class HeatmapLoss(paddle.nn.Layer):
    """热力图损失函数 - MSE + Focal"""

    def __init__(
        self,
        mse_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ):
        super(HeatmapLoss, self).__init__()
        self.mse_weight = mse_weight
        self.focal_weight = focal_weight
        self.mse_loss = paddle.nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, beta=focal_beta)

    def forward(
        self,
        pred: paddle.Tensor,
        target: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Args:
            pred: 预测热力图 (B, K, H, W)
            target: 目标热力图 (B, K, H, W)
            mask: 掩膜 (B, K, H, W)

        Returns:
            loss: 总损失
        """
        mse_loss = self.mse_loss(pred, target)
        focal_loss = self.focal_loss(pred, target, mask)
        total_loss = self.mse_weight * mse_loss + self.focal_weight * focal_loss
        return total_loss


class CornerLoss(paddle.nn.Layer):
    """角点坐标损失函数 - L1 loss"""

    def __init__(self, reduction: str = "mean"):
        super(CornerLoss, self).__init__()
        self.l1_loss = paddle.nn.L1Loss(reduction=reduction)

    def forward(
        self,
        pred_corners: paddle.Tensor,
        target_corners: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Args:
            pred_corners: 预测角点 (B, 4, 2)
            target_corners: 目标角点 (B, 4, 2)
            mask: 角点有效性mask (B, 4)

        Returns:
            loss: 角点损失
        """
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            loss = self.l1_loss(pred_corners * mask, target_corners * mask)
        else:
            loss = self.l1_loss(pred_corners, target_corners)
        return loss


class OffsetLoss(paddle.nn.Layer):
    """偏移量损失函数"""

    def __init__(self, reduction: str = "mean"):
        super(OffsetLoss, self).__init__()
        self.l1_loss = paddle.nn.L1Loss(reduction=reduction)

    def forward(
        self,
        pred_offset: paddle.Tensor,
        target_offset: paddle.Tensor,
        heatmap: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Args:
            pred_offset: 预测偏移量 (B, 8, H, W)
            target_offset: 目标偏移量 (B, 8, H, W)
            heatmap: 热力图，用于加权 (B, 4, H, W)

        Returns:
            loss: 偏移量损失
        """
        if heatmap is not None:
            weights = heatmap.view(heatmap.shape[0], -1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-06)
            loss = (
                self.l1_loss(pred_offset, target_offset, reduction="none")
                * weights.unsqueeze(1)
            ).sum()
        else:
            loss = self.l1_loss(pred_offset, target_offset)
        return loss


class DocAlignerLoss(paddle.nn.Layer):
    """
    DocAligner 总损失函数
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 0.1,
        use_focal: bool = True,
    ):
        super(DocAlignerLoss, self).__init__()
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.use_focal = use_focal
        if use_focal:
            self.heatmap_loss = HeatmapLoss(mse_weight=1.0, focal_weight=1.0)
        else:
            self.heatmap_loss = paddle.nn.MSELoss()
        self.offset_loss = OffsetLoss()

    def forward(
        self, pred: Dict[str, paddle.Tensor], target: Dict[str, paddle.Tensor]
    ) -> Dict[str, paddle.Tensor]:
        """
        Args:
            pred: 预测结果
                - heatmap: 预测热力图
                - offset: 预测偏移量（可选）
            target: 目标结果
                - heatmap: 目标热力图
                - offset: 目标偏移量（可选）

        Returns:
            losses: 损失字典
                - total: 总损失
                - heatmap: 热力图损失
                - offset: 偏移量损失（如果存在）
        """
        losses = {}
        heatmap_loss = self.heatmap_loss(pred["heatmap"], target["heatmap"])
        losses["heatmap"] = heatmap_loss
        if "offset" in pred and "offset" in target:
            offset_loss = self.offset_loss(
                pred["offset"], target["offset"], pred["heatmap"]
            )
            losses["offset"] = offset_loss
            losses["total"] = (
                self.heatmap_weight * heatmap_loss + self.offset_weight * offset_loss
            )
        else:
            losses["total"] = self.heatmap_weight * heatmap_loss
        return losses


def compute_iou(
    pred_corners: paddle.Tensor, target_corners: paddle.Tensor
) -> paddle.Tensor:
    """
    计算IoU (Intersection over Union)

    Args:
        pred_corners: 预测角点 (B, 4, 2)
        target_corners: 目标角点 (B, 4, 2)

    Returns:
        iou: IoU值 (B,)
    """
    batch_size = pred_corners.shape[0]
    pred_area = polygon_area(pred_corners)
    target_area = polygon_area(target_corners)
    iou_values = []
    for i in range(batch_size):
        try:
            from shapely.geometry import Polygon

            pred_poly = Polygon(pred_corners[i].cpu().numpy())
            target_poly = Polygon(target_corners[i].cpu().numpy())
            intersection = pred_poly.intersection(target_poly).area
            union = pred_poly.union(target_poly).area
            iou_value = intersection / (union + 1e-06)
            iou_values.append(iou_value)
        except:
            iou_values.append(0.0)
    return paddle.tensor(iou_values, device=pred_corners.device)


def polygon_area(corners: paddle.Tensor) -> paddle.Tensor:
    """
    计算多边形面积（Shoelace公式）

    Args:
        corners: 角点 (B, 4, 2)

    Returns:
        area: 面积 (B,)
    """
    batch_size = corners.shape[0]
    areas = []
    for i in range(batch_size):
        pts = corners[i]
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * paddle.abs(
            paddle.dot(x, paddle.roll(y, -1)) - paddle.dot(y, paddle.roll(x, -1))
        )
        areas.append(area)
    return paddle.stack(areas)


def compute_nme(
    pred_corners: paddle.Tensor, target_corners: paddle.Tensor
) -> paddle.Tensor:
    """
    计算归一化平均误差 (Normalized Mean Error)

    Args:
        pred_corners: 预测角点 (B, 4, 2)
        target_corners: 目标角点 (B, 4, 2)

    Returns:
        nme: NME值 (B,)
    """
    distance = paddle.norm(pred_corners - target_corners, dim=-1)
    tl = target_corners[:, 0]
    br = target_corners[:, 2]
    diagonal = paddle.norm(br - tl, dim=-1)
    nme = distance.mean(dim=-1) / (diagonal + 1e-06)
    return nme


if __name__ == "__main__":
    device = paddle.device("cuda" if paddle.cuda.is_available() else "cpu")
    batch_size = 4
    pred_heatmap = paddle.rand(batch_size, 4, 128, 128).to(device)
    target_heatmap = paddle.rand(batch_size, 4, 128, 128).to(device)
    target_heatmap[target_heatmap > 0.8] = 1.0
    focal_loss = FocalLoss()
    loss = focal_loss(pred_heatmap, target_heatmap)
    print(f"Focal Loss: {loss.item():.4f}")
    heatmap_loss = HeatmapLoss()
    loss = heatmap_loss(pred_heatmap, target_heatmap)
    print(f"Heatmap Loss: {loss.item():.4f}")
    pred_corners = paddle.rand(batch_size, 4, 2).to(device)
    target_corners = paddle.rand(batch_size, 4, 2).to(device)
    corner_loss = CornerLoss()
    loss = corner_loss(pred_corners, target_corners)
    print(f"Corner Loss: {loss.item():.4f}")
    iou = compute_iou(pred_corners, target_corners)
    print(f"IoU: {iou.mean().item():.4f}")
    nme = compute_nme(pred_corners, target_corners)
    print(f"NME: {nme.mean().item():.4f}")
