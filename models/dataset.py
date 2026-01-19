import os

import paddle

"""
SmartDoc 2015 数据集类
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import paddle.nn.functional as F


class SmartDocDataset(paddle.io.Dataset):
    """SmartDoc 2015 训练数据集"""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        input_size: Tuple[int, int] = (512, 512),
        target_size: Optional[Tuple[int, int]] = None,  # 添加参数
        use_heatmap: bool = True,
        heatmap_sigma: float = 10.0,           #5.0
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: "train" or "val" 
            transform: 图像变换
            input_size: 输入尺寸 (H, W)
            use_heatmap: 是否使用热力图标注
            heatmap_sigma: 热力图高斯核大小
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.input_size = input_size
        self.use_heatmap = use_heatmap
        self.heatmap_sigma = heatmap_sigma
        self.samples = self._load_samples()
        self.target_size = target_size if target_size else input_size  # 默认与input_size相同

    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        split_file = self.root_dir / "split.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r") as f:
            split_data = json.load(f)
        indices = split_data[self.split]
        samples = []
        for idx in indices:
            image_path = self.root_dir / "images" / f"{idx:06d}.jpg"
            annotation_path = self.root_dir / "annotations" / f"{idx:06d}.json"
            if image_path.exists() and annotation_path.exists():
                samples.append(
                    {
                        "image": str(image_path),
                        "annotation": str(annotation_path),
                        "id": idx,
                    }
                )
        print(f"Loaded {len(samples)} {self.split} samples")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """获取样本"""
        sample = self.samples[idx]
        image = cv2.imread(sample["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with open(sample["annotation"], "r") as f:
            annotation = json.load(f)
        corners = np.array(annotation["corners"], dtype=np.float32)
        original_height, original_width = image.shape[:2]
        image_resized, corners_resized = self._resize_image_and_corners(
            image, corners, self.input_size
        )
        if self.use_heatmap:
            heatmap = self._generate_heatmap(corners_resized)
        else:
            heatmap = None
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (
            image_normalized - np.array([0.485, 0.456, 0.406])
        ) / np.array([0.229, 0.224, 0.225])
        image_normalized = image_normalized.transpose(2, 0, 1)
        corners_normalized = corners_resized.copy()
        corners_normalized[:, 0] /= self.input_size[1]
        corners_normalized[:, 1] /= self.input_size[0]
        result = {
            "image": paddle.to_tensor(image_normalized).float(),
            "corners": paddle.to_tensor(corners_normalized).float(),
            "original_size": (original_width, original_height),
            "image_id": sample["id"],
        }
        if heatmap is not None:
            result["heatmap"] = paddle.to_tensor(heatmap).float()
        return result

    def _resize_image_and_corners(
        self, image: np.ndarray, corners: np.ndarray, target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        调整图像和角点大小

        Args:
            image: 原始图像
            corners: 原始角点 (4, 2)
            target_size: 目标尺寸 (H, W)

        Returns:
            resized_image: 调整后的图像
            resized_corners: 调整后的角点
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size
        scale_x = target_w / w
        scale_y = target_h / h
        resized_image = cv2.resize(
            image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        resized_corners = corners.copy()
        resized_corners[:, 0] *= scale_x
        resized_corners[:, 1] *= scale_y
        return resized_image, resized_corners

    def _generate_heatmap(
        self, corners: np.ndarray, num_keypoints: int = 4
    ) -> np.ndarray:
        """
        生成关键点热力图

        Args:
            corners: 角点坐标 (4, 2)
            num_keypoints: 关键点数量

        Returns:
            heatmap: 热力图 (num_keypoints, H, W)
        """
        h, w = 64, 64  # 匹配模型输出分辨率
        # 重新缩放角点坐标到64x64范围
        scale_x = 64 / self.target_size[1]
        scale_y = 64 / self.target_size[0]
        scaled_corners = corners.copy()
        scaled_corners[:, 0] *= scale_x
        scaled_corners[:, 1] *= scale_y
        
        heatmap = np.zeros((num_keypoints, h, w), dtype=np.float32)
        for i, (x, y) in enumerate(scaled_corners):
            x = int(np.clip(x, 0, w - 1))
            y = int(np.clip(y, 0, h - 1))
            heatmap[i] = self._generate_single_heatmap((x, y), (h, w), sigma=1.5)  # 减小sigma值
        return heatmap

    def _generate_single_heatmap(
        self,
        point: Tuple[int, int],
        size: Tuple[int, int],
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        为单个点生成高斯热力图

        Args:
            point: 点坐标 (x, y)
            size: 图像尺寸 (H, W)
            sigma: 高斯核大小

        Returns:
            heatmap: 单个热力图 (H, W)
        """
        if sigma is None:
            sigma = self.heatmap_sigma
        h, w = size
        x, y = point
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        return heatmap


def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    input_size: Tuple[int, int] = (512, 512),
    target_size: Optional[Tuple[int, int]] = None,  # 添加
    use_heatmap: bool = True,
) -> Tuple[paddle.io.DataLoader, paddle.io.DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        root_dir: 数据集根目录
        batch_size: 批次大小
        num_workers: 工作线程数
        input_size: 输入尺寸
        use_heatmap: 是否使用热力图

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    train_dataset = SmartDocDataset(
        root_dir=root_dir, split="train", input_size=input_size,target_size=target_size,use_heatmap=use_heatmap
    )
    val_dataset = SmartDocDataset(
        root_dir=root_dir, split="val", input_size=input_size, target_size=target_size, use_heatmap=use_heatmap
    )
    train_loader = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = paddle.io.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    root_dir = "C:/Users/dengyu/smartdoc15_train_samples"
    dataset = SmartDocDataset(root_dir, split="train")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Corners shape: {sample['corners'].shape}")
    if "heatmap" in sample:
        print(f"Heatmap shape: {sample['heatmap'].shape}")
