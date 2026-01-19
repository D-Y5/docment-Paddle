import os
import paddle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

class SmartDocDataset(paddle.io.Dataset):
    """修改后的数据集类，直接返回角点坐标"""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        input_size: Tuple[int, int] = (512, 512)
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.input_size = input_size
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        split_file = self.root_dir / "split.json"
        with open(split_file, "r") as f:
            split_data = json.load(f)
        indices = split_data[self.split]
        samples = []
        for idx in indices[:1000]:  # 示例只加载部分数据
            image_path = self.root_dir / "images" / f"{idx:06d}.jpg"
            annotation_path = self.root_dir / "annotations" / f"{idx:06d}.json"
            if image_path.exists() and annotation_path.exists():
                samples.append({
                    "image_path": str(image_path),
                    "annotation_path": str(annotation_path),
                    "id": idx
                })
        return samples

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载图像和标注
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with open(sample["annotation_path"], "r") as f:
            annotation = json.load(f)
        
        # 获取原始角点并归一化 
        orig_corners = np.array(annotation["corners"], dtype=np.float32)
        
        # 归一化到[0,1]范围
        h, w = image.shape[:2]
        normalized_corners = orig_corners.copy()
        normalized_corners[:, 0] /= w  # x坐标归一化
        normalized_corners[:, 1] /= h  # y坐标归一化
        
        # 调整图像大小
        resized_image = cv2.resize(image, self.input_size)
        
        # 归一化图像
        image_tensor = paddle.to_tensor(resized_image.transpose(2,0,1)) / 255.0
        
        return {
            "image": image_tensor,
            "corners": paddle.to_tensor(normalized_corners),
            "image_id": sample["id"]
        }
        
def create_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    input_size: Tuple[int, int] = (512, 512)
):
    """创建数据加载器"""
    train_dataset = SmartDocDataset(root_dir, "train", input_size)
    val_dataset = SmartDocDataset(root_dir, "val", input_size)
    
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = paddle.io.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader
