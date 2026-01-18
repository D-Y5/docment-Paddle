import os

"""
SmartDoc 2015 数据集预处理脚本
从 CSV 标注文件生成训练样本（帧 + 四点标注）
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "C:/Users/dengyu/scikit_learn_data/smartdoc15-ch1"
OUTPUT_ROOT = "C:/Users/dengyu/smartdoc15_train_samples"
FRAMES_CSV = "frames/frames_metadata.csv"
MODELS_CSV = "models/model_metadata.csv"


class SmartDocDataProcessor:
    """SmartDoc 2015 数据集处理器"""

    def __init__(self, data_root, output_root):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.frames_csv = self.data_root / FRAMES_CSV
        self.models_csv = self.data_root / MODELS_CSV
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "images").mkdir(exist_ok=True)
        (self.output_root / "annotations").mkdir(exist_ok=True)
        self.stats = {
            "total_frames": 0,
            "total_documents": 0,
            "failed": 0,
            "skipped": 0,
        }
        print("Loading CSV files...")
        self.frames_df = pd.read_csv(self.frames_csv)
        self.models_df = pd.read_csv(self.models_csv)
        print(f"Loaded {len(self.frames_df)} frame annotations")
        print(f"Loaded {len(self.models_df)} model annotations")

    def load_image(self, image_path):
        """
        加载图像文件

        Args:
            image_path: 图像相对路径（从 CSV 的 image_path 列）

        Returns:
            image: PIL Image 对象
        """
        full_path = self.data_root / "frames" / image_path
        if not full_path.exists():
            alt_path = self.data_root / image_path
            if alt_path.exists():
                full_path = alt_path
            else:
                print(f"Warning: Image not found: {image_path}")
                return None
        try:
            image = Image.open(full_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            return None

    def get_corners_from_csv(self, row):
        """
        从 CSV 行中提取四个角点坐标

        Args:
            row: DataFrame 行

        Returns:
            corners: 四个角点坐标 [tl, tr, br, bl]，每个点是 [x, y]
        """
        tl = [row["tl_x"], row["tl_y"]]
        tr = [row["tr_x"], row["tr_y"]]
        br = [row["br_x"], row["br_y"]]
        bl = [row["bl_x"], row["bl_y"]]
        corners = [tl, tr, br, bl]
        corners = np.array(corners, dtype=np.float32)
        return corners

    def generate_training_sample(self, image, corners, metadata, sample_id):
        """
        生成训练样本并保存

        Args:
            image: PIL Image 对象
            corners: 四个角点坐标 (4, 2)
            metadata: 元数据字典
            sample_id: 样本ID

        Returns:
            success: 是否成功
        """
        try:
            image_path = self.output_root / "images" / f"{sample_id:06d}.jpg"
            image.save(image_path, quality=95)
            annotation = {
                "image_path": str(image_path.relative_to(self.output_root)),
                "corners": corners.tolist(),
                "image_width": image.width,
                "image_height": image.height,
                "metadata": {
                    "bg_name": metadata.get("bg_name", ""),
                    "bg_id": int(metadata.get("bg_id", 0)),
                    "model_name": metadata.get("model_name", ""),
                    "model_id": int(metadata.get("model_id", 0)),
                    "modeltype_name": metadata.get("modeltype_name", ""),
                    "modeltype_id": int(metadata.get("modeltype_id", 0)),
                    "model_width": float(metadata.get("model_width", 0)),
                    "model_height": float(metadata.get("model_height", 0)),
                    "frame_index": int(metadata.get("frame_index", 0)),
                },
            }
            annotation_path = self.output_root / "annotations" / f"{sample_id:06d}.json"
            with open(annotation_path, "w") as f:
                json.dump(annotation, f, indent=2)
            self.stats["total_frames"] += 1
            return True
        except Exception as e:
            print(f"Error saving sample {sample_id}: {e}")
            self.stats["failed"] += 1
            return False

    def process_dataset(self, max_samples=None, split_ratio=0.8):
        """
        处理整个数据集

        Args:
            max_samples: 最大样本数（用于测试，None 表示全部）
            split_ratio: 训练集/验证集划分比例
        """
        print(f"\nProcessing dataset from CSV...")
        print(f"Total frames: {len(self.frames_df)}")
        if max_samples:
            frames_to_process = self.frames_df.head(max_samples)
        else:
            frames_to_process = self.frames_df
        sample_id = 0
        skipped_count = 0
        for idx, row in tqdm(
            frames_to_process.iterrows(),
            total=len(frames_to_process),
            desc="Processing",
        ):
            image_rel_path = row["image_path"]
            image = self.load_image(image_rel_path)
            if image is None:
                skipped_count += 1
                continue
            corners = self.get_corners_from_csv(row)
            metadata = row.to_dict()
            success = self.generate_training_sample(image, corners, metadata, sample_id)
            if success:
                sample_id += 1
            else:
                self.stats["failed"] += 1
        self.stats["skipped"] = skipped_count
        print(f"\n{'=' * 60}")
        print(f"Processing complete!")
        print(f"{'=' * 60}")
        print(f"Total samples generated: {sample_id}")
        print(f"Skipped (image not found): {skipped_count}")
        print(f"Failed (save error): {self.stats['failed']}")
        if sample_id > 0:
            self.create_dataset_split(sample_id, split_ratio)
        else:
            print("Warning: No samples generated!")

    def create_dataset_split(self, total_samples, split_ratio=0.8):
        """创建训练集/验证集划分"""
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        split_idx = int(total_samples * split_ratio)
        train_indices = indices[:split_idx].tolist()
        val_indices = indices[split_idx:].tolist()
        split = {
            "train": train_indices,
            "val": val_indices,
            "total_train": len(train_indices),
            "total_val": len(val_indices),
        }
        split_path = self.output_root / "split.json"
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"\nDataset split created:")
        print(
            f"  Train: {len(train_indices)} samples ({len(train_indices) / total_samples * 100:.1f}%)"
        )
        print(
            f"  Val: {len(val_indices)} samples ({len(val_indices) / total_samples * 100:.1f}%)"
        )
        print(f"\nSplit file saved to: {split_path}")

    def verify_samples(self, num_samples=5):
        """验证生成的样本"""
        print(f"\n{'=' * 60}")
        print(f"Verifying {num_samples} samples...")
        print(f"{'=' * 60}\n")
        image_dir = self.output_root / "images"
        annotation_dir = self.output_root / "annotations"
        image_files = sorted(image_dir.glob("*.jpg"))[:num_samples]
        if not image_files:
            print("No samples found!")
            return
        for img_path in image_files:
            sample_id = img_path.stem
            ann_path = annotation_dir / f"{sample_id}.json"
            print(f"Sample {sample_id}:")
            print(f"  Image: {img_path.name}")
            if not ann_path.exists():
                print(f"  ❌ Missing annotation!")
                continue
            with open(ann_path, "r") as f:
                ann = json.load(f)
            print(f"  ✅ Annotation found")
            print(f"  Image size: {ann['image_width']} x {ann['image_height']}")
            print(f"  Corners:")
            corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
            for i, (name, corner) in enumerate(zip(corner_names, ann["corners"])):
                print(f"    {name}: ({corner[0]:.2f}, {corner[1]:.2f})")
            print(f"  Metadata:")
            print(
                f"    Model: {ann['metadata']['model_name']} (ID: {ann['metadata']['model_id']})"
            )
            print(f"    Background: {ann['metadata']['bg_name']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SmartDoc 2015 dataset for training"
    )
    parser.add_argument(
        "--data_root", type=str, default=DATA_ROOT, help="Path to SmartDoc dataset root"
    )
    parser.add_argument(
        "--output_root", type=str, default=OUTPUT_ROOT, help="Path to output directory"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.8, help="Train/validation split ratio"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify generated samples"
    )
    args = parser.parse_args()
    processor = SmartDocDataProcessor(args.data_root, args.output_root)
    processor.process_dataset(
        max_samples=args.max_samples, split_ratio=args.split_ratio
    )
    if args.verify:
        processor.verify_samples()
    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
