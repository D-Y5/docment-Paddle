import os
import cv2
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

class SampleGenerator:
    """训练样本和验证集生成器"""

    def __init__(self, frames_root, output_root, image_size=(640, 640), val_ratio=0.2):
        self.frames_root = frames_root
        self.output_root = output_root
        self.image_size = image_size
        self.val_ratio = val_ratio

        os.makedirs(os.path.join(output_root, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "train", "annotations"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "train", "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "val", "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "val", "annotations"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "val", "masks"), exist_ok=True)

        self.frame_metadata = self._load_frame_metadata()

    def _load_frame_metadata(self):
        """加载帧元数据"""
        metadata_path = os.path.join(self.frames_root, "frames_metadata.csv")
        if os.path.exists(metadata_path):
            print(f"Loading frame metadata from: {metadata_path}")
            try:
                metadata = pd.read_csv(metadata_path)
                print(f"Loaded frame metadata with {len(metadata)} entries")
                return metadata
            except Exception as e:
                print(f"Error loading frame metadata: {e}")
                return None
        else:
            print(f"Warning: Frame metadata not found at: {metadata_path}")
            return None

    def generate_mask(self, image_shape, corners):
        """生成分割掩码"""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _get_corners_from_metadata(self, bg_name, doc_id, frame_index):
        """从元数据中获取四角点坐标"""
        if self.frame_metadata is not None:
            try:
                if 'model_name' in self.frame_metadata.columns and 'bg_name' in self.frame_metadata.columns:
                    mask = (self.frame_metadata['model_name'] == doc_id) & (self.frame_metadata['bg_name'] == bg_name)
                    frame_doc_metadata = self.frame_metadata[mask]

                    if len(frame_doc_metadata) > 0:
                        frame_doc_metadata = frame_doc_metadata.sort_values('frame_index')
                        if frame_index < len(frame_doc_metadata):
                            metadata_entry = frame_doc_metadata.iloc[frame_index]
                        else:
                            metadata_entry = frame_doc_metadata.iloc[0]

                        required_columns = ['tl_x', 'tl_y', 'tr_x', 'tr_y', 'br_x', 'br_y', 'bl_x', 'bl_y']
                        if all(col in self.frame_metadata.columns for col in required_columns):
                            corners = [
                                [int(metadata_entry['tl_x']), int(metadata_entry['tl_y'])],
                                [int(metadata_entry['tr_x']), int(metadata_entry['tr_y'])],
                                [int(metadata_entry['br_x']), int(metadata_entry['br_y'])],
                                [int(metadata_entry['bl_x']), int(metadata_entry['bl_y'])]
                            ]
                            return corners
            except Exception as e:
                print(f"Error extracting corners from metadata: {e}")

        h, w = self.image_size
        margin = int(min(h, w) * 0.1)
        default_corners = [
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ]
        return default_corners

    def _process_frame(self, frame_path, bg_name, doc_id, frame_index, split="train"):
        """处理单个帧图像"""
        image_name = os.path.basename(frame_path)
        output_image_name = f"{bg_name}_{doc_id}_{frame_index:04d}.jpg"
        image_output_path = os.path.join(self.output_root, split, "images", output_image_name)

        mask_name = output_image_name.replace(".jpg", ".png")
        mask_output_path = os.path.join(self.output_root, split, "masks", mask_name)

        annotation_name = output_image_name.replace(".jpg", ".json")
        annotation_output_path = os.path.join(self.output_root, split, "annotations", annotation_name)

        if os.path.exists(image_output_path) and os.path.exists(annotation_output_path) and os.path.exists(mask_output_path):
            return "skipped"

        frame_image = cv2.imread(frame_path)
        if frame_image is None:
            return None

        resized_frame = cv2.resize(frame_image, self.image_size)

        corners = self._get_corners_from_metadata(bg_name, doc_id, frame_index)
        mask = self.generate_mask(self.image_size, corners)

        cv2.imwrite(image_output_path, resized_frame)
        cv2.imwrite(mask_output_path, mask)

        annotation = {
            "image_path": output_image_name,
            "mask_path": mask_name,
            "corners": corners,
            "bg_name": bg_name,
            "doc_id": doc_id,
            "frame_index": frame_index
        }

        with open(annotation_output_path, "w") as f:
            json.dump(annotation, f)

        return image_output_path

    def generate_samples(self):
        """生成训练集和验证集"""
        total_train = 0
        total_val = 0

        background_dirs = [d for d in os.listdir(self.frames_root) if d.startswith("background")]
        print(f"Found background directories: {background_dirs}")

        for background_dir in background_dirs:
            background_path = os.path.join(self.frames_root, background_dir)
            print(f"Processing background: {background_path}")

            doc_types = [d for d in os.listdir(background_path) if os.path.isdir(os.path.join(background_path, d))]
            print(f"Found doc types: {doc_types}")

            for doc_type in doc_types:
                doc_path = os.path.join(background_path, doc_type)
                print(f"Processing doc type: {doc_path}")

                if not os.path.isdir(doc_path):
                    continue

                all_files = os.listdir(doc_path)
                frames = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg"))]
                print(f"Found {len(frames)} frames in {doc_path}")

                if len(frames) == 0:
                    continue

                frames.sort()

                val_count = int(len(frames) * self.val_ratio)
                train_count = len(frames) - val_count

                train_frames = frames[:train_count]
                val_frames = frames[train_count:]

                skipped_train = 0
                for frame_id, frame_name in enumerate(tqdm(train_frames, desc=f"Train {doc_type}")):
                    frame_path = os.path.join(doc_path, frame_name)
                    result = self._process_frame(frame_path, background_dir, doc_type, frame_id, "train")
                    if result == "skipped":
                        skipped_train += 1
                    elif result:
                        total_train += 1

                if skipped_train > 0:
                    print(f"Skipped {skipped_train} existing train files in {doc_type}")

                skipped_val = 0
                for frame_id, frame_name in enumerate(tqdm(val_frames, desc=f"Val {doc_type}")):
                    frame_path = os.path.join(doc_path, frame_name)
                    result = self._process_frame(frame_path, background_dir, doc_type, frame_id, "val")
                    if result == "skipped":
                        skipped_val += 1
                    elif result:
                        total_val += 1

                if skipped_val > 0:
                    print(f"Skipped {skipped_val} existing val files in {doc_type}")

        print(f"Total train samples: {total_train}")
        print(f"Total val samples: {total_val}")
        print(f"Metadata used: {self.frame_metadata is not None}")

def main():
    parser = argparse.ArgumentParser(description="Generate train and validation samples from frames")
    parser.add_argument("--frames_root", type=str, default="frames", help="Path to frames directory")
    parser.add_argument("--output_root", type=str, default="work", help="Path to output directory")
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 640], help="Image size (width, height)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")

    args = parser.parse_args()

    if not os.path.exists(args.frames_root):
        print(f"Error: Frames directory not found: {args.frames_root}")
        return

    print(f"Using frames root: {args.frames_root}")
    print(f"Using output root: {args.output_root}")
    print(f"Validation ratio: {args.val_ratio}")

    generator = SampleGenerator(args.frames_root, args.output_root, args.image_size, args.val_ratio)
    generator.generate_samples()

if __name__ == "__main__":
    main()
