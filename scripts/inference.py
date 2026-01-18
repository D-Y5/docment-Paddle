import sys

sys.path.append("/home/aistudio/paddle_project")
import os

import paddle
from paddle_utils import *

"""
DocAligner推理脚本
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from models.docaligner import create_docaligner_model
from models.transform import PerspectiveTransformer, draw_corners


class DocAlignerInference:
    """DocAligner推理器"""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "base",
        input_size: tuple = (512, 512),
        device: str = "cuda",
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            model_type: 模型类型
            input_size: 输入尺寸 (H, W)
            device: 设备
        """
        self.device = paddle.device(device if paddle.cuda.is_available() else "cpu")
        self.input_size = input_size
        print(f"Loading model from {checkpoint_path}...")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = paddle.load(path=str(checkpoint_path))
        self.model = create_docaligner_model(model_type=model_type, num_corners=4)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        self.transformer = PerspectiveTransformer()

    def preprocess(self, image: np.ndarray) -> paddle.Tensor:
        """
        预处理图像

        Args:
            image: 输入图像 (H, W, 3) RGB

        Returns:
            tensor: 预处理后的张量 (1, 3, H, W)
        """
        image_resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        image_tensor = paddle.from_numpy(image_normalized).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def extract_corners(self, heatmap: np.ndarray) -> np.ndarray:
        """
        从热力图中提取角点坐标

        Args:
            heatmap: 热力图 (4, H, W)

        Returns:
            corners: 角点坐标 (4, 2), 归一化到[0, 1]
        """
        corners = []
        for i in range(4):
            h = heatmap[i]
            max_val = h._max()
            if max_val > 0.1:
                y, x = np.unravel_index(h.argmax(), h.shape)
                x_norm = x / (h.shape[1] - 1)
                y_norm = y / (h.shape[0] - 1)
                corners.append([x_norm, y_norm])
            else:
                corners.append([0.5, 0.5])
        return np.array(corners, dtype=np.float32)

    def refine_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        使用角点检测器精化角点（可选）

        Args:
            image: 原始图像
            corners: 初始角点

        Returns:
            refined_corners: 精化后的角点
        """
        h, w = image.shape[:2]
        corners_pixel = corners.copy()
        corners_pixel[:, 0] *= w
        corners_pixel[:, 1] *= h
        refined = []
        for corner in corners_pixel:
            search_radius = 20
            x, y = corner.astype(int)
            x_min = max(0, x - search_radius)
            x_max = min(w, x + search_radius)
            y_min = max(0, y - search_radius)
            y_max = min(h, y + search_radius)
            roi = image[y_min:y_max, x_min:x_max]
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            _, _, _, max_loc = cv2.minMaxLoc(dst)
            refined_x = x_min + max_loc[0]
            refined_y = y_min + max_loc[1]
            refined.append([refined_x, refined_y])
        refined_corners = np.array(refined, dtype=np.float32)
        refined_corners[:, 0] /= w
        refined_corners[:, 1] /= h
        return refined_corners

    def detect_corners(self, image: np.ndarray) -> np.ndarray:
        """
        检测文档角点

        Args:
            image: 输入图像 (H, W, 3) RGB

        Returns:
            corners: 角点坐标 (4, 2), 归一化到[0, 1]
        """
        image_tensor = self.preprocess(image)
        image_tensor = image_tensor.to(self.device)
        with paddle.no_grad():
            output = self.model(image_tensor)
        heatmap = output["heatmap"][0].cpu().numpy()
        corners = self.extract_corners(heatmap)
        return corners

    def correct_perspective(
        self, image: np.ndarray, corners: np.ndarray, maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        矫正透视变形

        Args:
            image: 输入图像 (H, W, 3)
            corners: 角点坐标 (4, 2), 归一化到[0, 1]
            maintain_aspect: 是否保持宽高比

        Returns:
            corrected_image: 矫正后的图像
        """
        h, w = image.shape[:2]
        corners_pixel = corners.copy()
        corners_pixel[:, 0] *= w
        corners_pixel[:, 1] *= h
        corrected_image = self.transformer.transform(
            image, corners_pixel, maintain_aspect=maintain_aspect
        )
        return corrected_image

    def process_image(
        self,
        input_path: str,
        output_path: str,
        save_corners: bool = False,
        save_visualization: bool = False,
    ):
        """
        处理单张图像

        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            save_corners: 是否保存角点标注
            save_visualization: 是否保存可视化结果
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to read image from {input_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corners = self.detect_corners(image)
        corrected = self.correct_perspective(image, corners)
        corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, corrected_bgr)
        print(f"Corrected image saved to: {output_path}")
        if save_corners:
            output_dir = Path(output_path).parent
            corners_path = output_dir / f"{Path(output_path).stem}_corners.json"
            corners_data = {
                "image_path": str(input_path),
                "corners": corners.tolist(),
                "image_width": image.shape[1],
                "image_height": image.shape[0],
            }
            with open(corners_path, "w") as f:
                json.dump(corners_data, f, indent=2)
            print(f"Corners saved to: {corners_path}")
        if save_visualization:
            h, w = image.shape[:2]
            corners_pixel = corners.copy()
            corners_pixel[:, 0] *= w
            corners_pixel[:, 1] *= h
            visualization = draw_corners(
                image, corners_pixel, labels=["TL", "TR", "BR", "BL"]
            )
            output_dir = Path(output_path).parent
            vis_path = output_dir / f"{Path(output_path).stem}_vis.jpg"
            cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Visualization saved to: {vis_path}")

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_corners: bool = False,
        save_visualization: bool = False,
    ):
        """
        处理目录中的所有图像

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            save_corners: 是否保存角点标注
            save_visualization: 是否保存可视化结果
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in image_extensions
        ]
        print(f"Found {len(image_files)} images")
        for idx, image_file in enumerate(image_files):
            print(f"\n[{idx + 1}/{len(image_files)}] Processing: {image_file.name}")
            output_file = output_path / image_file.name
            self.process_image(
                str(image_file),
                str(output_file),
                save_corners=save_corners,
                save_visualization=save_visualization,
            )
        print(f"\nAll images processed! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DocAligner Inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input image or directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output image or directory"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="base",
        choices=["base", "offset"],
        help="Model type",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Input size (height width)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device"
    )
    parser.add_argument(
        "--save-corners", action="store_true", help="Save corner annotations"
    )
    parser.add_argument(
        "--save-visualization", action="store_true", help="Save visualization results"
    )
    args = parser.parse_args()
    inference = DocAlignerInference(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        input_size=tuple(args.input_size),
        device=args.device,
    )
    input_path = Path(args.input)
    if input_path.is_file():
        inference.process_image(
            args.input,
            args.output,
            save_corners=args.save_corners,
            save_visualization=args.save_visualization,
        )
    elif input_path.is_dir():
        inference.process_directory(
            args.input,
            args.output,
            save_corners=args.save_corners,
            save_visualization=args.save_visualization,
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
