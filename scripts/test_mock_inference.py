"""
模拟推理脚本 - 用于测试Web界面
当没有训练好的模型时，使用此脚本模拟文档矫正
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def mock_detect_corners(image_shape):
    """模拟角点检测 - 返回图像的四个角"""
    h, w = image_shape[:2]
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    corners = np.array(
        [
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y],
        ],
        dtype=np.float32,
    )
    return corners


def mock_correct_perspective(image, corners):
    """模拟透视变换 - 简单裁剪"""
    x_min = int(np.min(corners[:, 0]))
    y_min = int(np.min(corners[:, 1]))
    x_max = int(np.max(corners[:, 0]))
    y_max = int(np.max(corners[:, 1]))
    corrected = image[y_min:y_max, x_min:x_max]
    return corrected


def process_image(input_path, output_path, save_corners=True, save_visualization=True):
    """处理图像"""
    print(f"Processing: {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to read image from {input_path}")
    corners = mock_detect_corners(image.shape)
    corrected = mock_correct_perspective(image, corners)
    cv2.imwrite(output_path, corrected)
    print(f"Corrected image saved to: {output_path}")
    if save_corners:
        corners_path = Path(output_path).with_suffix(".json")
        corners_path = str(corners_path).replace(".jpg", "_corners.json")
        h, w = image.shape[:2]
        corners_normalized = corners / np.array([w, h])
        corners_data = {
            "image_path": str(input_path),
            "corners": corners_normalized.tolist(),
            "image_width": w,
            "image_height": h,
        }
        with open(corners_path, "w") as f:
            json.dump(corners_data, f, indent=2)
        print(f"Corners saved to: {corners_path}")
    if save_visualization:
        vis_path = str(output_path).replace(".jpg", "_vis.jpg")
        vis_image = image.copy()
        corners_int = corners.astype(np.int32)
        cv2.polylines(vis_image, [corners_int], True, (0, 255, 0), 2)
        labels = ["TL", "TR", "BR", "BL"]
        for i, corner in enumerate(corners_int):
            cv2.circle(vis_image, tuple(corner), 5, (0, 0, 255), -1)
            cv2.putText(
                vis_image,
                labels[i],
                (corner[0] + 10, corner[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(vis_path, vis_image)
        print(f"Visualization saved to: {vis_path}")


def main():
    parser = argparse.ArgumentParser(description="Mock inference for testing")
    parser.add_argument("--input", type=str, required=True, help="Input image")
    parser.add_argument("--output", type=str, required=True, help="Output image")
    parser.add_argument(
        "--save-corners", action="store_true", help="Save corner annotations"
    )
    parser.add_argument(
        "--save-visualization", action="store_true", help="Save visualization"
    )
    args = parser.parse_args()
    process_image(
        args.input,
        args.output,
        save_corners=args.save_corners,
        save_visualization=args.save_visualization,
    )


if __name__ == "__main__":
    main()
