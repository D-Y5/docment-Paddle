import os
import cv2
import paddle
import argparse
import numpy as np
import json
from tqdm import tqdm
from model import DocDetector
from align import DocAligner

class Evaluator:
    """评估器 - 使用文档区域IoU作为性能指标"""

    def __init__(self, model_path, input_size=(640, 640)):
        self.aligner = DocAligner(model_path, input_size)
        self.input_size = input_size

    def calculate_iou(self, pred_corners, gt_corners, image_size=None):
        """计算文档区域IoU"""
        if image_size is None:
            image_size = self.input_size

        h, w = image_size

        pred_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(pred_mask, [np.array(pred_corners, dtype=np.int32)], 1)

        gt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_mask, [np.array(gt_corners, dtype=np.int32)], 1)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def evaluate(self, test_root, batch_size=1):
        """评估模型性能"""
        image_dir = os.path.join(test_root, "images")
        annotation_dir = os.path.join(test_root, "annotations")

        total_iou = 0
        num_samples = 0
        iou_list = []

        image_paths = []
        annotation_paths = []

        for image_name in os.listdir(image_dir):
            if not image_name.endswith(".jpg"):
                continue

            image_path = os.path.join(image_dir, image_name)
            annotation_name = image_name.replace(".jpg", ".json")
            annotation_path = os.path.join(annotation_dir, annotation_name)

            if os.path.exists(annotation_path):
                image_paths.append(image_path)
                annotation_paths.append(annotation_path)

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Evaluating"):
            batch_image_paths = image_paths[i:i+batch_size]
            batch_annotation_paths = annotation_paths[i:i+batch_size]

            for image_path, annotation_path in zip(batch_image_paths, batch_annotation_paths):
                image = cv2.imread(image_path)
                if image is None:
                    continue

                with open(annotation_path, "r") as f:
                    annotation = json.load(f)
                gt_corners = annotation["corners"]

                pred_corners = self.aligner.predict_corners(image)

                iou = self.calculate_iou(pred_corners, gt_corners, image.shape[:2])
                total_iou += iou
                iou_list.append(iou)

                num_samples += 1

                image_name = os.path.basename(image_path)
                print(f"Sample: {image_name}, IoU: {iou:.4f}")

        if num_samples > 0:
            avg_iou = total_iou / num_samples
            std_iou = np.std(iou_list) if len(iou_list) > 1 else 0

            iou_threshold = 0.85
            iou_above_threshold = sum(1 for iou in iou_list if iou >= iou_threshold) / num_samples

            print(f"\n=== Evaluation Report ===")
            print(f"Total samples: {num_samples}")
            print(f"Average IoU: {avg_iou:.4f} (±{std_iou:.4f})")
            print(f"IoU ≥ {iou_threshold}: {iou_above_threshold:.2%}")
            print(f"=========================")

            if avg_iou >= iou_threshold:
                print("✓ IoU requirement met!")
            else:
                print("✗ IoU requirement not met!")

            evaluation_result = {
                "total_samples": num_samples,
                "average_iou": avg_iou,
                "std_iou": std_iou,
                "iou_above_threshold": iou_above_threshold,
                "iou_threshold": iou_threshold
            }

            result_path = os.path.join(os.path.dirname(test_root), "evaluation_result.json")
            with open(result_path, "w") as f:
                json.dump(evaluation_result, f, indent=2)
            print(f"Evaluation result saved to: {result_path}")

            return evaluation_result
        else:
            print("No samples found!")
            return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate document segmentation model")
    parser.add_argument("--model", type=str, default="work/best_model.pdparams", help="Path to trained model")
    parser.add_argument("--test_root", type=str, default="work/val_samples", help="Path to test samples")
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 640], help="Model input size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")

    args = parser.parse_args()

    evaluator = Evaluator(args.model, args.image_size)
    evaluator.evaluate(args.test_root, args.batch_size)

if __name__ == "__main__":
    main()
