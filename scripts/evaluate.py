import sys

sys.path.append("/home/aistudio/paddle_project")
import os

import paddle
from paddle_utils import *

"""
SmartDoc Challenge 评测脚本
参考官方评测标准
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from models.dataset import SmartDocDataset
from models.docaligner import create_docaligner_model
from models.loss import compute_iou, compute_nme


class SmartDocEvaluator:
    """SmartDoc Challenge 评测器"""

    def __init__(
        self,
        checkpoint_path: str,
        data_root: str,
        model_type: str = "base",
        device: str = "cuda",
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            data_root: 数据集根目录
            model_type: 模型类型
            device: 设备
        """
        self.device = paddle.device(device if paddle.cuda.is_available() else "cpu")
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = paddle.load(path=str(checkpoint_path))
        self.model = create_docaligner_model(model_type=model_type, num_corners=4)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        self.val_dataset = SmartDocDataset(
            root_dir=data_root, split="val", input_size=(512, 512), use_heatmap=True
        )

    def evaluate(self, batch_size=8):
        """
        执行评测

        Args:
            batch_size: 批次大小

        Returns:
            results: 评测结果
        """
        self.model.eval()
        all_iou = []
        all_nme = []
        all_corners_pred = []
        all_corners_gt = []
        print(f"Evaluating on {len(self.val_dataset)} samples...")
        with paddle.no_grad():
            for i in tqdm(range(0, len(self.val_dataset), batch_size)):
                end_idx = min(i + batch_size, len(self.val_dataset))
                batch_data = [self.val_dataset[j] for j in range(i, end_idx)]
                images = paddle.stack([d["image"] for d in batch_data]).to(self.device)
                corners_gt = paddle.stack([d["corners"] for d in batch_data])
                original_sizes = [d["original_size"] for d in batch_data]
                output = self.model(images)
                corners_pred = self._extract_corners(output)
                iou = compute_iou(corners_pred, corners_gt)
                nme = compute_nme(corners_pred, corners_gt)
                all_iou.extend(iou.cpu().numpy())
                all_nme.extend(nme.cpu().numpy())
                all_corners_pred.extend(corners_pred.cpu().numpy())
                all_corners_gt.extend(corners_gt.numpy())
        results = {
            "mean_iou": np.mean(all_iou),
            "std_iou": np.std(all_iou),
            "mean_nme": np.mean(all_nme),
            "std_nme": np.std(all_nme),
            "iou_threshold_085": (np.array(all_iou) >= 0.85).mean(),
            "nme_threshold_003": (np.array(all_nme) <= 0.03).mean(),
            "samples": len(all_iou),
            "all_iou": all_iou,
            "all_nme": all_nme,
        }
        return results

    def _extract_corners(self, output):
        """从模型输出中提取角点"""
        heatmap = output["heatmap"]
        batch_size = heatmap.shape[0]
        num_corners = 4
        corners = paddle.zeros(batch_size, num_corners, 2)
        for b in range(batch_size):
            for i in range(num_corners):
                h = heatmap[b, i]
                max_val = h._max()
                if max_val > 0.1:
                    y, x = paddle.where(h == max_val)
                    y, x = y[0].item(), x[0].item()
                    corners[b, i, 0] = x / (h.shape[1] - 1)
                    corners[b, i, 1] = y / (h.shape[0] - 1)
                else:
                    corners[b, i] = paddle.tensor([0.5, 0.5])
        return corners

    def print_results(self, results):
        """打印评测结果"""
        print("\n" + "=" * 60)
        print("SmartDoc Challenge 评测结果")
        print("=" * 60)
        print(f"\n样本数量: {results['samples']}")
        print(f"\nIoU 指标:")
        print(f"  平均 IoU: {results['mean_iou']:.4f} ± {results['std_iou']:.4f}")
        print(f"  IoU ≥ 0.85 比例: {results['iou_threshold_085'] * 100:.2f}%")
        print(f"\nNME 指标:")
        print(f"  平均 NME: {results['mean_nme']:.4f} ± {results['std_nme']:.4f}")
        print(f"  NME ≤ 0.03 比例: {results['nme_threshold_003'] * 100:.2f}%")
        print(f"\n性能目标:")
        iou_target = results["mean_iou"] >= 0.85
        nme_target = results["mean_nme"] <= 0.03
        print(f"  IoU ≥ 0.85: {'✓ 达标' if iou_target else '✗ 未达标'}")
        print(f"  NME ≤ 0.03: {'✓ 达标' if nme_target else '✗ 未达标'}")
        print("\n" + "=" * 60)

    def save_results(self, results, output_path):
        """保存评测结果"""
        output_data = {
            "mean_iou": float(results["mean_iou"]),
            "std_iou": float(results["std_iou"]),
            "mean_nme": float(results["mean_nme"]),
            "std_nme": float(results["std_nme"]),
            "iou_threshold_085": float(results["iou_threshold_085"]),
            "nme_threshold_003": float(results["nme_threshold_003"]),
            "samples": int(results["samples"]),
            "all_iou": [float(v) for v in results["all_iou"]],
            "all_nme": [float(v) for v in results["all_nme"]],
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {output_path}")


def compare_with_official(pred_corners_path: str, gt_corners_path: str):
    """
    与官方评测结果对比

    Args:
        pred_corners_path: 预测角点文件
        gt_corners_path: 官方标注文件
    """
    with open(pred_corners_path, "r") as f:
        pred_data = json.load(f)
    with open(gt_corners_path, "r") as f:
        gt_data = json.load(f)
    pred_corners = np.array(pred_data["corners"])
    gt_corners = np.array(gt_data["corners"])
    iou = compute_iou(
        paddle.from_numpy(pred_corners).unsqueeze(0),
        paddle.from_numpy(gt_corners).unsqueeze(0),
    ).item()
    nme = compute_nme(
        paddle.from_numpy(pred_corners).unsqueeze(0),
        paddle.from_numpy(gt_corners).unsqueeze(0),
    ).item()
    print(f"\n与官方评测对比:")
    print(f"  IoU: {iou:.4f}")
    print(f"  NME: {nme:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DocAligner on SmartDoc dataset"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-root", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="base",
        choices=["base", "offset"],
        help="Model type",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output path for results",
    )
    args = parser.parse_args()
    evaluator = SmartDocEvaluator(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        model_type=args.model_type,
        device=args.device,
    )
    results = evaluator.evaluate(batch_size=args.batch_size)
    evaluator.print_results(results)
    evaluator.save_results(results, args.output)
    iou_passed = results["mean_iou"] >= 0.85
    nme_passed = results["mean_nme"] <= 0.03
    if iou_passed and nme_passed:
        print("\n✓ 所有性能指标均达标！")
        sys.exit(0)
    else:
        print("\n✗ 部分性能指标未达标，需要进一步优化")
        sys.exit(1)


if __name__ == "__main__":
    main()
