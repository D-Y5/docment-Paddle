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
    """评估器"""
    
    def __init__(self, model_path, input_size=(640, 640)):
        self.aligner = DocAligner(model_path, input_size)
        self.input_size = input_size
    
    def calculate_iou(self, pred_corners, gt_corners, image_size=None):
        """计算文档区域IoU"""
        if image_size is None:
            image_size = self.input_size
        
        h, w = image_size
        
        # 创建预测区域掩码
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(pred_mask, [np.array(pred_corners, dtype=np.int32)], 1)
        
        # 创建真实区域掩码
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_mask, [np.array(gt_corners, dtype=np.int32)], 1)
        
        # 计算交集和并集
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_nme(self, pred_corners, gt_corners):
        """计算四点归一化平均误差"""
        # 计算真实四角点的对角线长度
        gt_corners = np.array(gt_corners)
        width = np.max(gt_corners[:, 0]) - np.min(gt_corners[:, 0])
        height = np.max(gt_corners[:, 1]) - np.min(gt_corners[:, 1])
        diagonal = np.sqrt(width ** 2 + height ** 2)
        
        if diagonal == 0:
            return float('inf')
        
        # 计算每个点的误差
        pred_corners = np.array(pred_corners)
        errors = np.sqrt(np.sum((pred_corners - gt_corners) ** 2, axis=1))
        
        # 计算平均误差
        nme = np.mean(errors) / diagonal
        
        return nme
    
    def evaluate(self, test_root, batch_size=1):
        """评估模型性能"""
        # 加载测试数据集
        image_dir = os.path.join(test_root, "images")
        annotation_dir = os.path.join(test_root, "annotations")
        
        total_iou = 0
        total_nme = 0
        num_samples = 0
        iou_list = []
        nme_list = []
        
        # 获取所有测试图像路径
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
        
        # 批处理评估
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Evaluating"):
            batch_image_paths = image_paths[i:i+batch_size]
            batch_annotation_paths = annotation_paths[i:i+batch_size]
            
            for image_path, annotation_path in zip(batch_image_paths, batch_annotation_paths):
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # 读取真实标注
                with open(annotation_path, "r") as f:
                    annotation = json.load(f)
                gt_corners = annotation["corners"]
                
                # 预测四角点
                pred_corners = self.aligner.predict_corners(image)
                
                # 计算IoU
                iou = self.calculate_iou(pred_corners, gt_corners, image.shape[:2])
                total_iou += iou
                iou_list.append(iou)
                
                # 计算NME
                nme = self.calculate_nme(pred_corners, gt_corners)
                total_nme += nme
                nme_list.append(nme)
                
                num_samples += 1
                
                # 打印样本评估结果
                image_name = os.path.basename(image_path)
                print(f"Sample: {image_name}, IoU: {iou:.4f}, NME: {nme:.4f}")
        
        # 计算平均指标
        if num_samples > 0:
            avg_iou = total_iou / num_samples
            avg_nme = total_nme / num_samples
            
            # 计算IoU和NME的标准差
            std_iou = np.std(iou_list) if len(iou_list) > 1 else 0
            std_nme = np.std(nme_list) if len(nme_list) > 1 else 0
            
            # 计算IoU大于阈值的比例
            iou_threshold = 0.85
            iou_above_threshold = sum(1 for iou in iou_list if iou >= iou_threshold) / num_samples
            
            # 计算NME小于阈值的比例
            nme_threshold = 0.03
            nme_below_threshold = sum(1 for nme in nme_list if nme <= nme_threshold) / num_samples
            
            # 生成评估报告
            print(f"\n=== Evaluation Report ===")
            print(f"Total samples: {num_samples}")
            print(f"Average IoU: {avg_iou:.4f} (±{std_iou:.4f})")
            print(f"Average NME: {avg_nme:.4f} (±{std_nme:.4f})")
            print(f"IoU ≥ {iou_threshold}: {iou_above_threshold:.2f}%")
            print(f"NME ≤ {nme_threshold}: {nme_below_threshold:.2f}%")
            print(f"=========================")
            
            # 检查是否满足性能要求
            if avg_iou >= iou_threshold:
                print("✓ IoU requirement met!")
            else:
                print("✗ IoU requirement not met!")
            
            if avg_nme <= nme_threshold:
                print("✓ NME requirement met!")
            else:
                print("✗ NME requirement not met!")
            
            # 保存评估结果
            evaluation_result = {
                "total_samples": num_samples,
                "average_iou": avg_iou,
                "average_nme": avg_nme,
                "std_iou": std_iou,
                "std_nme": std_nme,
                "iou_above_threshold": iou_above_threshold,
                "nme_below_threshold": nme_below_threshold,
                "iou_threshold": iou_threshold,
                "nme_threshold": nme_threshold
            }
            
            # 保存评估结果到文件
            result_path = os.path.join(os.path.dirname(test_root), "evaluation_result.json")
            with open(result_path, "w") as f:
                json.dump(evaluation_result, f, indent=2)
            print(f"Evaluation result saved to: {result_path}")
            
            return evaluation_result
        else:
            print("No samples found!")
            return None
    
    def evaluate_image(self, image_path, gt_corners=None):
        """评估单个图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # 预测四角点
        pred_corners = self.aligner.predict_corners(image)
        
        if gt_corners is not None:
            # 计算评估指标
            iou = self.calculate_iou(pred_corners, gt_corners, image.shape[:2])
            nme = self.calculate_nme(pred_corners, gt_corners)
            
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Detected corners: {pred_corners}")
            print(f"Ground truth corners: {gt_corners}")
            print(f"IoU: {iou:.4f}")
            print(f"NME: {nme:.4f}")
            
            return pred_corners, iou, nme
        else:
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Detected corners: {pred_corners}")
            return pred_corners

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate document boundary detection model")
    parser.add_argument("--model", type=str, default="work/best_model.pdparams", help="Path to trained model")
    parser.add_argument("--test_root", type=str, default="work/val_samples", help="Path to test samples")
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 640], help="Model input size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = Evaluator(args.model, args.image_size)
    
    # 评估模型
    evaluator.evaluate(args.test_root, args.batch_size)

if __name__ == "__main__":
    main()
