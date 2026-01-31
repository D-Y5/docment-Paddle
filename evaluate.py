import os
import cv2
import paddle
import argparse
import numpy as np
from model import DocDetector
from align import DocAligner

class Evaluator:
    """评估器"""
    
    def __init__(self, model_path, input_size=(640, 640)):
        self.aligner = DocAligner(model_path, input_size)
    
    def calculate_iou(self, pred_corners, gt_corners):
        """计算文档区域IoU"""
        import cv2
        
        # 创建预测区域掩码
        h, w = 640, 640  # 假设图像尺寸
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
    
    def evaluate(self, test_root):
        """评估模型性能"""
        # 加载测试数据集
        image_dir = os.path.join(test_root, "images")
        annotation_dir = os.path.join(test_root, "annotations")
        
        total_iou = 0
        total_nme = 0
        num_samples = 0
        
        for image_name in os.listdir(image_dir):
            if not image_name.endswith(".jpg"):
                continue
            
            # 读取图像
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            
            # 读取真实标注
            annotation_name = image_name.replace(".jpg", ".json")
            annotation_path = os.path.join(annotation_dir, annotation_name)
            
            if not os.path.exists(annotation_path):
                continue
            
            import json
            with open(annotation_path, "r") as f:
                annotation = json.load(f)
            gt_corners = annotation["corners"]
            
            # 预测四角点
            pred_corners = self.aligner.predict_corners(image)
            
            # 计算IoU
            iou = self.calculate_iou(pred_corners, gt_corners)
            total_iou += iou
            
            # 计算NME
            nme = self.calculate_nme(pred_corners, gt_corners)
            total_nme += nme
            
            num_samples += 1
            
            print(f"Sample: {image_name}, IoU: {iou:.4f}, NME: {nme:.4f}")
        
        # 计算平均指标
        if num_samples > 0:
            avg_iou = total_iou / num_samples
            avg_nme = total_nme / num_samples
            
            print(f"\nAverage IoU: {avg_iou:.4f}")
            print(f"Average NME: {avg_nme:.4f}")
            
            # 检查是否满足性能要求
            if avg_iou >= 0.85:
                print("✓ IoU requirement met!")
            else:
                print("✗ IoU requirement not met!")
            
            if avg_nme <= 0.03:
                print("✓ NME requirement met!")
            else:
                print("✗ NME requirement not met!")
            
            return avg_iou, avg_nme
        else:
            print("No samples found!")
            return 0.0, float('inf')

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate document boundary detection model")
    parser.add_argument("--model", type=str, default="work/models/model_epoch_100.pdparams", help="Path to trained model")
    parser.add_argument("--test_root", type=str, default="work/val_samples", help="Path to test samples")
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 640], help="Model input size")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = Evaluator(args.model, args.image_size)
    
    # 评估模型
    evaluator.evaluate(args.test_root)

if __name__ == "__main__":
    main()
