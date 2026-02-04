import os
import cv2
import paddle
import argparse
import numpy as np
from model import DocDetector

class DocAligner:
    """文档对齐器"""
    
    def __init__(self, model_path, input_size=(640, 640)):
        self.input_size = input_size
        
        # 初始化模型
        self.model = DocDetector(input_size=input_size)
        
        # 加载模型权重
        state_dict = paddle.load(model_path)
        self.model.set_state_dict(state_dict)
        self.model.eval()
    
    def preprocess(self, image):
        """预处理图像"""
        # 调整图像尺寸
        resized_image = cv2.resize(image, self.input_size)
        # 归一化
        resized_image = resized_image.astype(np.float32) / 255.0
        # 转换为CHW格式
        resized_image = np.transpose(resized_image, (2, 0, 1))
        # 增加批次维度
        resized_image = np.expand_dims(resized_image, axis=0)
        return resized_image
    
    def predict_corners(self, image):
        """预测四角点"""
        # 预处理图像
        preprocessed = self.preprocess(image)
        # 转换为paddle张量
        input_tensor = paddle.to_tensor(preprocessed)
        
        # 使用模型的predict_corners方法预测四角点
        with paddle.no_grad():
            corners = self.model.predict_corners(input_tensor)
        
        # 归一化到原始图像尺寸
        h, w = image.shape[:2]
        input_h, input_w = self.input_size
        
        normalized_corners = []
        for corner in corners:
            # 将模型输出的坐标映射到原始图像尺寸
            x = int(corner[0] / input_w * w)
            y = int(corner[1] / input_h * h)
            normalized_corners.append([x, y])
        
        return normalized_corners
    
    def sort_corners(self, corners):
        """排序四角点（左上、右上、右下、左下）"""
        # 计算中心点
        center = np.mean(corners, axis=0)
        
        # 计算每个点相对于中心点的角度
        angles = []
        for corner in corners:
            angle = np.arctan2(corner[1] - center[1], corner[0] - center[0])
            angles.append(angle)
        
        # 按角度排序
        sorted_indices = np.argsort(angles)
        sorted_corners = [corners[i] for i in sorted_indices]
        
        # 确保顺序正确（左上、右上、右下、左下）
        # 计算左上角（x最小且y最小）
        top_left = min(sorted_corners, key=lambda p: p[0] + p[1])
        top_left_idx = sorted_corners.index(top_left)
        
        # 重新排序
        sorted_corners = sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx]
        
        return sorted_corners
    
    def align_document(self, image, corners):
        """对齐文档"""
        # 排序四角点
        sorted_corners = self.sort_corners(corners)
        
        # 计算目标尺寸
        width = int(max(
            np.sqrt((sorted_corners[0][0] - sorted_corners[1][0]) ** 2 + (sorted_corners[0][1] - sorted_corners[1][1]) ** 2),
            np.sqrt((sorted_corners[2][0] - sorted_corners[3][0]) ** 2 + (sorted_corners[2][1] - sorted_corners[3][1]) ** 2)
        ))
        
        height = int(max(
            np.sqrt((sorted_corners[0][0] - sorted_corners[3][0]) ** 2 + (sorted_corners[0][1] - sorted_corners[3][1]) ** 2),
            np.sqrt((sorted_corners[1][0] - sorted_corners[2][0]) ** 2 + (sorted_corners[1][1] - sorted_corners[2][1]) ** 2)
        ))
        
        # 目标四角点
        dst_corners = [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ]
        
        # 执行透视变换
        src_points = np.float32(sorted_corners)
        dst_points = np.float32(dst_corners)
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        aligned = cv2.warpPerspective(image, M, (width, height))
        
        return aligned
    
    def process(self, image_path, output_path, export_format="jpeg"):
        """处理图像并导出"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # 预测四角点
        corners = self.predict_corners(image)
        
        # 对齐文档
        aligned = self.align_document(image, corners)
        
        # 导出结果
        if export_format == "jpeg":
            cv2.imwrite(output_path, aligned)
        elif export_format == "pdf":
            # 保存为PDF（需要PIL库）
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
            img.save(output_path, "PDF")
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        return corners

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Align document using trained model")
    parser.add_argument("--model", type=str, default="work/models/model_epoch_100.pdparams", help="Path to trained model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to output image")
    parser.add_argument("--format", type=str, default="jpeg", choices=["jpeg", "pdf"], help="Export format")
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 640], help="Model input size")
    
    args = parser.parse_args()
    
    # 初始化对齐器
    aligner = DocAligner(args.model, args.image_size)
    
    # 处理图像
    corners = aligner.process(args.input, args.output, args.format)
    
    print(f"Processing completed!")
    print(f"Detected corners: {corners}")
    print(f"Aligned document saved to: {args.output}")

if __name__ == "__main__":
    main()
