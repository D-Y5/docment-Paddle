import os
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm

class SampleGenerator:
    """训练样本生成器"""
    
    def __init__(self, data_root, output_root, image_size=(640, 640)):
        self.data_root = data_root
        self.output_root = output_root
        self.image_size = image_size
        
        # 确保输出目录存在
        os.makedirs(os.path.join(output_root, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "annotations"), exist_ok=True)
    
    def generate_heatmap(self, image_shape, corner, sigma=5):
        """生成热力图"""
        h, w = image_shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # 确保角点在图像范围内
        x, y = int(corner[0]), int(corner[1])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        
        # 生成高斯热力图
        for i in range(max(0, y - 3 * sigma), min(h, y + 3 * sigma)):
            for j in range(max(0, x - 3 * sigma), min(w, x + 3 * sigma)):
                distance = np.sqrt((i - y) ** 2 + (j - x) ** 2)
                if distance < 3 * sigma:
                    heatmap[i, j] = np.exp(-distance ** 2 / (2 * sigma ** 2))
        
        return heatmap
    
    def process_frame(self, frame_path, doc_type, doc_id, frame_id):
        """处理单帧图像"""
        # 读取图像
        image = cv2.imread(frame_path)
        if image is None:
            return None
        
        # 调整图像尺寸
        resized_image = cv2.resize(image, self.image_size)
        
        # 生成四角点标注（这里使用模拟标注，实际应用中需要使用真实标注）
        # 假设文档占据图像的中心区域
        h, w = self.image_size
        margin = int(min(h, w) * 0.1)
        corners = [
            [margin, margin],  # 左上角
            [w - margin, margin],  # 右上角
            [w - margin, h - margin],  # 右下角
            [margin, h - margin]  # 左下角
        ]
        
        # 生成热力图
        heatmaps = []
        for corner in corners:
            heatmap = self.generate_heatmap(self.image_size, corner)
            heatmaps.append(heatmap)
        
        # 保存图像
        image_name = f"{doc_type}{doc_id}_{frame_id}.jpg"
        image_output_path = os.path.join(self.output_root, "images", image_name)
        cv2.imwrite(image_output_path, resized_image)
        
        # 保存标注
        annotation_name = f"{doc_type}{doc_id}_{frame_id}.json"
        annotation_output_path = os.path.join(self.output_root, "annotations", annotation_name)
        annotation = {
            "image_path": image_name,
            "corners": corners,
            "heatmaps": [heatmap.tolist() for heatmap in heatmaps]
        }
        
        with open(annotation_output_path, "w") as f:
            json.dump(annotation, f)
        
        return image_output_path
    
    def generate_samples(self):
        """生成所有训练样本"""
        # 遍历所有背景场景
        background_dirs = [d for d in os.listdir(self.data_root) if d.startswith("background")]
        
        total_frames = 0
        for background_dir in background_dirs:
            background_path = os.path.join(self.data_root, background_dir)
            
            # 遍历所有文档类型
            doc_types = os.listdir(background_path)
            for doc_type in doc_types:
                doc_path = os.path.join(background_path, doc_type)
                
                # 提取文档类型和ID
                if doc_type.startswith("datasheet"):
                    doc_type_name = "datasheet"
                    doc_id = doc_type.replace("datasheet", "")
                elif doc_type.startswith("letter"):
                    doc_type_name = "letter"
                    doc_id = doc_type.replace("letter", "")
                elif doc_type.startswith("magazine"):
                    doc_type_name = "magazine"
                    doc_id = doc_type.replace("magazine", "")
                elif doc_type.startswith("paper"):
                    doc_type_name = "paper"
                    doc_id = doc_type.replace("paper", "")
                elif doc_type.startswith("patent"):
                    doc_type_name = "patent"
                    doc_id = doc_type.replace("patent", "")
                elif doc_type.startswith("tax"):
                    doc_type_name = "tax"
                    doc_id = doc_type.replace("tax", "")
                else:
                    continue
                
                # 遍历所有帧图像
                frames = [f for f in os.listdir(doc_path) if f.endswith(".jpg")]
                frames.sort()
                
                for frame_id, frame_name in enumerate(tqdm(frames, desc=f"Processing {doc_type}")):
                    frame_path = os.path.join(doc_path, frame_name)
                    self.process_frame(frame_path, doc_type_name, doc_id, frame_id)
                    total_frames += 1
        
        print(f"Total frames processed: {total_frames}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Generate training samples for document boundary detection")
    parser.add_argument("--data_root", type=str, default="work/frames", help="Path to frames directory")
    parser.add_argument("--output_root", type=str, default="work/train_samples", help="Path to output directory")
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 640], help="Image size (width, height)")
    
    args = parser.parse_args()
    
    generator = SampleGenerator(args.data_root, args.output_root, args.image_size)
    generator.generate_samples()

if __name__ == "__main__":
    main()
