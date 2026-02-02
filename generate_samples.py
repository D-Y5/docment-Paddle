import os
import cv2
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

class SampleGenerator:
    """训练样本生成器"""
    
    def __init__(self, frames_root, models_root, output_root, image_size=(640, 640)):
        self.frames_root = frames_root
        self.models_root = models_root
        self.output_root = output_root
        self.image_size = image_size
        
        # 确保输出目录存在
        os.makedirs(os.path.join(output_root, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "annotations"), exist_ok=True)
        
        # 加载元数据
        self.model_metadata = self._load_model_metadata()
        self.frame_metadata = self._load_frame_metadata()
        
        # 存储标准模型图像路径的映射
        self.model_images_map = self._build_model_images_map()
    
    def _load_model_metadata(self):
        """加载模型元数据"""
        metadata_path = os.path.join(self.models_root, "model_metadata.csv")
        if os.path.exists(metadata_path):
            print(f"Loading model metadata from: {metadata_path}")
            try:
                metadata = pd.read_csv(metadata_path)
                print(f"Loaded model metadata with {len(metadata)} entries")
                print(f"Model metadata columns: {list(metadata.columns)}")
                return metadata
            except Exception as e:
                print(f"Error loading model metadata: {e}")
                return None
        else:
            print(f"Warning: Model metadata not found at: {metadata_path}")
            return None
    
    def _load_frame_metadata(self):
        """加载帧元数据"""
        metadata_path = os.path.join(self.frames_root, "frames_metadata.csv")
        if os.path.exists(metadata_path):
            print(f"Loading frame metadata from: {metadata_path}")
            try:
                metadata = pd.read_csv(metadata_path)
                print(f"Loaded frame metadata with {len(metadata)} entries")
                print(f"Frame metadata columns: {list(metadata.columns)}")
                return metadata
            except Exception as e:
                print(f"Error loading frame metadata: {e}")
                return None
        else:
            print(f"Warning: Frame metadata not found at: {metadata_path}")
            return None
    
    def _build_model_images_map(self):
        """构建标准模型图像路径的映射"""
        model_images_map = {}
        
        # 使用02-edited目录中的统一规格图像作为标准
        edited_dir = os.path.join(self.models_root, "02-edited")
        
        if os.path.exists(edited_dir):
            print(f"Building model images map from: {edited_dir}")
            
            for file_name in os.listdir(edited_dir):
                if file_name.endswith(".png"):
                    # 提取文档类型和ID，例如datasheet001.png → datasheet001
                    doc_id = file_name.split(".")[0]
                    model_images_map[doc_id] = os.path.join(edited_dir, file_name)
            
            print(f"Found {len(model_images_map)} model images")
        else:
            print(f"Warning: 02-edited directory not found at: {edited_dir}")
        
        return model_images_map
    
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
    
    def _get_corners_from_metadata(self, doc_id):
        """从元数据中获取四角点坐标"""
        # 首先尝试从frames_metadata.csv中获取四角点
        if self.frame_metadata is not None:
            try:
                # 从frame_metadata中查找对应文档的四角点
                # 优先使用model_name列来匹配doc_id
                if 'model_name' in self.frame_metadata.columns:
                    frame_doc_metadata = self.frame_metadata[self.frame_metadata['model_name'] == doc_id]
                    
                    if len(frame_doc_metadata) > 0:
                        # 从frame_metadata中提取四角点坐标
                        try:
                            # 检查是否有必要的坐标列
                            required_columns = ['tl_x', 'tl_y', 'tr_x', 'tr_y', 'br_x', 'br_y', 'bl_x', 'bl_y']
                            if all(col in self.frame_metadata.columns for col in required_columns):
                                # 使用第一个匹配的条目
                                metadata_entry = frame_doc_metadata.iloc[0]
                                
                                # 提取四角点坐标（左上、右上、右下、左下）
                                corners = [
                                    [int(metadata_entry['tl_x']), int(metadata_entry['tl_y'])],  # 左上角
                                    [int(metadata_entry['tr_x']), int(metadata_entry['tr_y'])],  # 右上角
                                    [int(metadata_entry['br_x']), int(metadata_entry['br_y'])],  # 右下角
                                    [int(metadata_entry['bl_x']), int(metadata_entry['bl_y'])]   # 左下角
                                ]
                                
                                print(f"Using corners from frame metadata for {doc_id}: {corners}")
                                return corners
                            else:
                                print(f"Missing required coordinate columns in frame metadata")
                        except Exception as e:
                            print(f"Error extracting corners from frame metadata: {e}")
                    else:
                        print(f"No frame metadata found for model_name: {doc_id}")
            except Exception as e:
                print(f"Error processing frame metadata: {e}")
        
        # 如果frame_metadata不可用，尝试从model_metadata中获取
        if self.model_metadata is not None:
            try:
                # 从model_metadata中查找对应文档
                if 'model_name' in self.model_metadata.columns:
                    model_doc_metadata = self.model_metadata[self.model_metadata['model_name'] == doc_id]
                    
                    if len(model_doc_metadata) > 0:
                        print(f"Found model metadata for {doc_id}, but using default corners")
                        # model_metadata中没有四角点，使用默认值
            except Exception as e:
                print(f"Error processing model metadata: {e}")
        
        # 如果元数据不可用，使用默认四角点
        h, w = self.image_size
        margin = int(min(h, w) * 0.1)
        default_corners = [
            [margin, margin],  # 左上角
            [w - margin, margin],  # 右上角
            [w - margin, h - margin],  # 右下角
            [margin, h - margin]  # 左下角
        ]
        print(f"Using default corners for {doc_id}: {default_corners}")
        return default_corners
    
    def process_pair(self, frame_path, model_path, doc_id, frame_id):
        """处理配对的帧图像和模型图像"""
        # 生成输出文件名
        image_name = f"{doc_id}_{frame_id}.jpg"
        image_output_path = os.path.join(self.output_root, "images", image_name)
        
        annotation_name = f"{doc_id}_{frame_id}.json"
        annotation_output_path = os.path.join(self.output_root, "annotations", annotation_name)
        
        # 检查文件是否已存在（断点续传）
        if os.path.exists(image_output_path) and os.path.exists(annotation_output_path):
            return "skipped"
        
        # 读取拍摄帧图像
        frame_image = cv2.imread(frame_path)
        if frame_image is None:
            return None
        
        # 读取标准模型图像
        model_image = cv2.imread(model_path)
        if model_image is None:
            return None
        
        # 调整图像尺寸
        resized_frame = cv2.resize(frame_image, self.image_size)
        
        # 从元数据或默认值获取四角点标注
        corners = self._get_corners_from_metadata(doc_id)
        
        # 保存拍摄帧图像
        cv2.imwrite(image_output_path, resized_frame)
        
        # 保存标注
        annotation = {
            "image_path": image_name,
            "corners": corners,
            "model_path": os.path.basename(model_path),
            "doc_id": doc_id,
            "metadata_used": self.model_metadata is not None
        }
        
        with open(annotation_output_path, "w") as f:
            json.dump(annotation, f)
        
        return image_output_path
    
    def generate_samples(self):
        """生成所有训练样本"""
        total_frames = 0
        
        # 遍历 frames 目录中的背景场景
        background_dirs = [d for d in os.listdir(self.frames_root) if d.startswith("background")]
        print(f"Found background directories: {background_dirs}")
        
        for background_dir in background_dirs:
            background_path = os.path.join(self.frames_root, background_dir)
            print(f"Processing background: {background_path}")
            
            # 遍历所有文档类型
            doc_types = os.listdir(background_path)
            print(f"Found doc types: {doc_types}")
            
            for doc_type in doc_types:
                doc_path = os.path.join(background_path, doc_type)
                print(f"Processing doc type: {doc_path}")
                
                # 提取文档类型和ID，例如datasheet005 → datasheet005
                doc_id = doc_type
                
                # 检查是否有对应的标准模型图像
                if doc_id not in self.model_images_map:
                    print(f"Warning: No model image found for doc_id: {doc_id}")
                    continue
                
                model_path = self.model_images_map[doc_id]
                print(f"Using model image: {model_path}")
                
                # 遍历所有帧图像
                try:
                    if not os.path.isdir(doc_path):
                        print(f"Skipping non-directory: {doc_path}")
                        continue
                    
                    # 列出目录中的所有文件
                    all_files = os.listdir(doc_path)
                    frames = [f for f in all_files if f.endswith(".jpg") or f.endswith(".jpeg")]
                    print(f"Found {len(frames)} frames in {doc_path}")
                    
                    frames.sort()
                    
                    skipped_count = 0
                    for frame_id, frame_name in enumerate(tqdm(frames, desc=f"Processing {doc_type}")):
                        frame_path = os.path.join(doc_path, frame_name)
                        
                        result = self.process_pair(frame_path, model_path, doc_id, frame_id)
                        if result == "skipped":
                            skipped_count += 1
                        elif result:
                            total_frames += 1
                    
                    if skipped_count > 0:
                        print(f"Skipped {skipped_count} existing files in {doc_type}")
                except Exception as e:
                    print(f"Error processing {doc_path}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Total frames processed: {total_frames}")
        print(f"Metadata used: {self.model_metadata is not None}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Generate training samples for document boundary detection")
    parser.add_argument("--frames_root", type=str, default="frames", help="Path to frames directory")
    parser.add_argument("--models_root", type=str, default="models", help="Path to models directory")
    parser.add_argument("--output_root", type=str, default="work/train_samples", help="Path to output directory")
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 640], help="Image size (width, height)")
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.frames_root):
        print(f"Error: Frames directory not found: {args.frames_root}")
        return
    
    if not os.path.exists(args.models_root):
        print(f"Error: Models directory not found: {args.models_root}")
        return
    
    print(f"Using frames root: {args.frames_root}")
    print(f"Using models root: {args.models_root}")
    print(f"Using output root: {args.output_root}")
    
    generator = SampleGenerator(args.frames_root, args.models_root, args.output_root, args.image_size)
    generator.generate_samples()

if __name__ == "__main__":
    main()
