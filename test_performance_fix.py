import os
import subprocess
import time

# 创建必要的输出目录
os.makedirs('work/train_samples/images', exist_ok=True)
os.makedirs('work/train_samples/annotations', exist_ok=True)
os.makedirs('work/val_samples/images', exist_ok=True)
os.makedirs('work/val_samples/annotations', exist_ok=True)

# 测试生成训练样本的性能
print("Testing generate_samples.py performance fix...")
start_time = time.time()

result = subprocess.run(
    ['python', 'generate_samples.py', '--data_root', 'work/frames', '--output_root', 'work/train_samples'],
    capture_output=True,
    text=True,
    timeout=300  # 设置5分钟超时
)

end_time = time.time()
runtime = end_time - start_time

print(f"\nRuntime: {runtime:.2f} seconds")
print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

# 检查输出目录
print("\nChecking output directory...")
train_images_dir = 'work/train_samples/images'
train_annotations_dir = 'work/train_samples/annotations'

if os.path.exists(train_images_dir):
    images = os.listdir(train_images_dir)
    print(f"Found {len(images)} images in {train_images_dir}")
    if images:
        print(f"Sample image files: {images[:5]}...")
else:
    print(f"Directory not found: {train_images_dir}")

if os.path.exists(train_annotations_dir):
    annotations = os.listdir(train_annotations_dir)
    print(f"Found {len(annotations)} annotations in {train_annotations_dir}")
    if annotations:
        print(f"Sample annotation files: {annotations[:5]}...")
        # 检查一个标注文件的内容
        sample_annotation = os.path.join(train_annotations_dir, annotations[0])
        if os.path.exists(sample_annotation):
            with open(sample_annotation, 'r') as f:
                import json
                data = json.load(f)
                print(f"\nSample annotation content:")
                print(f"Image path: {data.get('image_path')}")
                print(f"Corners: {data.get('corners')}")
                print(f"Has heatmaps: {'heatmaps' in data}")
else:
    print(f"Directory not found: {train_annotations_dir}")
