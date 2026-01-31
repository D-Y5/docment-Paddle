import os
import subprocess

# 检查当前目录结构
print("Current directory:", os.getcwd())
print("Directory contents:", os.listdir('.'))

# 检查frames目录是否存在
if os.path.exists('frames'):
    print("Found frames directory")
    print("Frames directory contents:", os.listdir('frames')[:10])  # 只显示前10个项目
else:
    print("Frames directory not found")

# 检查是否存在smartdoc15-ch1目录
if os.path.exists('smartdoc15-ch1'):
    print("Found smartdoc15-ch1 directory")
    print("smartdoc15-ch1 contents:", os.listdir('smartdoc15-ch1'))
    
    # 检查smartdoc15-ch1/frames目录
    if os.path.exists('smartdoc15-ch1/frames'):
        print("Found smartdoc15-ch1/frames directory")
        print("smartdoc15-ch1/frames contents:", os.listdir('smartdoc15-ch1/frames')[:10])

# 创建work目录
os.makedirs('work/train_samples/images', exist_ok=True)
os.makedirs('work/train_samples/annotations', exist_ok=True)
os.makedirs('work/val_samples/images', exist_ok=True)
os.makedirs('work/val_samples/annotations', exist_ok=True)

# 运行生成训练样本的命令
print("\nRunning generate_samples.py...")

# 尝试不同的frames目录路径
frames_paths = ['frames', 'smartdoc15-ch1/frames']

for frames_path in frames_paths:
    if os.path.exists(frames_path):
        print(f"Trying frames path: {frames_path}")
        cmd = f"python generate_samples.py --data_root {frames_path} --output_root work/train_samples"
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        break
else:
    print("No valid frames directory found")
