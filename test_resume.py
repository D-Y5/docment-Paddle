import os
import subprocess

# 测试断点续传功能
print("Testing resume functionality...")
print("=" * 50)

# 第一次运行（处理部分文件）
print("\nFirst run (should process all files):")
result1 = subprocess.run(
    ['python', 'generate_samples.py', '--data_root', 'work/frames', '--output_root', 'work/train_samples'],
    capture_output=True,
    text=True,
    timeout=60
)

print("STDOUT (last 500 chars):")
print(result1.stdout[-500:] if len(result1.stdout) > 500 else result1.stdout)
print(f"\nReturn code: {result1.returncode}")

# 检查生成的文件数量
train_images_dir = 'work/train_samples/images'
train_annotations_dir = 'work/train_samples/annotations'

if os.path.exists(train_images_dir):
    images = os.listdir(train_images_dir)
    print(f"\nGenerated {len(images)} images")
else:
    print(f"\nImages directory not found: {train_images_dir}")

if os.path.exists(train_annotations_dir):
    annotations = os.listdir(train_annotations_dir)
    print(f"Generated {len(annotations)} annotations")
else:
    print(f"Annotations directory not found: {train_annotations_dir}")

# 第二次运行（应该跳过已存在的文件）
print("\n" + "=" * 50)
print("\nSecond run (should skip existing files):")
result2 = subprocess.run(
    ['python', 'generate_samples.py', '--data_root', 'work/frames', '--output_root', 'work/train_samples'],
    capture_output=True,
    text=True,
    timeout=60
)

print("STDOUT (last 500 chars):")
print(result2.stdout[-500:] if len(result2.stdout) > 500 else result2.stdout)
print(f"\nReturn code: {result2.returncode}")

# 检查是否显示了跳过信息
if "Skipped" in result2.stdout or "skipped" in result2.stdout.lower():
    print("\n✓ Resume functionality is working correctly!")
    print("  The script skipped existing files.")
else:
    print("\n✗ Resume functionality may not be working as expected.")
    print("  No 'Skipped' messages found in output.")

# 最终文件数量检查
if os.path.exists(train_images_dir):
    final_images = os.listdir(train_images_dir)
    print(f"\nFinal image count: {len(final_images)}")

if os.path.exists(train_annotations_dir):
    final_annotations = os.listdir(train_annotations_dir)
    print(f"Final annotation count: {len(final_annotations)}")
