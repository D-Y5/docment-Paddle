import os

"""
SmartDoc 数据集诊断工具 - 快速版本
用于快速识别数据集结构
"""
import sys
from pathlib import Path


def quick_diagnose(data_root):
    """快速诊断数据集结构"""
    data_root = Path(data_root)
    print("=" * 80)
    print("SmartDoc 数据集快速诊断")
    print("=" * 80)
    print(f"\n路径: {data_root}")
    print(f"存在: {'✅' if data_root.exists() else '❌'}")
    if not data_root.exists():
        print("\n❌ 路径不存在！请检查路径是否正确。")
        print(f"\n当前工作目录: {os.getcwd()}")
        print(f"请输入完整的路径，例如:")
        print("  C:\\Users\\dengyu\\scikit_learn_data\\smartdoc15-ch1")
        return
    print("\n" + "-" * 80)
    print("根目录内容:")
    print("-" * 80)
    items = list(data_root.iterdir())
    print(f"共 {len(items)} 项")
    dirs = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file()]
    print(f"\n📁 目录 ({len(dirs)} 个):")
    for d in dirs[:20]:
        print(f"  - {d.name}")
    print(f"\n📄 文件 ({len(files)} 个):")
    for f in files[:20]:
        print(f"  - {f.name}")
    print("\n" + "-" * 80)
    print("关键文件检查:")
    print("-" * 80)
    key_files = {
        "metadata.pkl": False,
        "metadata.mat": False,
        "annotations.csv": False,
        "groundtruth.txt": False,
        "README.md": False,
    }
    for name in key_files.keys():
        found = any(f.name.lower() == name.lower() for f in files)
        key_files[name] = found
        status = "✅" if found else "❌"
        print(f"{status} {name}")
    print("\n" + "-" * 80)
    print("图像文件检查:")
    print("-" * 80)
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
    image_files = []
    for ext in image_extensions:
        found = list(data_root.rglob(f"*{ext}"))
        image_files.extend(found)
        print(f"📄 {ext:8s}: {len(found)} 个")
    if image_files:
        print(f"\n总共找到 {len(image_files)} 个图像文件")
        depth_distribution = {}
        for img in image_files:
            rel_path = img.relative_to(data_root)
            depth = len(rel_path.parts) - 1
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
        print("\n图像分布（按路径深度）:")
        for depth, count in sorted(depth_distribution.items()):
            print(f"  深度 {depth}: {count} 个文件")
        print("\n示例路径:")
        for img in image_files[:10]:
            rel_path = img.relative_to(data_root)
            print(f"  🖼️  {rel_path}")
    print("\n" + "-" * 80)
    print("标注文件检查:")
    print("-" * 80)
    annotation_files = []
    for ext in [".pkl", ".mat", ".csv", ".json", ".txt"]:
        found = list(data_root.rglob(f"*{ext}"))
        for f in found:
            f_lower = f.name.lower()
            if any(kw in f_lower for kw in ["meta", "annot", "ground", "label", "gt"]):
                annotation_files.append(f)
    if annotation_files:
        print(f"找到 {len(annotation_files)} 个可能的标注文件:")
        for f in annotation_files[:20]:
            rel_path = f.relative_to(data_root)
            print(f"  📄 {rel_path}")
    else:
        print("❌ 未找到标注文件")
    print("\n" + "-" * 80)
    print("数据集类型推断:")
    print("-" * 80)
    if any(f.name.lower() == "metadata.pkl" for f in files):
        print("✅ 检测到 sklearn 格式 (metadata.pkl)")
        print("   建议：使用 fetch_mldata 或直接加载 .pkl 文件")
    elif any(f.name.lower() == "metadata.mat" for f in files):
        print("✅ 检测到 MATLAB 格式 (metadata.mat)")
        print("   建议：使用 scipy.io.loadmat 加载")
    elif any("video" in d.name.lower() for d in dirs):
        print("✅ 检测到视频序列结构")
        print("   建议：遍历视频目录读取帧")
    elif any("images" in d.name.lower() for d in dirs):
        print("✅ 检测到图像目录结构")
        print("   建议：读取 images 目录中的图像")
    print("\n" + "-" * 80)
    print("下一步建议:")
    print("-" * 80)
    if annotation_files:
        print("1. 📝 查看标注文件内容，了解标注格式")
        print("2. 🖼️ 确认图像路径与标注的对应关系")
        print("3. 📊 运行数据预处理脚本")
    else:
        print("1. ⚠️  未找到标注文件，请确认数据集完整性")
        print("2. 🔍 检查是否需要下载额外的标注数据")
        print("3. 📚 查看数据集的 README 文件")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    DATA_ROOT = "C:\\Users\\dengyu\\scikit_learn_data\\smartdoc15-ch1"
    if len(sys.argv) > 1:
        DATA_ROOT = sys.argv[1]
    try:
        quick_diagnose(DATA_ROOT)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print(f"请检查路径是否正确: {DATA_ROOT}")
        import traceback

        traceback.print_exc()
    input("\n按回车键退出...")
