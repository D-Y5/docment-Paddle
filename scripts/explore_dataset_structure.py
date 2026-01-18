import os

"""
探索 SmartDoc 2015 数据集的实际文件结构
"""
from collections import defaultdict
from pathlib import Path

DATA_ROOT = "C:\\Users\\dengyu\\scikit_learn_data\\smartdoc15-ch1"


def explore_structure(data_root, max_depth=4):
    """探索目录结构"""
    data_root = Path(data_root)
    print("=" * 80)
    print("SmartDoc 2015 数据集文件结构探索")
    print("=" * 80)
    print(f"\n数据集路径: {data_root}")
    print(f"路径是否存在: {data_root.exists()}")
    print()
    if not data_root.exists():
        print("❌ 路径不存在！")
        return
    stats = {
        "total_dirs": 0,
        "total_files": 0,
        "file_types": defaultdict(int),
        "root_items": [],
    }
    print("根目录内容:")
    print("-" * 80)
    for item in sorted(data_root.iterdir()):
        if item.is_dir():
            stats["total_dirs"] += 1
            size = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"📁 {item.name:<40} [目录, 包含 {size} 个文件]")
        else:
            stats["total_files"] += 1
            ext = item.suffix.lower()
            stats["file_types"][ext] += 1
            print(f"📄 {item.name:<40} [文件, {item.suffix}]")
        stats["root_items"].append(item)
    print("\n" + "=" * 80)
    print("统计信息:")
    print("-" * 80)
    print(f"根目录项数: {len(stats['root_items'])}")
    print(f"目录总数: {stats['total_dirs']}")
    print(f"文件总数: {stats['total_files']}")
    if stats["file_types"]:
        print("\n文件类型分布:")
        for ext, count in sorted(
            stats["file_types"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {ext or '无扩展名':<15} : {count} 个")
    print("\n" + "=" * 80)
    print("详细目录结构 (前3层):")
    print("-" * 80)

    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        try:
            items = sorted(
                directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
            )
        except PermissionError:
            print(f"{prefix}⚠️  (权限不足)")
            return
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            if item.is_dir():
                print(f"{prefix}{current_prefix}📁 {item.name}")
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
            else:
                ext = item.suffix.lower()
                icon = (
                    "🖼️ " if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"] else "📄 "
                )
                print(
                    f"{prefix}{current_prefix}{icon}{item.name} ({item.suffix or '无扩展名'})"
                )

    print_tree(data_root, max_depth=max_depth)
    print("\n" + "=" * 80)
    print("查找标注文件:")
    print("-" * 80)
    annotation_keywords = [
        "annotation",
        "label",
        "ground",
        "truth",
        "metadata",
        "meta",
        "gt",
        "seg",
    ]
    annotation_files = []
    for ext in [
        ".txt",
        ".csv",
        ".json",
        ".xml",
        ".mat",
        ".pkl",
        ".pickle",
        ".yaml",
        ".yml",
    ]:
        for file in data_root.rglob(f"*{ext}"):
            file_lower = file.name.lower()
            if any(keyword in file_lower for keyword in annotation_keywords):
                annotation_files.append(file)
    if annotation_files:
        print(f"找到 {len(annotation_files)} 个可能的标注文件:")
        for file in annotation_files[:20]:
            rel_path = file.relative_to(data_root)
            print(f"  📄 {rel_path}")
            try:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(200)
                    print(f"      内容预览: {content[:100]}...")
            except:
                pass
    else:
        print("❌ 未找到明显的标注文件")
    print("\n" + "=" * 80)
    print("查找图像文件:")
    print("-" * 80)
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_root.rglob(f"*{ext}"))
    if image_files:
        print(f"找到 {len(image_files)} 个图像文件")
        print(f"分布示例 (前20个):")
        for file in image_files[:20]:
            rel_path = file.relative_to(data_root)
            print(f"  🖼️  {rel_path}")
        print("\n图像路径模式分析:")
        path_depths = defaultdict(int)
        for file in image_files:
            depth = len(file.relative_to(data_root).parts) - 1
            path_depths[depth] += 1
        for depth, count in sorted(path_depths.items()):
            print(f"  深度 {depth}: {count} 个文件")
        if image_files:
            sample = image_files[0].relative_to(data_root)
            print(f"\n完整路径示例:")
            print(f"  {sample}")
            print(f"  结构: {' / '.join(sample.parts)}")
    else:
        print("❌ 未找到图像文件")
    print("\n" + "=" * 80)
    print("查找特殊文件:")
    print("-" * 80)
    special_files = {"README": [], "LICENSE": [], "meta": [], "data": []}
    for file in data_root.rglob("*"):
        if file.is_file():
            name_lower = file.name.lower()
            if "readme" in name_lower:
                special_files["README"].append(file)
            elif "license" in name_lower:
                special_files["LICENSE"].append(file)
            elif "meta" in name_lower:
                special_files["meta"].append(file)
    for key, files in special_files.items():
        if files:
            print(f"\n{key} 文件:")
            for file in files[:5]:
                rel_path = file.relative_to(data_root)
                print(f"  📄 {rel_path}")
    print("\n" + "=" * 80)
    print("探索完成！")
    print("=" * 80)


if __name__ == "__main__":
    explore_structure(DATA_ROOT)
