"""
分析 SmartDoc 数据集的 CSV 标注文件
"""
from pathlib import Path

import pandas as pd


def analyze_csv(csv_path, name):
    """分析 CSV 文件"""
    print(f"\n{'=' * 80}")
    print(f"分析文件: {name}")
    print(f"{'=' * 80}\n")
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return
    print(f"✅ 文件路径: {csv_path}")
    print(f"📊 总行数: {len(df)}")
    print(f"📝 总列数: {len(df.columns)}\n")
    print("列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    print(f"\n前 10 行数据:")
    print(df.head(10).to_string())
    print(f"\n数据类型:")
    print(df.dtypes)
    print(f"\n统计信息:")
    print(df.describe())
    print(f"\n缺失值统计:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✅ 无缺失值")
    print(f"\n角点坐标列检查:")
    corner_keywords = ["tl", "tr", "br", "bl", "corner", "point", "x", "y", "coord"]
    found_corner_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in corner_keywords):
            found_corner_cols.append(col)
    if found_corner_cols:
        print(f"✅ 找到可能的角点坐标列:")
        for col in found_corner_cols:
            print(f"  - {col}")
            unique_count = df[col].nunique()
            print(f"    唯一值数量: {unique_count}")
            print(f"    示例值: {df[col].head(3).tolist()}")
    else:
        print("❌ 未找到明显的角点坐标列")
    print(f"\n图像路径列检查:")
    path_keywords = ["path", "image", "file", "name", "filename"]
    found_path_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in path_keywords):
            found_path_cols.append(col)
    if found_path_cols:
        print(f"✅ 找到可能的图像路径列:")
        for col in found_path_cols:
            print(f"  - {col}")
            print(f"    示例值: {df[col].head(3).tolist()}")
    else:
        print("❌ 未找到明显的图像路径列")
    return df


def main():
    DATA_ROOT = "C:\\Users\\dengyu\\scikit_learn_data\\smartdoc15-ch1"
    frames_csv = Path(DATA_ROOT) / "frames" / "frames_metadata.csv"
    models_csv = Path(DATA_ROOT) / "models" / "model_metadata.csv"
    df_frames = analyze_csv(frames_csv, "frames_metadata.csv")
    df_models = analyze_csv(models_csv, "model_metadata.csv")
    print(f"\n{'=' * 80}")
    print("总结")
    print(f"{'=' * 80}\n")
    if df_frames is not None:
        print(f"frames_metadata.csv:")
        print(f"  - 包含 {len(df_frames)} 条记录")
        print(f"  - 列数: {len(df_frames.columns)}")
        print(f"  - 主要列: {', '.join(df_frames.columns[:5])}...")
    if df_models is not None:
        print(f"\nmodel_metadata.csv:")
        print(f"  - 包含 {len(df_models)} 条记录")
        print(f"  - 列数: {len(df_models.columns)}")
        print(f"  - 主要列: {', '.join(df_models.columns[:5])}...")
    print(f"\n{'=' * 80}")
    print("建议")
    print(f"{'=' * 80}\n")
    print("1. 查看上述分析结果，确认 CSV 格式")
    print("2. 检查是否包含角点坐标（tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y）")
    print("3. 检查是否包含图像路径信息")
    print("4. 将结果发送给开发者，以便修改预处理脚本")
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()
    input("\n按回车键退出...")
