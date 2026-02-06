"""创建小规模测试数据集用于试运行。

从原始数据中提取少量样本，创建测试数据集。
"""

from pathlib import Path

import pyarrow.parquet as pq


def create_test_dataset(
    source_dir: Path,
    output_dir: Path,
    max_files: int = 2,
    max_rows_per_file: int = 1000,
) -> None:
    """创建测试数据集。

    Args:
        source_dir: 源数据目录
        output_dir: 输出目录
        max_files: 最大文件数
        max_rows_per_file: 每个文件最大行数
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有 parquet 文件
    parquet_files = list(source_dir.rglob("*.parquet"))
    selected_files = parquet_files[:max_files]

    print(f"选择 {len(selected_files)} 个文件用于测试")

    for i, file_path in enumerate(selected_files):
        # 读取 parquet 文件
        table = pq.read_table(file_path)
        df = table.to_pandas()

        # 只取前 N 行
        df = df.head(max_rows_per_file)

        # 构建输出路径，保持目录结构
        relative_path = file_path.relative_to(source_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为 parquet
        df.to_parquet(output_path, compression="zstd")

        print(f"  [{i + 1}/{len(selected_files)}] {relative_path}: {len(df)} 行")

    print(f"测试数据集已创建: {output_dir}")


if __name__ == "__main__":
    source = Path("data/datasets/HuggingFaceFW/fineweb-edu/data")
    output = Path("data/datasets/test_fineweb_input")

    create_test_dataset(source, output, max_files=2, max_rows_per_file=1000)
