"""Parquet 文件合并工具。

在 Datatrove 处理完成后，将多个小文件合并成指定大小的大文件。
"""

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config_loader import Compression

logger = logging.getLogger(__name__)


def merge_bucket_files(
    bucket_dir: Path,
    target_file_size: int,
    compression: Compression = "zstd",
    remove_source: bool = True,
) -> list[Path]:
    """将桶内的小文件合并成指定大小的大文件。

    Args:
        bucket_dir: 桶目录路径
        target_file_size: 目标文件大小（字节）
        compression: 压缩格式
        remove_source: 是否删除源文件

    Returns:
        合并后的文件路径列表
    """
    if not bucket_dir.exists():
        logger.warning(f"桶目录不存在: {bucket_dir}")
        return []

    parquet_files = sorted(bucket_dir.glob("*.parquet"))
    if not parquet_files:
        logger.info(f"桶目录为空: {bucket_dir}")
        return []

    # 如果只有一个文件且大小已经合适，不需要合并
    if len(parquet_files) == 1:
        file_size = parquet_files[0].stat().st_size
        if file_size >= target_file_size * 0.5:  # 至少达到目标大小的50%
            logger.info(f"桶 {bucket_dir.name} 只有一个合适大小的文件，跳过合并")
            return parquet_files

    logger.info(
        f"合并桶 {bucket_dir.name}: {len(parquet_files)} 个文件 -> 目标大小 {target_file_size / 1024 / 1024:.0f}MB"
    )

    merged_files: list[Path] = []
    current_tables: list[pa.Table] = []
    current_size = 0
    file_counter = 0

    for parquet_file in parquet_files:
        try:
            table = pq.read_table(parquet_file)
            table_size = sum(col.nbytes for col in table.columns)

            # 检查添加此表后是否会超过目标大小
            if current_tables and current_size + table_size > target_file_size:
                # 写入当前累积的数据
                merged_file = _write_merged_file(
                    bucket_dir, current_tables, file_counter, compression
                )
                merged_files.append(merged_file)
                file_counter += 1

                # 重置累积状态
                current_tables = []
                current_size = 0

            current_tables.append(table)
            current_size += table_size

        except Exception as e:
            logger.error(f"读取文件失败 {parquet_file}: {e}")
            continue

    # 写入剩余的数据
    if current_tables:
        merged_file = _write_merged_file(
            bucket_dir, current_tables, file_counter, compression
        )
        merged_files.append(merged_file)

    # 删除源文件
    if remove_source and merged_files:
        for parquet_file in parquet_files:
            try:
                parquet_file.unlink()
            except Exception as e:
                logger.warning(f"删除源文件失败 {parquet_file}: {e}")

    logger.info(f"桶 {bucket_dir.name} 合并完成: {len(merged_files)} 个文件")
    return merged_files


def _write_merged_file(
    bucket_dir: Path,
    tables: list[pa.Table],
    counter: int,
    compression: Compression,
) -> Path:
    """将多个表写入合并后的文件。"""
    if not tables:
        raise ValueError("表列表不能为空")

    # 合并所有表
    merged_table = pa.concat_tables(tables)

    # 生成文件名: {counter:05d}.parquet
    output_path = bucket_dir / f"{counter:05d}.parquet"

    # 写入文件
    pq.write_table(merged_table, output_path, compression=compression)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    row_count = len(merged_table)
    logger.info(
        f"写入合并文件: {output_path.name} ({file_size_mb:.1f}MB, {row_count} 行)"
    )

    return output_path


def merge_all_buckets(
    output_dir: Path,
    target_file_size: int,
    compression: Compression = "zstd",
    remove_source: bool = True,
) -> dict[str, list[Path]]:
    """合并所有桶的文件。

    Args:
        output_dir: 输出目录（包含各个桶的子目录）
        target_file_size: 目标文件大小（字节）
        compression: 压缩格式
        remove_source: 是否删除源文件

    Returns:
        每个桶的合并后文件路径字典
    """
    results = {}

    if not output_dir.exists():
        logger.warning(f"输出目录不存在: {output_dir}")
        return results

    # 遍历所有桶目录
    for bucket_dir in output_dir.iterdir():
        if bucket_dir.is_dir():
            merged = merge_bucket_files(
                bucket_dir, target_file_size, compression, remove_source
            )
            if merged:
                results[bucket_dir.name] = merged

    return results
