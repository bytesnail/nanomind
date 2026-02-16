"""Parquet 文件合并工具。

在 Datatrove 处理完成后，将多个小文件合并成指定大小的大文件。
"""

import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config_loader import Compression

logger = logging.getLogger(__name__)

MAX_READ_WORKERS = 32


def _read_parquet_file(file_path: Path) -> tuple[Path, pa.Table | None]:
    """读取单个 parquet 文件。"""
    try:
        table = pq.read_table(file_path)
        return file_path, table
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {e}")
        return file_path, None


def _delete_file(file_path: Path) -> bool:
    """删除单个文件。"""
    try:
        file_path.unlink()
        return True
    except Exception as e:
        logger.warning(f"删除源文件失败 {file_path}: {e}")
        return False


def merge_bucket_files(
    bucket_dir: Path,
    target_file_size: int,
    compression: Compression = "zstd",
    remove_source: bool = True,
    max_workers: int | None = None,
) -> list[Path]:
    """将桶内的小文件合并成指定大小的大文件。

    采用流式处理方式：
    1. 批量并行读取小文件（控制内存占用）
    2. 使用 ParquetWriter 边读边写
    3. 达到目标大小后立即写入，不累积全部数据

    Args:
        bucket_dir: 桶目录路径
        target_file_size: 目标文件大小（字节）
        compression: 压缩格式
        remove_source: 是否删除源文件
        max_workers: 并行读取的工作线程数

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

    if len(parquet_files) == 1:
        source_file = parquet_files[0]
        target_file = bucket_dir / "00000.parquet"
        if source_file.name != target_file.name:
            source_file.rename(target_file)
            logger.info(
                f"桶 {bucket_dir.name} 只有一个文件，重命名为 {target_file.name}"
            )
        else:
            logger.info(f"桶 {bucket_dir.name} 只有一个文件，无需处理")
        return [target_file]

    logger.info(
        f"合并桶 {bucket_dir.name}: {len(parquet_files)} 个文件 -> 目标大小 {target_file_size / 1024 / 1024:.0f}MB"
    )

    if max_workers is None:
        max_workers = min(MAX_READ_WORKERS, multiprocessing.cpu_count() * 2)

    # 获取 schema
    try:
        first_table = pq.read_table(parquet_files[0])
        schema = first_table.schema
    except Exception as e:
        logger.error(f"无法读取第一个文件获取 schema: {e}")
        return []

    merged_files: list[Path] = []
    file_counter = 0
    current_writer: pq.ParquetWriter | None = None
    current_path: Path | None = None
    current_size = 0
    batch_size = max_workers * 2
    processed_sources: list[Path] = []

    for i in range(0, len(parquet_files), batch_size):
        batch = parquet_files[i : i + batch_size]

        # 并行读取批次
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(_read_parquet_file, pf): pf for pf in batch
            }
            batch_tables = []
            for future in as_completed(future_to_file):
                file_path, table = future.result()
                if table is not None:
                    batch_tables.append((file_path, table))

        batch_tables.sort(key=lambda x: x[0].name)

        for file_path, table in batch_tables:
            table_size = sum(col.nbytes for col in table.columns)

            if (
                current_writer is not None
                and current_size + table_size > target_file_size
            ):
                current_writer.close()
                if current_path is None:
                    raise RuntimeError(
                        "Internal error: current_path should not be None after write"
                    )
                file_size_mb = current_path.stat().st_size / 1024 / 1024
                logger.info(f"写入合并文件: {current_path.name} ({file_size_mb:.1f}MB)")
                merged_files.append(current_path)
                file_counter += 1
                current_writer = None
                current_size = 0

            if current_writer is None:
                current_path = bucket_dir / f"{file_counter:05d}.parquet"
                current_writer = pq.ParquetWriter(
                    current_path, schema, compression=compression
                )

            current_writer.write_table(table)
            current_size += table_size
            processed_sources.append(file_path)

    if current_writer is not None:
        current_writer.close()
        if current_path is None:
            raise RuntimeError(
                "Internal error: current_path should not be None after write"
            )
        file_size_mb = current_path.stat().st_size / 1024 / 1024
        logger.info(f"写入合并文件: {current_path.name} ({file_size_mb:.1f}MB)")
        merged_files.append(current_path)

    if remove_source and merged_files:
        logger.debug(f"删除 {len(processed_sources)} 个源文件")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_delete_file, processed_sources))

    logger.info(f"桶 {bucket_dir.name} 合并完成: {len(merged_files)} 个文件")
    return merged_files


def merge_all_buckets(
    output_dir: Path,
    target_file_size: int,
    compression: Compression = "zstd",
    remove_source: bool = True,
    max_workers: int | None = None,
) -> dict[str, list[Path]]:
    """串行合并所有桶的文件。

    Args:
        output_dir: 输出目录（包含各个桶的子目录）
        target_file_size: 目标文件大小（字节）
        compression: 压缩格式
        remove_source: 是否删除源文件
        max_workers: 每个桶的并行工作线程数

    Returns:
        每个桶的合并后文件路径字典
    """
    results = {}

    if not output_dir.exists():
        logger.warning(f"输出目录不存在: {output_dir}")
        return results

    for bucket_dir in output_dir.iterdir():
        if bucket_dir.is_dir():
            merged = merge_bucket_files(
                bucket_dir, target_file_size, compression, remove_source, max_workers
            )
            if merged:
                results[bucket_dir.name] = merged

    return results
