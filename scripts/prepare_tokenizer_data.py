#!/usr/bin/env python3
"""准备 Tokenizer 训练数据。

从多个数据源按配置比例采样，输出为 Parquet 格式。
使用真正的流式处理避免内存累积。

改进点:
1. 使用 ParquetFile.iter_batches() 实现真正的流式读取
2. 使用生成器链避免中间数据累积
3. 修复并行处理中的内存泄漏问题
4. 统一哈希计算逻辑，减少重复代码
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import logging
import os
import sys
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/tokenizer_data.yaml")
DEFAULT_WORKERS = min(32, (os.cpu_count() or 4))
DEFAULT_IO_WORKERS = DEFAULT_WORKERS * 2
DEFAULT_MAX_ROWS = 500_000
DEFAULT_BATCH_SIZE = 50_000
COMPRESSION = "zstd"

HASH_MODULUS = 2**64
MD5_BYTES_USED = 8


class DocHash(NamedTuple):
    """文档哈希信息。"""

    hash_value: int
    doc_id: str
    file_path: Path
    row_index: int


SampleDoc = dict[str, Any]


@dataclass
class SamplingConfig:
    """单个数据源的采样配置。"""

    name: str
    source: Path
    samples: int
    buckets: dict[str, int] = field(default_factory=dict)
    stars_filter: dict[str, int] = field(default_factory=dict)

    def get_all_counts(self) -> dict[str, int]:
        """获取所有分桶的样本数。"""
        if self.buckets:
            return self.buckets
        if self.stars_filter:
            return self.stars_filter
        return {}


@dataclass
class TokenizerDataConfig:
    """Tokenizer 数据准备的整体配置。"""

    datasets: dict[str, SamplingConfig]
    random_seed: int
    output_format: str
    output_dir: Path


@dataclass
class SamplingInfo:
    """采样信息统计。"""

    total_requested: int
    total_sampled: int
    sources: dict[str, dict[str, Any]]
    random_seed: int


def load_config(config_path: Path) -> TokenizerDataConfig:
    """从 YAML 文件加载采样配置。

    Args:
        config_path: 配置文件路径

    Returns:
        TokenizerDataConfig 实例

    Raises:
        FileNotFoundError: 配置文件不存在时
        ValueError: 配置格式错误时
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    datasets = {}
    for key, cfg in raw.get("datasets", {}).items():
        buckets = cfg.get("buckets", {})
        stars_filter = cfg.get("stars_filter", {})

        datasets[key] = SamplingConfig(
            name=cfg["name"],
            source=Path(cfg["source"]),
            samples=cfg["samples"],
            buckets={str(k): int(v["count"]) for k, v in buckets.items()}
            if buckets
            else {},
            stars_filter={str(k): int(v["count"]) for k, v in stars_filter.items()}
            if stars_filter
            else {},
        )

    return TokenizerDataConfig(
        datasets=datasets,
        random_seed=raw.get("random_seed", 42),
        output_format=raw.get("output_format", "parquet"),
        output_dir=Path(raw.get("output_dir", "data/datasets/nanomind_tokenizer")),
    )


def compute_doc_hash(doc_id: str, seed: int) -> int:
    """计算文档的确定性哈希值。

    Args:
        doc_id: 文档唯一标识
        seed: 随机种子

    Returns:
        64位哈希值
    """
    data = f"{seed}_{doc_id}".encode()
    return int.from_bytes(
        hashlib.md5(data, usedforsecurity=False).digest()[:MD5_BYTES_USED],
        "big",
    )


def get_bucket_files(source_dir: Path, bucket_name: str) -> list[Path]:
    """获取指定桶目录下的所有 Parquet 文件。

    Args:
        source_dir: 数据源根目录
        bucket_name: 桶名称（如 "4.0", "above_2"）

    Returns:
        Parquet 文件路径列表
    """
    bucket_dir = source_dir / bucket_name
    if not bucket_dir.exists():
        logger.warning(f"桶目录不存在: {bucket_dir}")
        return []

    files = sorted(bucket_dir.rglob("*.parquet"))
    logger.info(f"  [{bucket_name}] 找到 {len(files)} 个文件")
    return files


@lru_cache(maxsize=1024)
def get_file_row_count(file_path: Path) -> int:
    """获取 Parquet 文件的行数（快速统计，已缓存）。

    使用流式方式读取，避免将整个文件加载到内存。
    结果被缓存以避免对同一文件的重复统计。

    Args:
        file_path: Parquet 文件路径

    Returns:
        文件行数
    """
    try:
        metadata = pq.read_metadata(file_path)
        return metadata.num_rows
    except Exception as e:
        logger.warning(f"无法读取 {file_path} 元数据: {e}，使用流式统计")
        # 使用流式读取避免内存暴涨
        total_rows = 0
        with pq.ParquetFile(file_path) as pf:
            for batch in pf.iter_batches(batch_size=10000):
                total_rows += len(batch)
        return total_rows


def stream_file_rows(
    file_path: Path,
    text_column: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    indices: set[int] | None = None,
) -> Generator[tuple[int, str], None, None]:
    """流式读取文件的行，产生 (row_index, text) 对。

    关键改进: 使用 ParquetFile.iter_batches() 而不是 read_table()，
    避免一次性加载整个文件到内存。

    Args:
        file_path: Parquet 文件路径
        text_column: 文本列名
        batch_size: 每批次读取的行数
        indices: 可选的行索引集合，如果提供则只返回这些索引的行

    Yields:
        (行索引, 文本内容) 对
    """
    if indices is not None and not indices:
        return

    try:
        with pq.ParquetFile(file_path) as pf:
            row_idx = 0
            for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
                for text in batch.column(text_column).to_pylist():
                    if indices is None or row_idx in indices:
                        yield row_idx, text
                    row_idx += 1
    except Exception as e:
        action = "指定行" if indices else ""
        logger.warning(f"流式读取{action}文件失败 {file_path}: {e}")


def count_bucket_samples_parallel(
    files: list[Path], bucket_name: str, max_workers: int = 10
) -> int:
    """并行统计桶内的总样本数。

    Args:
        files: Parquet文件列表
        bucket_name: 桶名称
        max_workers: 最大并发数

    Returns:
        总样本数
    """
    if not files:
        return 0

    total = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_file_row_count, f): f for f in files}
        for future in tqdm(
            as_completed(futures),
            total=len(files),
            desc=f"统计 {bucket_name}",
            leave=False,
        ):
            try:
                total += future.result()
            except Exception as e:
                logger.warning(f"统计文件失败 {futures[future]}: {e}")

    return total


def select_top_k_document_hashes(
    files: list[Path],
    bucket_name: str,
    seed: int,
    target_count: int,
) -> dict[Path, set[int]]:
    """流式选择哈希值最小的前 target_count 个文档。

    内存优化设计:
    1. 使用文件索引(int)代替Path对象存储在堆中，大幅减少内存占用
    2. 串行处理文件避免并行内存累积
    3. 仅在最后构建文件到索引的映射

    Args:
        files: Parquet 文件列表
        bucket_name: 桶名称
        seed: 随机种子
        target_count: 目标采样数

    Returns:
        文件路径到行索引集合的映射
    """
    if not files:
        return {}

    total_count = sum(get_file_row_count(f) for f in files)

    if target_count >= total_count:
        logger.info(
            f"  [{bucket_name}] 目标数({target_count:,}) >= 总数({total_count:,})，跳过哈希计算"
        )
        result: dict[Path, set[int]] = {}
        for fp in files:
            num_rows = get_file_row_count(fp)
            if num_rows > 0:
                result[fp] = set(range(num_rows))
        return result

    max_heap: list[tuple[int, int, int]] = []
    total_scanned = 0
    file_list = list(files)

    logger.info(
        f"  [{bucket_name}] 流式选择 Top-{target_count:,} (总数 {total_count:,})"
    )

    for file_idx, fp in enumerate(
        tqdm(file_list, desc=f"哈希计算 {bucket_name}", leave=False)
    ):
        try:
            num_rows = get_file_row_count(fp)
            base_doc_id = f"{bucket_name}#{fp.name}#"

            for row_idx in range(num_rows):
                doc_id = f"{base_doc_id}{row_idx}"
                doc_hash = compute_doc_hash(doc_id, seed)
                total_scanned += 1

                if len(max_heap) < target_count:
                    heapq.heappush(max_heap, (-doc_hash, file_idx, row_idx))
                elif doc_hash < -max_heap[0][0]:
                    heapq.heapreplace(max_heap, (-doc_hash, file_idx, row_idx))

        except Exception as e:
            logger.warning(f"处理文件失败 {fp}: {e}")

    file_to_indices: dict[Path, set[int]] = {}
    for _, file_idx, row_idx in max_heap:
        fp = file_list[file_idx]
        file_to_indices.setdefault(fp, set()).add(row_idx)

    # 显式释放大对象，避免内存峰值
    del max_heap
    gc.collect()

    logger.info(
        f"  [{bucket_name}] 扫描 {total_scanned:,} 个文档，选中 {len(file_to_indices):,} 个"
    )

    return file_to_indices


def _read_all_files_streaming(
    files: list[Path],
    bucket_name: str,
    dataset_name: str,
    text_column: str,
    writer: StreamingParquetWriter,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """真正的流式读取所有文件，边读边写。

    关键改进: 使用 stream_file_rows() 生成器，避免加载整个文件。
    """
    total_sampled = 0

    for file_path in tqdm(files, desc=f"流式读取 {bucket_name}", leave=False):
        try:
            for _, text in stream_file_rows(file_path, text_column, batch_size):
                writer.write(create_sample_doc(text, dataset_name, bucket_name))
                total_sampled += 1
        except Exception as e:
            logger.warning(f"处理文件失败 {file_path}: {e}")

    return total_sampled


def _read_selected_files_streaming(
    file_to_indices: dict[Path, set[int]],
    bucket_name: str,
    dataset_name: str,
    text_column: str,
    writer: StreamingParquetWriter,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """真正的流式读取指定行，边读边写。"""
    total_sampled = 0

    logger.info(f"  [{bucket_name}] 流式读取 {len(file_to_indices)} 个文件")

    for file_path, indices in tqdm(
        file_to_indices.items(), desc=f"读取 {bucket_name}", leave=False
    ):
        try:
            for _, text in stream_file_rows(
                file_path, text_column, batch_size, indices
            ):
                writer.write(create_sample_doc(text, dataset_name, bucket_name))
                total_sampled += 1
        except Exception as e:
            logger.warning(f"读取文件失败 {file_path}: {e}")

    return total_sampled


def sample_from_bucket_streaming(
    source_dir: Path,
    bucket_name: str,
    target_count: int,
    seed: int,
    dataset_name: str,
    writer: StreamingParquetWriter,
    text_column: str = "text",
    io_workers: int = 8,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """从指定桶中流式采样指定数量的样本。

    采样策略：
    1. 如果目标数 >= 总数，直接流式读取全部数据
    2. 否则，使用流式 Top-K 选择哈希值最小的文档

    Args:
        source_dir: 数据源根目录
        bucket_name: 桶名称
        target_count: 目标采样数
        seed: 随机种子
        dataset_name: 数据集名称
        writer: 流式写入器
        text_column: 文本字段名
        io_workers: IO 统计阶段 workers（仅用于统计文件行数）
        batch_size: 流式读取批次大小

    Returns:
        实际采样的文档数
    """
    files = get_bucket_files(source_dir, bucket_name)
    if not files:
        logger.warning(f"桶 {bucket_name} 无数据文件")
        return 0

    # 首先统计总样本数
    total_count = count_bucket_samples_parallel(files, bucket_name, io_workers)
    if total_count == 0:
        return 0

    logger.info(
        f"  [{bucket_name}] 总样本: {total_count:,}, 目标: {target_count:,} "
        f"(采样率: {target_count / total_count:.1%})"
    )

    # 如果目标数超过总数，流式读取全部
    if target_count >= total_count:
        logger.info(f"  [{bucket_name}] 目标数超过总数，全部流式采样")
        sampled = _read_all_files_streaming(
            files, bucket_name, dataset_name, text_column, writer, batch_size
        )
        logger.info(f"  [{bucket_name}] 实际采样: {sampled:,}")
        return sampled

    # 计算哈希值并排序 - 使用流式 Top-K
    logger.info(f"  [{bucket_name}] 流式计算哈希...")
    file_to_indices = select_top_k_document_hashes(
        files, bucket_name, seed, target_count
    )

    selected_count = sum(len(v) for v in file_to_indices.values())
    logger.info(f"  [{bucket_name}] 已选择 {selected_count:,} 个文档，开始流式读取...")

    # 流式读取选中的行
    sampled = _read_selected_files_streaming(
        file_to_indices, bucket_name, dataset_name, text_column, writer, batch_size
    )

    logger.info(f"  [{bucket_name}] 实际采样: {sampled:,}")
    return sampled


def determine_text_column(dataset_name: str) -> str:
    """根据数据集名称确定文本字段名。

    Args:
        dataset_name: 数据集名称

    Returns:
        文本字段名
    """
    if "github" in dataset_name.lower():
        return "content"
    return "text"


def create_sample_doc(text: str, dataset_name: str, bucket_name: str) -> SampleDoc:
    return {
        "text": text,
        "source_dataset": dataset_name,
        "source_bucket": bucket_name,
    }


def process_dataset_streaming(
    config: SamplingConfig,
    seed: int,
    writer: StreamingParquetWriter,
    io_workers: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """流式处理单个数据集的所有桶.

    Args:
        config: 采样配置
        seed: 随机种子
        writer: 流式写入器
        io_workers: IO 统计阶段 workers
        batch_size: 流式读取批次大小

    Returns:
        统计信息
    """
    logger.info(f"处理数据集: {config.name}")
    logger.info(f"  源目录: {config.source}")

    bucket_stats = {}
    text_column = determine_text_column(config.name)

    counts = config.get_all_counts()
    for bucket_name, target_count in counts.items():
        sampled = sample_from_bucket_streaming(
            source_dir=config.source,
            bucket_name=bucket_name,
            target_count=target_count,
            seed=seed,
            dataset_name=config.name,
            writer=writer,
            text_column=text_column,
            io_workers=io_workers,
            batch_size=batch_size,
        )
        bucket_stats[bucket_name] = {
            "requested": target_count,
            "sampled": sampled,
        }

    total_requested = sum(s["requested"] for s in bucket_stats.values())
    total_sampled = sum(s["sampled"] for s in bucket_stats.values())

    logger.info(
        f"  [{config.name}] 总计: 请求 {total_requested:,}, 实际 {total_sampled:,}"
    )

    return {
        "name": config.name,
        "source": str(config.source),
        "requested": total_requested,
        "sampled": total_sampled,
        "buckets": bucket_stats,
    }


class StreamingParquetWriter:
    """流式 Parquet 写入器，边生成边写入，避免内存累积。"""

    def __init__(
        self,
        output_dir: Path,
        prefix: str = "train",
        max_rows_per_file: int = DEFAULT_MAX_ROWS,
        buffer_size: int = 5000,
    ):
        """初始化流式写入器。

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
            max_rows_per_file: 每个文件的最大行数
            buffer_size: 内存缓冲区的行数，达到此值即写入磁盘
        """
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_rows_per_file = max_rows_per_file
        self.buffer_size = buffer_size
        self.output_files: list[Path] = []
        self.current_batch: list[SampleDoc] = []
        self.total_written = 0
        self.file_idx = 0
        self._rows_in_current_file = 0

        output_dir.mkdir(parents=True, exist_ok=True)

    def _write_batch(self) -> None:
        """将当前批次写入文件。"""
        if not self.current_batch:
            return

        texts = [s["text"] for s in self.current_batch]
        source_datasets = [s["source_dataset"] for s in self.current_batch]
        source_buckets = [s["source_bucket"] for s in self.current_batch]

        table = pa.table(
            {
                "text": texts,
                "source_dataset": source_datasets,
                "source_bucket": source_buckets,
            }
        )

        # 使用临时文件名，在 close() 中统一重命名
        temp_filename = f"{self.prefix}-{self.file_idx:05d}.part"
        temp_path = self.output_dir / temp_filename

        pq.write_table(table, temp_path, compression=COMPRESSION)
        self.output_files.append(temp_path)

        logger.info(f"  写入批次 {self.file_idx} ({len(self.current_batch):,} 行)")

        self.total_written += len(self.current_batch)
        self._rows_in_current_file += len(self.current_batch)
        self.current_batch = []

        # 如果当前文件达到上限，开始新文件
        if self._rows_in_current_file >= self.max_rows_per_file:
            self.file_idx += 1
            self._rows_in_current_file = 0

    def write(self, sample: SampleDoc) -> None:
        """写入单个样本。

        Args:
            sample: 单个样本文档
        """
        self.current_batch.append(sample)

        # 检查是否需要刷盘：缓冲区满或当前文件达到上限
        if len(self.current_batch) >= self.buffer_size:
            self._write_batch()
        elif (
            self._rows_in_current_file + len(self.current_batch)
            >= self.max_rows_per_file
        ):
            self._write_batch()

    def close(self) -> list[Path]:
        """关闭写入器，写入剩余数据并返回所有文件路径。"""
        if self.current_batch:
            self._write_batch()

        total_files = len(self.output_files)
        for i, temp_path in enumerate(self.output_files):
            final_path = (
                self.output_dir / f"{self.prefix}-{i:05d}-of-{total_files:05d}.parquet"
            )
            temp_path.rename(final_path)
            self.output_files[i] = final_path

        logger.info(f"总计写入 {self.total_written:,} 个样本到 {total_files} 个文件")
        return self.output_files

    def __enter__(self) -> StreamingParquetWriter:
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出。"""
        self.close()


def save_sampling_info(
    info: SamplingInfo,
    output_dir: Path,
) -> Path:
    """保存采样信息到 JSON 文件。

    Args:
        info: 采样信息
        output_dir: 输出目录

    Returns:
        写入的文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = output_dir / "sampling_info.json"

    data = {
        "total_requested": info.total_requested,
        "total_sampled": info.total_sampled,
        "random_seed": info.random_seed,
        "sources": info.sources,
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"采样信息已保存: {info_path}")
    return info_path


def _log_section(title: str, width: int = 60) -> None:
    """记录带分隔线的章节标题。

    Args:
        title: 章节标题
        width: 分隔线宽度
    """
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def _log_config_info(
    config: TokenizerDataConfig,
    workers: int,
    io_workers: int,
    batch_size: int,
) -> None:
    """记录配置信息。"""
    _log_section("准备 Tokenizer 训练数据")
    logger.info(f"配置文件: {CONFIG_PATH}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"随机种子: {config.random_seed}")
    logger.info(f"整体 workers 基准: {workers}")
    logger.info(f"  - IO 统计阶段: {io_workers} workers")
    logger.info(f"  - 流式读取批次: {batch_size:,} 行")
    logger.info(f"数据集数量: {len(config.datasets)}")
    logger.info("流式处理: 已启用 (边读取边写入)")


def _calculate_totals(
    source_stats: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """Helper: 计算请求总数和采样总数。"""
    total_requested = sum(s["requested"] for s in source_stats.values())
    total_sampled = sum(s["sampled"] for s in source_stats.values())
    return total_requested, total_sampled


def _log_final_report(
    source_stats: dict[str, dict[str, Any]],
    output_files: list[Path],
    output_dir: Path,
    total_requested: int,
    total_sampled: int,
) -> None:
    """记录最终报告。"""
    print()
    _log_section("处理完成")
    logger.info(f"总请求样本: {total_requested:,}")
    logger.info(f"总采样样本: {total_sampled:,}")
    logger.info(f"采样率: {total_sampled / total_requested:.1%}")
    logger.info(f"输出文件: {len(output_files)} 个")
    logger.info(f"输出目录: {output_dir}")

    print()
    logger.info("各数据源采样详情:")
    for source_key, stats in source_stats.items():
        logger.info(f"  [{source_key}] {stats['name']}")
        logger.info(f"    请求: {stats['requested']:,}, 采样: {stats['sampled']:,}")
        if stats["buckets"]:
            for bucket_name, bucket_stats in stats["buckets"].items():
                logger.info(
                    f"      {bucket_name}: {bucket_stats['sampled']:,} "
                    f"(目标: {bucket_stats['requested']:,})"
                )


def prepare_tokenizer_data(
    workers: int,
    io_workers: int,
    max_rows_per_file: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """主函数：准备 tokenizer 训练数据。

    使用流式处理避免内存累积，边采样边写入文件。

    Args:
        workers: 整体并行度基准
        io_workers: 文件统计阶段的 IO 并行度
        max_rows_per_file: 每个输出文件的行数上限
        batch_size: 流式读取的批次大小

    Returns:
        退出码 (0=成功)
    """
    try:
        config = load_config(CONFIG_PATH)
        _log_config_info(config, workers, io_workers, batch_size)

        source_stats = {}
        with StreamingParquetWriter(
            output_dir=config.output_dir,
            prefix="train",
            max_rows_per_file=max_rows_per_file,
        ) as writer:
            for source_key, dataset_config in config.datasets.items():
                print()
                stats = process_dataset_streaming(
                    config=dataset_config,
                    seed=config.random_seed,
                    writer=writer,
                    io_workers=io_workers,
                    batch_size=batch_size,
                )
                source_stats[source_key] = stats

                logger.info(f"  [{source_key}] 已流式写入 {stats['sampled']:,} 个样本")

            output_files = writer.output_files

        total_requested, total_sampled = _calculate_totals(source_stats)

        info = SamplingInfo(
            total_requested=total_requested,
            total_sampled=total_sampled,
            sources=source_stats,
            random_seed=config.random_seed,
        )
        save_sampling_info(info, config.output_dir)
        _log_final_report(
            source_stats,
            output_files,
            config.output_dir,
            total_requested,
            total_sampled,
        )

        return 0

    except Exception as e:
        logger.exception(f"处理失败: {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="准备 Tokenizer 训练数据 - 多数据源采样",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/prepare_tokenizer_data.py

  # 指定整体 workers 数量（自动分配到各阶段）
  python scripts/prepare_tokenizer_data.py --workers 4

  # 调整流式读取批次大小以优化内存
  python scripts/prepare_tokenizer_data.py --batch-size 5000

  # 调整每个文件的行数
  python scripts/prepare_tokenizer_data.py --max-rows 100000
        """,
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"整体并行度基准。 (默认: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=None,
        help=f"文件统计阶段的 IO 并行度 (默认: {DEFAULT_IO_WORKERS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"流式读取的批次大小 (默认: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"每个输出文件的最大行数 (默认: {DEFAULT_MAX_ROWS})",
    )

    args = parser.parse_args()

    io_workers = args.io_workers if args.io_workers is not None else DEFAULT_IO_WORKERS

    return prepare_tokenizer_data(
        workers=args.workers,
        io_workers=io_workers,
        max_rows_per_file=args.max_rows,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    sys.exit(main())
