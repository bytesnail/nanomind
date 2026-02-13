#!/usr/bin/env python3
"""准备 Tokenizer 训练数据。

从多个数据源按配置比例采样，输出为 Parquet 格式。
详见项目文档了解使用方法和输出结构。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

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

# 默认配置
CONFIG_PATH = Path("config/tokenizer_data.yaml")
DEFAULT_WORKERS = 32
DEFAULT_MAX_ROWS = 200_000
COMPRESSION = "zstd"
IO_WORKERS_MULTIPLIER = 10

HASH_MODULUS = 2**64
MD5_BYTES_USED = 8

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    items: list[T],
    worker_func: Callable[[T], R],
    max_workers: int,
    desc: str,
    on_error: Callable[[T, Exception], R] | None = None,
) -> list[R]:
    """并行执行函数，统一处理 ThreadPoolExecutor 和 tqdm 进度条。

    Args:
        items: 待处理的项目列表
        worker_func: 处理每个项目的函数
        max_workers: 最大并行数
        desc: 进度条描述
        on_error: 错误处理回调，接收 (item, exception) 返回默认值

    Returns:
        处理结果列表（保持与输入相同的顺序）
    """
    results: list[R | None] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(worker_func, item): idx for idx, item in enumerate(items)
        }
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(items),
            desc=desc,
            leave=False,
        ):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                if on_error:
                    results[idx] = on_error(items[idx], e)
                else:
                    logger.warning(f"处理失败 {items[idx]}: {e}")

    return [r for r in results if r is not None]


class DocHash(NamedTuple):
    """文档哈希信息。"""

    hash_value: int
    doc_id: str
    file_path: Path
    row_index: int


class SampleDoc(dict[str, Any]):
    """样本文档字典。"""

    pass


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


def deterministic_sample(doc_hash: int, target_count: int, total_count: int) -> bool:
    """基于哈希值的确定性采样。

    Args:
        doc_hash: 文档哈希值
        target_count: 目标采样数
        total_count: 总数

    Returns:
        是否选中该文档
    """
    if target_count >= total_count:
        return True
    rate = target_count / total_count
    return doc_hash / HASH_MODULUS < rate


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


def get_file_row_count(file_path: Path) -> int:
    """获取 Parquet 文件的行数（快速统计）。

    Args:
        file_path: Parquet 文件路径

    Returns:
        文件行数
    """
    try:
        metadata = pq.read_metadata(file_path)
        return metadata.num_rows
    except Exception as e:
        logger.warning(f"无法读取 {file_path} 元数据: {e}")
        return pq.read_table(file_path).num_rows


def _get_file_info(
    file_path: Path, bucket_name: str, seed: int
) -> tuple[Path, int, list[DocHash]]:
    """获取文件信息：行数和文档哈希列表。

    Returns:
        (文件路径, 行数, [(哈希值, doc_id, 文件路径, 行索引), ...])
    """
    try:
        num_rows = get_file_row_count(file_path)
        hashes = []
        for i in range(num_rows):
            doc_id = f"{bucket_name}#{file_path.name}#{i}"
            doc_hash = compute_doc_hash(doc_id, seed)
            hashes.append((doc_hash, doc_id, file_path, i))
        return file_path, num_rows, hashes
    except Exception as e:
        logger.warning(f"处理文件失败 {file_path}: {e}")
        return file_path, 0, []


def count_bucket_samples_parallel(
    files: list[Path], bucket_name: str, max_workers: int = 32
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

    counts = parallel_map(
        items=files,
        worker_func=get_file_row_count,
        max_workers=max_workers,
        desc=f"统计 {bucket_name}",
        on_error=_on_file_error,
    )
    return sum(counts)


def _table_to_sample_docs(
    table: pa.Table,
    dataset_name: str,
    bucket_name: str,
    text_column: str,
    indices: list[int] | None = None,
) -> list[SampleDoc]:
    """将 PyArrow Table 转换为 SampleDoc 列表。

    Args:
        table: PyArrow Table
        dataset_name: 数据集名称
        bucket_name: 桶名称
        text_column: 文本字段名
        indices: 要读取的行索引列表，None 表示全部

    Returns:
        SampleDoc 列表
    """
    texts = table.column(text_column).to_pylist()
    if indices is not None:
        texts = [texts[i] for i in indices if i < len(texts)]
    return [create_sample_doc(t, dataset_name, bucket_name) for t in texts]


def _read_all_files(
    files: list[Path],
    bucket_name: str,
    dataset_name: str,
    text_column: str,
) -> list[SampleDoc]:
    """读取所有文件的全部内容。

    Args:
        files: Parquet 文件列表
        bucket_name: 桶名称
        dataset_name: 数据集名称
        text_column: 文本字段名

    Returns:
        所有文档列表
    """
    sampled: list[SampleDoc] = []
    for file_path in tqdm(files, desc=f"采样 {bucket_name}", leave=False):
        try:
            table = pq.read_table(file_path, columns=[text_column])
            sampled.extend(
                _table_to_sample_docs(table, dataset_name, bucket_name, text_column)
            )
        except Exception as e:
            logger.warning(f"处理文件失败 {file_path}: {e}")
    return sampled


def _merge_file_info_results(
    results: Sequence[tuple[Path | None, int, list[DocHash]]],
) -> list[DocHash]:
    """合并文件信息结果，提取所有文档哈希。"""
    doc_hashes: list[DocHash] = []
    for _, _, hashes in results:
        doc_hashes.extend(hashes)
    return doc_hashes


def _create_error_handler(
    default_return: R, log_template: str
) -> Callable[[Any, Exception], R]:
    """创建错误处理回调函数工厂。

    Args:
        default_return: 错误时返回的默认值
        log_template: 日志消息模板，应包含 {item} 和 {error} 占位符

    Returns:
        错误处理回调函数
    """

    def handler(item: Any, error: Exception) -> R:
        # 从 item 中提取文件路径（支持 Path 或 tuple）
        file_path = item[0] if isinstance(item, tuple) else item
        logger.warning(log_template.format(item=file_path, error=error))
        return default_return

    return handler


# 使用工厂函数创建具体的错误处理回调
_on_file_error = _create_error_handler(0, "处理文件失败 {item}: {error}")
_on_compute_hash_error = _create_error_handler(
    (None, 0, []), "计算哈希失败 {item}: {error}"
)
_on_read_error = _create_error_handler([], "读取文件失败 {item}: {error}")


def _compute_doc_hashes(
    files: list[Path],
    bucket_name: str,
    seed: int,
    max_workers: int,
) -> list[DocHash]:
    """并行计算所有文档的哈希值。

    Args:
        files: Parquet 文件列表
        bucket_name: 桶名称
        seed: 随机种子
        max_workers: 最大并行数

    Returns:
        文档哈希列表，每个元素为 (哈希值, doc_id, 文件路径, 行索引)
    """
    from functools import partial

    worker = partial(_get_file_info, bucket_name=bucket_name, seed=seed)

    results = parallel_map(
        items=files,
        worker_func=worker,
        max_workers=max_workers,
        desc=f"计算哈希 {bucket_name}",
        on_error=_on_compute_hash_error,
    )
    return _merge_file_info_results(results)


def _read_selected_rows(
    file_path: Path,
    indices: list[int],
    dataset_name: str,
    bucket_name: str,
    text_column: str,
) -> list[SampleDoc]:
    """从文件中读取指定行的数据。"""
    try:
        table = pq.read_table(file_path, columns=[text_column])
        return _table_to_sample_docs(
            table, dataset_name, bucket_name, text_column, indices
        )
    except Exception as err:
        logger.warning(f"读取文件失败 {file_path}: {err}")
        return []


ReadJob = tuple[Path, list[int]]


def _read_selected_files(
    file_to_indices: dict[Path, list[int]],
    bucket_name: str,
    dataset_name: str,
    text_column: str,
    max_workers: int,
) -> list[SampleDoc]:
    """并行读取多个文件的指定行。

    Args:
        file_to_indices: 文件路径到行索引列表的映射
        bucket_name: 桶名称
        dataset_name: 数据集名称
        text_column: 文本字段名
        max_workers: 最大并行数

    Returns:
        采样的文档列表
    """
    jobs: list[ReadJob] = list(file_to_indices.items())

    def _worker(job: ReadJob) -> list[SampleDoc]:
        file_path, indices = job
        return _read_selected_rows(
            file_path, indices, dataset_name, bucket_name, text_column
        )

    results = parallel_map(
        items=jobs,
        worker_func=_worker,
        max_workers=max_workers,
        desc=f"读取 {bucket_name}",
        on_error=_on_read_error,
    )

    sampled: list[SampleDoc] = []
    for samples in results:
        sampled.extend(samples)
    return sampled


def sample_from_bucket(
    source_dir: Path,
    bucket_name: str,
    target_count: int,
    seed: int,
    dataset_name: str,
    text_column: str = "text",
    max_workers: int = 32,
) -> list[SampleDoc]:
    """从指定桶中采样指定数量的样本。

    采样策略：
    1. 如果目标数 >= 总数，直接读取全部数据
    2. 否则，基于哈希值排序后取前 target_count 个

    Args:
        source_dir: 数据源根目录
        bucket_name: 桶名称
        target_count: 目标采样数
        seed: 随机种子
        dataset_name: 数据集名称
        text_column: 文本字段名
        max_workers: 文件读取并行度

    Returns:
        采样的文档列表
    """
    files = get_bucket_files(source_dir, bucket_name)
    if not files:
        logger.warning(f"桶 {bucket_name} 无数据文件")
        return []

    # IO密集型操作使用更高并发
    io_workers = max_workers * IO_WORKERS_MULTIPLIER

    # 首先统计总样本数
    total_count = count_bucket_samples_parallel(files, bucket_name, io_workers)
    if total_count == 0:
        return []

    logger.info(
        f"  [{bucket_name}] 总样本: {total_count:,}, 目标: {target_count:,} "
        f"(采样率: {target_count / total_count:.1%})"
    )

    # 如果目标数超过总数，直接读取全部
    if target_count >= total_count:
        logger.info(f"  [{bucket_name}] 目标数超过总数，全部采样")
        sampled = _read_all_files(files, bucket_name, dataset_name, text_column)
        logger.info(f"  [{bucket_name}] 实际采样: {len(sampled):,}")
        return sampled

    # 计算哈希值并排序
    logger.info(f"  [{bucket_name}] 并行计算哈希 (max_workers={io_workers})...")
    doc_hashes = _compute_doc_hashes(files, bucket_name, seed, io_workers)
    doc_hashes.sort(key=lambda x: x[0])

    # 构建文件到行索引的映射
    file_to_indices: dict[Path, list[int]] = {}
    for doc_hash in doc_hashes[:target_count]:
        _, _, file_path, row_idx = doc_hash
        file_to_indices.setdefault(file_path, []).append(row_idx)

    logger.info(f"  [{bucket_name}] 已选择 {target_count:,} 个文档，开始读取...")

    # 并行读取选中的行
    sampled = _read_selected_files(
        file_to_indices, bucket_name, dataset_name, text_column, max_workers
    )

    logger.info(f"  [{bucket_name}] 实际采样: {len(sampled):,}")
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
    return SampleDoc(
        text=text,
        source_dataset=dataset_name,
        source_bucket=bucket_name,
    )


def process_dataset(
    config: SamplingConfig,
    seed: int,
    max_workers: int = 32,
) -> tuple[list[SampleDoc], dict[str, Any]]:
    """处理单个数据集的所有桶。

    Args:
        config: 采样配置
        seed: 随机种子
        max_workers: 文件读取并行度

    Returns:
        (采样的文档列表, 统计信息)
    """
    logger.info(f"处理数据集: {config.name}")
    logger.info(f"  源目录: {config.source}")

    all_samples = []
    bucket_stats = {}
    text_column = determine_text_column(config.name)

    counts = config.get_all_counts()
    for bucket_name, target_count in counts.items():
        samples = sample_from_bucket(
            source_dir=config.source,
            bucket_name=bucket_name,
            target_count=target_count,
            seed=seed,
            dataset_name=config.name,
            text_column=text_column,
            max_workers=max_workers,
        )
        all_samples.extend(samples)
        bucket_stats[bucket_name] = {
            "requested": target_count,
            "sampled": len(samples),
        }

    total_requested = sum(s["requested"] for s in bucket_stats.values())
    total_sampled = sum(s["sampled"] for s in bucket_stats.values())

    logger.info(
        f"  [{config.name}] 总计: 请求 {total_requested:,}, 实际 {total_sampled:,}"
    )

    return all_samples, {
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
    ):
        """初始化流式写入器。

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
            max_rows_per_file: 每个文件的最大行数
        """
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_rows_per_file = max_rows_per_file
        self.output_files: list[Path] = []
        self.current_batch: list[SampleDoc] = []
        self.total_written = 0
        self.file_idx = 0

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
        self.current_batch = []
        self.file_idx += 1

    def write(self, sample: SampleDoc) -> None:
        """写入单个样本。

        Args:
            sample: 单个样本文档
        """
        self.current_batch.append(sample)

        if len(self.current_batch) >= self.max_rows_per_file:
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

    def __enter__(self) -> "StreamingParquetWriter":
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


def _log_config_info(config: TokenizerDataConfig, workers: int) -> None:
    """记录配置信息。"""
    _log_section("准备 Tokenizer 训练数据")
    logger.info(f"配置文件: {CONFIG_PATH}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"随机种子: {config.random_seed}")
    logger.info(f"并行 workers: {workers}")
    logger.info(f"数据集数量: {len(config.datasets)}")
    logger.info("流式写入: 是 (避免内存累积)")


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
) -> None:
    """记录最终报告。"""
    total_requested, total_sampled = _calculate_totals(source_stats)

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
    max_rows_per_file: int,
) -> int:
    """主函数：准备 tokenizer 训练数据。

    使用流式写入避免内存累积，边采样边写入文件。

    Args:
        workers: 并行工作进程数（预留）
        max_rows_per_file: 每个文件的行数上限

    Returns:
        退出码 (0=成功)
    """
    try:
        config = load_config(CONFIG_PATH)
        _log_config_info(config, workers)

        source_stats = {}
        with StreamingParquetWriter(
            output_dir=config.output_dir,
            prefix="train",
            max_rows_per_file=max_rows_per_file,
        ) as writer:
            for source_key, dataset_config in config.datasets.items():
                print()
                samples, stats = process_dataset(
                    config=dataset_config,
                    seed=config.random_seed,
                    max_workers=workers,
                )
                source_stats[source_key] = stats

                for sample in samples:
                    writer.write(sample)

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
        _log_final_report(source_stats, output_files, config.output_dir)

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

  # 指定 workers 数量
  python scripts/prepare_tokenizer_data.py --workers 32

  # 调整每个文件的行数
  python scripts/prepare_tokenizer_data.py --max-rows 100000
        """,
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"并行文件读取数，控制统计/哈希计算/数据读取的并发度"
            f" (默认: {DEFAULT_WORKERS})"
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"每个输出文件的最大行数 (默认: {DEFAULT_MAX_ROWS})",
    )

    args = parser.parse_args()

    return prepare_tokenizer_data(
        workers=args.workers,
        max_rows_per_file=args.max_rows,
    )


if __name__ == "__main__":
    sys.exit(main())
