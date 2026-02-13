#!/usr/bin/env python3
"""
准备 Tokenizer 训练数据

从多个数据源（FineWeb-EN, FineWeb-ZH, GitHub Code, Nemotron-CC-Math）
按配置比例采样 40M 样本，输出为 Parquet 格式。

用法:
    python scripts/prepare_tokenizer_data.py
    python scripts/prepare_tokenizer_data.py --workers 32

输出:
    data/datasets/nanomind_tokenizer/
    ├── train-{idx:05d}-of-{total:05d}.parquet
    └── sampling_info.json

采样结果统计分析:
    输出文件包含以下字段:
        - text: 文档文本内容
        - source_dataset: 来源数据集名称 (如 fineweb_edu_en, github_code)
        - source_bucket: 来源分桶标识 (如 4.0, above_2)

    使用 pandas 进行统计分析示例:
        import pandas as pd

        # 读取所有 parquet 文件
        df = pd.read_parquet("data/datasets/nanomind_tokenizer/")

        # 1. 按数据集统计样本数
        df.groupby("source_dataset").size()

        # 2. 按分桶统计样本数
        df.groupby("source_bucket").size()

        # 3. 按数据集+分桶交叉统计
        df.groupby(["source_dataset", "source_bucket"]).size()

        # 4. 验证采样比例是否符合设计目标
        sampling_info = pd.read_json(
            "data/datasets/nanomind_tokenizer/sampling_info.json"
        )
        for source, stats in sampling_info["sources"].items():
            print(f"{source}: 目标 {stats['requested']}, 实际 {stats['sampled']}")
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

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
DEFAULT_MAX_ROWS = 200000
COMPRESSION = "zstd"

HASH_MODULUS = 2**64
MD5_BYTES_USED = 8

DocHash: TypeAlias = tuple[int, str, Path, int]
SampleDoc: TypeAlias = dict[str, Any]


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

    total = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(get_file_row_count, f): f for f in files}
        for future in tqdm(
            as_completed(future_to_file),
            total=len(files),
            desc=f"统计 {bucket_name}",
            leave=False,
        ):
            try:
                total += future.result()
            except Exception as e:
                file_path = future_to_file[future]
                logger.warning(f"统计文件失败 {file_path}: {e}")

    return total


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

    # 首先统计总样本数（IO密集型，使用 2x 并发）
    io_workers = max_workers * 2
    total_count = count_bucket_samples_parallel(files, bucket_name, io_workers)
    if total_count == 0:
        return []

    logger.info(
        f"  [{bucket_name}] 总样本: {total_count:,}, 目标: {target_count:,} "
        f"(采样率: {target_count / total_count:.1%})"
    )

    if target_count >= total_count:
        logger.info(f"  [{bucket_name}] 目标数超过总数，全部采样")
        sampled = []
        for file_path in tqdm(files, desc=f"采样 {bucket_name}", leave=False):
            try:
                table = pq.read_table(file_path, columns=[text_column])
                texts = table.column(text_column).to_pylist()
                sampled.extend(
                    create_sample_doc(t, dataset_name, bucket_name) for t in texts
                )
            except Exception as e:
                logger.warning(f"处理文件失败 {file_path}: {e}")
        logger.info(f"  [{bucket_name}] 实际采样: {len(sampled):,}")
        return sampled

    logger.info(f"  [{bucket_name}] 并行计算哈希 (max_workers={io_workers})...")
    doc_hashes = []
    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        future_to_file = {
            executor.submit(_get_file_info, f, bucket_name, seed): f for f in files
        }
        for future in tqdm(
            as_completed(future_to_file),
            total=len(files),
            desc=f"计算哈希 {bucket_name}",
            leave=False,
        ):
            try:
                _, _, hashes = future.result()
                doc_hashes.extend(hashes)
            except Exception as e:
                file_path = future_to_file[future]
                logger.warning(f"计算哈希失败 {file_path}: {e}")

    doc_hashes.sort(key=lambda x: x[0])
    selected_ids = set(doc_hash[1] for doc_hash in doc_hashes[:target_count])

    logger.info(f"  [{bucket_name}] 已选择 {len(selected_ids):,} 个文档，开始读取...")

    file_to_indices: dict[Path, list[int]] = {}
    for doc_hash in doc_hashes[:target_count]:
        _, doc_id, file_path, row_idx = doc_hash
        if file_path not in file_to_indices:
            file_to_indices[file_path] = []
        file_to_indices[file_path].append(row_idx)

    def read_selected_rows(fp: Path, indices: list[int]) -> list[SampleDoc]:
        try:
            tbl = pq.read_table(fp, columns=[text_column])
            text_list = tbl.column(text_column).to_pylist()
            return [
                create_sample_doc(text_list[i], dataset_name, bucket_name)
                for i in indices
                if i < len(text_list)
            ]
        except Exception as err:
            logger.warning(f"读取文件失败 {fp}: {err}")
            return []

    sampled = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(read_selected_rows, fp, row_indices): fp
            for fp, row_indices in file_to_indices.items()
        }
        for future in tqdm(
            as_completed(future_to_file),
            total=len(file_to_indices),
            desc=f"读取 {bucket_name}",
            leave=False,
        ):
            try:
                samples = future.result()
                sampled.extend(samples)
            except Exception as e:
                file_path = future_to_file[future]
                logger.warning(f"处理文件失败 {file_path}: {e}")

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
    return {
        "text": text,
        "source_dataset": dataset_name,
        "source_bucket": bucket_name,
    }


def shuffle_samples(samples: list[SampleDoc], seed: int) -> list[SampleDoc]:
    samples_with_hash = [
        (hashlib.md5(f"{seed}_{i}_{s['text'][:50]}".encode()).digest(), s)
        for i, s in enumerate(samples)
    ]
    samples_with_hash.sort(key=lambda x: x[0])
    return [s for _, s in samples_with_hash]


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


def write_parquet_files(
    samples: list[SampleDoc],
    output_dir: Path,
    prefix: str = "train",
    max_rows_per_file: int = 100000,
) -> list[Path]:
    """将样本写入 Parquet 文件。

    Args:
        samples: 样本列表
        output_dir: 输出目录
        prefix: 文件名前缀
        max_rows_per_file: 每个文件的最大行数

    Returns:
        写入的文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    total = len(samples)
    num_files = (total + max_rows_per_file - 1) // max_rows_per_file

    logger.info(f"写入 {total:,} 个样本到 {num_files} 个文件...")

    for file_idx in range(num_files):
        start = file_idx * max_rows_per_file
        end = min(start + max_rows_per_file, total)
        batch = samples[start:end]

        # 构建 PyArrow 表
        texts = [s["text"] for s in batch]
        source_datasets = [s["source_dataset"] for s in batch]
        source_buckets = [s["source_bucket"] for s in batch]

        table = pa.table(
            {
                "text": texts,
                "source_dataset": source_datasets,
                "source_bucket": source_buckets,
            }
        )

        filename = f"{prefix}-{file_idx:05d}-of-{num_files:05d}.parquet"
        output_path = output_dir / filename

        pq.write_table(table, output_path, compression=COMPRESSION)
        output_files.append(output_path)

        logger.info(f"  写入: {filename} ({len(batch):,} 行)")

        # 定期 GC
        if file_idx % 10 == 0:
            gc.collect()

    return output_files


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


def prepare_tokenizer_data(
    workers: int,
    max_rows_per_file: int,
) -> int:
    """主函数：准备 tokenizer 训练数据。

    Args:
        workers: 并行工作进程数（预留）
        max_rows_per_file: 每个文件的行数上限

    Returns:
        退出码 (0=成功)
    """
    try:
        # 加载配置
        config = load_config(CONFIG_PATH)
        output_dir = config.output_dir
        seed = config.random_seed

        logger.info("=" * 60)
        logger.info("准备 Tokenizer 训练数据")
        logger.info("=" * 60)
        logger.info(f"配置文件: {CONFIG_PATH}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"随机种子: {seed}")
        logger.info(f"并行 workers: {workers}")
        logger.info(f"数据集数量: {len(config.datasets)}")

        # 处理所有数据集
        all_samples = []
        source_stats = {}

        for source_key, dataset_config in config.datasets.items():
            print()
            samples, stats = process_dataset(dataset_config, seed, workers)
            all_samples.extend(samples)
            source_stats[source_key] = stats

            # 每个数据集处理后触发 GC
            gc.collect()

        # 打乱顺序（确定性）
        logger.info("\n打乱样本顺序...")
        all_samples = shuffle_samples(all_samples, seed)

        # 写入文件
        print()
        output_files = write_parquet_files(
            samples=all_samples,
            output_dir=output_dir,
            prefix="train",
            max_rows_per_file=max_rows_per_file,
        )

        # 保存采样信息
        total_requested = sum(s["requested"] for s in source_stats.values())
        total_sampled = sum(s["sampled"] for s in source_stats.values())

        info = SamplingInfo(
            total_requested=total_requested,
            total_sampled=total_sampled,
            sources=source_stats,
            random_seed=seed,
        )
        save_sampling_info(info, output_dir)

        # 最终报告
        print()
        logger.info("=" * 60)
        logger.info("处理完成")
        logger.info("=" * 60)
        logger.info(f"总请求样本: {total_requested:,}")
        logger.info(f"总采样样本: {total_sampled:,}")
        logger.info(f"采样率: {total_sampled / total_requested:.1%}")
        logger.info(f"输出文件: {len(output_files)} 个")
        logger.info(f"输出目录: {output_dir}")

        # 详细统计
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
        help=f"并行文件读取数，控制统计/哈希计算/数据读取的并发度 (默认: {DEFAULT_WORKERS})",
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
