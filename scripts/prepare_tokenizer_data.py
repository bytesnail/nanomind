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
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def deterministic_sample(
    doc_id: str, target_count: int, total_count: int, seed: int
) -> bool:
    """基于 MD5 哈希的确定性采样。

    Args:
        doc_id: 文档唯一标识
        target_count: 目标采样数
        total_count: 总数
        seed: 随机种子

    Returns:
        是否选中该文档
    """
    if target_count >= total_count:
        return True

    rate = target_count / total_count

    # 基于 MD5 哈希的确定性采样
    data = f"{seed}_{doc_id}".encode()
    h = int.from_bytes(hashlib.md5(data, usedforsecurity=False).digest()[:8], "big")
    return h / (2**64) < rate


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


def count_bucket_samples(source_dir: Path, bucket_name: str) -> int:
    """统计桶内的总样本数。

    Args:
        source_dir: 数据源根目录
        bucket_name: 桶名称

    Returns:
        总样本数
    """
    files = get_bucket_files(source_dir, bucket_name)
    if not files:
        return 0

    total = 0
    for f in tqdm(files, desc=f"统计 {bucket_name}", leave=False):
        try:
            total += get_file_row_count(f)
        except Exception as e:
            logger.warning(f"统计文件失败 {f}: {e}")

    return total


def sample_from_bucket(
    source_dir: Path,
    bucket_name: str,
    target_count: int,
    seed: int,
    text_column: str = "text",
) -> list[dict[str, Any]]:
    """从指定桶中采样指定数量的样本。

    Args:
        source_dir: 数据源根目录
        bucket_name: 桶名称
        target_count: 目标采样数
        seed: 随机种子
        text_column: 文本字段名

    Returns:
        采样的文档列表
    """
    files = get_bucket_files(source_dir, bucket_name)
    if not files:
        logger.warning(f"桶 {bucket_name} 无数据文件")
        return []

    # 首先统计总样本数
    total_count = count_bucket_samples(source_dir, bucket_name)
    if total_count == 0:
        return []

    logger.info(
        f"  [{bucket_name}] 总样本: {total_count:,}, 目标: {target_count:,} "
        f"(采样率: {target_count / total_count:.1%})"
    )

    sampled = []
    doc_counter = 0

    for file_path in tqdm(files, desc=f"采样 {bucket_name}", leave=False):
        try:
            table = pq.read_table(file_path, columns=[text_column])
            texts = table.column(text_column).to_pylist()

            for text in texts:
                doc_id = f"{bucket_name}#{file_path.name}#{doc_counter}"
                if deterministic_sample(doc_id, target_count, total_count, seed):
                    sampled.append({"text": text, "source": bucket_name})

                doc_counter += 1

                # 提前退出：如果已采集足够样本
                if len(sampled) >= target_count * 1.1:  # 允许 10% 超额
                    break

            # 定期触发 GC
            if doc_counter % 100000 == 0:
                gc.collect()

        except Exception as e:
            logger.warning(f"处理文件失败 {file_path}: {e}")

    # 如果采样过多，截断到目标数量
    if len(sampled) > target_count:
        # 使用确定性随机打乱后截断
        sampled_hash = [
            (hashlib.md5(f"{seed}_{i}_{s['text'][:50]}".encode()).digest(), s)
            for i, s in enumerate(sampled)
        ]
        sampled_hash.sort(key=lambda x: x[0])
        sampled = [s for _, s in sampled_hash[:target_count]]

    logger.info(f"  [{bucket_name}] 实际采样: {len(sampled):,}")
    return sampled


def determine_text_column(dataset_name: str) -> str:
    """根据数据集名称确定文本字段名。

    Args:
        dataset_name: 数据集名称

    Returns:
        文本字段名
    """
    # GitHub Code 使用 "content" 字段
    if "github" in dataset_name.lower():
        return "content"
    # FineWeb 和 Nemotron 使用 "text" 字段
    return "text"


def process_dataset(
    config: SamplingConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """处理单个数据集的所有桶。

    Args:
        config: 采样配置
        seed: 随机种子

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
            text_column=text_column,
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
    samples: list[dict[str, Any]],
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
    import pyarrow as pa

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
        sources = [s["source"] for s in batch]

        table = pa.table(
            {
                "text": texts,
                "source": sources,
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
            samples, stats = process_dataset(dataset_config, seed)
            all_samples.extend(samples)
            source_stats[source_key] = stats

            # 每个数据集处理后触发 GC
            gc.collect()

        # 打乱顺序（确定性）
        logger.info("\n打乱样本顺序...")
        samples_with_hash = [
            (hashlib.md5(f"{seed}_{i}_{s['text'][:50]}".encode()).digest(), s)
            for i, s in enumerate(all_samples)
        ]
        samples_with_hash.sort(key=lambda x: x[0])
        all_samples = [s for _, s in samples_with_hash]

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
        help=f"并行工作进程数 (默认: {DEFAULT_WORKERS})",
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
