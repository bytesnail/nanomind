"""FineWeb-Edu 数据集重组主入口。

实现按评分桶独立处理数据集的完整 Pipeline，包括：
- 数据读取和字段筛选
- 评分过滤和确定性采样
- 进程内去重
- 按 CC-MAIN 批次组织输出
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader

from .adapters import fineweb_adapter_safe
from .bucket_config import BucketConfig, get_all_bucket_configs, get_bucket_config
from .cc_main_path_writer import CCMainPathWriter
from .score_filter import ScoreFilter

# 默认 Pipeline 配置常量
DEFAULT_WORKERS: int = 8
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_COMPRESSION: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "zstd"
DEFAULT_MAX_FILE_SIZE: int = 512 * 1024 * 1024  # 512MB

# Bloom Filter 默认配置
BLOOM_CAPACITY: int = 2_000_000_000
BLOOM_ERROR_RATE: float = 0.001

# 日志格式
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 跟踪日志是否已初始化
_logging_initialized: bool = False


def setup_logging(log_dir: Path, bucket_name: str) -> None:
    """设置日志记录（线程/进程安全，避免重复添加处理器）。

    Args:
        log_dir: 日志目录
        bucket_name: 评分桶名称
    """
    global _logging_initialized

    if _logging_initialized:
        return

    log_dir = log_dir / bucket_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "processing.log"

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _logging_initialized = True


def create_bucket_pipeline(
    input_path: Path,
    output_path: Path,
    bucket: BucketConfig,
    workers: int = DEFAULT_WORKERS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    compression: Literal[
        "snappy", "gzip", "brotli", "lz4", "zstd"
    ] = DEFAULT_COMPRESSION,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> LocalPipelineExecutor:
    """创建单个评分桶的 Pipeline。

    Args:
        input_path: 源数据目录（包含 CC-MAIN 批次子目录）
        output_path: 输出目录
        bucket: 评分桶配置
        workers: 并行 worker 数量
        random_seed: 随机种子（用于确定性采样）
        compression: 输出文件压缩格式
        max_file_size: 单个输出文件最大大小（字节）

    Returns:
        LocalPipelineExecutor: 配置好的执行器
    """
    pipeline = [
        ParquetReader(
            str(input_path),
            adapter=fineweb_adapter_safe,
            glob_pattern="**/*.parquet",
        ),
        ScoreFilter(
            bucket=bucket,
            random_seed=random_seed,
            use_bloom_filter=True,
            bloom_capacity=BLOOM_CAPACITY,
            bloom_error_rate=BLOOM_ERROR_RATE,
        ),
        CCMainPathWriter(
            output_folder=str(output_path),
            compression=compression,
            max_file_size=max_file_size,
        ),
    ]

    return LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=workers,
        logging_dir=str(output_path.parent / "logs" / bucket.name),
    )


def process_single_bucket(
    input_path: Path,
    output_base: Path,
    bucket: BucketConfig,
    workers: int = DEFAULT_WORKERS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    compression: Literal[
        "snappy", "gzip", "brotli", "lz4", "zstd"
    ] = DEFAULT_COMPRESSION,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> str:
    """处理单个评分桶。

    Args:
        input_path: 源数据目录
        output_base: 输出基础目录
        bucket: 评分桶配置
        workers: 并行 worker 数量
        random_seed: 随机种子
        compression: 压缩格式
        max_file_size: 单个文件最大大小

    Returns:
        str: 完成的桶名称
    """
    output_path = output_base / bucket.name
    output_path.mkdir(parents=True, exist_ok=True)

    setup_logging(output_base.parent / "logs", bucket.name)

    logger = logging.getLogger(__name__)
    logger.info(f"开始处理桶 {bucket.name}: {bucket}")

    executor = create_bucket_pipeline(
        input_path=input_path,
        output_path=output_path,
        bucket=bucket,
        workers=workers,
        random_seed=random_seed,
        compression=compression,
        max_file_size=max_file_size,
    )

    executor.run()

    logger.info(f"桶 {bucket.name} 处理完成")
    return bucket.name


def process_all_buckets(
    input_path: Path,
    output_base: Path,
    workers_per_bucket: int = DEFAULT_WORKERS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    parallel_buckets: int = 1,
    compression: Literal[
        "snappy", "gzip", "brotli", "lz4", "zstd"
    ] = DEFAULT_COMPRESSION,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    buckets: list[BucketConfig] | None = None,
) -> list[str]:
    """处理所有评分桶。

    Args:
        input_path: 源数据目录
        output_base: 输出基础目录
        workers_per_bucket: 每个桶的 worker 数量
        random_seed: 随机种子
        parallel_buckets: 同时运行的桶数量（1=顺序，4=并行）
        compression: 压缩格式
        max_file_size: 单个文件最大大小
        buckets: 要处理的桶列表，None 表示处理所有默认桶

    Returns:
        list[str]: 完成的桶名称列表
    """
    if buckets is None:
        buckets = get_all_bucket_configs()

    if parallel_buckets == 1:
        results = []
        for bucket in buckets:
            result = process_single_bucket(
                input_path=input_path,
                output_base=output_base,
                bucket=bucket,
                workers=workers_per_bucket,
                random_seed=random_seed,
                compression=compression,
                max_file_size=max_file_size,
            )
            results.append(result)
        return results

    with ProcessPoolExecutor(max_workers=parallel_buckets) as executor:
        futures = [
            executor.submit(
                process_single_bucket,
                input_path,
                output_base,
                bucket,
                workers_per_bucket,
                random_seed,
                compression,
                max_file_size,
            )
            for bucket in buckets
        ]
        return [f.result() for f in futures]


def main() -> int:
    """主入口函数。

    Returns:
        int: 退出码（0=成功，1=失败）
    """
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 数据集质量评分分桶重组工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理所有评分桶（顺序）
  python -m src.data_processing.fineweb_reorganizer

  # 处理指定评分桶
  python -m src.data_processing.fineweb_reorganizer --bucket 3.0

  # 指定 workers 和随机种子
  python -m src.data_processing.fineweb_reorganizer --workers 16 --seed 42

  # 并行处理多个桶
  python -m src.data_processing.fineweb_reorganizer --parallel-buckets 4
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/datasets/HuggingFaceFW/fineweb-edu"),
        help="源数据目录（默认：data/datasets/HuggingFaceFW/fineweb-edu）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/fineweb/en"),
        help="输出目录（默认：data/datasets/fineweb/en）",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        choices=["2.8", "3.0", "3.5", "4.0"],
        help="只处理指定评分桶（默认处理所有桶）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"每个桶的并行 worker 数量（默认：{DEFAULT_WORKERS}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"随机种子，用于确定性采样（默认：{DEFAULT_RANDOM_SEED}）",
    )
    parser.add_argument(
        "--parallel-buckets",
        type=int,
        default=1,
        help="同时运行的桶数量（默认：1，顺序处理）",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=DEFAULT_COMPRESSION,
        choices=["zstd", "gzip", "snappy", "brotli", "lz4"],
        help=f"输出文件压缩格式（默认：{DEFAULT_COMPRESSION}）",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help=f"单个输出文件最大大小（字节，默认：{DEFAULT_MAX_FILE_SIZE // (1024 * 1024)}MB）",
    )

    args = parser.parse_args()

    # 验证输入目录
    if not args.input.exists():
        print(f"错误：输入目录不存在：{args.input}", file=sys.stderr)
        return 1

    # 创建输出目录
    args.output.mkdir(parents=True, exist_ok=True)

    # 确定要处理的桶
    if args.bucket:
        buckets = [get_bucket_config(args.bucket)]
    else:
        buckets = None  # 处理所有默认桶

    compression_value: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = (
        args.compression
    )  # type: ignore[assignment]

    try:
        results = process_all_buckets(
            input_path=args.input,
            output_base=args.output,
            workers_per_bucket=args.workers,
            random_seed=args.seed,
            parallel_buckets=args.parallel_buckets,
            compression=compression_value,
            max_file_size=args.max_file_size,
            buckets=buckets,
        )
        print(f"处理完成：{', '.join(results)}")
        return 0
    except Exception as e:
        print(f"处理失败：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
