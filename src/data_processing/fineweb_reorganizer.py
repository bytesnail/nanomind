"""FineWeb-Edu 数据集重组主入口。"""

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

DEFAULT_WORKERS = 8
DEFAULT_SEED = 42
DEFAULT_COMPRESSION: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "zstd"
DEFAULT_MAX_SIZE = 512 * 1024 * 1024
BLOOM_CAPACITY = 2_000_000_000
BLOOM_ERROR_RATE = 0.001
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _setup_logging(log_dir: Path, bucket_name: str) -> logging.Logger:
    """设置日志记录。"""
    log_dir = log_dir / bucket_name
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"fineweb_{bucket_name}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler = logging.FileHandler(log_dir / "processing.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def _create_pipeline(
    input_path: Path,
    output_path: Path,
    bucket: BucketConfig,
    workers: int = DEFAULT_WORKERS,
    seed: int = DEFAULT_SEED,
    compression: Literal[
        "snappy", "gzip", "brotli", "lz4", "zstd"
    ] = DEFAULT_COMPRESSION,
    max_size: int = DEFAULT_MAX_SIZE,
) -> LocalPipelineExecutor:
    """创建单个评分桶的 Pipeline。"""
    pipeline = [
        ParquetReader(
            str(input_path),
            adapter=fineweb_adapter_safe,
            glob_pattern="**/*.parquet",
        ),
        ScoreFilter(
            bucket=bucket,
            random_seed=seed,
            use_bloom_filter=True,
            bloom_capacity=BLOOM_CAPACITY,
            bloom_error_rate=BLOOM_ERROR_RATE,
        ),
        CCMainPathWriter(
            output_folder=str(output_path),
            compression=compression,
            max_file_size=max_size,
        ),
    ]

    return LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=workers,
        logging_dir=str(output_path.parent / "logs" / bucket.name),
    )


def _process_single_bucket(
    input_path: Path,
    output_base: Path,
    bucket: BucketConfig,
    workers: int = DEFAULT_WORKERS,
    seed: int = DEFAULT_SEED,
    compression: Literal[
        "snappy", "gzip", "brotli", "lz4", "zstd"
    ] = DEFAULT_COMPRESSION,
    max_size: int = DEFAULT_MAX_SIZE,
) -> str:
    """处理单个评分桶。"""
    output_path = output_base / bucket.name
    output_path.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(output_base.parent / "logs", bucket.name)
    logger.info(f"开始处理桶 {bucket.name}: {bucket}")

    executor = _create_pipeline(
        input_path=input_path,
        output_path=output_path,
        bucket=bucket,
        workers=workers,
        seed=seed,
        compression=compression,
        max_size=max_size,
    )

    executor.run()
    logger.info(f"桶 {bucket.name} 处理完成")
    return bucket.name


def process_all_buckets(
    input_path: Path,
    output_base: Path,
    workers_per_bucket: int = DEFAULT_WORKERS,
    random_seed: int = DEFAULT_SEED,
    parallel_buckets: int = 1,
    compression: Literal[
        "snappy", "gzip", "brotli", "lz4", "zstd"
    ] = DEFAULT_COMPRESSION,
    max_file_size: int = DEFAULT_MAX_SIZE,
    buckets: list[BucketConfig] | None = None,
) -> list[str]:
    """处理所有评分桶。"""
    buckets = buckets or get_all_bucket_configs()

    if parallel_buckets == 1:
        return [
            _process_single_bucket(
                input_path=input_path,
                output_base=output_base,
                bucket=bucket,
                workers=workers_per_bucket,
                seed=random_seed,
                compression=compression,
                max_size=max_file_size,
            )
            for bucket in buckets
        ]

    with ProcessPoolExecutor(max_workers=parallel_buckets) as executor:
        futures = [
            executor.submit(
                _process_single_bucket,
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
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 数据集质量评分分桶重组工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.data_processing.fineweb_reorganizer
  python -m src.data_processing.fineweb_reorganizer --bucket 3.0
  python -m src.data_processing.fineweb_reorganizer --workers 16 --seed 42
  python -m src.data_processing.fineweb_reorganizer --parallel-buckets 4
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/datasets/HuggingFaceFW/fineweb-edu"),
        help="源数据目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/fineweb/en"),
        help="输出目录",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        choices=["2.8", "3.0", "3.5", "4.0"],
        help="只处理指定评分桶",
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
        default=DEFAULT_SEED,
        help=f"随机种子（默认：{DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--parallel-buckets",
        type=int,
        default=1,
        help="同时运行的桶数量（默认：1）",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=DEFAULT_COMPRESSION,
        choices=["zstd", "gzip", "snappy", "brotli", "lz4"],
        help=f"压缩格式（默认：{DEFAULT_COMPRESSION}）",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_SIZE,
        help=f"单个输出文件最大大小（字节，默认：{DEFAULT_MAX_SIZE // (1024 * 1024)}MB）",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入目录不存在：{args.input}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    buckets = [get_bucket_config(args.bucket)] if args.bucket else None

    try:
        results = process_all_buckets(
            input_path=args.input,
            output_base=args.output,
            workers_per_bucket=args.workers,
            random_seed=args.seed,
            parallel_buckets=args.parallel_buckets,
            compression=args.compression,  # type: ignore[arg-type]
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
