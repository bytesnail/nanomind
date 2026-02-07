"""FineWeb-Edu 数据集重组主入口。"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader

from .adapters import fineweb_adapter
from .bucket_config import (
    BucketConfig,
    get_all_bucket_configs,
    get_bucket_config,
    get_bucket_names,
)
from .bucket_path_writer import BucketPathWriter
from .config_loader import get_config
from .score_filter import ScoreFilter

Compression = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]

_cfg = get_config()
_processing_cfg = _cfg.processing
_paths_cfg = _cfg.paths

DEFAULT_WORKERS = _processing_cfg.get("workers", 8)
DEFAULT_SEED = _processing_cfg.get("random_seed", 42)
DEFAULT_COMPRESSION: Compression = _processing_cfg.get("compression", "zstd")
DEFAULT_MAX_SIZE = _processing_cfg.get("max_file_size_bytes", 512 * 1024 * 1024)
LOG_FMT = _processing_cfg.get("logging", {}).get(
    "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
DEFAULT_INPUT_DIR = Path(
    _paths_cfg.get("input_dir", "data/datasets/HuggingFaceFW/fineweb-edu")
)
DEFAULT_OUTPUT_DIR = Path(_paths_cfg.get("output_dir", "data/datasets/fineweb/en"))


def _setup_logging(log_dir: Path, bucket: str) -> logging.Logger:
    (path := log_dir / bucket).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"fineweb_{bucket}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(LOG_FMT)
        logger.addHandler(h := logging.FileHandler(path / "processing.log"))
        h.setFormatter(fmt)
        logger.addHandler(ch := logging.StreamHandler(sys.stdout))
        ch.setFormatter(fmt)
    return logger


def _create_pipeline(
    input_dir: Path,
    output: Path,
    bucket: BucketConfig,
    workers: int = DEFAULT_WORKERS,
    seed: int = DEFAULT_SEED,
    compression: Compression = DEFAULT_COMPRESSION,
    max_size: int = DEFAULT_MAX_SIZE,
) -> LocalPipelineExecutor:
    pipeline = [
        ParquetReader(
            str(input_dir), adapter=fineweb_adapter, glob_pattern="**/*.parquet"
        ),
        ScoreFilter(bucket=bucket, random_seed=seed),
        BucketPathWriter(
            output_folder=str(output), compression=compression, max_file_size=max_size
        ),
    ]
    return LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=workers,
        logging_dir=str(output.parent / "logs" / bucket.name),
    )


def _process_bucket(
    input_dir: Path,
    out_base: Path,
    bucket: BucketConfig,
    workers: int = DEFAULT_WORKERS,
    seed: int = DEFAULT_SEED,
    compression: Compression = DEFAULT_COMPRESSION,
    max_size: int = DEFAULT_MAX_SIZE,
) -> str:
    out = out_base / bucket.name
    out.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(out_base.parent / "logs", bucket.name)
    logger.info(f"开始处理桶 {bucket.name}: {bucket}")
    _create_pipeline(input_dir, out, bucket, workers, seed, compression, max_size).run()
    logger.info(f"桶 {bucket.name} 处理完成")
    return bucket.name


def process_all_buckets(
    input_dir: Path,
    output_dir: Path,
    workers: int = DEFAULT_WORKERS,
    seed: int = DEFAULT_SEED,
    parallel: int = 1,
    compression: Compression = DEFAULT_COMPRESSION,
    max_size: int = DEFAULT_MAX_SIZE,
    buckets: list[BucketConfig] | None = None,
) -> list[str]:
    buckets = buckets or get_all_bucket_configs()
    if parallel == 1:
        return [
            _process_bucket(
                input_dir, output_dir, b, workers, seed, compression, max_size
            )
            for b in buckets
        ]
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = [
            pool.submit(
                _process_bucket,
                input_dir,
                output_dir,
                b,
                workers,
                seed,
                compression,
                max_size,
            )
            for b in buckets
        ]
        return [f.result() for f in futures]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 数据集质量评分分桶重组工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python -m src.data_processing.fineweb_reorganizer
  python -m src.data_processing.fineweb_reorganizer --bucket 3.0
  python -m src.data_processing.fineweb_reorganizer --workers 16 --seed 42
  python -m src.data_processing.fineweb_reorganizer --parallel-buckets 4""",
    )

    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_DIR, help="源数据目录"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录"
    )
    parser.add_argument(
        "--bucket", type=str, choices=get_bucket_names(), help="只处理指定评分桶"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"每个桶的 worker 数量（默认：{DEFAULT_WORKERS}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机种子（默认：{DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--parallel-buckets", type=int, default=1, help="同时运行的桶数量（默认：1）"
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

    try:
        buckets = [get_bucket_config(args.bucket)] if args.bucket else None
        results = process_all_buckets(
            args.input,
            args.output,
            args.workers,
            args.seed,
            args.parallel_buckets,
            args.compression,
            args.max_file_size,
            buckets,  # type: ignore[arg-type]
        )
        print(f"处理完成：{', '.join(results)}")
        return 0
    except Exception as e:
        print(f"处理失败：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
