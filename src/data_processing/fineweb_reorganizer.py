import argparse
import logging
import sys
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _defaults():
    cfg = get_config()
    processing = cfg.processing
    paths = cfg.paths
    return {
        "workers": processing.get("workers", 8),
        "tasks": processing.get("tasks", 8),
        "seed": processing.get("random_seed", 42),
        "compression": processing.get("compression", "zstd"),
        "max_size": processing.get("max_file_size_bytes", 512 * 1024 * 1024),
        "input_dir": Path(
            paths.get("input_dir", "data/datasets/HuggingFaceFW/fineweb-edu")
        ),
        "output_dir": Path(paths.get("output_dir", "data/datasets/fineweb/en")),
        "log_format": processing.get("logging", {}).get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
    }


def _setup_logging(log_dir: Path, name: str) -> logging.Logger:
    log_path = log_dir / name
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"fineweb_{name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(_defaults()["log_format"])
    logger.addHandler(logging.FileHandler(log_path / "processing.log"))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(fmt)
    return logger


def _create_pipeline(
    input_dir: Path,
    output_dir: Path,
    buckets: list[BucketConfig],
    workers: int = 8,
    tasks: int = 8,
    seed: int = 42,
    compression: Compression = "zstd",
    max_size: int = 512 * 1024 * 1024,
) -> LocalPipelineExecutor:
    log_name = "multi_bucket" if len(buckets) > 1 else buckets[0].name
    return LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                str(input_dir), adapter=fineweb_adapter, glob_pattern="**/*.parquet"
            ),
            ScoreFilter(buckets=buckets, random_seed=seed),
            BucketPathWriter(
                output_dir=str(output_dir),
                buckets=buckets,
                compression=compression,
                max_file_size=max_size,
            ),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(output_dir.parent / "logs" / log_name),
    )


def process_all_buckets(
    input_dir: Path,
    output_dir: Path,
    workers: int = 8,
    tasks: int = 8,
    seed: int = 42,
    compression: Compression = "zstd",
    max_size: int = 512 * 1024 * 1024,
    buckets: list[BucketConfig] | None = None,
) -> list[str]:
    buckets = buckets or get_all_bucket_configs()
    is_multi = len(buckets) > 1
    log_name = "multi_bucket" if is_multi else buckets[0].name
    logger = _setup_logging(output_dir.parent / "logs", log_name)
    names = ", ".join(b.name for b in buckets)
    logger.info(f"开始处理{'多桶' if is_multi else '单桶'}: {names}")

    out = output_dir if is_multi else output_dir / buckets[0].name
    out.mkdir(parents=True, exist_ok=True)

    _create_pipeline(
        input_dir, out, buckets, workers, tasks, seed, compression, max_size
    ).run()
    logger.info(f"处理完成: {names}")
    return [b.name for b in buckets]


def main() -> int:
    defaults = _defaults()
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 数据集质量评分分桶重组工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python -m src.data_processing.fineweb_reorganizer
  python -m src.data_processing.fineweb_reorganizer --bucket 3.0
  python -m src.data_processing.fineweb_reorganizer --workers 16 --seed 42""",
    )
    parser.add_argument(
        "--input", type=Path, default=defaults["input_dir"], help="源数据目录"
    )
    parser.add_argument(
        "--output", type=Path, default=defaults["output_dir"], help="输出目录"
    )
    parser.add_argument(
        "--bucket", type=str, choices=get_bucket_names(), help="只处理指定评分桶"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=defaults["workers"],
        help=f"worker数量(默认:{defaults['workers']})",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=defaults["tasks"],
        help=f"tasks数量(默认:{defaults['tasks']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults["seed"],
        help=f"随机种子(默认:{defaults['seed']})",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=defaults["compression"],
        choices=["zstd", "gzip", "snappy", "brotli", "lz4"],
        help=f"压缩格式(默认:{defaults['compression']})",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=defaults["max_size"],
        help=f"单文件最大字节(默认:{defaults['max_size'] // (1024 * 1024)}MB)",
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
            args.tasks,
            args.seed,
            args.compression,
            args.max_file_size,
            buckets,
        )
        print(f"处理完成：{', '.join(results)}")
        return 0
    except Exception as e:
        print(f"处理失败：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
