import logging
import sys
from functools import lru_cache
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader

from .adapters import fineweb_adapter
from .bucket_config import BucketConfig, get_all_bucket_configs
from .bucket_path_writer import BucketPathWriter
from .config_loader import (
    DEFAULT_COMPRESSION,
    DEFAULT_LOG_FORMAT,
    DEFAULT_MAX_FILE_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TASKS,
    DEFAULT_WORKERS,
    Compression,
    get_dataset_configs,
    get_processing_config,
)
from .parquet_merger import merge_all_buckets
from .score_filter import ScoreFilter

__all__ = [
    "create_pipeline",
    "get_default_config",
    "main",
    "process_all_datasets",
    "process_single_dataset",
    "setup_logging",
]


@lru_cache(maxsize=1)
def get_default_config():
    processing = get_processing_config()
    return {
        "workers": processing.get("workers", DEFAULT_WORKERS),
        "tasks": processing.get("tasks", DEFAULT_TASKS),
        "random_seed": processing.get("random_seed", DEFAULT_RANDOM_SEED),
        "compression": processing.get("compression", DEFAULT_COMPRESSION),
        "max_size": processing.get("max_file_size_bytes", DEFAULT_MAX_FILE_SIZE),
        "log_format": processing.get("logging", {}).get("format", DEFAULT_LOG_FORMAT),
    }


def setup_logging(log_dir: Path, name: str) -> logging.Logger:
    log_path = log_dir / name
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"fineweb_{name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(get_default_config()["log_format"])
    logger.addHandler(logging.FileHandler(log_path / "processing.log"))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(fmt)
    return logger


def create_pipeline(
    input_dir: Path,
    output_dir: Path,
    buckets: list[BucketConfig],
    workers: int = DEFAULT_WORKERS,
    tasks: int = DEFAULT_TASKS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    compression: Compression = DEFAULT_COMPRESSION,
    max_size: int = DEFAULT_MAX_FILE_SIZE,
) -> LocalPipelineExecutor:
    # 使用 output_dir 的名称（语言代码）来区分不同数据集的日志目录
    log_name = f"multi_bucket_{output_dir.name}"
    return LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                str(input_dir), adapter=fineweb_adapter, glob_pattern="**/*.parquet"
            ),
            ScoreFilter(buckets=buckets, random_seed=random_seed),
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


def process_single_dataset(
    input_dir: Path,
    output_dir: Path,
    buckets: list[BucketConfig],
    workers: int = DEFAULT_WORKERS,
    tasks: int = DEFAULT_TASKS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    compression: Compression = DEFAULT_COMPRESSION,
    max_size: int = DEFAULT_MAX_FILE_SIZE,
) -> list[str]:
    log_name = f"multi_bucket_{output_dir.name}"
    logger = setup_logging(output_dir.parent / "logs", log_name)
    names = ", ".join(b.name for b in buckets)
    logger.info(f"开始处理数据集: {names}")

    output_dir.mkdir(parents=True, exist_ok=True)

    create_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        buckets=buckets,
        workers=workers,
        tasks=tasks,
        random_seed=random_seed,
        compression=compression,
        max_size=max_size,
    ).run()

    # 合并小文件
    logger.info(f"开始合并文件，目标大小: {max_size / 1024 / 1024:.0f}MB")
    merge_all_buckets(
        output_dir=output_dir,
        target_file_size=max_size,
        compression=compression,
        remove_source=True,
    )

    logger.info(f"处理完成: {names}")
    return [b.name for b in buckets]


def process_all_datasets(
    workers: int = 0,
    tasks: int = 0,
    random_seed: int = 0,
    compression: Compression = DEFAULT_COMPRESSION,
    max_size: int = 0,
) -> dict[str, list[str]]:
    defaults = get_default_config()
    workers = workers if workers > 0 else defaults["workers"]
    tasks = tasks if tasks > 0 else defaults["tasks"]
    random_seed = random_seed if random_seed != 0 else defaults["random_seed"]
    compression = compression if compression else defaults["compression"]
    max_size = max_size if max_size > 0 else defaults["max_size"]

    dataset_configs = get_dataset_configs()

    results = {}
    for lang, dataset_config in dataset_configs.items():
        input_dir = Path(dataset_config.get("input_dir", ""))
        output_dir = Path(dataset_config.get("output_dir", ""))

        if not input_dir.exists():
            print(f"警告：输入目录不存在，跳过 {lang}: {input_dir}", file=sys.stderr)
            continue

        buckets = get_all_bucket_configs(lang)
        if not buckets:
            print(f"警告：未找到评分桶配置，跳过 {lang}", file=sys.stderr)
            continue

        print(f"\n处理数据集 [{lang}]: {dataset_config.get('name', lang)}")
        print(f"  输入: {input_dir}")
        print(f"  输出: {output_dir}")

        result = process_single_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            buckets=buckets,
            workers=workers,
            tasks=tasks,
            random_seed=random_seed,
            compression=compression,
            max_size=max_size,
        )
        results[lang] = result

    return results


def main() -> int:
    print("FineWeb-Edu 数据集质量评分分桶重组工具")
    print("=" * 50)

    try:
        results = process_all_datasets()

        print("\n" + "=" * 50)
        print("所有数据集处理完成")
        for lang, buckets in results.items():
            print(f"  [{lang}]: {', '.join(buckets)}")
        return 0
    except Exception as e:
        print(f"处理失败：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
