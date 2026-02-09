"""数据处理模块。

提供数据集预处理和重组功能，支持多种数据集格式。

子模块：
    - fineweb_edu: FineWeb-Edu 数据集处理
"""

from .bucket_config import BucketConfig, find_bucket_for_score, get_all_bucket_configs
from .bucket_path_writer import BucketPathWriter
from .config_loader import Compression
from .fineweb_edu import (
    fineweb_adapter,
    normalize_score,
    process_all_datasets,
    process_single_dataset,
)
from .parquet_merger import merge_all_buckets, merge_bucket_files
from .score_filter import ScoreFilter

__all__ = [
    "BucketConfig",
    "BucketPathWriter",
    "Compression",
    "ScoreFilter",
    "find_bucket_for_score",
    "fineweb_adapter",
    "get_all_bucket_configs",
    "merge_all_buckets",
    "merge_bucket_files",
    "normalize_score",
    "process_all_datasets",
    "process_single_dataset",
]
