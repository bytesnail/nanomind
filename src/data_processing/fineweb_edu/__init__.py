"""FineWeb-Edu 数据集处理子模块。

提供 FineWeb-Edu 数据集的质量评分分桶重组功能，
支持多语言、分层采样和高性能并行处理。
"""

from .adapters import fineweb_adapter, normalize_score
from .reorganizer import (
    create_pipeline,
    get_default_config,
    main,
    process_all_datasets,
    process_single_dataset,
    setup_logging,
)

__all__ = [
    "create_pipeline",
    "fineweb_adapter",
    "get_default_config",
    "main",
    "normalize_score",
    "process_all_datasets",
    "process_single_dataset",
    "setup_logging",
]
