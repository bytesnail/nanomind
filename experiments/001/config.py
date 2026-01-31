"""数据集配置管理模块。

本模块定义了数据集配置类和数据集注册表，用于统一管理不同数据集的
字段映射、分组策略和文件匹配模式。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    """数据集配置类。

    Attributes:
        name: 数据集唯一标识。
        path: 数据集根路径。
        text_key: 文本字段名。
        id_key: ID 字段名（可选）。
        group_field: 分组字段（metadata 字段名，可选）。
        group_by: 分组策略（None, "directory"）。
        score_field: Score 字段名或路径（支持嵌套）。
        int_score_field: int_score 字段名或路径（可选）。
        glob_pattern: 文件匹配模式。
    """

    name: str
    path: str
    text_key: str = "text"
    id_key: Optional[str] = "id"
    group_field: Optional[str] = None
    group_by: Optional[str] = None
    score_field: Optional[str] = "score"
    int_score_field: Optional[str] = "int_score"
    glob_pattern: str = "**/*.parquet"


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "HuggingFaceFW/fineweb-edu": DatasetConfig(
        name="HuggingFaceFW/fineweb-edu",
        path="data/datasets/HuggingFaceFW/fineweb-edu/data/",
        text_key="text",
        id_key="id",
        group_field="dump",
        group_by=None,
        score_field="score",
        int_score_field="int_score",
        glob_pattern="**/*.parquet",
    ),
    "opencsg/Fineweb-Edu-Chinese-V2.1": DatasetConfig(
        name="opencsg/Fineweb-Edu-Chinese-V2.1",
        path="data/datasets/opencsg/Fineweb-Edu-Chinese-V2.1/",
        text_key="text",
        id_key=None,
        group_field=None,
        group_by="directory",
        score_field="score",
        int_score_field=None,
        glob_pattern="**/*.parquet",
    ),
    "HuggingFaceTB/finemath": DatasetConfig(
        name="HuggingFaceTB/finemath",
        path="data/datasets/HuggingFaceTB/finemath/",
        text_key="text",
        id_key=None,
        group_field="snapshot_type",
        group_by=None,
        score_field="score",
        int_score_field="int_score",
        glob_pattern="**/*.parquet",
    ),
    "nick007x/github-code-2025": DatasetConfig(
        name="nick007x/github-code-2025",
        path="data/datasets/nick007x/github-code-2025/",
        text_key="content",
        id_key=None,
        group_field="repo_id",
        group_by=None,
        score_field=None,
        int_score_field=None,
        glob_pattern="**/*.parquet",
    ),
    "nvidia/Nemotron-CC-Math-v1": DatasetConfig(
        name="nvidia/Nemotron-CC-Math-v1",
        path="data/datasets/nvidia/Nemotron-CC-Math-v1/",
        text_key="text",
        id_key="id",
        group_field=None,
        group_by=None,
        score_field="metadata.finemath_scores",
        int_score_field="metadata.finemath_int_scores",
        glob_pattern="**/*.parquet",
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """获取数据集配置。

    Args:
        dataset_name: 数据集名称。

    Returns:
        数据集配置对象。

    Raises:
        ValueError: 如果数据集名称不存在。
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def get_datasets_with_score() -> List[str]:
    """获取有 score 字段的数据集列表。

    Returns:
        有 score 字段的数据集名称列表。
    """
    return [
        name for name, cfg in DATASET_CONFIGS.items() if cfg.score_field is not None
    ]
