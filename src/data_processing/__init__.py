"""FineWeb-Edu 数据处理工具包。

提供按评分桶重组 FineWeb-Edu 数据集的功能，包括：
- 数据读取和字段筛选
- 评分过滤和确定性采样
- 进程内去重
- 按 CC-MAIN 批次组织输出
"""

from .adapters import fineweb_adapter, fineweb_adapter_safe
from .bucket_config import BucketConfig, get_all_bucket_configs, get_bucket_config
from .cc_main_path_writer import CCMainPathWriter
from .score_filter import ScoreFilter

__all__ = [
    "BucketConfig",
    "CCMainPathWriter",
    "ScoreFilter",
    "fineweb_adapter",
    "fineweb_adapter_safe",
    "get_all_bucket_configs",
    "get_bucket_config",
]
