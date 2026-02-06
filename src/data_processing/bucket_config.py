"""评分桶配置模块。

定义评分桶配置数据类和默认的评分桶配置。
"""

from dataclasses import dataclass
from typing import Final

# 浮点数比较精度阈值
EPSILON: Final[float] = 1e-9


@dataclass(frozen=True)
class BucketConfig:
    """评分桶配置。

    定义评分桶的名称、评分区间和采样率。
    评分区间采用左闭右开方式 [min_score, max_score)。

    Attributes:
        name: 桶名称（如 "2.8"）
        min_score: 评分下限（包含）
        max_score: 评分上限（不包含），None 表示无上限
        sampling_rate: 采样率（0-1）
    """

    name: str
    min_score: float
    max_score: float | None
    sampling_rate: float

    def contains(self, score: float) -> bool:
        """检查评分是否在区间内（左闭右开，考虑浮点数精度）。

        Args:
            score: 要检查的评分值

        Returns:
            bool: 如果评分在区间内返回 True，否则返回 False
        """
        if self.max_score is None:
            return score >= self.min_score - EPSILON
        return (score >= self.min_score - EPSILON) and (score < self.max_score)

    def __repr__(self) -> str:
        """返回桶配置的字符串表示。"""
        if self.max_score is None:
            interval = f"[{self.min_score}, +∞)"
        else:
            interval = f"[{self.min_score}, {self.max_score})"
        return f"BucketConfig(name='{self.name}', interval={interval}, sampling_rate={self.sampling_rate:.0%})"


# 默认评分桶配置（按设计文档定义）
DEFAULT_BUCKETS: Final[list[BucketConfig]] = [
    BucketConfig("2.8", 2.8, 3.0, 0.30),  # 2.8 ≤ score < 3.0, 采样率 30%
    BucketConfig("3.0", 3.0, 3.5, 0.60),  # 3.0 ≤ score < 3.5, 采样率 60%
    BucketConfig("3.5", 3.5, 4.0, 0.80),  # 3.5 ≤ score < 4.0, 采样率 80%
    BucketConfig("4.0", 4.0, None, 1.0),  # score ≥ 4.0, 采样率 100%
]

# 评分桶名称到配置的映射
BUCKET_NAME_MAP: Final[dict[str, BucketConfig]] = {
    bucket.name: bucket for bucket in DEFAULT_BUCKETS
}


def get_bucket_config(name: str) -> BucketConfig:
    """根据桶名称获取配置。

    Args:
        name: 桶名称（如 "2.8", "3.0", "3.5", "4.0"）

    Returns:
        BucketConfig: 对应的桶配置

    Raises:
        ValueError: 如果桶名称不存在
    """
    if name not in BUCKET_NAME_MAP:
        available = ", ".join(BUCKET_NAME_MAP.keys())
        raise ValueError(f"Unknown bucket name: {name}. Available: {available}")
    return BUCKET_NAME_MAP[name]


def get_all_bucket_configs() -> list[BucketConfig]:
    """获取所有默认评分桶配置。

    Returns:
        list[BucketConfig]: 所有默认桶配置的列表
    """
    return list(DEFAULT_BUCKETS)


def find_bucket_for_score(score: float) -> BucketConfig | None:
    """根据评分值找到对应的桶配置。

    Args:
        score: 评分值

    Returns:
        BucketConfig | None: 对应的桶配置，如果没有匹配的桶则返回 None
    """
    return next((bucket for bucket in DEFAULT_BUCKETS if bucket.contains(score)), None)
