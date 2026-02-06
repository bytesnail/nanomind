"""评分桶配置。"""

from dataclasses import dataclass
from typing import Final

EPSILON: Final[float] = 1e-9


@dataclass(frozen=True)
class BucketConfig:
    """评分桶配置。

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
        """检查评分是否在区间内（左闭右开，考虑浮点数精度）。"""
        if self.max_score is None:
            return score >= self.min_score - EPSILON
        return (score >= self.min_score - EPSILON) and (score < self.max_score)

    def __repr__(self) -> str:
        """返回桶配置的字符串表示。"""
        interval = (
            f"[{self.min_score}, +∞)"
            if self.max_score is None
            else f"[{self.min_score}, {self.max_score})"
        )
        return f"BucketConfig(name='{self.name}', interval={interval}, sampling_rate={self.sampling_rate:.0%})"


DEFAULT_BUCKETS: Final[list[BucketConfig]] = [
    BucketConfig("2.8", 2.8, 3.0, 0.30),
    BucketConfig("3.0", 3.0, 3.5, 0.60),
    BucketConfig("3.5", 3.5, 4.0, 0.80),
    BucketConfig("4.0", 4.0, None, 1.0),
]

BUCKET_NAME_MAP: Final[dict[str, BucketConfig]] = {b.name: b for b in DEFAULT_BUCKETS}


def get_bucket_config(name: str) -> BucketConfig:
    """根据桶名称获取配置。"""
    if name not in BUCKET_NAME_MAP:
        raise ValueError(
            f"Unknown bucket: {name}. Available: {', '.join(BUCKET_NAME_MAP.keys())}"
        )
    return BUCKET_NAME_MAP[name]


def get_all_bucket_configs() -> list[BucketConfig]:
    """获取所有默认评分桶配置。"""
    return list(DEFAULT_BUCKETS)


def find_bucket_for_score(score: float) -> BucketConfig | None:
    """根据评分值找到对应的桶配置。"""
    return next((b for b in DEFAULT_BUCKETS if b.contains(score)), None)
