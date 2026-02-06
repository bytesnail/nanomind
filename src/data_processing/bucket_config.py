"""评分桶配置模块。"""

from dataclasses import dataclass
from typing import Final

EPSILON: Final = 1e-9


@dataclass(frozen=True)
class BucketConfig:
    name: str
    min_score: float
    max_score: float | None
    sampling_rate: float

    def contains(self, score: float) -> bool:
        if self.max_score is None:
            return score >= self.min_score - EPSILON
        return self.min_score - EPSILON <= score < self.max_score

    def __repr__(self) -> str:
        interval = (
            f"[{self.min_score}, +∞)"
            if self.max_score is None
            else f"[{self.min_score}, {self.max_score})"
        )
        return f"BucketConfig(name='{self.name}', interval={interval}, sampling_rate={self.sampling_rate:.0%})"


DEFAULT_BUCKETS: Final = [
    BucketConfig("2.8", 2.8, 3.0, 0.30),
    BucketConfig("3.0", 3.0, 3.5, 0.60),
    BucketConfig("3.5", 3.5, 4.0, 0.80),
    BucketConfig("4.0", 4.0, None, 1.0),
]

_BUCKET_MAP: Final = {b.name: b for b in DEFAULT_BUCKETS}


def get_bucket_config(name: str) -> BucketConfig:
    if name not in _BUCKET_MAP:
        raise ValueError(
            f"Unknown bucket: {name}. Available: {', '.join(_BUCKET_MAP.keys())}"
        )
    return _BUCKET_MAP[name]


def get_all_bucket_configs() -> list[BucketConfig]:
    return list(DEFAULT_BUCKETS)


def find_bucket_for_score(score: float) -> BucketConfig | None:
    return next((b for b in DEFAULT_BUCKETS if b.contains(score)), None)
