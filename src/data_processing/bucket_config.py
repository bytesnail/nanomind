from dataclasses import dataclass
from functools import lru_cache

from .config_loader import get_config


@dataclass(frozen=True)
class BucketConfig:
    name: str
    min_score: float
    max_score: float | None
    sampling_rate: float

    def contains(self, score: float) -> bool:
        if score < self.min_score:
            return False
        return self.max_score is None or score < self.max_score

    def __repr__(self) -> str:
        interval = (
            f"[{self.min_score}, +∞)"
            if self.max_score is None
            else f"[{self.min_score}, {self.max_score})"
        )
        return f"BucketConfig(name='{self.name}', interval={interval}, sampling_rate={self.sampling_rate:.0%})"


@lru_cache(maxsize=1)
def _load_buckets() -> tuple[tuple[BucketConfig, ...], dict[str, BucketConfig]]:
    buckets = tuple(
        BucketConfig(
            name=b["name"],
            min_score=b["min_score"],
            max_score=b.get("max_score"),
            sampling_rate=b["sampling_rate"],
        )
        for b in get_config().get_bucket_configs()
    )
    return buckets, {b.name: b for b in buckets}


def get_bucket_config(name: str) -> BucketConfig:
    _, name_map = _load_buckets()
    if name not in name_map:
        raise ValueError(f"Unknown bucket: {name}")
    return name_map[name]


def get_all_bucket_configs() -> list[BucketConfig]:
    return list(_load_buckets()[0])


def find_bucket_for_score(score: float) -> BucketConfig | None:
    return next((b for b in _load_buckets()[0] if b.contains(score)), None)


def get_bucket_names() -> list[str]:
    return list(_load_buckets()[1].keys())
