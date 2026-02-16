from dataclasses import dataclass
from functools import lru_cache

from .config_loader import get_raw_bucket_configs


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
        return (
            f"BucketConfig(name='{self.name}', interval={interval}, "
            f"sampling_rate={self.sampling_rate:.0%})"
        )


def _create_bucket_configs(
    bucket_list: list[dict],
) -> tuple[tuple[BucketConfig, ...], dict[str, BucketConfig]]:
    buckets = tuple(
        BucketConfig(
            name=b["name"],
            min_score=b["min_score"],
            max_score=b.get("max_score"),
            sampling_rate=b["sampling_rate"],
        )
        for b in bucket_list
    )
    return buckets, {b.name: b for b in buckets}


@lru_cache(maxsize=4)
def get_bucket_configs_for_dataset(
    dataset_key: str,
) -> tuple[tuple[BucketConfig, ...], dict[str, BucketConfig]]:
    bucket_list = get_raw_bucket_configs(dataset_key)
    if not bucket_list:
        raise ValueError(f"No bucket configuration found for dataset: {dataset_key}")
    return _create_bucket_configs(bucket_list)


def get_all_bucket_configs(dataset_key: str) -> list[BucketConfig]:
    configs, _ = get_bucket_configs_for_dataset(dataset_key)
    return list(configs)


def find_bucket_in_sorted(
    score: float,
    buckets: tuple[BucketConfig, ...] | list[BucketConfig],
) -> BucketConfig | None:
    """在已排序的桶列表中二分查找评分对应的桶。"""
    left, right = 0, len(buckets)
    while left < right:
        mid = (left + right) // 2
        bucket = buckets[mid]
        if score < bucket.min_score:
            right = mid
        elif bucket.max_score is not None and score >= bucket.max_score:
            left = mid + 1
        else:
            return bucket
    return None


def find_bucket_for_score(
    score: float,
    dataset_key: str,
) -> BucketConfig | None:
    """查找评分对应的桶。"""
    buckets = get_bucket_configs_for_dataset(dataset_key)[0]
    return find_bucket_in_sorted(score, buckets)
