from dataclasses import dataclass

from .config_loader import get_config


@dataclass(frozen=True)
class BucketConfig:
    name: str
    min_score: float
    max_score: float | None
    sampling_rate: float

    def contains(self, score: float) -> bool:
        epsilon = get_config().get_epsilon()
        ok = score >= self.min_score - epsilon
        return ok if self.max_score is None else ok and score < self.max_score

    def __repr__(self) -> str:
        interval = (
            f"[{self.min_score}, +∞)"
            if self.max_score is None
            else f"[{self.min_score}, {self.max_score})"
        )
        return f"BucketConfig(name='{self.name}', interval={interval}, sampling_rate={self.sampling_rate:.0%})"


def _load_buckets() -> list[BucketConfig]:
    config = get_config()
    bucket_data = config.get_bucket_configs()
    return [
        BucketConfig(
            name=b["name"],
            min_score=b["min_score"],
            max_score=b.get("max_score"),
            sampling_rate=b["sampling_rate"],
        )
        for b in bucket_data
    ]


_DEFAULT_BUCKETS: list[BucketConfig] | None = None
_BUCKET_MAP: dict[str, BucketConfig] | None = None


def _ensure_loaded():
    global _DEFAULT_BUCKETS, _BUCKET_MAP
    if _DEFAULT_BUCKETS is None:
        _DEFAULT_BUCKETS = _load_buckets()
        _BUCKET_MAP = {b.name: b for b in _DEFAULT_BUCKETS}


def get_bucket_config(name: str) -> BucketConfig:
    _ensure_loaded()
    if _BUCKET_MAP is None:
        available = "None"
        raise ValueError(f"Unknown bucket: {name}. Available: {available}")
    bucket_map = _BUCKET_MAP
    bucket_config = bucket_map.get(name)
    if bucket_config is None:
        available = ", ".join(bucket_map.keys())
        raise ValueError(f"Unknown bucket: {name}. Available: {available}")
    return bucket_config


def get_all_bucket_configs() -> list[BucketConfig]:
    _ensure_loaded()
    return list(_DEFAULT_BUCKETS) if _DEFAULT_BUCKETS else []


def find_bucket_for_score(score: float) -> BucketConfig | None:
    return next((b for b in get_all_bucket_configs() if b.contains(score)), None)


def get_bucket_names() -> list[str]:
    return [b.name for b in get_all_bucket_configs()]


def get_sampling_rates() -> dict[str, float]:
    return {b.name: b.sampling_rate for b in get_all_bucket_configs()}
