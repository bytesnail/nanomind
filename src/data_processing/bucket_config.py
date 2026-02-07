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
    if bucket := _BUCKET_MAP.get(name):
        return bucket
    available = ", ".join(_BUCKET_MAP.keys()) if _BUCKET_MAP else "None"
    raise ValueError(f"Unknown bucket: {name}. Available: {available}")


def get_all_bucket_configs() -> list[BucketConfig]:
    _ensure_loaded()
    return list(_DEFAULT_BUCKETS)


def find_bucket_for_score(score: float) -> BucketConfig | None:
    for b in get_all_bucket_configs():
        if b.contains(score):
            return b
    return None


def get_bucket_names() -> list[str]:
    return [b.name for b in get_all_bucket_configs()]


def get_sampling_rates() -> dict[str, float]:
    return {b.name: b.sampling_rate for b in get_all_bucket_configs()}
