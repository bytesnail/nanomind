from dataclasses import dataclass

EPSILON = 1e-6


@dataclass(frozen=True)
class BucketConfig:
    name: str
    min_score: float
    max_score: float | None
    sampling_rate: float

    def contains(self, score: float) -> bool:
        ok = score >= self.min_score - EPSILON
        return ok if self.max_score is None else ok and score < self.max_score

    def __repr__(self) -> str:
        interval = (
            f"[{self.min_score}, +∞)"
            if self.max_score is None
            else f"[{self.min_score}, {self.max_score})"
        )
        return f"BucketConfig(name='{self.name}', interval={interval}, sampling_rate={self.sampling_rate:.0%})"


DEFAULT_BUCKETS = [
    BucketConfig("2.8", 2.8, 3.0, 0.30),
    BucketConfig("3.0", 3.0, 3.5, 0.60),
    BucketConfig("3.5", 3.5, 4.0, 0.80),
    BucketConfig("4.0", 4.0, None, 1.0),
]

BUCKET_NAMES = [b.name for b in DEFAULT_BUCKETS]
SAMPLING_RATES = {b.name: b.sampling_rate for b in DEFAULT_BUCKETS}
_BUCKET_MAP = {b.name: b for b in DEFAULT_BUCKETS}


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
