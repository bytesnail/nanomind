"""评分过滤器。"""

import hashlib

from datatrove.pipeline.base import PipelineStep
from pybloom_live import ScalableBloomFilter

from .bucket_config import BucketConfig


class ScoreFilter(PipelineStep):
    def __init__(
        self,
        bucket: BucketConfig,
        random_seed: int = 42,
        use_bloom_filter: bool = True,
        bloom_capacity: int = 2_000_000_000,
        bloom_error_rate: float = 0.001,
    ):
        super().__init__()
        self.bucket = bucket
        self.random_seed = random_seed
        self.use_bloom_filter = use_bloom_filter
        self.bloom_capacity = bloom_capacity
        self.bloom_error_rate = bloom_error_rate
        self._bloom: ScalableBloomFilter | None = None

    def _init_bloom(self) -> ScalableBloomFilter:
        if self._bloom is None:
            self._bloom = ScalableBloomFilter(
                initial_capacity=self.bloom_capacity,
                error_rate=self.bloom_error_rate,
            )
        return self._bloom

    def _is_duplicate(self, doc_id: str) -> bool:
        if not self.use_bloom_filter:
            return False
        bloom = self._init_bloom()
        if doc_id in bloom:
            return True
        bloom.add(doc_id)
        return False

    def _should_sample(self, doc_id: str, rate: float) -> bool:
        if rate >= 1.0:
            return True
        hash_bytes = hashlib.md5(
            f"{self.random_seed}_{doc_id}".encode(), usedforsecurity=False
        ).digest()
        random_val = int.from_bytes(hash_bytes[:8], byteorder="big") / (2**64)
        return random_val < rate

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            score = doc.metadata.get("score")

            if score is None:
                self.stat_update("missing_score", value=1)
                continue

            if not isinstance(score, (int, float)):
                self.stat_update("invalid_score", value=1)
                continue

            if not self.bucket.contains(float(score)):
                self.stat_update("filtered_out", value=1)
                continue

            if self._is_duplicate(doc.id):
                self.stat_update("duplicates_removed", value=1)
                continue

            if self._should_sample(doc.id, self.bucket.sampling_rate):
                self.stat_update("kept", value=1)
                yield doc
            else:
                self.stat_update("sampled_out", value=1)
