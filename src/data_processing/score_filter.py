import hashlib

from datatrove.pipeline.base import PipelineStep

from .bucket_config import BucketConfig, _find_bucket_in_sorted


class ScoreFilter(PipelineStep):
    name = "Score Filter"
    type = "🎯 - FILTER"

    def __init__(self, buckets: list[BucketConfig], random_seed: int = 42):
        super().__init__()
        self.buckets = buckets
        self.random_seed = random_seed

    def _should_sample(self, doc_id: str, rate: float) -> bool:
        if rate >= 1.0:
            return True
        data = f"{self.random_seed}_{doc_id}".encode()
        h = int.from_bytes(hashlib.md5(data, usedforsecurity=False).digest()[:8], "big")
        return h / (2**64) < rate

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            score = doc.metadata.get("score")
            if score is None:
                self.stat_update("missing_score", value=1)
                continue

            score = float(score)
            bucket = _find_bucket_in_sorted(score, self.buckets)
            if bucket is None:
                self.stat_update("filtered_out", value=1)
                continue

            if self._should_sample(doc.id, bucket.sampling_rate):
                doc.metadata["__target_bucket"] = bucket.name
                self.stat_update(f"kept_{bucket.name}", value=1)
                yield doc
            else:
                self.stat_update(f"sampled_out_{bucket.name}", value=1)
