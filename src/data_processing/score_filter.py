import hashlib

from datatrove.pipeline.base import PipelineStep

from .bucket_config import BucketConfig


class ScoreFilter(PipelineStep):
    def __init__(self, bucket: BucketConfig, random_seed: int = 42):
        super().__init__()
        self.bucket = bucket
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
            if not isinstance(score, (int, float)):
                self.stat_update("invalid_score", value=1)
                continue
            if not self.bucket.contains(float(score)):
                self.stat_update("filtered_out", value=1)
                continue

            if self._should_sample(doc.id, self.bucket.sampling_rate):
                self.stat_update("kept", value=1)
                yield doc
            else:
                self.stat_update("sampled_out", value=1)
