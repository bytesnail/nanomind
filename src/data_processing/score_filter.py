"""评分过滤器。"""

import hashlib
import logging
from typing import TYPE_CHECKING

from datatrove.pipeline.base import PipelineStep

if TYPE_CHECKING:
    from pybloom_live import ScalableBloomFilter

    from .bucket_config import BucketConfig

logger = logging.getLogger(__name__)


class ScoreFilter(PipelineStep):
    """根据评分区间过滤文档并进行确定性采样和去重。"""

    def __init__(
        self,
        bucket: "BucketConfig",
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
        self._bloom_filter: "ScalableBloomFilter | None" = None

    def _init_bloom_filter(self) -> None:
        """延迟初始化 Bloom Filter。"""
        if self._bloom_filter is not None:
            return

        try:
            from pybloom_live import ScalableBloomFilter

            self._bloom_filter = ScalableBloomFilter(
                initial_capacity=self.bloom_capacity,
                error_rate=self.bloom_error_rate,
            )
        except ImportError as e:
            raise ImportError(
                "pybloom-live is required. Install: pip install pybloom-live"
            ) from e

    def _is_duplicate(self, doc_id: str) -> bool:
        """检查文档是否重复（进程内）。"""
        if not self.use_bloom_filter:
            return False

        self._init_bloom_filter()

        if doc_id in self._bloom_filter:  # type: ignore[operator]
            return True
        self._bloom_filter.add(doc_id)  # type: ignore[union-attr]
        return False

    def _should_sample(self, doc_id: str, rate: float) -> bool:
        """确定性采样：使用 MD5 哈希生成伪随机数。"""
        if rate >= 1.0:
            return True

        hash_input = f"{self.random_seed}_{doc_id}"
        hash_bytes = hashlib.md5(hash_input.encode(), usedforsecurity=False).digest()
        random_val = int.from_bytes(hash_bytes[:8], byteorder="big") / (2**64)

        return random_val < rate

    def run(self, data, rank: int = 0, world_size: int = 1):
        """处理文档流，过滤符合评分区间的文档。"""
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
