"""评分过滤器模块。

实现基于评分区间的文档过滤、确定性采样和进程内去重功能。
"""

import hashlib
import logging
from typing import TYPE_CHECKING

from datatrove.pipeline.base import PipelineStep

if TYPE_CHECKING:
    from pybloom_live import ScalableBloomFilter

    from .bucket_config import BucketConfig


logger = logging.getLogger(__name__)


class ScoreFilter(PipelineStep):
    """评分过滤器：根据评分区间过滤文档并进行确定性采样。

    每个桶独立运行一个 ScoreFilter 实例，实现按桶独立处理。
    支持进程内去重（使用 Bloom Filter）和确定性采样（基于 MD5 哈希）。
    """

    def __init__(
        self,
        bucket: "BucketConfig",
        random_seed: int = 42,
        use_bloom_filter: bool = True,
        bloom_capacity: int = 2_000_000_000,
        bloom_error_rate: float = 0.001,
    ):
        """初始化评分过滤器。

        Args:
            bucket: 评分桶配置
            random_seed: 随机种子，用于确定性采样
            use_bloom_filter: 是否使用 Bloom Filter 进行去重
            bloom_capacity: Bloom Filter 预估容量
            bloom_error_rate: Bloom Filter 误报率
        """
        super().__init__()
        self.bucket = bucket
        self.random_seed = random_seed
        self.use_bloom_filter = use_bloom_filter
        self.bloom_capacity = bloom_capacity
        self.bloom_error_rate = bloom_error_rate
        self._bloom_filter: ScalableBloomFilter | None = None

    def _ensure_bloom_filter_initialized(self) -> None:
        """确保 Bloom Filter 已初始化（按需初始化）。"""
        if self._bloom_filter is not None:
            return

        try:
            from pybloom_live import ScalableBloomFilter

            self._bloom_filter = ScalableBloomFilter(
                initial_capacity=self.bloom_capacity,
                error_rate=self.bloom_error_rate,
            )
        except ImportError as e:
            msg = (
                "pybloom-live is required for Bloom Filter deduplication. "
                "Install with: pip install pybloom-live"
            )
            raise ImportError(msg) from e

    def _is_duplicate(self, doc_id: str) -> bool:
        """检查文档是否重复（进程内）。

        Args:
            doc_id: 文档 ID

        Returns:
            bool: 如果是重复文档返回 True，否则返回 False
        """
        if not self.use_bloom_filter:
            return False

        self._ensure_bloom_filter_initialized()

        if doc_id in self._bloom_filter:  # type: ignore[operator]
            return True
        self._bloom_filter.add(doc_id)  # type: ignore[union-attr]
        return False

    def _should_sample(self, doc_id: str, rate: float) -> bool:
        """确定性采样：使用 MD5 哈希生成伪随机数。

        基于文档 ID 和随机种子生成确定性随机数，确保多进程一致性。
        使用 MD5 的前 8 字节生成 [0, 1) 范围内的伪随机数。

        Args:
            doc_id: 文档 ID
            rate: 采样率（0-1）

        Returns:
            bool: 是否保留该文档
        """
        if rate >= 1.0:
            return True

        hash_input = f"{self.random_seed}_{doc_id}"
        hash_bytes = hashlib.md5(hash_input.encode(), usedforsecurity=False).digest()
        hash_val = int.from_bytes(hash_bytes[:8], byteorder="big")
        random_val = hash_val / (2**64)

        return random_val < rate

    def run(
        self,
        data,
        rank: int = 0,
        world_size: int = 1,
    ):
        """处理文档流，过滤符合评分区间的文档。

        Args:
            data: 输入文档流
            rank: 当前进程编号
            world_size: 总进程数

        Yields:
            Document: 符合评分区间且通过采样的文档
        """
        for doc in data:
            # score 在 metadata 中，这是 Datatrove Document 的设计
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
