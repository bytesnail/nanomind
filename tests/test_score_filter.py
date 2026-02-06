"""测试评分过滤器模块。"""

from datatrove.data import Document

from src.data_processing.bucket_config import BucketConfig
from src.data_processing.score_filter import ScoreFilter


class TestScoreFilter:
    """测试 ScoreFilter 类。"""

    def create_document(self, doc_id: str, score: float) -> Document:
        """创建测试文档。"""
        return Document(
            text=f"Test text {doc_id}",
            id=doc_id,
            metadata={"score": score, "cc_main": "CC-MAIN-2024-10"},
        )

    def test_score_filtering(self):
        """测试评分过滤功能。"""
        bucket = BucketConfig("3.0", 3.0, 3.5, 1.0)
        filter_step = ScoreFilter(bucket, use_bloom_filter=False)

        docs = [
            self.create_document("doc1", 2.9),  # 低于范围
            self.create_document("doc2", 3.0),  # 在范围内
            self.create_document("doc3", 3.4),  # 在范围内
            self.create_document("doc4", 3.5),  # 等于上限（不包含）
            self.create_document("doc5", 4.0),  # 高于范围
        ]

        result = list(filter_step.run(iter(docs)))

        assert len(result) == 2
        assert result[0].id == "doc2"
        assert result[1].id == "doc3"

    def test_missing_score(self):
        """测试缺少 score 字段。"""
        bucket = BucketConfig("3.0", 3.0, 3.5, 1.0)
        filter_step = ScoreFilter(bucket, use_bloom_filter=False)

        docs = [
            Document(text="Test", id="doc1", metadata={"cc_main": "CC-MAIN-2024-10"}),
            self.create_document("doc2", 3.2),
        ]

        result = list(filter_step.run(iter(docs)))

        assert len(result) == 1
        assert result[0].id == "doc2"

    def test_invalid_score_type(self):
        """测试无效的 score 类型。"""
        bucket = BucketConfig("3.0", 3.0, 3.5, 1.0)
        filter_step = ScoreFilter(bucket, use_bloom_filter=False)

        docs = [
            Document(
                text="Test", id="doc1", metadata={"score": "invalid", "cc_main": "test"}
            ),
            self.create_document("doc2", 3.2),
        ]

        result = list(filter_step.run(iter(docs)))

        assert len(result) == 1
        assert result[0].id == "doc2"

    def test_deterministic_sampling(self):
        """测试确定性采样的一致性。"""
        bucket = BucketConfig("3.0", 3.0, 3.5, 0.5)
        filter_step1 = ScoreFilter(bucket, random_seed=42, use_bloom_filter=False)
        filter_step2 = ScoreFilter(bucket, random_seed=42, use_bloom_filter=False)

        docs = [self.create_document(f"doc{i}", 3.2) for i in range(100)]

        result1 = list(filter_step1.run(iter(docs)))
        result2 = list(filter_step2.run(iter(docs)))

        # 相同种子应该产生相同结果
        assert len(result1) == len(result2)
        assert [d.id for d in result1] == [d.id for d in result2]

    def test_different_seeds(self):
        """测试不同种子产生不同采样结果。"""
        bucket = BucketConfig("3.0", 3.0, 3.5, 0.5)
        filter_step1 = ScoreFilter(bucket, random_seed=42, use_bloom_filter=False)
        filter_step2 = ScoreFilter(bucket, random_seed=24, use_bloom_filter=False)

        docs = [self.create_document(f"doc{i}", 3.2) for i in range(1000)]

        result1 = list(filter_step1.run(iter(docs)))
        result2 = list(filter_step2.run(iter(docs)))

        # 不同种子应该产生不同结果（概率极高，使用大样本避免偶然相同）
        ids1 = {d.id for d in result1}
        ids2 = {d.id for d in result2}
        assert ids1 != ids2, "不同随机种子应该产生不同的采样结果"

    def test_full_sampling(self):
        """测试 100% 采样率。"""
        bucket = BucketConfig("4.0", 4.0, None, 1.0)
        filter_step = ScoreFilter(bucket, use_bloom_filter=False)

        docs = [self.create_document(f"doc{i}", 4.5) for i in range(100)]

        result = list(filter_step.run(iter(docs)))

        assert len(result) == 100

    def test_zero_sampling(self):
        """测试 0% 采样率。"""
        bucket = BucketConfig("3.0", 3.0, 3.5, 0.0)
        filter_step = ScoreFilter(bucket, use_bloom_filter=False)

        docs = [self.create_document(f"doc{i}", 3.2) for i in range(100)]

        result = list(filter_step.run(iter(docs)))

        assert len(result) == 0

    def test_deduplication(self):
        """测试去重功能。"""
        bucket = BucketConfig("3.0", 3.0, 3.5, 1.0)
        filter_step = ScoreFilter(bucket, use_bloom_filter=True)

        docs = [
            self.create_document("doc1", 3.2),
            self.create_document("doc1", 3.2),  # 重复 ID
            self.create_document("doc2", 3.3),
        ]

        result = list(filter_step.run(iter(docs)))

        assert len(result) == 2
        assert result[0].id == "doc1"
        assert result[1].id == "doc2"

    def test_bucket_28(self):
        """测试 2.8 桶的边界条件。"""
        bucket = BucketConfig("2.8", 2.8, 3.0, 1.0)
        filter_step = ScoreFilter(bucket, use_bloom_filter=False)

        docs = [
            self.create_document("doc1", 2.79),  # 低于
            self.create_document("doc2", 2.8),  # 边界（包含）
            self.create_document("doc3", 2.99),  # 范围内
            self.create_document("doc4", 3.0),  # 边界（不包含）
            self.create_document("doc5", 3.1),  # 高于
        ]

        result = list(filter_step.run(iter(docs)))

        assert len(result) == 2
        assert result[0].id == "doc2"
        assert result[1].id == "doc3"

    def test_bucket_40(self):
        """测试 4.0 桶（无上限）。"""
        bucket = BucketConfig("4.0", 4.0, None, 1.0)
        filter_step = ScoreFilter(bucket, use_bloom_filter=False)

        docs = [
            self.create_document("doc1", 3.9),  # 低于
            self.create_document("doc2", 4.0),  # 边界（包含）
            self.create_document("doc3", 5.0),  # 范围内
            self.create_document("doc4", 10.0),  # 很高
        ]

        result = list(filter_step.run(iter(docs)))

        assert len(result) == 3
        assert result[0].id == "doc2"
        assert result[1].id == "doc3"
        assert result[2].id == "doc4"
