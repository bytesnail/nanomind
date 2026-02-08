import pytest
from datatrove.data import Document

from src.data_processing.bucket_config import BucketConfig
from src.data_processing.score_filter import ScoreFilter


class TestScoreFilter:
    @staticmethod
    def _doc(doc_id: str, score: float) -> Document:
        return Document(
            text=f"Test {doc_id}",
            id=doc_id,
            metadata={"score": score, "cc_main": "CC-MAIN-2024-10"},
        )

    def test_score_filtering(self):
        bucket = BucketConfig("3.0", 3.0, 3.5, 1.0)
        filter_step = ScoreFilter([bucket])
        docs = [
            self._doc("doc1", 2.9),
            self._doc("doc2", 3.0),
            self._doc("doc3", 3.4),
            self._doc("doc4", 3.5),
            self._doc("doc5", 4.0),
        ]
        result = list(filter_step.run(iter(docs)))
        assert len(result) == 2
        assert result[0].id == "doc2"
        assert result[1].id == "doc3"

    def test_missing_score(self):
        bucket = BucketConfig("3.0", 3.0, 3.5, 1.0)
        filter_step = ScoreFilter([bucket])
        docs = [
            Document(text="Test", id="doc1", metadata={"cc_main": "test"}),
            self._doc("doc2", 3.2),
        ]
        result = list(filter_step.run(iter(docs)))
        assert len(result) == 1
        assert result[0].id == "doc2"

    def test_deterministic_sampling(self):
        bucket = BucketConfig("3.0", 3.0, 3.5, 0.5)
        f1 = ScoreFilter([bucket], random_seed=42)
        f2 = ScoreFilter([bucket], random_seed=42)
        docs = [self._doc(f"doc{i}", 3.2) for i in range(100)]
        r1, r2 = list(f1.run(iter(docs))), list(f2.run(iter(docs)))
        assert len(r1) == len(r2)
        assert [d.id for d in r1] == [d.id for d in r2]

    def test_different_seeds(self):
        bucket = BucketConfig("3.0", 3.0, 3.5, 0.5)
        f1 = ScoreFilter([bucket], random_seed=42)
        f2 = ScoreFilter([bucket], random_seed=24)
        docs = [self._doc(f"doc{i}", 3.2) for i in range(1000)]
        ids1 = {d.id for d in f1.run(iter(docs))}
        ids2 = {d.id for d in f2.run(iter(docs))}
        assert ids1 != ids2

    def test_full_sampling(self):
        bucket = BucketConfig("4.0", 4.0, None, 1.0)
        filter_step = ScoreFilter([bucket])
        docs = [self._doc(f"doc{i}", 4.5) for i in range(100)]
        assert len(list(filter_step.run(iter(docs)))) == 100

    def test_zero_sampling(self):
        bucket = BucketConfig("3.0", 3.0, 3.5, 0.0)
        filter_step = ScoreFilter([bucket])
        docs = [self._doc(f"doc{i}", 3.2) for i in range(100)]
        assert len(list(filter_step.run(iter(docs)))) == 0

    @pytest.mark.parametrize(
        "name,min_score,max_score,scores,expected_ids",
        [
            ("2.8", 2.8, 3.0, [2.79, 2.8, 2.99, 3.0], ["doc2", "doc3"]),
            ("4.0", 4.0, None, [3.9, 4.0, 5.0, 10.0], ["doc2", "doc3", "doc4"]),
        ],
    )
    def test_bucket_boundaries(self, name, min_score, max_score, scores, expected_ids):
        bucket = BucketConfig(name, min_score, max_score, 1.0)
        filter_step = ScoreFilter([bucket])
        docs = [self._doc(f"doc{i + 1}", s) for i, s in enumerate(scores)]
        result = list(filter_step.run(iter(docs)))
        assert len(result) == len(expected_ids)
        assert [d.id for d in result] == expected_ids
