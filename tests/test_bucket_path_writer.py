from typing import Any

from datatrove.data import Document

from src.data_processing.bucket_config import BucketConfig
from src.data_processing.bucket_path_writer import BucketPathWriter


class TestBucketPathWriter:
    @staticmethod
    def _make_doc(doc_id: str, text: str, score: float, bucket: str | None) -> Document:
        metadata: dict[str, Any] = {"score": score}
        if bucket:
            metadata["__target_bucket"] = bucket
        return Document(text=text, id=doc_id, metadata=metadata)

    @staticmethod
    def _count_parquet_files(path) -> int:
        return len(list(path.glob("*.parquet")))

    @staticmethod
    def _create_writer(tmp_path, buckets):
        return BucketPathWriter(output_dir=str(tmp_path / "output"), buckets=buckets)

    def test_single_bucket_writing(self, tmp_path):
        writer = self._create_writer(tmp_path, [BucketConfig("3.0", 3.0, 3.5, 1.0)])
        doc = self._make_doc("doc1", "Test content", 3.2, "3.0")
        writer.run([doc], rank=0)

        assert self._count_parquet_files(tmp_path / "output" / "3.0") == 1

    def test_multi_bucket_initialization(self, tmp_path):
        self._create_writer(
            tmp_path,
            [
                BucketConfig("2.8", 2.8, 3.0, 1.0),
                BucketConfig("3.0", 3.0, 3.5, 1.0),
            ],
        )

        assert (tmp_path / "output" / "2.8").exists()
        assert (tmp_path / "output" / "3.0").exists()

    def test_multi_bucket_file_creation(self, tmp_path):
        writer = self._create_writer(
            tmp_path,
            [
                BucketConfig("3.0", 3.0, 3.5, 1.0),
                BucketConfig("3.5", 3.5, 4.0, 1.0),
            ],
        )
        doc = self._make_doc("doc1", "Test content", 3.2, "3.0")
        writer.run([doc], rank=0)

        assert self._count_parquet_files(tmp_path / "output" / "3.0") == 1

    def test_multi_bucket_routing(self, tmp_path):
        writer = self._create_writer(
            tmp_path,
            [
                BucketConfig("2.8", 2.8, 3.0, 1.0),
                BucketConfig("3.0", 3.0, 3.5, 1.0),
            ],
        )
        docs = [
            self._make_doc("doc1", "Content 1", 2.9, "2.8"),
            self._make_doc("doc2", "Content 2", 3.2, "3.0"),
            self._make_doc("doc3", "Content 3", 2.85, "2.8"),
        ]
        writer.run(docs, rank=0)

        assert self._count_parquet_files(tmp_path / "output" / "2.8") == 1
        assert self._count_parquet_files(tmp_path / "output" / "3.0") == 1

    def test_missing_bucket_tag(self, tmp_path):
        writer = self._create_writer(
            tmp_path,
            [
                BucketConfig("3.0", 3.0, 3.5, 1.0),
                BucketConfig("3.5", 3.5, 4.0, 1.0),
            ],
        )
        doc = self._make_doc("doc1", "Test content", 3.2, None)
        writer.run([doc], rank=0)

        assert self._count_parquet_files(tmp_path / "output" / "3.0") == 0
