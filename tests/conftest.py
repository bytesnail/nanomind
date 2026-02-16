import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from datatrove.data import Document

from src.data_processing.bucket_config import BucketConfig


@pytest.fixture
def sample_document():
    return Document(
        text="This is a test document.",
        id="test_doc_001",
        metadata={"score": 4.5},
    )


@pytest.fixture
def sample_documents():
    def _create(num_docs: int = 10, score: float = 3.5):
        return [
            Document(text=f"text_{i}", id=f"doc_{i}", metadata={"score": score})
            for i in range(num_docs)
        ]

    return _create


@pytest.fixture
def sample_buckets():
    return [
        BucketConfig("3.0", 3.0, 3.5, 0.5),
        BucketConfig("3.5", 3.5, 4.0, 0.8),
    ]


@pytest.fixture
def create_parquet(tmp_path):
    def _create(
        num_rows: int = 10, filename: str = "test.parquet", columns: dict = None
    ):
        if columns is None:
            columns = {"text": [f"text_{i}" for i in range(num_rows)]}
        table = pa.table(columns)
        file_path = tmp_path / filename
        pq.write_table(table, file_path)
        return file_path

    return _create
