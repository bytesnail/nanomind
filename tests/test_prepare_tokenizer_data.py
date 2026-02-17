"""Tests for prepare_tokenizer_data module."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from datatrove.data import Document

from scripts.prepare_tokenizer_data import (
    IndexFilter,
    SamplingConfig,
    SamplingInfo,
    SourceTagger,
    TokenizerDataConfig,
    TokenizerDataWriter,
    calculate_tasks,
    create_row_index_adapter,
    find_bucket_dir,
    compute_doc_hash,
    count_total_rows_fast,
    determine_text_column,
    load_config,
    precompute_sampling_indices,
    save_sampling_info,
)


def create_test_parquet(path: Path, num_rows: int, text_key: str = "text") -> Path:
    """辅助函数：创建测试 Parquet 文件。"""
    table = pa.table({text_key: [f"text_{i}" for i in range(num_rows)]})
    file_path = path / "test.parquet"
    pq.write_table(table, file_path)
    return file_path


def create_test_documents(file_path: Path, num_docs: int) -> list[Document]:
    """辅助函数：创建测试 Document 列表。"""
    return [
        Document(
            text=f"text_{i}",
            id=f"doc_{i}",
            metadata={"file_path": str(file_path), "row_idx": i},
        )
        for i in range(num_docs)
    ]


@pytest.fixture
def valid_config_dict():
    return {
        "datasets": {
            "fineweb_en": {
                "name": "fineweb_edu_en",
                "source": "data/fineweb/en",
                "samples": 1000,
                "buckets": {
                    "4.0": {"count": 500},
                    "3.0": {"count": 500},
                },
            }
        },
        "random_seed": 42,
        "output_format": "parquet",
        "output_dir": "data/output",
    }


@pytest.fixture
def sample_document():
    return Document(
        text="This is a test document.",
        id="test_doc_001",
        metadata={"score": 4.5},
    )


class TestComputeDocHash:
    def test_determinism(self):
        hash1 = compute_doc_hash("doc_123", seed=42)
        hash2 = compute_doc_hash("doc_123", seed=42)
        assert hash1 == hash2

    def test_different_seeds(self):
        hash1 = compute_doc_hash("doc_123", seed=42)
        hash2 = compute_doc_hash("doc_123", seed=24)
        assert hash1 != hash2

    def test_valid_int(self):
        result = compute_doc_hash("doc_123", seed=42)
        assert isinstance(result, int)
        assert 0 <= result < 2**64


class TestDetermineTextColumn:
    @pytest.mark.parametrize(
        "dataset_name,expected",
        [
            ("github_code", "content"),
            ("GitHub-Code-2025", "content"),
            ("github-code", "content"),
        ],
    )
    def test_github_code(self, dataset_name, expected):
        assert determine_text_column(dataset_name) == expected

    @pytest.mark.parametrize(
        "dataset_name",
        ["fineweb_edu_en", "fineweb_edu_zh", "other_dataset"],
    )
    def test_other_datasets(self, dataset_name):
        assert determine_text_column(dataset_name) == "text"


class TestLoadConfig:
    def test_load_valid(self, tmp_path, valid_config_dict):
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(valid_config_dict, f)

        config = load_config(config_path)
        assert isinstance(config, TokenizerDataConfig)
        assert config.random_seed == 42
        assert len(config.datasets) == 1

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestSamplingConfig:
    def test_get_all_counts_with_buckets(self):
        config = SamplingConfig(
            name="test",
            source=Path("data/test"),
            samples=1000,
            buckets={"4.0": 500, "3.0": 500},
        )
        assert config.get_all_counts() == {"4.0": 500, "3.0": 500}

    def test_get_all_counts_empty(self):
        config = SamplingConfig(
            name="test",
            source=Path("data/test"),
            samples=1000,
        )
        assert config.get_all_counts() == {}


class TestPrecomputeSamplingIndices:
    def test_selects_correct_number(self, tmp_path):
        file_path = create_test_parquet(tmp_path, 20)

        indices = precompute_sampling_indices(
            files=[file_path],
            bucket_name="test",
            seed=42,
            target_count=5,
        )

        assert sum(len(v) for v in indices.values()) == 5

    def test_determinism(self, tmp_path):
        file_path = create_test_parquet(tmp_path, 20)

        indices1 = precompute_sampling_indices(
            files=[file_path],
            bucket_name="test",
            seed=42,
            target_count=5,
        )
        indices2 = precompute_sampling_indices(
            files=[file_path],
            bucket_name="test",
            seed=42,
            target_count=5,
        )

        assert indices1 == indices2

    def test_empty_files(self):
        indices = precompute_sampling_indices(
            files=[],
            bucket_name="test",
            seed=42,
            target_count=5,
        )
        assert indices == {}

    def test_target_exceeds_total(self, tmp_path):
        file_path = create_test_parquet(tmp_path, 5)

        indices = precompute_sampling_indices(
            files=[file_path],
            bucket_name="test",
            seed=42,
            target_count=100,
        )

        assert sum(len(v) for v in indices.values()) == 5


class TestIndexFilter:
    def test_filters_by_indices(self, tmp_path):
        file_path = create_test_parquet(tmp_path, 10)
        filter_step = IndexFilter(indices={file_path: {1, 3, 5}})
        docs = create_test_documents(file_path, 10)

        result = list(filter_step.run(iter(docs), rank=0, world_size=1))
        assert len(result) == 3
        assert {doc.metadata["row_idx"] for doc in result} == {1, 3, 5}

    def test_passes_all_when_all_in_indices(self, tmp_path):
        file_path = create_test_parquet(tmp_path, 5)
        filter_step = IndexFilter(indices={file_path: {0, 1, 2, 3, 4}})
        docs = create_test_documents(file_path, 5)

        result = list(filter_step.run(iter(docs), rank=0, world_size=1))
        assert len(result) == 5

    def test_filters_all_when_none_in_indices(self, tmp_path):
        file_path = create_test_parquet(tmp_path, 5)
        filter_step = IndexFilter(indices={file_path: set()})
        docs = create_test_documents(file_path, 5)

        result = list(filter_step.run(iter(docs), rank=0, world_size=1))
        assert len(result) == 0

    def test_works_with_parquet_reader(self, tmp_path):
        from datatrove.pipeline.readers import ParquetReader

        from scripts.prepare_tokenizer_data import create_row_index_adapter

        file_path = create_test_parquet(tmp_path, 10)

        indices = precompute_sampling_indices(
            files=[file_path],
            bucket_name="test",
            seed=42,
            target_count=5,
        )

        filter_step = IndexFilter(indices=indices)
        adapter = create_row_index_adapter("text")
        reader = ParquetReader(
            data_folder=str(tmp_path),
            glob_pattern="*.parquet",
            adapter=adapter,
        )

        docs = list(reader.read_file("test.parquet"))
        filtered = list(filter_step.run(iter(docs), rank=0, world_size=1))

        assert len(filtered) == 5


class TestCreateRowIndexAdapter:
    def test_fineweb_id_preserved(self):
        adapter = create_row_index_adapter("text", "id")
        result = adapter(
            None,
            {"text": "test", "id": "data/datasets/fineweb/file.parquet#123"},
            "/path/to/file.parquet",
            0,
        )
        assert result["id"] == "data/datasets/fineweb/file.parquet#123"

    def test_full_path_id_format(self):
        adapter = create_row_index_adapter("text", "id")
        result = adapter(
            None,
            {"text": "test"},
            "/data/datasets/github/repo/file.parquet",
            42,
        )
        assert result["id"] == "/data/datasets/github/repo/file.parquet#42"

    def test_id_format_matches_fineweb_adapter(self):
        adapter = create_row_index_adapter("text", "id")
        path = "data/datasets/nick007x/github-code-2025/bucket/data.parquet"
        result = adapter(None, {"text": "test"}, path, 5)
        assert result["id"] == f"{path}#5"


class TestCountTotalRowsFast:
    def test_counts_rows_correctly(self, tmp_path):
        file_path = create_test_parquet(tmp_path, 100)
        assert count_total_rows_fast([file_path]) == 100

    def test_empty_files(self):
        assert count_total_rows_fast([]) == 0

    def test_multiple_files(self, tmp_path):
        for i in range(3):
            table = pa.table({"text": [f"text_{j}" for j in range(10)]})
            pq.write_table(table, tmp_path / f"test_{i}.parquet")

        files = list(tmp_path.glob("*.parquet"))
        assert count_total_rows_fast(files) == 30


class TestSourceTagger:
    def test_adds_source_info(self, sample_document):
        tagger = SourceTagger(dataset_name="fineweb", bucket_name="4.0")
        result = list(tagger.run(iter([sample_document]), rank=0, world_size=1))
        assert result[0].metadata["source_dataset"] == "fineweb"
        assert result[0].metadata["source_bucket"] == "4.0"

    def test_preserves_existing_metadata(self):
        doc = Document(text="Test", id="test", metadata={"existing": "value"})
        tagger = SourceTagger(dataset_name="fineweb", bucket_name="4.0")
        result = list(tagger.run(iter([doc]), rank=0, world_size=1))
        assert result[0].metadata["existing"] == "value"


class TestTokenizerDataWriter:
    def test_creates_output_directory(self, tmp_path):
        output_dir = tmp_path / "output"
        TokenizerDataWriter(
            output_dir=str(output_dir), max_rows_per_file=1000, buffer_size=100
        )
        assert output_dir.exists()

    def test_creates_single_file(self, tmp_path):
        output_dir = tmp_path / "output"
        writer = TokenizerDataWriter(
            output_dir=str(output_dir),
            dataset_name="test_dataset",
            bucket_name="4.0",
            max_rows_per_file=1000,
            buffer_size=10,
        )
        docs = [
            Document(
                text=f"text_{i}",
                id=f"doc_{i}",
                metadata={"source_dataset": "test_dataset", "source_bucket": "4.0"},
            )
            for i in range(5)
        ]
        writer.run(iter(docs), rank=0, world_size=1)
        files = list(output_dir.glob("*.parquet"))
        assert len(files) == 1
        assert files[0].name.startswith("test_dataset-4.0-")

    def test_correct_schema(self, tmp_path):
        output_dir = tmp_path / "output"
        writer = TokenizerDataWriter(
            output_dir=str(output_dir),
            dataset_name="fineweb_edu_en",
            bucket_name="4.0",
            max_rows_per_file=100,
            buffer_size=10,
        )
        doc = Document(
            text="hello world",
            id="doc_1",
            metadata={"source_dataset": "fineweb_edu_en", "source_bucket": "4.0"},
        )
        writer.run(iter([doc]), rank=0, world_size=1)
        files = list(output_dir.glob("*.parquet"))
        table = pq.read_table(files[0])
        assert "text" in table.column_names
        assert "source_dataset" in table.column_names
        assert "id" in table.column_names
        assert table.num_rows == 1

    def test_filename_format(self, tmp_path):
        output_dir = tmp_path / "output"
        writer = TokenizerDataWriter(
            output_dir=str(output_dir),
            dataset_name="fineweb_edu_zh",
            bucket_name="3.5",
            max_rows_per_file=100,
            buffer_size=10,
        )
        doc = Document(
            text="test",
            id="test_id",
            metadata={"source_dataset": "fineweb_edu_zh", "source_bucket": "3.5"},
        )
        writer.run(iter([doc]), rank=0, world_size=1)
        files = list(output_dir.glob("*.parquet"))
        assert len(files) == 1
        assert files[0].name == "fineweb_edu_zh-3.5-00000-rank-00000.parquet"


class TestSaveSamplingInfo:
    def test_creates_file(self, tmp_path):
        info = SamplingInfo(
            total_requested=1000,
            total_sampled=950,
            sources={"dataset1": {"requested": 500, "sampled": 480}},
            random_seed=42,
        )
        output_path = save_sampling_info(info, tmp_path)
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["total_requested"] == 1000


class TestCalculateTasks:
    def test_explicit_tasks_value(self):
        assert calculate_tasks(tasks=10, workers=4, item_count=1000) == 10

    def test_auto_with_item_count(self):
        assert calculate_tasks(tasks=0, workers=4, item_count=50000) == 4
        assert calculate_tasks(tasks=0, workers=8, item_count=50000) == 5
        assert calculate_tasks(tasks=0, workers=16, item_count=5000) == 1

    def test_auto_without_item_count(self):
        assert calculate_tasks(tasks=0, workers=8) == 8
        assert calculate_tasks(tasks=0, workers=1) == 1


class TestFindBucketDir:
    def test_finds_bucket_dir(self, tmp_path):
        bucket_dir = tmp_path / "4.0"
        bucket_dir.mkdir()
        file_path = bucket_dir / "data.parquet"

        result = find_bucket_dir([file_path], "4.0")
        assert result == bucket_dir

    def test_finds_nested_bucket_dir(self, tmp_path):
        bucket_dir = tmp_path / "fineweb" / "en" / "4.0"
        bucket_dir.mkdir(parents=True)
        file_path = bucket_dir / "data.parquet"

        result = find_bucket_dir([file_path], "4.0")
        assert result == bucket_dir
