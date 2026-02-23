"""Tests for prepare_tokenizer_data module."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from datatrove.data import Document

from scripts.prepare_tokenizer_data import (
    ALLOWED_LANGUAGES,
    IndexFilter,
    LANGUAGE_EXTENSIONS,
    LanguageTagger,
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
    get_file_extension,
    load_config,
    precompute_sampling_indices,
    save_sampling_info,
)


def create_test_parquet(
    path: Path, num_rows: int, text_key: str = "text", file_path_col: str | None = None
) -> Path:
    """辅助函数：创建测试 Parquet 文件。"""
    data = {text_key: [f"text_{i}" for i in range(num_rows)]}
    if file_path_col:
        data[file_path_col] = [f"path/to/file_{i}.py" for i in range(num_rows)]
    table = pa.table(data)
    file_path = path / "test.parquet"
    pq.write_table(table, file_path)
    return file_path


def create_test_documents(file_path: Path, num_docs: int) -> list[Document]:
    """辅助函数：创建测试 Document 列表。"""
    return [
        Document(
            text=f"text_{i}",
            id=f"doc_{i}",
            metadata={"parquet_path": str(file_path), "row_idx": i},
        )
        for i in range(num_docs)
    ]


def create_source_document(
    text: str = "test",
    doc_id: str = "doc_1",
    dataset_name: str = "test_dataset",
    bucket_name: str = "test_bucket",
) -> Document:
    """辅助函数：创建带有 source_dataset 和 source_bucket 的测试 Document。"""
    return Document(
        text=text,
        id=doc_id,
        metadata={
            "source_dataset": dataset_name,
            "source_bucket": bucket_name,
        },
    )


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


class TestGetFileExtension:
    """测试全局函数 get_file_extension。"""

    def test_common_extensions(self):
        assert get_file_extension("test.py") == ".py"
        assert get_file_extension("/path/to/file.cpp") == ".cpp"
        assert get_file_extension("script.js") == ".js"
        assert get_file_extension("App.tsx") == ".tsx"

    def test_empty_and_none(self):
        assert get_file_extension("") is None

    def test_no_extension(self):
        assert get_file_extension("Makefile") is None
        assert get_file_extension("/path/to/README") is None

    def test_case_insensitive(self):
        assert get_file_extension("file.PY") == ".py"
        assert get_file_extension("file.CPP") == ".cpp"


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

    def test_with_allowed_extensions(self, tmp_path):
        """测试带扩展名过滤的预计算采样。"""
        # 创建包含 file_path 列的 parquet 文件
        file_path = create_test_parquet(tmp_path, 20, file_path_col="file_path")

        indices = precompute_sampling_indices(
            files=[file_path],
            bucket_name="test",
            seed=42,
            target_count=5,
            allowed_extensions={".py", ".js"},
        )

        # 应该返回索引（所有测试数据都是 .py 扩展名）
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

        # 创建测试文件（结果未使用，但为了创建文件）
        _ = create_test_parquet(tmp_path, 10)
        # 使用相对路径创建索引，与 ParquetReader 行为一致
        relative_path = "test.parquet"
        indices = {relative_path: {0, 1, 2, 3, 4}}
        filter_step = IndexFilter(indices=indices)
        adapter = create_row_index_adapter("text")
        reader = ParquetReader(
            data_folder=str(tmp_path),
            glob_pattern="*.parquet",
            adapter=adapter,
        )
        docs = list(reader.read_file("test.parquet"))
        result = list(filter_step.run(iter(docs), rank=0, world_size=1))
        assert len(result) == 5

    def test_accepts_list_indices(self, tmp_path):
        """测试 IndexFilter 接受 list 格式的索引（JSON 序列化场景）。"""
        file_path = create_test_parquet(tmp_path, 10)
        filter_step = IndexFilter(indices={str(file_path): [1, 3, 5]})
        docs = create_test_documents(file_path, 10)

        result = list(filter_step.run(iter(docs), rank=0, world_size=1))
        assert len(result) == 3
        assert {doc.metadata["row_idx"] for doc in result} == {1, 3, 5}

    def test_accepts_string_indices(self, tmp_path):
        """测试 IndexFilter 接受字符串格式的 set（JSON 序列化恢复场景）。"""
        file_path = create_test_parquet(tmp_path, 10)
        filter_step = IndexFilter(indices={str(file_path): "{1, 3, 5, 7}"})
        docs = create_test_documents(file_path, 10)

        result = list(filter_step.run(iter(docs), rank=0, world_size=1))
        assert len(result) == 4
        assert {doc.metadata["row_idx"] for doc in result} == {1, 3, 5, 7}

    def test_string_indices_type_conversion(self):
        filter_step = IndexFilter(indices={"test.parquet": "{0, 2, 4}"})

        assert isinstance(filter_step.indices["test.parquet"], set)
        assert filter_step.indices["test.parquet"] == {0, 2, 4}

    def test_invalid_string_indices_returns_empty_set(self, tmp_path):
        filter_step = IndexFilter(indices={"test.parquet": "not a valid set"})

        assert filter_step.indices["test.parquet"] == set()

    def test_matches_by_parquet_path(self, tmp_path):
        """测试 IndexFilter 使用 parquet_path 匹配索引。"""
        file_path = create_test_parquet(tmp_path, 10)
        # 使用完整路径创建索引
        filter_step = IndexFilter(indices={file_path: {1, 3, 5}})

        # 文档使用 parquet_path 和 row_idx
        docs = [
            Document(
                text=f"text_{i}",
                id=f"doc_{i}",
                metadata={"parquet_path": str(file_path), "row_idx": i},
            )
            for i in range(10)
        ]

        result = list(filter_step.run(iter(docs), rank=0, world_size=1))
        assert len(result) == 3
        assert {doc.metadata["row_idx"] for doc in result} == {1, 3, 5}

    def test_matches_different_paths_same_filename(self, tmp_path):
        """测试 IndexFilter 区分不同路径但文件名相同的情况。"""
        # 创建两个不同目录下的同名文件
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        file1 = create_test_parquet(dir1, 10)

        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        file2 = create_test_parquet(dir2, 10)

        # 只为 file1 创建索引
        filter_step = IndexFilter(indices={file1: {2, 4}})

        # file1 的文档应该通过
        docs1 = [
            Document(
                text=f"text_{i}",
                id=f"doc_{i}",
                metadata={"parquet_path": str(file1), "row_idx": i},
            )
            for i in range(10)
        ]

        # file2 的文档不应该通过
        docs2 = [
            Document(
                text=f"text_{i}",
                id=f"doc_{i}",
                metadata={"parquet_path": str(file2), "row_idx": i},
            )
            for i in range(10)
        ]

        result1 = list(filter_step.run(iter(docs1), rank=0, world_size=1))
        result2 = list(filter_step.run(iter(docs2), rank=0, world_size=1))

        assert len(result1) == 2
        assert len(result2) == 0


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

    def test_metadata_priority_over_data_columns(self):
        """测试 adapter 正确区分 file_path 和 parquet_path。

        对于 github_code 数据集：
        - file_path 保留原始代码文件路径（如 "README.md"），供 LanguageTagger 使用
        - parquet_path 存储 parquet 文件路径（如 "train_000.parquet"），供 IndexFilter 使用
        """
        adapter = create_row_index_adapter("content")
        parquet_path = "data/datasets/github-code/train_000.parquet"
        data = {
            "content": "def hello(): pass",
            "file_path": "README.md",  # 数据集自带的文件路径列
            "repo_id": "github/user/repo",
        }

        result = adapter(None, data, parquet_path, 42)

        # 关键验证：file_path 保留原始文件路径，parquet_path 存储 parquet 路径
        assert result["metadata"]["file_path"] == "README.md"
        assert result["metadata"]["parquet_path"] == parquet_path
        assert result["metadata"]["row_idx"] == 42
        assert result["metadata"]["repo_id"] == "github/user/repo"  # 其他列保留

    def test_row_idx_in_metadata(self):
        """测试 row_idx 被正确设置到 metadata 中。"""
        adapter = create_row_index_adapter("text")
        result = adapter(None, {"text": "test"}, "/path/to/file.parquet", 123)

        assert result["metadata"]["row_idx"] == 123
        assert result["metadata"]["parquet_path"] == "/path/to/file.parquet"


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
        doc = create_source_document(
            text="hello world",
            doc_id="doc_1",
            dataset_name="fineweb_edu_en",
            bucket_name="4.0",
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

    def test_multiple_batches_no_overwrite(self, tmp_path):
        output_dir = tmp_path / "output"
        writer = TokenizerDataWriter(
            output_dir=str(output_dir),
            dataset_name="test_dataset",
            bucket_name="test_bucket",
            max_rows_per_file=1000,
            buffer_size=3,
        )
        docs = [
            Document(
                text=f"text_{i}",
                id=f"doc_{i}",
                metadata={
                    "source_dataset": "test_dataset",
                    "source_bucket": "test_bucket",
                },
            )
            for i in range(10)
        ]
        writer.run(iter(docs), rank=0, world_size=1)

        files = sorted(output_dir.glob("*.parquet"))
        assert len(files) == 4

        total_rows = sum(pq.read_table(f).num_rows for f in files)
        assert total_rows == 10

        expected_names = [
            "test_dataset-test_bucket-00000-rank-00000.parquet",
            "test_dataset-test_bucket-00001-rank-00000.parquet",
            "test_dataset-test_bucket-00002-rank-00000.parquet",
            "test_dataset-test_bucket-00003-rank-00000.parquet",
        ]
        assert [f.name for f in files] == expected_names


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


class TestLanguageTagger:
    def test_tags_python_files(self):
        doc = Document(
            text="print('hello')",
            id="doc_1",
            metadata={"file_path": "/path/to/script.py"},
        )
        tagger = LanguageTagger(allowed_extensions=ALLOWED_LANGUAGES)
        result = list(tagger.run(iter([doc]), rank=0, world_size=1))
        assert len(result) == 1
        assert result[0].metadata["language"] == "python"

    def test_tags_cpp_files(self):
        doc = Document(
            text="int main() {}",
            id="doc_1",
            metadata={"file_path": "/path/to/main.cpp"},
        )
        tagger = LanguageTagger(allowed_extensions=ALLOWED_LANGUAGES)
        result = list(tagger.run(iter([doc]), rank=0, world_size=1))
        assert result[0].metadata["language"] == "cpp"

    def test_filters_unsupported_extensions(self):
        doc = Document(
            text="some content",
            id="doc_1",
            metadata={"file_path": "/path/to/file.unknown"},
        )
        tagger = LanguageTagger(allowed_extensions=ALLOWED_LANGUAGES)
        result = list(tagger.run(iter([doc]), rank=0, world_size=1))
        assert len(result) == 0

    def test_filters_empty_file_path(self):
        doc = Document(
            text="some content",
            id="doc_1",
            metadata={"file_path": ""},
        )
        tagger = LanguageTagger(allowed_extensions=ALLOWED_LANGUAGES)
        result = list(tagger.run(iter([doc]), rank=0, world_size=1))
        assert len(result) == 0

    def test_filters_no_extension(self):
        doc = Document(
            text="some content",
            id="doc_1",
            metadata={"file_path": "/path/to/Makefile"},
        )
        tagger = LanguageTagger(allowed_extensions=ALLOWED_LANGUAGES)
        result = list(tagger.run(iter([doc]), rank=0, world_size=1))
        assert len(result) == 0

    def test_tags_javascript_and_typescript(self):
        docs = [
            Document(text="js", id="1", metadata={"file_path": "app.js"}),
            Document(text="ts", id="2", metadata={"file_path": "app.ts"}),
            Document(text="jsx", id="3", metadata={"file_path": "App.jsx"}),
            Document(text="tsx", id="4", metadata={"file_path": "App.tsx"}),
        ]
        tagger = LanguageTagger(allowed_extensions=ALLOWED_LANGUAGES)
        result = list(tagger.run(iter(docs), rank=0, world_size=1))
        assert len(result) == 4
        assert result[0].metadata["language"] == "javascript"
        assert result[1].metadata["language"] == "typescript"
        assert result[2].metadata["language"] == "javascript"
        assert result[3].metadata["language"] == "typescript"

    def test_language_extensions_constant(self):
        assert ".py" in LANGUAGE_EXTENSIONS
        assert ".cpp" in LANGUAGE_EXTENSIONS
        assert ".js" in LANGUAGE_EXTENSIONS
        assert ".ts" in LANGUAGE_EXTENSIONS
        assert LANGUAGE_EXTENSIONS[".py"] == "python"
        assert LANGUAGE_EXTENSIONS[".cpp"] == "cpp"

    def test_allowed_languages_constant(self):
        assert ".py" in ALLOWED_LANGUAGES
        assert ".cpp" in ALLOWED_LANGUAGES
        assert ".unknown" not in ALLOWED_LANGUAGES

    def test_includes_language_column_when_enabled(self, tmp_path):
        output_dir = tmp_path / "output"
        writer = TokenizerDataWriter(
            output_dir=str(output_dir),
            dataset_name="github_code",
            bucket_name="test_bucket",
            max_rows_per_file=100,
            buffer_size=10,
            include_language=True,
        )
        doc = Document(
            text="def hello(): pass",
            id="doc_1",
            metadata={
                "source_dataset": "github_code",
                "source_bucket": "test_bucket",
                "language": "python",
            },
        )
        writer.run(iter([doc]), rank=0, world_size=1)
        files = list(output_dir.glob("*.parquet"))
        table = pq.read_table(files[0])
        assert "language" in table.column_names
        assert table["language"][0].as_py() == "python"

    def test_excludes_language_column_when_disabled(self, tmp_path):
        output_dir = tmp_path / "output"
        writer = TokenizerDataWriter(
            output_dir=str(output_dir),
            dataset_name="fineweb_edu_en",
            bucket_name="4.0",
            max_rows_per_file=100,
            buffer_size=10,
            include_language=False,
        )
        doc = create_source_document(
            text="hello world",
            doc_id="doc_1",
            dataset_name="fineweb_edu_en",
            bucket_name="4.0",
        )
        writer.run(iter([doc]), rank=0, world_size=1)
        files = list(output_dir.glob("*.parquet"))
        table = pq.read_table(files[0])
        assert "language" not in table.column_names

    def test_default_language_is_unknown(self, tmp_path):
        output_dir = tmp_path / "output"
        writer = TokenizerDataWriter(
            output_dir=str(output_dir),
            dataset_name="github_code",
            bucket_name="test_bucket",
            max_rows_per_file=100,
            buffer_size=10,
            include_language=True,
        )
        doc = Document(
            text="some text",
            id="doc_1",
            metadata={
                "source_dataset": "github_code",
                "source_bucket": "test_bucket",
            },
        )
        writer.run(iter([doc]), rank=0, world_size=1)
        files = list(output_dir.glob("*.parquet"))
        table = pq.read_table(files[0])
        assert table["language"][0].as_py() == "unknown"
