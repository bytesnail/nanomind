"""Tests for prepare_tokenizer_data module."""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.prepare_tokenizer_data import (
    SamplingConfig,
    SamplingInfo,
    StreamingParquetWriter,
    TokenizerDataConfig,
    compute_doc_hash,
    create_sample_doc,
    determine_text_column,
    get_file_row_count,
    load_config,
    save_sampling_info,
    select_top_k_document_hashes,
    stream_file_rows,
)


# Fixtures


@pytest.fixture
def sample_parquet_factory(tmp_path: Path):
    """创建测试 Parquet 文件的工厂。"""

    def _create(num_rows: int = 100, columns: dict[str, list] | None = None) -> Path:
        if columns is None:
            columns = {"text": [f"text_{i}" for i in range(num_rows)]}
        table = pa.table(columns)
        path = tmp_path / "test.parquet"
        pq.write_table(table, path)
        return path

    return _create


@pytest.fixture
def valid_config_dict() -> dict[str, Any]:
    """返回有效的配置字典。"""
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
def sample_config(valid_config_dict: dict[str, Any]) -> SamplingConfig:
    """返回示例 SamplingConfig。"""
    return SamplingConfig(
        name="test_dataset",
        source=Path("data/test"),
        samples=1000,
        buckets={"4.0": 500, "3.0": 500},
    )


# Tests


class TestComputeDocHash:
    """文档哈希计算函数的测试。"""

    def test_compute_doc_hash_determinism(self) -> None:
        """相同输入应该产生相同哈希值（确定性）。"""
        hash1 = compute_doc_hash("doc_123", seed=42)
        hash2 = compute_doc_hash("doc_123", seed=42)
        assert hash1 == hash2

    def test_compute_doc_hash_different_seeds(self) -> None:
        """不同种子应该产生不同哈希值。"""
        hash1 = compute_doc_hash("doc_123", seed=42)
        hash2 = compute_doc_hash("doc_123", seed=24)
        assert hash1 != hash2

    def test_compute_doc_hash_different_docs(self) -> None:
        """不同文档应该产生不同哈希值（大概率）。"""
        hash1 = compute_doc_hash("doc_123", seed=42)
        hash2 = compute_doc_hash("doc_456", seed=42)
        assert hash1 != hash2

    def test_compute_doc_hash_returns_valid_int(self) -> None:
        """应该返回有效的64位无符号整数。"""
        result = compute_doc_hash("doc_123", seed=42)
        assert isinstance(result, int)
        assert result >= 0
        assert result < 2**64


class TestDetermineTextColumn:
    """文本字段名检测的测试。"""

    @pytest.mark.parametrize(
        "dataset_name,expected",
        [
            ("github_code", "content"),
            ("GitHub-Code-2025", "content"),
            ("github-code", "content"),
            ("GITHUB_CODE", "content"),
            ("my_github_repo", "content"),
        ],
    )
    def test_github_code_uses_content(self, dataset_name: str, expected: str) -> None:
        """GitHub Code 数据集应该使用 'content' 字段。"""
        assert determine_text_column(dataset_name) == expected

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "fineweb_edu_en",
            "fineweb_edu_zh",
            "FineWeb-Edu",
            "nemotron_cc_math",
            "other_dataset",
        ],
    )
    def test_other_datasets_use_text(self, dataset_name: str) -> None:
        """非 GitHub 数据集应该使用 'text' 字段。"""
        assert determine_text_column(dataset_name) == "text"


def _write_yaml_config(tmp_path: Path, config_data: dict[str, Any]) -> Path:
    """Helper: 写入 YAML 配置到临时文件。"""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


class TestLoadConfig:
    """配置加载的测试。"""

    def test_load_valid_config(
        self, tmp_path: Path, valid_config_dict: dict[str, Any]
    ) -> None:
        """测试加载有效配置文件。"""
        config_path = _write_yaml_config(tmp_path, valid_config_dict)
        config = load_config(config_path)

        assert isinstance(config, TokenizerDataConfig)
        assert config.random_seed == 42
        assert config.output_format == "parquet"
        assert len(config.datasets) == 1

        fw_config = config.datasets["fineweb_en"]
        assert isinstance(fw_config, SamplingConfig)
        assert fw_config.name == "fineweb_edu_en"
        assert fw_config.samples == 1000
        assert fw_config.buckets == {"4.0": 500, "3.0": 500}

    def test_load_config_with_stars_filter(self, tmp_path: Path) -> None:
        """测试加载带 stars_filter 的配置。"""
        config_data = {
            "datasets": {
                "github_code": {
                    "name": "github_code",
                    "source": "data/github",
                    "samples": 1000,
                    "stars_filter": {
                        "above_2": {"count": 800},
                        "below_2": {"count": 200},
                    },
                }
            },
            "random_seed": 123,
            "output_format": "parquet",
            "output_dir": "data/output",
        }

        config_path = _write_yaml_config(tmp_path, config_data)
        config = load_config(config_path)

        gh_config = config.datasets["github_code"]
        assert gh_config.stars_filter == {"above_2": 800, "below_2": 200}
        assert gh_config.get_all_counts() == {"above_2": 800, "below_2": 200}

    def test_load_config_file_not_found(self, tmp_path: Path) -> None:
        """测试加载不存在的配置文件。"""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestGetFileRowCount:
    """文件行数统计的测试。"""

    def test_get_row_count_from_metadata(self, sample_parquet_factory) -> None:
        """测试从元数据获取行数。"""
        file_path = sample_parquet_factory(num_rows=3)
        count = get_file_row_count(file_path)
        assert count == 3

    def test_get_row_count_large_file(self, sample_parquet_factory) -> None:
        """测试较大文件的行数统计。"""
        file_path = sample_parquet_factory(num_rows=10000)
        count = get_file_row_count(file_path)
        assert count == 10000


class TestSamplingConfig:
    """SamplingConfig 数据类的测试。"""

    def test_get_all_counts_with_buckets(self, sample_config: SamplingConfig) -> None:
        """测试获取 buckets 的计数。"""
        assert sample_config.get_all_counts() == {"4.0": 500, "3.0": 500}

    def test_get_all_counts_with_stars_filter(self) -> None:
        """测试获取 stars_filter 的计数。"""
        config = SamplingConfig(
            name="test",
            source=Path("data/test"),
            samples=1000,
            stars_filter={"above_2": 800, "below_2": 200},
        )
        assert config.get_all_counts() == {"above_2": 800, "below_2": 200}

    def test_get_all_counts_empty(self) -> None:
        """测试空配置。"""
        config = SamplingConfig(
            name="test",
            source=Path("data/test"),
            samples=1000,
        )
        assert config.get_all_counts() == {}


class TestCreateSampleDoc:
    """创建样本文档函数的测试。"""

    def test_create_sample_doc_structure(self) -> None:
        """测试返回的文档结构正确。"""
        doc = create_sample_doc("test text", "fineweb", "4.0")
        assert doc["text"] == "test text"
        assert doc["source_dataset"] == "fineweb"
        assert doc["source_bucket"] == "4.0"

    def test_create_sample_doc_empty_text(self) -> None:
        """测试空文本的情况。"""
        doc = create_sample_doc("", "github", "main")
        assert doc["text"] == ""
        assert doc["source_dataset"] == "github"
        assert doc["source_bucket"] == "main"


class TestStreamingParquetWriter:
    """StreamingParquetWriter 的测试。"""

    def test_writer_creates_output_directory(self, tmp_path: Path) -> None:
        """测试写入器创建输出目录。"""
        output_dir = tmp_path / "output"
        with StreamingParquetWriter(output_dir, "test", 100):
            pass
        assert output_dir.exists()

    def test_writer_creates_single_file(self, tmp_path: Path) -> None:
        """测试写入单个文件。"""
        output_dir = tmp_path / "output"
        with StreamingParquetWriter(output_dir, "train", 1000) as writer:
            writer.write(create_sample_doc("text1", "ds", "bucket"))
            writer.write(create_sample_doc("text2", "ds", "bucket"))

        files = list(output_dir.glob("*.parquet"))
        assert len(files) == 1
        assert files[0].name == "train-00000-of-00001.parquet"

    def test_writer_creates_multiple_files(self, tmp_path: Path) -> None:
        """测试根据 max_rows 创建多个文件。"""
        output_dir = tmp_path / "output"
        with StreamingParquetWriter(output_dir, "train", 2) as writer:
            for i in range(5):
                writer.write(create_sample_doc(f"text{i}", "ds", "bucket"))

        files = sorted(output_dir.glob("*.parquet"))
        assert len(files) == 3
        assert [f.name for f in files] == [
            "train-00000-of-00003.parquet",
            "train-00001-of-00003.parquet",
            "train-00002-of-00003.parquet",
        ]

    def test_writer_content_has_correct_schema(self, tmp_path: Path) -> None:
        """测试写入的文件具有正确的 schema。"""
        output_dir = tmp_path / "output"
        with StreamingParquetWriter(output_dir, "train", 100) as writer:
            writer.write(create_sample_doc("hello", "fineweb", "4.0"))

        files = list(output_dir.glob("*.parquet"))
        table = pq.read_table(files[0])
        assert "text" in table.column_names
        assert "source_dataset" in table.column_names
        assert "source_bucket" in table.column_names
        assert table.num_rows == 1
        assert table.column("text").to_pylist() == ["hello"]

    def test_writer_context_manager_closes_properly(self, tmp_path: Path) -> None:
        """测试上下文管理器正确关闭。"""
        output_dir = tmp_path / "output"
        with StreamingParquetWriter(output_dir, "test", 10) as writer:
            writer.write(create_sample_doc("text", "ds", "bucket"))

        # 上下文退出后文件应该已经写入
        files = list(output_dir.glob("*.parquet"))
        assert len(files) == 1


class TestStreamFileRows:
    """流式文件读取函数的测试。"""

    def test_stream_file_rows_basic(self, tmp_path: Path) -> None:
        """测试基本流式读取功能。"""
        # 创建测试文件
        columns = {"text": [f"text_{i}" for i in range(100)]}
        table = pa.table(columns)
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        # 流式读取
        rows = list(stream_file_rows(file_path, "text", batch_size=10))

        assert len(rows) == 100
        for i, (idx, text) in enumerate(rows):
            assert idx == i
            assert text == f"text_{i}"

    def test_stream_file_rows_empty_file(self, tmp_path: Path) -> None:
        """测试空文件的流式读取。"""
        columns = {"text": []}
        table = pa.table(columns)
        file_path = tmp_path / "empty.parquet"
        pq.write_table(table, file_path)

        rows = list(stream_file_rows(file_path, "text"))
        assert len(rows) == 0

    def test_stream_file_rows_large_batch_size(self, tmp_path: Path) -> None:
        """测试批次大小大于文件大小时的情况。"""
        columns = {"text": [f"text_{i}" for i in range(5)]}
        table = pa.table(columns)
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        rows = list(stream_file_rows(file_path, "text", batch_size=1000))
        assert len(rows) == 5


class TestStreamFileRowsIndices:
    """带索引筛选的流式读取函数测试。"""

    def test_stream_with_indices_basic(self, tmp_path: Path) -> None:
        """测试基本索引筛选功能。"""
        columns = {"text": [f"text_{i}" for i in range(10)]}
        table = pa.table(columns)
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        indices = {1, 3, 5, 7}
        rows = list(stream_file_rows(file_path, "text", batch_size=3, indices=indices))

        assert len(rows) == 4
        assert {idx for idx, _ in rows} == indices

    def test_stream_with_indices_empty_indices(self, tmp_path: Path) -> None:
        """测试空索引集合的情况。"""
        columns = {"text": [f"text_{i}" for i in range(10)]}
        table = pa.table(columns)
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        rows = list(stream_file_rows(file_path, "text", indices=set()))
        assert len(rows) == 0

    def test_stream_with_indices_out_of_range(self, tmp_path: Path) -> None:
        """测试索引超出范围的情况（应忽略）。"""
        columns = {"text": [f"text_{i}" for i in range(5)]}
        table = pa.table(columns)
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        indices = {1, 100}  # 100 超出范围
        rows = list(stream_file_rows(file_path, "text", indices=indices))

        assert len(rows) == 1
        assert rows[0][0] == 1


class TestSelectTopKDocumentHashes:
    """Top-K 文档哈希选择函数测试。"""

    def test_select_top_k_basic(self, tmp_path: Path) -> None:
        """测试基本 Top-K 选择功能。"""
        # 创建测试文件
        for i in range(3):
            columns = {"text": [f"file{i}_text{j}" for j in range(10)]}
            table = pa.table(columns)
            pq.write_table(table, tmp_path / f"file_{i}.parquet")

        files = sorted(tmp_path.glob("*.parquet"))
        result = select_top_k_document_hashes(
            files, "test_bucket", seed=42, target_count=5
        )

        assert isinstance(result, dict)
        assert len(result) <= 3  # 最多 3 个文件
        total_indices = sum(len(indices) for indices in result.values())
        assert total_indices == 5

    def test_select_top_k_target_exceeds_total(self, tmp_path: Path) -> None:
        """测试目标数超过总数的情况。"""
        columns = {"text": [f"text_{i}" for i in range(5)]}
        table = pa.table(columns)
        pq.write_table(table, tmp_path / "small.parquet")

        files = [tmp_path / "small.parquet"]
        result = select_top_k_document_hashes(
            files, "test_bucket", seed=42, target_count=100
        )

        total_indices = sum(len(indices) for indices in result.values())
        assert total_indices == 5  # 应返回全部

    def test_select_top_k_empty_files(self, tmp_path: Path) -> None:
        """测试空文件列表的情况。"""
        result = select_top_k_document_hashes(
            [], "test_bucket", seed=42, target_count=10
        )
        assert result == {}

    def test_select_top_k_determinism(self, tmp_path: Path) -> None:
        """测试 Top-K 选择的确定性。"""
        columns = {"text": [f"text_{i}" for i in range(20)]}
        table = pa.table(columns)
        pq.write_table(table, tmp_path / "test.parquet")

        files = [tmp_path / "test.parquet"]
        result1 = select_top_k_document_hashes(
            files, "test_bucket", seed=42, target_count=5
        )
        result2 = select_top_k_document_hashes(
            files, "test_bucket", seed=42, target_count=5
        )

        assert result1 == result2


class TestSaveSamplingInfo:
    """保存采样信息函数的测试。"""

    def test_save_sampling_info_creates_file(self, tmp_path: Path) -> None:
        """测试保存采样信息创建 JSON 文件。"""
        info = SamplingInfo(
            total_requested=1000,
            total_sampled=950,
            sources={
                "dataset1": {"requested": 500, "sampled": 480},
            },
            random_seed=42,
        )

        output_path = save_sampling_info(info, tmp_path)

        assert output_path.exists()
        assert output_path.name == "sampling_info.json"

        # 验证内容
        with open(output_path) as f:
            data = json.load(f)
        assert data["total_requested"] == 1000
        assert data["total_sampled"] == 950
        assert data["random_seed"] == 42
        assert "sources" in data
