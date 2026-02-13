"""Tests for prepare_tokenizer_data module."""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.prepare_tokenizer_data import (
    DocHash,
    SampleDoc,
    SamplingConfig,
    StreamingParquetWriter,
    TokenizerDataConfig,
    compute_doc_hash,
    create_sample_doc,
    determine_text_column,
    deterministic_sample,
    get_file_row_count,
    load_config,
    save_sampling_info,
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


class TestDeterministicSample:
    """确定性采样函数的测试。"""

    def test_deterministic_sample_always_true_when_target_equals_total(self) -> None:
        """当目标数等于总数时，应该总是返回 True。"""
        doc_hash = compute_doc_hash("doc1", seed=42)
        assert deterministic_sample(doc_hash, 10, 10) is True
        doc_hash2 = compute_doc_hash("doc2", seed=123)
        assert deterministic_sample(doc_hash2, 100, 100) is True

    def test_deterministic_sample_always_true_when_target_exceeds_total(self) -> None:
        """当目标数超过总数时，应该总是返回 True。"""
        doc_hash = compute_doc_hash("doc1", seed=42)
        assert deterministic_sample(doc_hash, 100, 10) is True

    def test_deterministic_sample_determinism(self) -> None:
        """相同输入应该产生相同结果（确定性）。"""
        doc_hash = compute_doc_hash("doc_123", seed=42)
        results = [deterministic_sample(doc_hash, 50, 100) for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_deterministic_sample_rate_accuracy(self) -> None:
        """采样率应该接近目标比例（允许5%误差）。"""
        target_rate = 0.3
        total = 10000
        target = int(total * target_rate)

        sampled_count = sum(
            deterministic_sample(compute_doc_hash(f"doc_{i}", seed=42), target, total)
            for i in range(total)
        )

        actual_rate = sampled_count / total
        assert abs(actual_rate - target_rate) < 0.05


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


def _assert_dict_fields(
    data: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Helper: 断言字典字段值与期望值匹配。"""
    for key, value in expected.items():
        assert data[key] == value, f"Field '{key}' mismatch: {data[key]} != {value}"


class TestCreateSampleDoc:
    """创建样本文档函数的测试。"""

    def test_create_sample_doc_structure(self) -> None:
        """测试返回的文档结构正确。"""
        doc: SampleDoc = create_sample_doc("test text", "fineweb", "4.0")
        _assert_dict_fields(
            doc,
            {"text": "test text", "source_dataset": "fineweb", "source_bucket": "4.0"},
        )

    def test_create_sample_doc_empty_text(self) -> None:
        """测试空文本的情况。"""
        doc: SampleDoc = create_sample_doc("", "github", "main")
        _assert_dict_fields(
            doc, {"text": "", "source_dataset": "github", "source_bucket": "main"}
        )


class TestDocHash:
    """DocHash 命名元组的测试。"""

    def test_dochash_creation(self) -> None:
        """测试 DocHash 可以正确创建。"""
        dh = DocHash(
            hash_value=12345,
            doc_id="test_doc",
            file_path=Path("/tmp/test.parquet"),
            row_index=42,
        )
        assert dh.hash_value == 12345
        assert dh.doc_id == "test_doc"
        assert dh.file_path == Path("/tmp/test.parquet")
        assert dh.row_index == 42

    def test_dochash_unpacking(self) -> None:
        """测试 DocHash 可以解包。"""
        dh = DocHash(1, "doc", Path("file"), 0)
        h, d, f, r = dh
        assert h == 1
        assert d == "doc"
        assert f == Path("file")
        assert r == 0


def _assert_parquet_schema(table: pa.Table, expected_rows: int = 1) -> None:
    """Helper: 断言 Parquet 表结构正确。"""
    assert "text" in table.column_names
    assert "source_dataset" in table.column_names
    assert "source_bucket" in table.column_names
    assert table.num_rows == expected_rows


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
        _assert_parquet_schema(table, expected_rows=1)
        assert table.column("text").to_pylist() == ["hello"]

    def test_writer_context_manager_closes_properly(self, tmp_path: Path) -> None:
        """测试上下文管理器正确关闭。"""
        output_dir = tmp_path / "output"
        with StreamingParquetWriter(output_dir, "test", 10) as writer:
            writer.write(create_sample_doc("text", "ds", "bucket"))

        # 上下文退出后文件应该已经写入
        files = list(output_dir.glob("*.parquet"))
        assert len(files) == 1


class TestSaveSamplingInfo:
    """保存采样信息函数的测试。"""

    def test_save_sampling_info_creates_file(self, tmp_path: Path) -> None:
        """测试保存采样信息创建 JSON 文件。"""
        from scripts.prepare_tokenizer_data import SamplingInfo

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


class TestSampleDoc:
    """SampleDoc 字典子类的测试。"""

    def test_sampledoc_is_dict(self) -> None:
        """测试 SampleDoc 是 dict 的子类。"""
        doc = SampleDoc(text="hello", source_dataset="ds", source_bucket="4.0")
        assert isinstance(doc, dict)
        assert doc["text"] == "hello"

    def test_sampledoc_from_create_function(self) -> None:
        """测试 create_sample_doc 返回 SampleDoc。"""
        doc = create_sample_doc("text", "dataset", "bucket")
        assert isinstance(doc, SampleDoc)
        assert doc["source_dataset"] == "dataset"
