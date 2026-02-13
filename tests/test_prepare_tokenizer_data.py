"""Tests for prepare_tokenizer_data module."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.prepare_tokenizer_data import (
    SamplingConfig,
    TokenizerDataConfig,
    deterministic_sample,
    determine_text_column,
    get_file_row_count,
    load_config,
)


class TestDeterministicSample:
    """确定性采样函数的测试。"""

    def test_deterministic_sample_always_true_when_target_equals_total(self):
        """当目标数等于总数时，应该总是返回 True。"""
        assert deterministic_sample("doc1", 10, 10, seed=42) is True
        assert deterministic_sample("doc2", 100, 100, seed=123) is True

    def test_deterministic_sample_always_true_when_target_exceeds_total(self):
        """当目标数超过总数时，应该总是返回 True。"""
        assert deterministic_sample("doc1", 100, 10, seed=42) is True

    def test_deterministic_sample_determinism(self):
        """相同输入应该产生相同结果（确定性）。"""
        results = [deterministic_sample("doc_123", 50, 100, seed=42) for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_deterministic_sample_different_seeds(self):
        """不同种子应该产生不同结果。"""
        # 虽然单一样本可能相同，但概率极低
        # 使用多个样本进行统计检验
        count_42 = sum(
            deterministic_sample(f"doc_{i}", 50, 100, seed=42) for i in range(1000)
        )
        count_24 = sum(
            deterministic_sample(f"doc_{i}", 50, 100, seed=24) for i in range(1000)
        )
        # 两者都应该接近 500，但具体值可能不同
        assert 400 < count_42 < 600
        assert 400 < count_24 < 600

    def test_deterministic_sample_rate_accuracy(self):
        """采样率应该接近目标比例。"""
        target_rate = 0.3
        total = 10000
        target = int(total * target_rate)

        sampled_count = sum(
            deterministic_sample(f"doc_{i}", target, total, seed=42)
            for i in range(total)
        )

        actual_rate = sampled_count / total
        # 允许 5% 误差
        assert abs(actual_rate - target_rate) < 0.05


class TestDetermineTextColumn:
    """文本字段名检测的测试。"""

    def test_github_code_uses_content(self):
        """GitHub Code 数据集应该使用 'content' 字段。"""
        assert determine_text_column("github_code") == "content"
        assert determine_text_column("GitHub-Code-2025") == "content"
        assert determine_text_column("github-code") == "content"

    def test_fineweb_uses_text(self):
        """FineWeb 数据集应该使用 'text' 字段。"""
        assert determine_text_column("fineweb_edu_en") == "text"
        assert determine_text_column("fineweb_edu_zh") == "text"
        assert determine_text_column("FineWeb-Edu") == "text"

    def test_nemotron_uses_text(self):
        """Nemotron 数据集应该使用 'text' 字段。"""
        assert determine_text_column("nemotron_cc_math") == "text"


class TestLoadConfig:
    """配置加载的测试。"""

    def test_load_valid_config(self, tmp_path: Path):
        """测试加载有效配置文件。"""
        config_data = {
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

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

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

    def test_load_config_with_stars_filter(self, tmp_path: Path):
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

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)

        gh_config = config.datasets["github_code"]
        assert gh_config.stars_filter == {"above_2": 800, "below_2": 200}
        assert gh_config.get_all_counts() == {"above_2": 800, "below_2": 200}

    def test_load_config_file_not_found(self, tmp_path: Path):
        """测试加载不存在的配置文件。"""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestGetFileRowCount:
    """文件行数统计的测试。"""

    def test_get_row_count_from_metadata(self, tmp_path: Path):
        """测试从元数据获取行数。"""
        # 创建测试 Parquet 文件
        table = pa.table(
            {
                "text": ["line1", "line2", "line3"],
                "score": [1.0, 2.0, 3.0],
            }
        )

        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        count = get_file_row_count(file_path)
        assert count == 3

    def test_get_row_count_large_file(self, tmp_path: Path):
        """测试较大文件的行数统计。"""
        num_rows = 10000
        table = pa.table(
            {
                "text": [f"line_{i}" for i in range(num_rows)],
                "score": [float(i) for i in range(num_rows)],
            }
        )

        file_path = tmp_path / "large.parquet"
        pq.write_table(table, file_path)

        count = get_file_row_count(file_path)
        assert count == num_rows


class TestSamplingConfig:
    """SamplingConfig 数据类的测试。"""

    def test_get_all_counts_with_buckets(self):
        """测试获取 buckets 的计数。"""
        config = SamplingConfig(
            name="test",
            source=Path("data/test"),
            samples=1000,
            buckets={"4.0": 500, "3.0": 500},
        )
        assert config.get_all_counts() == {"4.0": 500, "3.0": 500}

    def test_get_all_counts_with_stars_filter(self):
        """测试获取 stars_filter 的计数。"""
        config = SamplingConfig(
            name="test",
            source=Path("data/test"),
            samples=1000,
            stars_filter={"above_2": 800, "below_2": 200},
        )
        assert config.get_all_counts() == {"above_2": 800, "below_2": 200}

    def test_get_all_counts_empty(self):
        """测试空配置。"""
        config = SamplingConfig(
            name="test",
            source=Path("data/test"),
            samples=1000,
        )
        assert config.get_all_counts() == {}


class TestIntegration:
    """集成测试。"""

    def test_full_sampling_workflow(self, tmp_path: Path):
        """测试完整采样流程（简化版）。"""
        # 创建模拟数据目录
        source_dir = tmp_path / "source" / "4.0"
        source_dir.mkdir(parents=True)

        # 创建测试 Parquet 文件
        table = pa.table(
            {
                "text": [f"text_{i}" for i in range(1000)],
                "score": [4.0] * 1000,
            }
        )
        pq.write_table(table, source_dir / "00000.parquet")

        # 验证文件存在且可读
        assert (source_dir / "00000.parquet").exists()
        assert get_file_row_count(source_dir / "00000.parquet") == 1000

    def test_config_matches_yaml_structure(self):
        """测试配置类与 YAML 结构匹配。"""
        # 验证 TokenizerDataConfig 可以容纳配置文件的完整结构
        config = TokenizerDataConfig(
            datasets={
                "fineweb_en": SamplingConfig(
                    name="fineweb_edu_en",
                    source=Path("data/fineweb/en"),
                    samples=12_000_000,
                    buckets={
                        "4.0": 5_400_000,
                        "3.5": 2_400_000,
                        "3.0": 2_400_000,
                        "2.5": 1_800_000,
                    },
                ),
                "github_code": SamplingConfig(
                    name="github_code",
                    source=Path("data/github"),
                    samples=12_000_000,
                    stars_filter={"above_2": 10_000_000, "below_2": 2_000_000},
                ),
            },
            random_seed=42,
            output_format="parquet",
            output_dir=Path("data/output"),
        )

        # 验证总样本数
        total = sum(d.samples for d in config.datasets.values())
        assert total == 24_000_000

        # 验证各数据集
        assert config.datasets["fineweb_en"].buckets["4.0"] == 5_400_000
        assert config.datasets["github_code"].stars_filter["above_2"] == 10_000_000
