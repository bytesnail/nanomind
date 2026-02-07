"""测试适配器。"""

import pytest

from src.data_processing.adapters import fineweb_adapter


@pytest.fixture
def reader():
    return object()


@pytest.fixture
def valid_raw():
    return {"text": "Test document", "score": 3.5, "dump": "CC-MAIN-2024-10"}


class TestIdGeneration:
    @pytest.mark.parametrize(
        "source,idx,expected",
        [
            ("train.parquet", 0, "train.parquet#0"),
            ("train.parquet", 42, "train.parquet#42"),
            (
                "data/CC-MAIN-2013-20/train.parquet",
                0,
                "data/CC-MAIN-2013-20/train.parquet#0",
            ),
            (
                "data/datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2013-20/train.parquet",
                1,
                "data/CC-MAIN-2013-20/train.parquet#1",
            ),
            (
                "data/other-dataset/train.parquet",
                0,
                "data/other-dataset/train.parquet#0",
            ),
        ],
    )
    def test_id_generation(self, reader, valid_raw, source, idx, expected):
        result = fineweb_adapter(reader, valid_raw, source, idx)
        assert result["id"] == expected


class TestDataExtraction:
    def test_full_extraction(self, reader):
        raw = {
            "text": "This is a test document.",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
            "url": "https://example.com",
        }
        result = fineweb_adapter(reader, raw, "test.parquet", 0)
        assert result["text"] == "This is a test document."
        assert result["metadata"]["score"] == 3.5
        assert result["metadata"]["cc_main"] == "CC-MAIN-2024-10"
        assert "url" not in result["metadata"]

    def test_default_score(self, reader):
        result = fineweb_adapter(
            reader, {"text": "Test", "dump": "CC-MAIN-2024-10"}, "test.parquet", 0
        )
        assert result["metadata"]["score"] == 0.0

    def test_unknown_cc_main_for_invalid_dump(self, reader):
        result = fineweb_adapter(
            reader,
            {"text": "Test", "score": 3.5, "dump": "invalid-format"},
            "test.parquet",
            0,
        )
        assert result["metadata"]["cc_main"] == "unknown"


class TestErrorHandling:
    @pytest.mark.parametrize(
        "invalid_data",
        [
            {"id": "test-id", "score": 3.5},
            {"text": "", "id": "test-id", "score": 3.5},
        ],
    )
    def test_missing_or_empty_text_raises(self, reader, invalid_data):
        with pytest.raises(ValueError, match="text is missing or empty"):
            fineweb_adapter(
                reader, invalid_data, "test.parquet", 0, raise_on_error=True
            )

    def test_returns_none_by_default(self, reader):
        assert fineweb_adapter(reader, {"score": 3.5}, "test.parquet", 0) is None
        assert (
            fineweb_adapter(reader, {"text": "", "score": 3.5}, "test.parquet", 0)
            is None
        )


class TestInputValidation:
    def test_negative_idx_raises(self, reader, valid_raw):
        with pytest.raises(ValueError, match="idx must be non-negative"):
            fineweb_adapter(reader, valid_raw, "test.parquet", -1)

    def test_empty_source_path_raises(self, reader, valid_raw):
        with pytest.raises(ValueError, match="source_path cannot be empty"):
            fineweb_adapter(reader, valid_raw, "", 0)
