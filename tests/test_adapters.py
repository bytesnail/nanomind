import pytest

from src.data_processing.adapters import fineweb_adapter, normalize_score


@pytest.fixture
def reader():
    return object()


@pytest.fixture
def valid_raw():
    return {"text": "Test document", "score": 3.5, "dump": "CC-MAIN-2024-10"}


class TestScoreNormalization:
    def test_no_normalization(self):
        assert normalize_score(3.5, None) == 3.5
        assert normalize_score(3.5, {"enabled": False}) == 3.5

    def test_with_normalization(self):
        config = {"enabled": True, "multiplier": 5.0}
        assert normalize_score(0.6, config) == 3.0
        assert normalize_score(0.8, config) == 4.0
        assert normalize_score(0.4, config) == 2.0

    def test_default_multiplier(self):
        config = {"enabled": True}
        assert normalize_score(0.5, config) == 0.5

    def test_none_score(self):
        assert normalize_score(None, None) == 0.0


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
                "data/datasets/opencsg/Fineweb-Edu-Chinese-V2.1/data/train.parquet",
                0,
                "data/train.parquet#0",
            ),
        ],
    )
    def test_id_generation(self, reader, valid_raw, source, idx, expected):
        result = fineweb_adapter(reader, valid_raw, source, idx)
        assert result is not None
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
        assert result is not None
        assert result["text"] == "This is a test document."
        assert result["metadata"]["score"] == 3.5
        assert result["metadata"]["cc_main"] == "CC-MAIN-2024-10"
        assert "url" not in result["metadata"]

    def test_default_score(self, reader):
        result = fineweb_adapter(
            reader, {"text": "Test", "dump": "CC-MAIN-2024-10"}, "test.parquet", 0
        )
        assert result is not None
        assert result["metadata"]["score"] == 0.0

    def test_unknown_cc_main_for_invalid_dump(self, reader):
        result = fineweb_adapter(
            reader,
            {"text": "Test", "score": 3.5, "dump": "invalid-format"},
            "test.parquet",
            0,
        )
        assert result is not None
        assert result["metadata"]["cc_main"] == "unknown"


class TestEdgeCases:
    def test_returns_none_for_missing_text(self, reader):
        assert fineweb_adapter(reader, {"score": 3.5}, "test.parquet", 0) is None

    def test_returns_none_for_empty_text(self, reader):
        assert (
            fineweb_adapter(reader, {"text": "", "score": 3.5}, "test.parquet", 0)
            is None
        )
