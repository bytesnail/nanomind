"""测试适配器。"""

import pytest

from src.data_processing.adapters import fineweb_adapter


class MockReader:
    pass


class TestFinewebAdapter:
    def test_valid_input(self):
        mock = MockReader()
        raw = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
            "url": "https://example.com",
        }
        result = fineweb_adapter(mock, raw, "test.parquet", 0)
        assert result is not None
        assert result["text"] == "This is a test document."
        assert result["id"] == "test-id-123"
        assert result["metadata"]["score"] == 3.5
        assert result["metadata"]["cc_main"] == "CC-MAIN-2024-10"

    def test_missing_text(self):
        mock = MockReader()
        raw = {"id": "test-id-123", "score": 3.5}
        with pytest.raises(ValueError):
            fineweb_adapter(mock, raw, "test.parquet", 0, raise_on_error=True)

    def test_missing_id(self):
        mock = MockReader()
        raw = {"text": "This is a test document.", "score": 3.5}
        with pytest.raises(ValueError):
            fineweb_adapter(mock, raw, "test.parquet", 0, raise_on_error=True)

    def test_empty_text(self):
        mock = MockReader()
        raw = {"text": "", "id": "test-id-123", "score": 3.5}
        with pytest.raises(ValueError):
            fineweb_adapter(mock, raw, "test.parquet", 0, raise_on_error=True)

    def test_missing_score(self):
        mock = MockReader()
        raw = {"text": "Test", "id": "test-id", "dump": "CC-MAIN-2024-10"}
        result = fineweb_adapter(mock, raw, "test.parquet", 0)
        assert result is not None
        assert result["metadata"]["score"] == 0.0

    def test_invalid_dump_format(self):
        mock = MockReader()
        raw = {"text": "Test", "id": "test-id", "score": 3.5, "dump": "invalid"}
        result = fineweb_adapter(mock, raw, "test.parquet", 0)
        assert result is not None
        assert result["metadata"]["cc_main"] == "unknown"

    def test_extra_fields_filtered(self):
        mock = MockReader()
        raw = {
            "text": "Test",
            "id": "test-id",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
            "url": "https://example.com",
        }
        result = fineweb_adapter(mock, raw, "test.parquet", 0)
        assert result is not None
        assert "url" not in result["metadata"]
        assert "score" in result["metadata"]
        assert "cc_main" in result["metadata"]


class TestFinewebAdapterSafeMode:
    """测试默认安全模式（raise_on_error=False）。"""

    def test_valid_input(self):
        mock = MockReader()
        raw = {"text": "Test", "id": "test-id", "score": 3.5, "dump": "CC-MAIN-2024-10"}
        result = fineweb_adapter(mock, raw, "test.parquet", 0)
        assert result is not None
        assert result["text"] == "Test"

    def test_invalid_input_returns_none(self):
        mock = MockReader()
        raw = {"id": "test-id", "score": 3.5}
        assert fineweb_adapter(mock, raw, "test.parquet", 0) is None

    def test_empty_text_returns_none(self):
        mock = MockReader()
        raw = {"text": "", "id": "test-id", "score": 3.5}
        assert fineweb_adapter(mock, raw, "test.parquet", 0) is None
