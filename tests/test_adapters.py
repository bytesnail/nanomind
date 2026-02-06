"""测试适配器模块。"""

import pytest

from src.data_processing.adapters import fineweb_adapter, fineweb_adapter_safe


class MockReader:
    pass


class TestFinewebAdapter:
    def test_valid_input(self):
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
            "url": "http://example.com",
        }

        result = fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

        assert result["text"] == "This is a test document."
        assert result["id"] == "test-id-123"
        assert result["metadata"]["score"] == 3.5
        assert result["metadata"]["cc_main"] == "CC-MAIN-2024-10"

    def test_missing_text(self):
        mock_reader = MockReader()
        raw_dict = {"id": "test-id-123", "score": 3.5}

        with pytest.raises(ValueError):
            fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

    def test_missing_id(self):
        mock_reader = MockReader()
        raw_dict = {"text": "This is a test document.", "score": 3.5}

        with pytest.raises(ValueError):
            fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

    def test_empty_text(self):
        mock_reader = MockReader()
        raw_dict = {"text": "", "id": "test-id-123", "score": 3.5}

        with pytest.raises(ValueError):
            fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

    def test_missing_score(self):
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "dump": "CC-MAIN-2024-10",
        }

        result = fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)
        assert result["metadata"]["score"] == 0.0

    def test_invalid_dump_format(self):
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "score": 3.5,
            "dump": "invalid-format",
        }

        result = fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)
        assert result["metadata"]["cc_main"] == "unknown"

    def test_extra_fields_filtered(self):
        mock_reader = MockReader()
        raw_dict = {
            "text": "Test",
            "id": "test-id",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
            "url": "http://example.com",
            "token_count": 100,
        }

        result = fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

        assert "url" not in result["metadata"]
        assert "token_count" not in result["metadata"]
        assert "score" in result["metadata"]
        assert "cc_main" in result["metadata"]


class TestFinewebAdapterSafe:
    def test_valid_input(self):
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
        }

        result = fineweb_adapter_safe(mock_reader, raw_dict, "test.parquet", 0)
        assert result is not None
        assert result["text"] == "This is a test document."

    def test_invalid_input_returns_none(self):
        mock_reader = MockReader()
        raw_dict = {"id": "test-id-123", "score": 3.5}

        result = fineweb_adapter_safe(mock_reader, raw_dict, "test.parquet", 0)
        assert result is None

    def test_empty_text_returns_none(self):
        mock_reader = MockReader()
        raw_dict = {"text": "", "id": "test-id-123", "score": 3.5}

        result = fineweb_adapter_safe(mock_reader, raw_dict, "test.parquet", 0)
        assert result is None
