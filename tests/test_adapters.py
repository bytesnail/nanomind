"""测试适配器模块。"""

import pytest

from src.data_processing.adapters import fineweb_adapter, fineweb_adapter_safe


class MockReader:
    """模拟 Datatrove Reader。"""

    pass


class TestFinewebAdapter:
    """测试 fineweb_adapter 函数。"""

    def test_valid_input(self):
        """测试有效的输入数据。"""
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
            "url": "http://example.com",
            "token_count": 100,
        }

        result = fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

        assert isinstance(result, dict)
        assert result["text"] == "This is a test document."
        assert result["id"] == "test-id-123"
        assert result["metadata"]["score"] == 3.5
        assert result["metadata"]["cc_main"] == "CC-MAIN-2024-10"

    def test_missing_text(self):
        """测试缺少 text 字段。"""
        mock_reader = MockReader()
        raw_dict = {
            "id": "test-id-123",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
        }

        with pytest.raises(ValueError, match="text"):
            fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

    def test_missing_id(self):
        """测试缺少 id 字段。"""
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
        }

        with pytest.raises(ValueError, match="id"):
            fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

    def test_empty_text(self):
        """测试空 text 字段。"""
        mock_reader = MockReader()
        raw_dict = {
            "text": "",
            "id": "test-id-123",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
        }

        with pytest.raises(ValueError, match="text"):
            fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

    def test_missing_score(self):
        """测试缺少 score 字段（使用默认值）。"""
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "dump": "CC-MAIN-2024-10",
        }

        result = fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)
        assert result["metadata"]["score"] == 0.0

    def test_invalid_dump_format(self):
        """测试无效的 dump 格式。"""
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "score": 3.5,
            "dump": "invalid-dump-format",
        }

        result = fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)
        assert result["metadata"]["cc_main"] == "unknown"

    def test_extra_fields_filtered(self):
        """测试额外字段被过滤。"""
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
            "url": "http://example.com",
            "token_count": 100,
            "language": "en",
            "language_score": 0.95,
        }

        result = fineweb_adapter(mock_reader, raw_dict, "test.parquet", 0)

        assert "url" not in result["metadata"]
        assert "token_count" not in result["metadata"]
        assert "language" not in result["metadata"]
        assert "score" in result["metadata"]
        assert "cc_main" in result["metadata"]


class TestFinewebAdapterSafe:
    """测试 fineweb_adapter_safe 函数。"""

    def test_valid_input(self):
        """测试有效的输入数据。"""
        mock_reader = MockReader()
        raw_dict = {
            "text": "This is a test document.",
            "id": "test-id-123",
            "score": 3.5,
            "dump": "CC-MAIN-2024-10",
        }

        result = fineweb_adapter_safe(mock_reader, raw_dict, "test.parquet", 0)

        assert isinstance(result, dict)
        assert result["text"] == "This is a test document."

    def test_invalid_input_returns_none(self):
        """测试无效输入返回 None。"""
        mock_reader = MockReader()
        raw_dict = {
            "id": "test-id-123",
            "score": 3.5,
        }

        result = fineweb_adapter_safe(mock_reader, raw_dict, "test.parquet", 0)
        assert result is None

    def test_empty_text_returns_none(self):
        """测试空 text 返回 None。"""
        mock_reader = MockReader()
        raw_dict = {
            "text": "",
            "id": "test-id-123",
            "score": 3.5,
        }

        result = fineweb_adapter_safe(mock_reader, raw_dict, "test.parquet", 0)
        assert result is None
