"""FineWeb-Edu 数据适配器模块。"""

from typing import Any


def _extract_data(raw: dict) -> dict[str, Any] | None:
    """从原始数据中提取有效文档数据。"""
    text, doc_id = raw.get("text", ""), raw.get("id", "")
    if not text or not doc_id:
        return None

    dump = raw.get("dump", "")
    return {
        "text": text,
        "id": doc_id,
        "metadata": {
            "score": raw.get("score", 0.0),
            "cc_main": dump if dump.startswith("CC-MAIN-") else "unknown",
        },
    }


def fineweb_adapter(
    self, raw_dict: dict, source_file: str, id_in_file: int
) -> dict[str, Any]:
    """适配器：无效数据时抛出异常。"""
    result = _extract_data(raw_dict)
    if result is None:
        raise ValueError("Missing required field: text or id")
    return result


def fineweb_adapter_safe(
    self, raw_dict: dict, source_file: str, id_in_file: int
) -> dict[str, Any] | None:
    """适配器：无效数据时返回 None。"""
    return _extract_data(raw_dict)
