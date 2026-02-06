"""FineWeb-Edu 数据适配器。"""

from typing import Any


def _extract_document_data(raw_dict: dict) -> dict[str, Any] | None:
    """从原始数据中提取文档数据。"""
    text = raw_dict.get("text", "")
    doc_id = raw_dict.get("id", "")

    if not text or not doc_id:
        return None

    dump = raw_dict.get("dump", "")
    cc_main = dump if dump.startswith("CC-MAIN-") else "unknown"

    return {
        "text": text,
        "id": doc_id,
        "metadata": {
            "score": raw_dict.get("score", 0.0),
            "cc_main": cc_main,
        },
    }


def fineweb_adapter(
    self,
    raw_dict: dict,
    source_file: str,
    id_in_file: int,
) -> dict[str, Any]:
    """将原始数据转换为 Document 字典，无效数据时抛出异常。"""
    result = _extract_document_data(raw_dict)
    if result is None:
        raise ValueError("Missing required field: text or id")
    return result


def fineweb_adapter_safe(
    self,
    raw_dict: dict,
    source_file: str,
    id_in_file: int,
) -> dict[str, Any] | None:
    """将原始数据转换为 Document 字典，无效数据时返回 None。"""
    return _extract_document_data(raw_dict)
