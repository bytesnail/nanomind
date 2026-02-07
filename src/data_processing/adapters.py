"""FineWeb-Edu 数据适配器。"""

from typing import Any


def _extract(raw: dict) -> dict[str, Any] | None:
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
    _reader, raw: dict, _source: str, _idx: int, *, raise_on_error: bool = False
) -> dict[str, Any] | None:
    """FineWeb-Edu 数据适配器。

    Args:
        _reader: 读取器实例（由 datatrove 传入，当前未使用）
        raw: 原始数据字典
        _source: 数据源文件路径（由 datatrove 传入，当前未使用）
        _idx: 数据在文件中的索引（由 datatrove 传入，当前未使用）
        raise_on_error: 如果为 True，当数据无效时抛出 ValueError；
                       如果为 False（默认），返回 None

    Returns:
        转换后的数据字典，如果 raise_on_error=False 且数据无效则返回 None

    Raises:
        ValueError: 如果 raise_on_error=True 且数据缺少必需的 text 或 id 字段
    """
    result = _extract(raw)
    if result is None and raise_on_error:
        raise ValueError("Missing required field: text or id")
    return result
