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


def fineweb_adapter(self, raw: dict, source: str, idx: int) -> dict[str, Any]:
    result = _extract(raw)
    if result is None:
        raise ValueError("Missing required field: text or id")
    return result


def fineweb_adapter_safe(
    self, raw: dict, source: str, idx: int
) -> dict[str, Any] | None:
    return _extract(raw)
