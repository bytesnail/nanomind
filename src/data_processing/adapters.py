"""FineWeb-Edu 数据适配器。"""

from pathlib import Path
from typing import Any

DATASET_ROOT_MARKER = "fineweb-edu"


def _generate_unique_id(source_path: str, idx: int) -> str:
    """从 source_path 提取相对路径并与 idx 组合为唯一 ID。"""
    if not source_path:
        raise ValueError("source_path cannot be empty")
    if idx < 0:
        raise ValueError(f"idx must be non-negative, got {idx}")

    path = Path(source_path)
    try:
        marker_idx = path.parts.index(DATASET_ROOT_MARKER)
        relative_path = "/".join(path.parts[marker_idx + 1 :])
    except ValueError:
        relative_path = path.as_posix()

    return f"{relative_path}#{idx}"


def fineweb_adapter(
    _reader: Any, raw: dict, source: str, idx: int, *, raise_on_error: bool = False
) -> dict[str, Any] | None:
    """FineWeb-Edu 数据适配器。text 缺失/为空时：raise_on_error 控制抛错或返回 None。"""
    if not (text := raw.get("text", "")):
        if raise_on_error:
            raise ValueError("text is missing or empty")
        return None

    dump = raw.get("dump", "")
    return {
        "text": text,
        "id": _generate_unique_id(source, idx),
        "metadata": {
            "score": raw.get("score", 0.0),
            "cc_main": dump if dump.startswith("CC-MAIN-") else "unknown",
        },
    }
