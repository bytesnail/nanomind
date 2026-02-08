from pathlib import Path
from typing import Any

from .config_loader import get_config


class _IdGenerator:
    _marker: str = ""

    @classmethod
    def get_marker(cls) -> str:
        if not cls._marker:
            cls._marker = (
                get_config()
                .dataset.get("fineweb_edu", {})
                .get("root_marker", "fineweb-edu")
            )
        return cls._marker

    @classmethod
    def generate(cls, source_path: str, idx: int) -> str:
        sep = f"{cls.get_marker()}/"
        rel_path = (
            source_path.split(sep, 1)[1]
            if sep in source_path
            else Path(source_path).as_posix()
        )
        return f"{rel_path}#{idx}"


def fineweb_adapter(
    _reader: Any, raw: dict, source: str, idx: int
) -> dict[str, Any] | None:
    text = raw.get("text", "")
    if not text:
        return None

    dump = raw.get("dump", "")
    return {
        "text": text,
        "id": _IdGenerator.generate(source, idx),
        "metadata": {
            "score": raw.get("score", 0.0),
            "cc_main": dump if dump.startswith("CC-MAIN-") else "unknown",
        },
    }
