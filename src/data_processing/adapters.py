from typing import Any

from .config_loader import get_dataset_configs


def _generate_id(source_path: str, idx: int, root_marker: str) -> str:
    sep = f"{root_marker}/"
    if sep in source_path:
        rel_path = source_path.split(sep, 1)[1]
    else:
        rel_path = source_path
    return f"{rel_path}#{idx}"


def normalize_score(
    raw_score: Any, normalization_config: dict[str, Any] | None
) -> float:
    score = float(raw_score) if raw_score is not None else 0.0
    if normalization_config and normalization_config.get("enabled", False):
        multiplier = normalization_config.get("multiplier", 1.0)
        score = score * multiplier
    return score


def _get_dataset_config_for_source(source: str) -> dict[str, Any]:
    datasets = get_dataset_configs()
    for _lang, config in datasets.items():
        marker = config.get("root_marker", "")
        if marker and marker in source:
            return config
    return datasets.get("en", {})


def fineweb_adapter(
    _reader: Any, raw: dict, source: str, idx: int
) -> dict[str, Any] | None:
    text = raw.get("text", "")
    if not text:
        return None

    dataset_config = _get_dataset_config_for_source(source)
    root_marker = dataset_config.get("root_marker", "fineweb-edu")
    score_normalization = dataset_config.get("score_normalization")

    dump = raw.get("dump", "")
    return {
        "text": text,
        "id": _generate_id(source, idx, root_marker),
        "metadata": {
            "score": normalize_score(raw.get("score"), score_normalization),
            "cc_main": dump if dump.startswith("CC-MAIN-") else "unknown",
        },
    }
