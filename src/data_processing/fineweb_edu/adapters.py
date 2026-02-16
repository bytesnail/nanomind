from typing import Any

from ..config_loader import get_dataset_configs


def normalize_score(
    raw_score: Any, normalization_config: dict[str, Any] | None
) -> float:
    score = float(raw_score) if raw_score is not None else 0.0
    if normalization_config and normalization_config.get("enabled", False):
        multiplier = normalization_config.get("multiplier", 1.0)
        score = score * multiplier
    return score


def _get_dataset_config_for_source(reader: Any | None = None) -> dict[str, Any]:
    if reader is None:
        return {}

    all_configs = get_dataset_configs()

    data_folder_path = str(reader.data_folder.path)

    for config in all_configs.values():
        input_dir = config.get("input_dir", "")
        if input_dir and input_dir in data_folder_path:
            return config

    return {}


def fineweb_adapter(
    reader: Any, raw: dict, source: str, idx: int
) -> dict[str, Any] | None:
    text = raw.get("text", "")
    if not text:
        return None

    dataset_config = _get_dataset_config_for_source(reader)
    score_normalization = dataset_config.get("score_normalization")

    dump = raw.get("dump", "")
    return {
        "text": text,
        "id": f"{source}#{idx}",
        "metadata": {
            "score": normalize_score(raw.get("score"), score_normalization),
            "cc_main": dump if dump.startswith("CC-MAIN-") else "unknown",
        },
    }
