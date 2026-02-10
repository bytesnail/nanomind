from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

Compression = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]

DEFAULT_WORKERS = 8
DEFAULT_TASKS = 8
DEFAULT_RANDOM_SEED = 42
DEFAULT_COMPRESSION: Compression = "zstd"
DEFAULT_MAX_FILE_SIZE = 512 * 1024 * 1024
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@lru_cache(maxsize=10)
def _load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_dataset_config_dict() -> dict[str, Any]:
    return _load_yaml("dataset")


@lru_cache(maxsize=1)
def get_processing_config() -> dict[str, Any]:
    return _load_yaml("processing")


@lru_cache(maxsize=1)
def get_paths_config() -> dict[str, Any]:
    return _load_yaml("paths")


def get_dataset_configs() -> dict[str, dict[str, Any]]:
    return get_dataset_config_dict().get("datasets", {})


def get_dataset_config(dataset_key: str) -> dict[str, Any]:
    return get_dataset_configs().get(dataset_key, {})


def get_raw_bucket_configs(dataset_key: str) -> list[dict[str, Any]]:
    return get_dataset_config(dataset_key).get("buckets", [])
