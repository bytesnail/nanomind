import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"

    @lru_cache(maxsize=10)
    def _load(self, name: str) -> dict[str, Any]:
        path = self.config_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def buckets(self) -> dict[str, Any]:
        return self._load("buckets")

    @property
    def processing(self) -> dict[str, Any]:
        return self._load("processing")

    @property
    def paths(self) -> dict[str, Any]:
        paths = dict(self._load("paths"))
        for key in paths:
            if env_var := os.getenv(f"FINEWEB_{key.upper()}"):
                paths[key] = env_var
        return paths

    @property
    def dataset(self) -> dict[str, Any]:
        return self._load("dataset")

    def get_bucket_configs(self) -> list[dict[str, Any]]:
        return self.buckets.get("buckets", [])


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
