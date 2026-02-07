"""FineWeb-Edu 配置加载器。"""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """配置管理器，支持从 YAML 文件加载和环境变量覆盖。"""

    def __init__(self, config_dir: Path | None = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "config"
        self.config_dir = config_dir
        self._cache: dict[str, Any] = {}

    def _load(self, name: str) -> dict[str, Any]:
        if name not in self._cache:
            path = self.config_dir / f"{name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"配置文件不存在: {path}")
            with open(path, encoding="utf-8") as f:
                self._cache[name] = yaml.safe_load(f) or {}
        return self._cache[name]

    @property
    def buckets(self) -> dict[str, Any]:
        return self._load("buckets")

    @property
    def processing(self) -> dict[str, Any]:
        return self._load("processing")

    @property
    def paths(self) -> dict[str, Any]:
        paths = self._load("paths").copy()
        for key in list(paths.keys()):
            env_key = f"FINEWEB_{key.upper()}"
            if env_var := os.getenv(env_key):
                paths[key] = env_var
        return paths

    @property
    def dataset(self) -> dict[str, Any]:
        return self._load("dataset")

    def get_bucket_configs(self) -> list[dict[str, Any]]:
        return self.buckets.get("buckets", [])

    def get_epsilon(self) -> float:
        return float(self.buckets.get("epsilon", 1e-6))

    def get_required_fields(self) -> set[str]:
        fineweb = self.dataset.get("fineweb_edu", {})
        return set(fineweb.get("required_fields", ["id", "text", "score"]))

    def get_root_marker(self) -> str:
        fineweb = self.dataset.get("fineweb_edu", {})
        return fineweb.get("root_marker", "fineweb-edu")


_config_instance: Config | None = None


def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
