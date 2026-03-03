#!/usr/bin/env python3
"""Scripts 公共工具模块。"""

from __future__ import annotations

import inspect
import json
import logging
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path

from src.data_processing.config_loader import DEFAULT_LOG_FORMAT


def setup_logging(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """配置日志并返回指定名称的 logger。

    Args:
        name: Logger 名称，默认使用调用者模块名
        level: 日志级别，默认为 INFO

    Returns:
        配置好的 logger 实例
    """
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)

    if name is None:
        # 获取调用者的模块名
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "__main__")
        else:
            name = "__main__"

    return logging.getLogger(name)


def read_json(path: Path) -> dict:
    """读取 JSON 文件并返回解析后的字典。

    Args:
        path: JSON 文件路径

    Returns:
        解析后的字典

    Raises:
        FileNotFoundError: 文件不存在时
        JSONDecodeError: JSON 解析失败时
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"JSON 文件不存在: {path}") from e
    except JSONDecodeError as e:
        raise JSONDecodeError(f"JSON 解析失败 {path}: {e.msg}", e.doc, e.pos) from e


def write_json(path: Path, data: dict, indent: int = 2) -> None:
    """写入 JSON 文件。

    Args:
        path: 输出文件路径
        data: 要写入的字典数据
        indent: 缩进空格数，默认 2

    Raises:
        OSError: 文件写入失败时
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except OSError as e:
        raise OSError(f"JSON 文件写入失败 {path}: {e}") from e


@lru_cache(maxsize=8)
def _get_separator(width: int) -> str:
    """缓存分隔符字符串。"""
    return "=" * width


def log_section(
    title: str, width: int = 60, logger: logging.Logger | None = None
) -> None:
    """输出格式化的分隔线和标题。"""
    log = logger or logging.getLogger(__name__)
    separator = _get_separator(width)
    log.info(separator)
    log.info(title)
    log.info(separator)


def log_directory_contents(
    directory: Path,
    title: str = "输出文件",
    logger: logging.Logger | None = None,
) -> None:
    """输出目录内容清单。"""
    log = logger or logging.getLogger(__name__)
    if not directory.exists():
        log.warning(f"目录不存在: {directory}")
        return

    files = list(directory.iterdir())
    log.info(f"{title} ({len(files)} 个):")
    for f in sorted(files):
        size = f.stat().st_size if f.is_file() else 0
        log.info(f"  - {f.name} ({size:,} bytes)")
