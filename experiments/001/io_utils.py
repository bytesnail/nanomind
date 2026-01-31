"""I/O 和数据处理辅助函数模块。

此模块包含文件 I/O 操作和数据处理的辅助函数，主要用于
实验 001 中的数据集统计分析。

提供的主要功能：
- 结果保存到 JSON 文件
- 嵌套字段解析
- 快照统计信息更新
- 数值字段收集
"""

import json
import logging
import os
from typing import Any, Dict, Optional


def resolve_nested_field(data: Dict[str, Any], field_path: str) -> Any:
    """解析嵌套字段（如 'metadata.finemath_scores'）。

    Args:
        data: 数据字典。
        field_path: 字段路径，用点号分隔。

    Returns:
        字段值，如果未找到则返回 None。
    """
    keys = field_path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def _update_snapshot_stats(
    snapshot_stats: Dict[str, Dict[str, Any]],
    group_key: str,
    file_name: Optional[str],
) -> None:
    """更新快照统计信息。

    Args:
        snapshot_stats: 快照统计字典。
        group_key: 分组键。
        file_name: 文件名。
    """
    if group_key not in snapshot_stats:
        snapshot_stats[group_key] = {"doc_count": 0, "files": set()}
    snapshot_stats[group_key]["doc_count"] += 1
    if file_name:
        snapshot_stats[group_key]["files"].add(file_name)


def _collect_numeric_field(
    data: Dict[str, Any],
    field_path: Optional[str],
    field_name: str,
    logger: logging.Logger,
) -> Optional[float]:
    """从文档数据中收集数值字段。

    Args:
        data: 文档数据字典。
        field_path: 字段路径（支持嵌套）。
        field_name: 字段名称（用于日志）。
        logger: 日志记录器。

    Returns:
        浮点数值，如果未找到或转换失败则返回 None。
    """
    if not field_path:
        return None

    if "." in field_path:
        value = resolve_nested_field(data, field_path)
    else:
        value = data.get(field_path)

    if value is None:
        return None

    try:
        return float(value)
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to convert {field_name} value: {value}, error: {e}")
        return None


def save_results(
    results: Dict[str, Any], output_dir: str, logger: logging.Logger
) -> None:
    """保存最终统计结果。

    Args:
        results: 结果数据字典。
        output_dir: 输出目录路径。
        logger: 日志记录器。
    """
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    results_file = os.path.join(output_dir, "results", "dataset_stats.json")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 结果已保存到: {results_file}")
    print(f"\n✅ 结果已保存到: {results_file}")
