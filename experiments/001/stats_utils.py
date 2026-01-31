"""统计计算工具模块。

本模块包含用于计算和聚合数据集统计信息的函数，主要用于
处理来自多个worker的统计结果并计算分数的统计指标。
"""

import json
from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np
from datatrove.io import DataFolder

try:
    from .config import DatasetConfig
except ImportError:
    from config import DatasetConfig

try:
    from ..utils.constants import PERCENTILE_STEPS
except ImportError:
    from experiments.utils.constants import PERCENTILE_STEPS


def compute_score_stats(scores: List[float], total_docs: int = 0) -> Dict[str, Any]:
    """计算 score 的统计信息。

    Args:
        scores: 分数列表。
        total_docs: 总文档数。

    Returns:
        统计信息字典，包含均值、中位数、标准差、百分位数等。
    """
    if not scores:
        return {"total_docs": total_docs, "all_scores": []}

    scores_array = np.array(scores)
    result = {
        "mean": float(np.mean(scores_array)),
        "median": float(np.median(scores_array)),
        "std": float(np.std(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "percentiles": {},
        "total_docs": total_docs,
        "all_scores": scores,
    }

    for i in range(1, 11):
        percentile = i * PERCENTILE_STEPS
        result["percentiles"][f"{percentile}%"] = float(
            np.percentile(scores_array, percentile)
        )

    return result


def _initialize_aggregated_stats() -> Dict[str, Any]:
    """初始化聚合统计结构。

    Returns:
        初始化的统计字典。
    """
    return {
        "snapshot_stats": Counter(),
        "score_stats": {"scores": [], "total_docs": 0},
        "int_score_distribution": Counter(),
    }


def _process_worker_snapshot_stats(
        aggregated: Dict[str, Any], worker_stats: Dict[str, Any]
) -> None:
    """处理单个worker的快照统计信息。

    Args:
        aggregated: 聚合统计字典。
        worker_stats: 单个worker的统计信息。
    """
    for snapshot, info in worker_stats.get("snapshot_stats", {}).items():
        aggregated["snapshot_stats"][f"{snapshot}_doc_count"] += info.get(
            "doc_count", 0
        )
        aggregated["snapshot_stats"][f"{snapshot}_file_count"] += info.get(
            "file_count", 0
        )


def _process_worker_score_stats(
        aggregated: Dict[str, Any], worker_stats: Dict[str, Any]
) -> None:
    """处理单个worker的分数统计信息。

    Args:
        aggregated: 聚合统计字典。
        worker_stats: 单个worker的统计信息。
    """
    score_stats = worker_stats.get("score_stats", {})
    aggregated["score_stats"]["scores"].extend(score_stats.get("all_scores", []))
    aggregated["score_stats"]["total_docs"] += score_stats.get("total_docs", 0)


def _process_worker_int_scores(
        aggregated: Dict[str, Any], worker_stats: Dict[str, Any]
) -> None:
    """处理单个worker的整数分数分布。

    Args:
        aggregated: 聚合统计字典。
        worker_stats: 单个worker的统计信息。
    """
    for int_score, count in worker_stats.get("int_score_distribution", {}).items():
        aggregated["int_score_distribution"][int_score] += count


def _finalize_snapshot_stats(aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """最终化快照统计信息格式。

    Args:
        aggregated: 聚合统计字典。

    Returns:
        格式化的快照统计信息。
    """
    snapshot_stats = {}
    snapshot_dict = dict(aggregated["snapshot_stats"])

    for key, value in snapshot_dict.items():
        if key.endswith("_doc_count"):
            snapshot_name = key[:-9]
            file_key = f"{snapshot_name}_file_count"
            snapshot_stats[snapshot_name] = {
                "doc_count": value,
                "file_count": snapshot_dict.get(file_key, 0),
            }

    return snapshot_stats


def _finalize_score_stats(aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """最终化分数统计信息格式。

    Args:
        aggregated: 聚合统计字典。

    Returns:
        格式化的分数统计信息。
    """
    scores = aggregated["score_stats"]["scores"]
    total_docs = aggregated["score_stats"]["total_docs"]

    if not scores:
        return {}

    score_stats = compute_score_stats(scores, total_docs)
    score_stats.pop("all_scores", None)
    return score_stats


def aggregate_worker_stats(
        data_folder: DataFolder, stats_files: Sequence[str]
) -> Dict[str, Any]:
    """聚合多个 worker 的统计结果。

    Args:
        data_folder: 数据文件夹对象。
        stats_files: 统计文件路径列表。

    Returns:
        聚合后的统计信息字典。
    """
    aggregated = _initialize_aggregated_stats()

    for stats_file in stats_files:
        with data_folder.open(stats_file, "r") as f:
            worker_stats = json.load(f)

        _process_worker_snapshot_stats(aggregated, worker_stats)
        _process_worker_score_stats(aggregated, worker_stats)
        _process_worker_int_scores(aggregated, worker_stats)

    final_result = {
        "snapshot_stats": _finalize_snapshot_stats(aggregated),
        "score_stats": _finalize_score_stats(aggregated),
        "int_score_distribution": dict(aggregated["int_score_distribution"]),
    }

    return final_result
