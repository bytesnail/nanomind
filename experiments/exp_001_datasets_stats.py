"""实验 001: 数据集统计与探索。

本脚本用于分析多种数据集的统计信息，包括：
- 文档计数和文件计数（按快照或目录分组）
- Score 分布统计（mean, median, std, min, max, percentiles）
- int_score 分布统计（整数分数的频次分布）

支持的数据集：
- HuggingFaceFW/fineweb-edu: 教育质量数据集
- opencsg/Fineweb-Edu-Chinese-V2.1: 中文教育数据集
- HuggingFaceTB/finemath: 数学数据集
- nvidia/Nemotron-CC-Math-v1: 多评分系统数据集

功能特性：
- 使用 datatrove pipeline 并行处理
- 支持多 worker 并发
- 自动聚合统计结果
- 生成 JSON 格式的详细报告

运行命令：
    # 探索模式：处理完整数据集
    python experiments/exp_001_datasets_stats.py --dataset HuggingFaceFW/fineweb-edu --data-dir <path> --workers 8

    # 快速模式：限制文档数用于快速测试
    python experiments/exp_001_datasets_stats.py --dataset HuggingFaceFW/fineweb-edu --data-dir <path> --limit 10000

    # 处理所有有 score 的数据集
    python experiments/exp_001_datasets_stats.py --dataset all --workers 8

    # Dry run：显示配置但不执行
    python experiments/exp_001_datasets_stats.py --dataset HuggingFaceFW/fineweb-edu --data-dir <path> --dry-run

注意：
- 使用 --help 查看所有可用参数和默认值
- 输出结果保存在 outputs/exp_001_datasets_stats/<dataset_name>/results/dataset_stats.json
- 支持断点续传（skip_completed=True）
"""

# 标准库
import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, override

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# YAML 配置和统计计算
import numpy as np

# DataTrove
from datatrove.data import Document
from datatrove.io import DataFolder, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader

# 自定义组件
from experiments.utils.common import (
    create_local_executor,
    get_timestamp,
    setup_logging,
)

StatsDict = Dict[str, Any]
DocumentGenerator = Generator[Document, None, None]

# 常量定义
SEPARATOR_LENGTH: int = 60
PERCENTILE_STEPS: int = 10

# ============================================================================
# 数据集配置系统
# ============================================================================


@dataclass
class DatasetConfig:
    """数据集配置类"""

    name: str  # 数据集唯一标识
    path: str  # 数据集根路径
    text_key: str = "text"  # 文本字段名
    id_key: Optional[str] = "id"  # ID 字段名
    group_field: Optional[str] = None  # 分组字段（metadata 字段名）
    group_by: Optional[str] = None  # 分组策略：None, "directory"
    score_field: Optional[str] = "score"  # Score 字段名或路径（支持嵌套）
    int_score_field: Optional[str] = "int_score"  # int_score 字段名或路径
    glob_pattern: str = "**/*.parquet"  # 文件匹配模式


# 数据集注册表
# 每个配置定义了数据集的字段映射和分组策略
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "HuggingFaceFW/fineweb-edu": DatasetConfig(
        name="HuggingFaceFW/fineweb-edu",
        path="data/datasets/HuggingFaceFW/fineweb-edu/data/",
        text_key="text",
        id_key="id",
        group_field="dump",  # 从 metadata.dump 分组
        group_by=None,
        score_field="score",
        int_score_field="int_score",
        glob_pattern="**/*.parquet",
    ),
    "opencsg/Fineweb-Edu-Chinese-V2.1": DatasetConfig(
        name="opencsg/Fineweb-Edu-Chinese-V2.1",
        path="data/datasets/opencsg/Fineweb-Edu-Chinese-V2.1/",
        text_key="text",
        id_key=None,  # 无 id 字段
        group_field=None,  # 无分组字段
        group_by="directory",  # 按目录名分组（如 2_3, 3_4, 4_5 表示质量等级）
        score_field="score",
        int_score_field=None,  # 无 int_score 字段
        glob_pattern="**/*.parquet",
    ),
    "HuggingFaceTB/finemath": DatasetConfig(
        name="HuggingFaceTB/finemath",
        path="data/datasets/HuggingFaceTB/finemath/",
        text_key="text",
        id_key=None,  # 需要验证是否有 id 字段
        group_field="snapshot_type",  # 从 metadata.snapshot_type 分组
        group_by=None,
        score_field="score",
        int_score_field="int_score",
        glob_pattern="**/*.parquet",
    ),
    # 无 Score 数据集 - 使用 --dataset all 时将被跳过
    "nick007x/github-code-2025": DatasetConfig(
        name="nick007x/github-code-2025",
        path="data/datasets/nick007x/github-code-2025/",
        text_key="content",  # 文本字段是 content 不是 text
        id_key=None,
        group_field="repo_id",
        group_by=None,
        score_field=None,  # 无 score 字段
        int_score_field=None,
        glob_pattern="**/*.parquet",
    ),
    # 多评分系统数据集 - 使用 finemath 评分（嵌套字段）
    "nvidia/Nemotron-CC-Math-v1": DatasetConfig(
        name="nvidia/Nemotron-CC-Math-v1",
        path="data/datasets/nvidia/Nemotron-CC-Math-v1/",
        text_key="text",
        id_key="id",
        group_field=None,  # 无分组
        group_by=None,
        score_field="metadata.finemath_scores",  # 使用嵌套的 finemath 评分
        int_score_field="metadata.finemath_int_scores",
        glob_pattern="**/*.parquet",
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """获取数据集配置，如果不存在则抛出错误"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def get_datasets_with_score() -> List[str]:
    """获取有 score 字段的数据集列表"""
    return [
        name for name, cfg in DATASET_CONFIGS.items() if cfg.score_field is not None
    ]


def resolve_nested_field(data: Dict[str, Any], field_path: str) -> Any:
    """解析嵌套字段（如 'metadata.finemath_scores'）。

    Args:
        data: 数据字典。
        field_path: 字段路径，用点号分隔（如 'metadata.finemath_scores'）。

    Returns:
        解析后的值，如果字段不存在则返回 None。

    Examples:
        >>> data = {"metadata": {"finemath_scores": 0.9}}
        >>> resolve_nested_field(data, "metadata.finemath_scores")
        0.9
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
        snapshot_stats: 快照统计字典（会被修改）。
        group_key: 分组键（快照名或目录名）。
        file_name: 文件名（可选）。
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

    支持简单字段和嵌套字段（点号分隔）。

    Args:
        data: 文档 metadata 字典。
        field_path: 字段路径（如 'score' 或 'metadata.finemath_scores'）。
        field_name: 字段名称（用于日志记录）。
        logger: 日志记录器。

    Returns:
        转换后的浮点数值，如果字段不存在或转换失败则返回 None。
    """
    if not field_path:
        return None

    # 解析字段值
    if "." in field_path:
        value = resolve_nested_field(data, field_path)
    else:
        value = data.get(field_path)

    if value is None:
        return None

    # 转换为 float
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to convert {field_name} value: {value}, error: {e}")
        return None


# ============================================================================
# 统计收集器核心实现
# ============================================================================


def compute_score_stats(scores: List[float], total_docs: int = 0) -> Dict[str, Any]:
    """计算 score 的统计信息。

    计算 mean, median, std, min, max 以及 10%、20%、...、90%、100% 的百分位数。

    Args:
        scores: 文档 score 列表。
        total_docs: 总文档数（默认为 0）。

    Returns:
        包含统计信息的字典，包括：
            - mean: 平均值
            - median: 中位数
            - std: 标准差
            - min: 最小值
            - max: 最大值
            - percentiles: 百分位数字典（10%, 20%, ..., 90%）
            - total_docs: 总文档数
            - all_scores: 所有 score 值列表

    Examples:
        >>> scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> stats = compute_score_stats(scores, total_docs=5)
        >>> print(stats["mean"])
        3.0
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


def aggregate_worker_stats(
    data_folder: DataFolder, stats_files: Sequence[str]
) -> StatsDict:
    """聚合多个 worker 的统计结果。

    读取多个 worker 生成的 JSON 统计文件，合并它们的统计信息。

    Args:
        data_folder: 数据文件夹对象（DataFolder）。
        stats_files: worker 统计文件路径列表。

    Returns:
        聚合后的统计信息字典，包括：
            - snapshot_stats: 快照统计（每个快照的文档数和文件数）
            - score_stats: score 统计（mean, median, std, min, max, percentiles）
            - int_score_distribution: int_score 分布（整数分数到计数的映射）

    Raises:
        json.JSONDecodeError: 如果统计文件格式错误。
        IOError: 如果无法读取统计文件。

    Examples:
        >>> from datatrove.io import get_datafolder
        >>> data_folder = get_datafolder("outputs/exp_001_datasets_stats")
        >>> stats = aggregate_worker_stats(data_folder, ["worker_00000.json", "worker_00001.json"])
        >>> print(stats["snapshot_stats"])
    """
    aggregated = {
        "snapshot_stats": Counter(),
        "score_stats": {"scores": [], "total_docs": 0},
        "int_score_distribution": Counter(),
    }

    for stats_file in stats_files:
        with data_folder.open(stats_file, "r") as f:
            worker_stats = json.load(f)

        for snapshot, info in worker_stats.get("snapshot_stats", {}).items():
            aggregated["snapshot_stats"][f"{snapshot}_doc_count"] += info.get(
                "doc_count", 0
            )
            aggregated["snapshot_stats"][f"{snapshot}_file_count"] += info.get(
                "file_count", 0
            )

        score_stats = worker_stats.get("score_stats", {})
        aggregated["score_stats"]["scores"].extend(score_stats.get("all_scores", []))
        aggregated["score_stats"]["total_docs"] += score_stats.get("total_docs", 0)

        for int_score, count in worker_stats.get("int_score_distribution", {}).items():
            aggregated["int_score_distribution"][int_score] += count

    final_result = {
        "snapshot_stats": {},
        "score_stats": {},
        "int_score_distribution": dict(aggregated["int_score_distribution"]),
    }

    snapshot_dict = dict(aggregated["snapshot_stats"])
    for key, value in snapshot_dict.items():
        if key.endswith("_doc_count"):
            snapshot_name = key[:-9]
            file_key = f"{snapshot_name}_file_count"
            final_result["snapshot_stats"][snapshot_name] = {
                "doc_count": value,
                "file_count": snapshot_dict.get(file_key, 0),
            }

    scores = aggregated["score_stats"]["scores"]
    total_docs = aggregated["score_stats"]["total_docs"]
    if scores:
        score_stats = compute_score_stats(scores, total_docs)
        score_stats.pop("all_scores", None)
        final_result["score_stats"] = score_stats

    return final_result


class DatasetStatsCollector(PipelineStep):
    """收集数据集统计信息的 PipelineStep（支持多种数据集配置）。

    该步骤收集以下统计信息：
    - 按快照或目录分组的文档计数和文件计数
    - Score 分布（mean, median, std, min, max, percentiles）
    - int_score 分布（整数分数的频次分布）

    支持的数据集配置：
    - 通过 DatasetConfig 配置字段映射和分组策略
    - 支持嵌套字段（如 metadata.finemath_scores）
    - 支持按 metadata 字段或目录名分组

    Attributes:
        name: 步骤名称。
        type: 步骤类型。
        config: 数据集配置对象。
        output_folder: 输出目录路径。
        data_folder: DataFolder 对象。
        snapshot_stats: 快照统计信息。
        scores: 收集的 score 值列表。
        int_score_counter: int_score 频次计数器。
        total_docs: 总文档数。

    Examples:
        >>> config = get_dataset_config("HuggingFaceFW/fineweb-edu")
        >>> collector = DatasetStatsCollector(config, "outputs/exp_001")
        >>> # 使用 datatrove pipeline 运行
    """

    name = "📊 DatasetStatsCollector"
    type = "STATS_COLLECTOR"

    def __init__(self, config: DatasetConfig, output_folder: str) -> None:
        """初始化统计收集器。

        Args:
            config: 数据集配置对象，包含字段映射和分组策略。
            output_folder: 输出目录路径，用于保存统计结果。
        """
        super().__init__()
        self.config: DatasetConfig = config
        self.output_folder: str = output_folder
        self.data_folder: DataFolder = get_datafolder(output_folder)
        self.snapshot_stats: Dict[str, Dict[str, Any]] = {}
        self.scores: List[float] = []
        self.int_score_counter: Counter = Counter()
        self.total_docs: int = 0
        self.logger = logging.getLogger(f"{self.name}")

    @override
    def run(
        self, data: Any, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """执行统计收集的流水线步骤。

        遍历所有文档，收集统计信息，最后保存到文件。

        Args:
            data: 文档生成器（Document generator）。
            rank: 当前 worker 的 rank（默认为 0）。
            world_size: 总 worker 数（默认为 1）。

        Yields:
            处理后的文档（原文档，不修改内容）。

        Raises:
            IOError: 如果无法保存统计文件。
        """
        for doc in data:
            with self.track_time():
                self._collect_document_stats(doc)
                self.stat_update("processed_docs")
            yield doc

        self._save_worker_stats(rank)

    def _collect_document_stats(self, doc: Document) -> None:
        """收集单个文档的统计信息。

        更新文档计数、快照统计、score 分布和 int_score 分布。

        Args:
            doc: 要收集统计信息的文档对象。
        """
        self.total_docs += 1

        # 分组逻辑
        if self.config.group_by == "directory":
            # 从文件路径提取目录名
            file_path = doc.metadata.get("file_path", "")
            if file_path:
                dir_name = os.path.basename(os.path.dirname(file_path))
                file_name = doc.metadata.get("file_name", "")
                _update_snapshot_stats(self.snapshot_stats, dir_name, file_name)
        elif self.config.group_field:
            # 从 metadata 字段分组
            group_value = doc.metadata.get(self.config.group_field, "")
            if group_value:
                file_name = doc.metadata.get("file_name", "")
                _update_snapshot_stats(self.snapshot_stats, group_value, file_name)

        # Score 收集（支持嵌套字段）
        score_value = _collect_numeric_field(
            doc.metadata, self.config.score_field, "score", self.logger
        )
        if score_value is not None:
            self.scores.append(score_value)

        # int_score 收集（支持嵌套字段）
        int_score_value = _collect_numeric_field(
            doc.metadata, self.config.int_score_field, "int_score", self.logger
        )
        if int_score_value is not None:
            self.int_score_counter[str(int(int_score_value))] += 1

    def _save_worker_stats(self, rank: int) -> None:
        """保存当前 worker 的统计结果。

        将统计结果保存为 JSON 文件，文件名格式为 worker_{rank:05d}.json。

        Args:
            rank: 当前 worker 的 rank。

        Raises:
            IOError: 如果无法写入统计文件。
            TypeError: 如果统计结果无法序列化为 JSON。
        """
        score_stats = compute_score_stats(self.scores, self.total_docs)

        snapshot_stats_final: Dict[str, Dict[str, Any]] = {}
        for snapshot_name, snapshot_info in self.snapshot_stats.items():
            files = snapshot_info.get("files", set())
            snapshot_stats_final[snapshot_name] = {
                "doc_count": snapshot_info["doc_count"],
                "file_count": len(files),
            }

        # 使用 config.name 作为目录名（替换 / 为 _）
        stats_dir_name = self.config.name.replace("/", "_")
        stats_dir = f"{stats_dir_name}_stats"

        worker_stats = {
            "snapshot_stats": snapshot_stats_final,
            "score_stats": score_stats,
            "int_score_distribution": dict(self.int_score_counter),
            "worker_rank": rank,
            "total_docs_processed": self.total_docs,
        }

        stats_file = f"{stats_dir}/worker_{rank:05d}.json"
        with self.data_folder.open(stats_file, "w") as f:
            json.dump(worker_stats, f, ensure_ascii=False, indent=2)

    @staticmethod
    def aggregate_stats(
        output_folder: str, config: DatasetConfig
    ) -> Optional[StatsDict]:
        """聚合所有 worker 的统计结果。

        读取所有 worker 生成的统计文件，合并为单一的聚合结果。

        Args:
            output_folder: 输出目录路径。
            config: 数据集配置对象。

        Returns:
            聚合后的统计信息字典，如果统计目录不存在或没有统计文件则返回 None。

        Examples:
            >>> config = get_dataset_config("HuggingFaceFW/fineweb-edu")
            >>> stats = DatasetStatsCollector.aggregate_stats("outputs/exp_001", config)
            >>> if stats:
            ...     print(stats["score_stats"])
        """
        data_folder = get_datafolder(output_folder)
        stats_dir_name = config.name.replace("/", "_")
        stats_dir = f"{stats_dir_name}_stats"

        if not data_folder.isdir(stats_dir):
            return None

        stats_files: List[str] = [
            file_path
            for file_path in data_folder.list_files(stats_dir)
            if file_path.endswith(".json") and "worker_" in file_path
        ]

        if not stats_files:
            return None

        aggregated_stats = aggregate_worker_stats(data_folder, stats_files)
        aggregated_file = f"{stats_dir}/aggregated_stats.json"
        with data_folder.open(aggregated_file, "w") as f:
            json.dump(aggregated_stats, f, ensure_ascii=False, indent=2)

        return aggregated_stats


# ============================================================================
# Pipeline 配置和执行
# ============================================================================


def parquet_to_doc_adapter(
    config: DatasetConfig,
    data: Dict[str, Any],
    source_file: str,
    id_in_file: int,
) -> Dict[str, Any]:
    """将 parquet 行转换为 Document 所需格式。

    根据 DatasetConfig 配置，提取 text、id 和 metadata 字段，转换为 Document 格式。

    Args:
        config: 数据集配置对象，包含字段映射信息。
        data: parquet 行数据（字典格式）。
        source_file: 源文件路径。
        id_in_file: 文件内的行 ID。

    Returns:
        包含 text、id 和 metadata 的字典，格式为：
            {
                "text": 文本内容,
                "id": 文档 ID（如果配置）,
                "metadata": 元数据字典
            }
    """
    # 提取 text 字段
    text_value = data.get(config.text_key)

    # 提取 id 字段（如果配置了）
    id_value = data.get(config.id_key) if config.id_key else None

    # 构建 metadata（保留所有非 text/id 字段）
    metadata = {}
    for key, value in data.items():
        if key != config.text_key and (config.id_key is None or key != config.id_key):
            metadata[key] = value

    # 添加文件路径
    if source_file:
        metadata["file_path"] = source_file

    result = {
        "text": text_value,
        "id": id_value,
        "metadata": metadata,
    }
    return result


def create_adapter(config: DatasetConfig):
    """创建配置好的适配器函数。

    创建一个适配器函数，用于将 parquet 行转换为 Document 格式。

    Args:
        config: 数据集配置对象。

    Returns:
        适配器函数，签名为 (reader, data, source_file, id_in_file) -> dict。

    Examples:
        >>> config = get_dataset_config("HuggingFaceFW/fineweb-edu")
        >>> adapter = create_adapter(config)
        >>> doc = adapter(None, {"text": "hello", "id": "123"}, "/path/to/file.parquet", 0)
    """

    def adapter(reader, data: Dict[str, Any], source_file: str, id_in_file: int):
        return parquet_to_doc_adapter(config, data, source_file, id_in_file)

    return adapter


def create_stats_collector(
    config: DatasetConfig, output_folder: str
) -> DatasetStatsCollector:
    """创建配置好的统计收集器。

    Args:
        config: 数据集配置对象。
        output_folder: 输出目录路径。

    Returns:
        配置好的 DatasetStatsCollector 实例。

    Examples:
        >>> config = get_dataset_config("HuggingFaceFW/fineweb-edu")
        >>> collector = create_stats_collector(config, "outputs/exp_001")
    """
    return DatasetStatsCollector(config=config, output_folder=output_folder)


def create_pipeline(
    config: DatasetConfig,
    output_dir: str,
    batch_size: int,
    limit: Optional[int],
) -> List:
    """创建 datatrove 处理流水线（参数化版本）。

    根据数据集配置创建 ParquetReader 和统计收集器流水线。

    Args:
        config: 数据集配置对象。
        output_dir: 输出目录路径。
        batch_size: 批量大小。
        limit: 文档数限制（None 表示无限制）。

    Returns:
        datatrove pipeline 步骤列表，包括 ParquetReader 和 DatasetStatsCollector。

    Examples:
        >>> config = get_dataset_config("HuggingFaceFW/fineweb-edu")
        >>> pipeline = create_pipeline(config, "outputs/exp_001", batch_size=5000, limit=None)
        >>> print(len(pipeline))
        2
    """
    reader_params = {
        "data_folder": config.path,
        "glob_pattern": config.glob_pattern,
        "batch_size": batch_size,
        "limit": limit if limit is not None else -1,
        "text_key": config.text_key,
        "adapter": create_adapter(config),
        "file_progress": True,
        "doc_progress": True,
    }

    # 只在 id_key 不为 None 时才传递
    if config.id_key is not None:
        reader_params["id_key"] = config.id_key

    return [
        ParquetReader(**reader_params),
        create_stats_collector(config, output_dir),
    ]


def run_pipeline(
    config: DatasetConfig,
    output_dir: str,
    workers: int,
    batch_size: int,
    limit: Optional[int],
    logger: logging.Logger,
) -> None:
    """运行 datatrove pipeline。

    创建并执行 datatrove pipeline，使用多 worker 并行处理数据。

    Args:
        config: 数据集配置对象。
        output_dir: 输出目录路径。
        workers: worker 数量。
        batch_size: 批量大小。
        limit: 文档数限制（None 表示无限制）。
        logger: 日志记录器。

    Raises:
        Exception: 如果 pipeline 执行失败（错误会被记录后重新抛出）。

    Examples:
        >>> config = get_dataset_config("HuggingFaceFW/fineweb-edu")
        >>> run_pipeline(config, "outputs/exp_001", workers=8, batch_size=5000, limit=None, logger)
    """
    pipeline = create_pipeline(config, output_dir, batch_size, limit)
    executor = create_local_executor(
        pipeline=pipeline,
        workers=workers,
        logging_dir=os.path.join(output_dir, "logs"),
        skip_completed=True,
    )

    logger.info(f"启动 {workers} 个 worker 开始处理...")
    try:
        executor.run()
        logger.info("pipeline 执行完成")
    except Exception as e:
        logger.error(f"pipeline 执行失败: {e}")
        raise


def save_results(
    results: Dict[str, Any], output_dir: str, logger: logging.Logger
) -> None:
    """保存最终统计结果。

    将统计结果保存为 JSON 文件到 results/dataset_stats.json。

    Args:
        results: 统计结果字典。
        output_dir: 输出目录路径。
        logger: 日志记录器。

    Raises:
        IOError: 如果无法写入结果文件。
        TypeError: 如果结果无法序列化为 JSON。

    Examples:
        >>> results = {"snapshot_stats": {}, "score_stats": {}}
        >>> save_results(results, "outputs/exp_001", logger)
    """
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    results_file = os.path.join(output_dir, "results", "dataset_stats.json")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 结果已保存到: {results_file}")
    print(f"\n✅ 结果已保存到: {results_file}")


# ============================================================================
# 输出辅助函数
# ============================================================================


def _setup_dataset_logging(
    config: DatasetConfig, dataset_output_dir: str, verbose: bool
) -> logging.Logger:
    log_dir = os.path.join(dataset_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = get_timestamp()
    log_file = os.path.join(log_dir, f"exp_001_datasets_stats_{timestamp}.log")
    logger = setup_logging(
        f"exp_001_datasets_stats_{config.name.replace('/', '_')}",
        "DEBUG" if verbose else "INFO",
        log_file,
    )
    return logger


def _print_dataset_info(config: DatasetConfig, dataset_output_dir: str) -> None:
    _print_section("处理数据集", config.name)
    print(f"  路径: {config.path}")
    print(f"  输出目录: {dataset_output_dir}")
    _print_separator()


def _build_results_dict(
    config: DatasetConfig, custom_stats: StatsDict, start_time: str, end_time: str
) -> Dict[str, Any]:
    return {
        "metadata": {
            "experiment": "多数据集统计",
            "dataset": config.name,
            "start_time": start_time,
            "end_time": end_time,
        },
        "snapshot_statistics": custom_stats.get("snapshot_stats", {}),
        "score_statistics": custom_stats.get("score_stats", {}),
        "int_score_distribution": custom_stats.get("int_score_distribution", {}),
    }


def _print_separator(char: str = "=", length: int = SEPARATOR_LENGTH) -> None:
    """打印分隔线。

    Args:
        char: 分隔线字符，默认为 "="。
        length: 分隔线长度，默认为 SEPARATOR_LENGTH。
    """
    print(char * length)


def _print_section(title: str, subtitle: str = "") -> None:
    """打印带标题的部分。

    Args:
        title: 主标题。
        subtitle: 副标题（可选）。
    """
    _print_separator()
    if subtitle:
        print(f"  {title}: {subtitle}")
    else:
        print(f"  {title}")
    _print_separator()


def _print_status(message: str, status: str = "info") -> None:
    """打印带状态图标的消息。

    Args:
        message: 要显示的消息。
        status: 状态类型 ("success", "error", "warning", "info")。
    """
    emoji_map = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
    }
    emoji = emoji_map.get(status, "")
    print(f"{emoji} {message}")


# ============================================================================
# 命令行接口
# ============================================================================


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="多数据集统计与探索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 全局参数
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出模式")

    # 数据集选择
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",  # 支持多个值
        required=True,
        help="数据集名称（支持多个或 'all' 表示所有有 score 的数据集）",
    )

    # 其他参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exp_001_datasets_stats",
        help="输出目录",
    )
    parser.add_argument("--workers", type=int, default=8, help="Worker 数量")
    parser.add_argument("--batch-size", type=int, default=5000, help="批量大小")
    parser.add_argument("--limit", type=int, default=None, help="限制文档数")
    parser.add_argument("--dry-run", action="store_true", help="演示模式")

    return parser


def _prepare_datasets(args: argparse.Namespace, logger: logging.Logger) -> List[str]:
    """准备数据集列表。

    Args:
        args: 命令行参数。
        logger: 日志记录器。

    Returns:
        要处理的数据集名称列表。
    """
    if "all" in args.dataset:
        dataset_names = get_datasets_with_score()
        logger.info(f"处理所有有 score 的数据集: {dataset_names}")
    else:
        dataset_names = args.dataset
        # 过滤掉无 score 的数据集
        dataset_names = [
            name
            for name in dataset_names
            if get_dataset_config(name).score_field is not None
        ]
        logger.info(f"处理数据集: {dataset_names}")

    return dataset_names


def _print_dry_run_summary(dataset_names: List[str]) -> None:
    """打印 dry run 模式下的配置摘要。

    Args:
        dataset_names: 数据集名称列表。
    """
    _print_section("Dry Run", "多数据集统计")
    for dataset_name in dataset_names:
        config = get_dataset_config(dataset_name)
        print(f"\n  数据集: {config.name}")
        print(f"    路径: {config.path}")
        print(f"    分组: {config.group_field or config.group_by or '无'}")
        print(f"    Score: {config.score_field or '无'}")
    _print_separator()
    print()


def _process_single_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    dataset_output_dir: str,
) -> None:
    """处理单个数据集。

    Args:
        dataset_name: 数据集名称。
        args: 命令行参数。
        dataset_output_dir: 数据集输出目录。
    """
    config = get_dataset_config(dataset_name)
    _print_dataset_info(config, dataset_output_dir)

    # 设置输出目录和日志
    os.makedirs(dataset_output_dir, exist_ok=True)
    logger = _setup_dataset_logging(config, dataset_output_dir, args.verbose)

    # 运行 pipeline
    start_time = datetime.now().isoformat()
    try:
        run_pipeline(
            config,
            dataset_output_dir,
            args.workers,
            args.batch_size,
            args.limit,
            logger,
        )

        # 聚合结果
        custom_stats = DatasetStatsCollector.aggregate_stats(dataset_output_dir, config)
        end_time = datetime.now().isoformat()

        if custom_stats:
            results = _build_results_dict(config, custom_stats, start_time, end_time)
            save_results(results, dataset_output_dir, logger)

            _print_section("总结")
            print("  数据集统计完成")
            print(f"  结果文件: {dataset_output_dir}/results/dataset_stats.json")
            print(f"  结束时间: {end_time}")
            _print_separator()
            print()
        else:
            logger.warning(f"数据集 {config.name} 没有生成统计结果")

    except Exception as e:
        logger.error(f"处理数据集 {config.name} 失败: {e}")
        _print_status(f"处理数据集 {config.name} 失败: {e}", "error")
        # 继续处理下一个数据集


def run_experiment(args: argparse.Namespace, logger: logging.Logger) -> None:
    """执行多数据集处理。

    Args:
        args: 命令行参数。
        logger: 日志记录器。
    """
    # 准备数据集列表
    dataset_names = _prepare_datasets(args, logger)

    # Dry run 模式
    if args.dry_run:
        _print_dry_run_summary(dataset_names)
        return

    # 处理每个数据集
    for dataset_name in dataset_names:
        dataset_output_dir = os.path.join(
            args.output_dir, dataset_name.replace("/", "_")
        )
        _process_single_dataset(dataset_name, args, dataset_output_dir)

    # 打印总结
    _print_section("全部完成")
    print(f"  已处理 {len(dataset_names)} 个数据集")
    _print_separator()
    print()


def main() -> None:
    """主函数，执行命令行入口流程。"""
    parser = create_parser()
    args = parser.parse_args()

    logger = logging.getLogger("exp_001_datasets_stats")
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    run_experiment(args, logger)


if __name__ == "__main__":
    main()
