"""统计收集器模块。"""

import json
import os
from collections import Counter
from typing import Any, Dict, Generator, List, Optional

from datatrove.data import Document
from datatrove.io import DataFolder, get_datafolder
from datatrove.pipeline.base import PipelineStep

from config import DatasetConfig
from io_utils import (
    _collect_numeric_field,
    _update_snapshot_stats,
)
from stats_utils import (
    aggregate_worker_stats,
    compute_score_stats,
)

StatsDict = Dict[str, Any]


class DatasetStatsCollector(PipelineStep):
    """收集数据集统计信息的 PipelineStep。"""

    name = "📊 DatasetStatsCollector"
    type = "STATS_COLLECTOR"

    def __init__(self, config: DatasetConfig, output_folder: str) -> None:
        """初始化统计收集器。

        Args:
            config: 数据集配置对象。
            output_folder: 输出文件夹路径。
        """
        super().__init__()
        self.config: DatasetConfig = config
        self.output_folder: str = output_folder
        self.data_folder: DataFolder = get_datafolder(output_folder)
        self.snapshot_stats: Dict[str, Dict[str, Any]] = {}
        self.scores: List[float] = []
        self.int_score_counter: Counter = Counter()
        self.total_docs: int = 0

    def run(
        self, data: Any, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """执行统计收集的流水线步骤。

        Args:
            data: 输入数据流。
            rank: 当前 worker 的 rank。
            world_size: 总 worker 数量。

        Yields:
            处理后的文档。
        """
        for doc in data:
            with self.track_time():
                self._collect_document_stats(doc)
                self.stat_update("processed_docs")
            yield doc

        self._save_worker_stats(rank)

    def _collect_document_stats(self, doc: Document) -> None:
        """收集单个文档的统计信息。

        Args:
            doc: 要收集统计信息的文档对象。
        """
        self.total_docs += 1

        if self.config.group_by == "directory":
            file_path = doc.metadata.get("file_path", "")
            if file_path:
                dir_name = os.path.basename(os.path.dirname(file_path))
                file_name = doc.metadata.get("file_name", "")
                _update_snapshot_stats(self.snapshot_stats, dir_name, file_name)
        elif self.config.group_field:
            group_value = doc.metadata.get(self.config.group_field, "")
            if group_value:
                file_name = doc.metadata.get("file_name", "")
                _update_snapshot_stats(self.snapshot_stats, group_value, file_name)

        score_value = _collect_numeric_field(
            doc.metadata, self.config.score_field, "score"
        )
        if score_value is not None:
            self.scores.append(score_value)

        int_score_value = _collect_numeric_field(
            doc.metadata, self.config.int_score_field, "int_score"
        )
        if int_score_value is not None:
            self.int_score_counter[str(int(int_score_value))] += 1

    def _save_worker_stats(self, rank: int) -> None:
        """保存当前 worker 的统计结果。

        Args:
            rank: worker 的 rank 编号。
        """
        score_stats = compute_score_stats(self.scores, self.total_docs)

        snapshot_stats_final: Dict[str, Dict[str, Any]] = {}
        for snapshot_name, snapshot_info in self.snapshot_stats.items():
            files = snapshot_info.get("files", set())
            snapshot_stats_final[snapshot_name] = {
                "doc_count": snapshot_info["doc_count"],
                "file_count": len(files),
            }

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

        Args:
            output_folder: 输出文件夹路径。
            config: 数据集配置对象。

        Returns:
            聚合后的统计信息字典，如果没有找到统计文件则返回 None。
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
