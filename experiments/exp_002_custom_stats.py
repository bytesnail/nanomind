"""实验 002: 自定义 PipelineStep 收集 Fineweb-Edu 统计信息

目的：
- 实现一个自定义的 PipelineStep 类 FinewebEduStatsCollector
- 收集域名分布、快照统计、score分布和int_score分布
- 支持多worker并发处理和结果聚合
- 将统计结果保存到文件系统

运行命令：
    python experiments/exp_002_custom_stats.py

输出：
- 每个worker的独立统计文件
- 聚合后的全局统计结果

"""

# 标准库
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Generator
from collections import Counter
from urllib.parse import urlparse
from pathlib import Path

# DataTrove
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolder, get_datafolder

# 辅助类型定义
StatsDict = Dict[str, Any]
DocumentGenerator = Generator[Document, None, None]


def extract_domain(url: str) -> str:
    """从URL中提取域名。

    Args:
        url: 完整的URL字符串

    Returns:
        域名部分（netloc），解析失败返回"unknown"
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return "unknown"


def aggregate_worker_stats(
    data_folder: DataFolder, stats_files: List[str]
) -> StatsDict:
    """聚合多个worker的统计结果。

    Args:
        data_folder: DataFolder实例
        stats_files: 各个worker统计文件路径列表

    Returns:
        聚合后的全局统计结果
    """
    # 初始化聚合结果结构
    aggregated = {
        "domain_stats": {"total_domains": set(), "top_1000": Counter()},
        "snapshot_stats": Counter(),
        "score_stats": {"scores": [], "total_docs": 0},
        "int_score_distribution": Counter(),
    }

    # 读取并聚合每个worker的统计
    for stats_file in stats_files:
        with data_folder.open(stats_file, "r") as f:
            worker_stats = json.load(f)

        # 聚合域名统计
        domain_stats = worker_stats.get("domain_stats", {})
        for domain, count in domain_stats.get("top_1000", {}).items():
            aggregated["domain_stats"]["top_1000"][domain] += count
            aggregated["domain_stats"]["total_domains"].add(domain)

        # 聚合快照统计
        for snapshot, info in worker_stats.get("snapshot_stats", {}).items():
            snapshot_key = f"{snapshot}_doc_count"
            aggregated["snapshot_stats"][snapshot_key] += info.get("doc_count", 0)

            snapshot_file_key = f"{snapshot}_file_count"
            aggregated["snapshot_stats"][snapshot_file_key] += info.get("file_count", 0)

        # 聚合score统计
        score_stats = worker_stats.get("score_stats", {})
        aggregated["score_stats"]["scores"].extend(score_stats.get("all_scores", []))
        aggregated["score_stats"]["total_docs"] += score_stats.get("total_docs", 0)

        # 聚合int_score分布
        for int_score, count in worker_stats.get("int_score_distribution", {}).items():
            aggregated["int_score_distribution"][int_score] += count

    # 计算最终统计结果
    final_result = {
        "domain_stats": {
            "total_domains": len(aggregated["domain_stats"]["total_domains"]),
            "top_1000": dict(aggregated["domain_stats"]["top_1000"].most_common(1000)),
        },
        "snapshot_stats": {},
        "score_stats": {},
        "int_score_distribution": dict(aggregated["int_score_distribution"]),
    }

    # 整理快照统计格式
    snapshot_dict = dict(aggregated["snapshot_stats"])
    for key, value in snapshot_dict.items():
        if key.endswith("_doc_count"):
            snapshot_name = key[:-9]  # 移除"_doc_count"后缀
            file_key = f"{snapshot_name}_file_count"
            final_result["snapshot_stats"][snapshot_name] = {
                "doc_count": value,
                "file_count": snapshot_dict.get(file_key, 0),
            }

    # 计算score统计
    scores = aggregated["score_stats"]["scores"]
    if scores:
        scores_array = np.array(scores)
        final_result["score_stats"] = {
            "mean": float(np.mean(scores_array)),
            "median": float(np.median(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "percentiles": {},
            "total_docs": aggregated["score_stats"]["total_docs"],
        }

        # 计算分位数
        for i in range(1, 11):
            percentile = i * 10
            value = float(np.percentile(scores_array, percentile))
            final_result["score_stats"]["percentiles"][f"{percentile}%"] = value

    return final_result


class FinewebEduStatsCollector(PipelineStep):
    """收集 Fineweb-Edu 数据集统计信息的 PipelineStep。

    统计信息包括：
    - 域名分布（Top-1000域名和总域名数）
    - 快照统计（每个快照的文档数和文件数）
    - score分布（summary统计和分位数）
    - int_score分布（计数分布）

    Args:
        output_folder: 统计结果输出目录
    """

    name = "📊 FinewebEduStatsCollector"
    type = "STATS_COLLECTOR"

    def __init__(self, output_folder: str):
        """初始化统计收集器。

        Args:
            output_folder: 统计结果输出目录
        """
        super().__init__()
        self.output_folder = output_folder
        self.data_folder = get_datafolder(output_folder)

        # 每个worker的独立统计状态
        self.domain_counter: Counter = Counter()
        self.snapshot_stats: Dict[str, Dict[str, int]] = {}
        self.scores: List[float] = []
        self.int_score_counter: Counter = Counter()
        self.total_docs: int = 0

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentGenerator:
        """执行统计收集的流水线步骤。

        Args:
            data: 输入文档流
            rank: 当前worker的rank
            world_size: 总worker数量

        Yields:
            原样传递的文档
        """
        for doc in data:
            # 收集统计信息
            with self.track_time():
                self._collect_document_stats(doc)
                self.stat_update("processed_docs")

            # 原样传递文档
            yield doc

        # 处理完成后保存统计结果
        self._save_worker_stats(rank)

    def _collect_document_stats(self, doc: Document) -> None:
        """收集单个文档的统计信息。

        Args:
            doc: 要统计的文档
        """
        self.total_docs += 1

        # 1. 域名统计
        url = doc.metadata.get("url", "")
        if url:
            domain = extract_domain(url)
            self.domain_counter[domain] += 1

        # 2. 快照统计
        dump = doc.metadata.get("dump", "")
        if dump:
            if dump not in self.snapshot_stats:
                self.snapshot_stats[dump] = {"doc_count": 0, "files": set()}
            self.snapshot_stats[dump]["doc_count"] += 1

            # 尝试获取文件信息
            file_name = doc.metadata.get("file_name", "")
            if file_name:
                self.snapshot_stats[dump]["files"].add(file_name)

        # 3. score统计
        score = doc.metadata.get("score")
        if score is not None:
            try:
                score_float = float(score)
                self.scores.append(score_float)
            except (ValueError, TypeError):
                pass

        # 4. int_score统计
        int_score = doc.metadata.get("int_score")
        if int_score is not None:
            try:
                int_score_int = int(int_score)
                self.int_score_counter[str(int_score_int)] += 1
            except (ValueError, TypeError):
                pass

    def _save_worker_stats(self, rank: int) -> None:
        """保存当前worker的统计结果。

        Args:
            rank: 当前worker的rank
        """
        # 计算score统计
        score_stats = {}
        if self.scores:
            scores_array = np.array(self.scores)
            score_stats = {
                "mean": float(np.mean(scores_array)),
                "median": float(np.median(scores_array)),
                "std": float(np.std(scores_array)),
                "min": float(np.min(scores_array)),
                "max": float(np.max(scores_array)),
                "percentiles": {},
                "total_docs": self.total_docs,
                "all_scores": self.scores,  # 保存所有score用于聚合
            }

            # 计算分位数
            for i in range(1, 11):
                percentile = i * 10
                value = float(np.percentile(scores_array, percentile))
                score_stats["percentiles"][f"{percentile}%"] = value
        else:
            score_stats = {"total_docs": self.total_docs, "all_scores": []}

        # 整理快照统计格式
        snapshot_stats_final = {}
        for snapshot_name, snapshot_info in self.snapshot_stats.items():
            snapshot_stats_final[snapshot_name] = {
                "doc_count": snapshot_info["doc_count"],
                "file_count": len(snapshot_info["files"]),
            }

        # 组装最终的统计结果
        worker_stats = {
            "domain_stats": {
                "total_domains": len(self.domain_counter),
                "top_1000": dict(self.domain_counter.most_common(1000)),
            },
            "snapshot_stats": snapshot_stats_final,
            "score_stats": score_stats,
            "int_score_distribution": dict(self.int_score_counter),
            "worker_rank": rank,
            "total_docs_processed": self.total_docs,
        }

        # 保存到文件
        stats_file = f"fineweb_edu_stats/worker_{rank:05d}.json"
        with self.data_folder.open(stats_file, "w") as f:
            json.dump(worker_stats, f, ensure_ascii=False, indent=2)

    @staticmethod
    def aggregate_stats(output_folder: str) -> Optional[StatsDict]:
        """聚合所有worker的统计结果。

        Args:
            output_folder: 统计结果目录

        Returns:
            聚合后的全局统计结果，如果没有找到统计文件则返回None
        """
        data_folder = get_datafolder(output_folder)
        stats_dir = "fineweb_edu_stats"

        # 获取所有worker统计文件
        if not data_folder.isdir(stats_dir):
            return None

        stats_files = []
        for file_path in data_folder.list_files(stats_dir):
            if file_path.endswith(".json"):
                stats_files.append(file_path)

        if not stats_files:
            return None

        # 构建完整的文件路径列表
        full_stats_files = []
        for file_path in stats_files:
            # 直接使用相对路径，因为DataFolder会正确处理
            full_stats_files.append(file_path)

        if not full_stats_files:
            return None

        # 聚合统计结果
        aggregated_stats = aggregate_worker_stats(data_folder, full_stats_files)

        # 保存聚合结果
        aggregated_file = "fineweb_edu_stats/aggregated_stats.json"
        with data_folder.open(aggregated_file, "w") as f:
            json.dump(aggregated_stats, f, ensure_ascii=False, indent=2)

        return aggregated_stats


# 示例使用函数
def demo_usage():
    """演示如何使用 FinewebEduStatsCollector 的示例函数。"""
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.executor.local import LocalPipelineExecutor

    # 创建统计收集器
    stats_collector = FinewebEduStatsCollector("outputs/stats_demo")

    # 创建简单的处理流水线
    pipeline = [JsonlReader("data/sample.jsonl"), stats_collector]

    # 执行流水线
    executor = LocalPipelineExecutor(
        pipeline=pipeline, tasks=1, logging_dir="outputs/stats_demo/logs", workers=1
    )

    # 运行并获取统计结果
    executor.run()

    # 聚合统计结果
    final_stats = FinewebEduStatsCollector.aggregate_stats("outputs/stats_demo")

    if final_stats:
        print("聚合统计结果:")
        print(json.dumps(final_stats, ensure_ascii=False, indent=2))
    else:
        print("未找到统计结果")


if __name__ == "__main__":
    demo_usage()
