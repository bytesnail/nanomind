"""实验 001: FineWeb-Edu 数据集统计与探索。

本脚本合并了以下原始功能：
- fineweb_stats_collector.py: 自定义 PipelineStep 收集统计信息
- fineweb_explore.py: 完整数据集探索脚本
- run_fineweb.py: 统一命令行入口

目的：
- 使用 datatrove 对 HuggingFaceFW/fineweb-edu 数据集进行分析
- 收集快照统计、score 分布和 int_score 分布
- 支持多 worker 并发处理和结果聚合
- 生成完整的统计报告

运行命令：
    python experiments/exp_001_datasets_stats.py explore --data-dir <path> --workers 8
    python experiments/exp_001_datasets_stats.py quick --data-dir <path> --limit 10000

注意：
- 脚本使用命令行参数，不依赖配置文件
- 使用 --help 查看所有可用参数和默认值
"""

# 标准库
import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# YAML 配置和统计计算
import yaml
import numpy as np

# DataTrove
from datatrove.data import Document, DocumentsPipeline
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


# ============================================================================
# 统计收集器核心实现
# ============================================================================


def compute_score_stats(scores: List[float], total_docs: int = 0) -> Dict[str, Any]:
    """计算 score 的统计信息（mean, median, std, min, max, percentiles）。"""
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
        percentile = i * 10
        result["percentiles"][f"{percentile}%"] = float(
            np.percentile(scores_array, percentile)
        )

    return result


def aggregate_worker_stats(
    data_folder: DataFolder, stats_files: List[str]
) -> StatsDict:
    """聚合多个 worker 的统计结果。"""
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


class FinewebEduStatsCollector(PipelineStep):
    """收集 Fineweb-Edu 数据集统计信息的 PipelineStep。"""

    name = "📊 FinewebEduStatsCollector"
    type = "STATS_COLLECTOR"

    def __init__(self, output_folder: str) -> None:
        super().__init__()
        self.output_folder: str = output_folder
        self.data_folder: DataFolder = get_datafolder(output_folder)
        self.snapshot_stats: Dict[str, Dict[str, Any]] = {}
        self.scores: List[float] = []
        self.int_score_counter: Counter = Counter()
        self.total_docs: int = 0

    def run(
        self, data: Any, rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """执行统计收集的流水线步骤。"""
        for doc in data:
            with self.track_time():
                self._collect_document_stats(doc)
                self.stat_update("processed_docs")
            yield doc

        self._save_worker_stats(rank)

    def _collect_document_stats(self, doc: Document) -> None:
        """收集单个文档的统计信息。"""
        self.total_docs += 1

        dump = doc.metadata.get("dump", "")
        if dump:
            if dump not in self.snapshot_stats:
                self.snapshot_stats[dump] = {"doc_count": 0, "files": set()}
            self.snapshot_stats[dump]["doc_count"] += 1
            file_name = doc.metadata.get("file_name", "")
            if file_name:
                self.snapshot_stats[dump]["files"].add(file_name)

        score = doc.metadata.get("score")
        if score is not None:
            try:
                self.scores.append(float(score))
            except (ValueError, TypeError):
                pass

        int_score = doc.metadata.get("int_score")
        if int_score is not None:
            try:
                self.int_score_counter[str(int(int_score))] += 1
            except (ValueError, TypeError):
                pass

    def _save_worker_stats(self, rank: int) -> None:
        """保存当前 worker 的统计结果。"""
        score_stats = compute_score_stats(self.scores, self.total_docs)

        snapshot_stats_final: Dict[str, Dict[str, Any]] = {}
        for snapshot_name, snapshot_info in self.snapshot_stats.items():
            files = snapshot_info.get("files", set())
            snapshot_stats_final[snapshot_name] = {
                "doc_count": snapshot_info["doc_count"],
                "file_count": len(files),
            }

        worker_stats = {
            "snapshot_stats": snapshot_stats_final,
            "score_stats": score_stats,
            "int_score_distribution": dict(self.int_score_counter),
            "worker_rank": rank,
            "total_docs_processed": self.total_docs,
        }

        stats_file = f"fineweb_edu_stats/worker_{rank:05d}.json"
        with self.data_folder.open(stats_file, "w") as f:
            json.dump(worker_stats, f, ensure_ascii=False, indent=2)

    @staticmethod
    def aggregate_stats(output_folder: str) -> Optional[StatsDict]:
        """聚合所有 worker 的统计结果。"""
        data_folder = get_datafolder(output_folder)
        stats_dir = "fineweb_edu_stats"

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
        aggregated_file = "fineweb_edu_stats/aggregated_stats.json"
        with data_folder.open(aggregated_file, "w") as f:
            json.dump(aggregated_stats, f, ensure_ascii=False, indent=2)

        return aggregated_stats


# ============================================================================
# Pipeline 配置和执行
# ============================================================================


def parquet_to_doc_adapter(
    reader, data: Dict[str, Any], source_file: str, id_in_file: int
) -> Dict[str, Any]:
    """将 parquet 行转换为 Document 所需格式。"""
    result = {
        "text": data.get("text"),
        "id": data.get("id"),
    }
    result["metadata"] = {
        "dump": data.get("dump"),
        "url": data.get("url"),
        "file_path": source_file,
        "language": data.get("language"),
        "language_score": data.get("language_score"),
        "token_count": data.get("token_count"),
        "score": data.get("score"),
        "int_score": data.get("int_score"),
    }
    return result


def create_pipeline(
    data_dir: str,
    output_dir: str,
    batch_size: int,
    limit: Optional[int],
) -> List:
    """创建 datatrove 处理流水线。"""
    return [
        ParquetReader(
            data_folder=data_dir,
            glob_pattern="**/*.parquet",
            batch_size=batch_size,
            limit=limit if limit is not None else -1,
            text_key="text",
            id_key="id",
            adapter=parquet_to_doc_adapter,
            file_progress=True,
            doc_progress=True,
        ),
        FinewebEduStatsCollector(output_folder=output_dir),
    ]


def run_pipeline(
    data_dir: str,
    output_dir: str,
    workers: int,
    batch_size: int,
    limit: Optional[int],
    logger: logging.Logger,
) -> None:
    """运行 datatrove pipeline。"""
    pipeline = create_pipeline(data_dir, output_dir, batch_size, limit)
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
    """保存最终统计结果。"""
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    results_file = os.path.join(output_dir, "results", "fineweb_stats.json")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 结果已保存到: {results_file}")
    print(f"\n✅ 结果已保存到: {results_file}")


# ============================================================================
# 命令行接口
# ============================================================================


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 数据集统计与探索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 全局参数
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出模式")

    # 子命令
    subparsers = parser.add_subparsers(dest="subcommand", title="子命令", required=True)

    # explore 子命令
    explore_parser = subparsers.add_parser("explore", help="完整数据集探索")
    explore_parser.add_argument(
        "--data-dir", type=str, required=True, help="数据集路径"
    )
    explore_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exp_001_datasets_stats",
        help="输出目录",
    )
    explore_parser.add_argument("--workers", type=int, default=8, help="Worker 数量")
    explore_parser.add_argument("--batch-size", type=int, default=5000, help="批量大小")
    explore_parser.add_argument("--dry-run", action="store_true", help="演示模式")

    # quick 子命令
    quick_parser = subparsers.add_parser("quick", help="快速统计")
    quick_parser.add_argument("--data-dir", type=str, required=True, help="数据集路径")
    quick_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exp_001_datasets_stats",
        help="输出目录",
    )
    quick_parser.add_argument("--limit", type=int, default=10000, help="限制文档数")
    quick_parser.add_argument("--workers", type=int, default=8, help="Worker 数量")
    quick_parser.add_argument("--batch-size", type=int, default=5000, help="批量大小")
    quick_parser.add_argument("--dry-run", action="store_true", help="演示模式")

    return parser


def run_explore_or_quick(
    args: argparse.Namespace,
    logger: logging.Logger,
    mode: str,
) -> None:
    """执行 explore 或 quick 子命令的统一处理逻辑。"""
    data_dir = args.data_dir
    output_dir = args.output_dir
    workers = args.workers
    batch_size = args.batch_size
    limit = None if mode == "explore" else args.limit

    # Dry run 模式
    if hasattr(args, "dry_run") and args.dry_run:
        print("\n" + "=" * 60)
        print(f"  Dry run: {mode} 子命令")
        print("=" * 60)
        print(f"  数据目录: {data_dir}")
        print(f"  输出目录: {output_dir}")
        print(f"  Worker 数量: {workers}")
        print(f"  批量大小: {batch_size}")
        print(f"  处理限制: {limit if limit else '全量'}")
        print("=" * 60 + "\n")
        return

    # 设置输出目录和日志
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = get_timestamp()
    log_file = os.path.join(log_dir, f"exp_001_datasets_stats_{timestamp}.log")
    logger = setup_logging(
        "exp_001_datasets_stats", "DEBUG" if args.verbose else "INFO", log_file
    )

    # 打印配置信息
    print("\n" + "=" * 60)
    print(f"  FineWeb-Edu {mode}")
    print("=" * 60)
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  Worker 数量: {workers}")
    print(f"  批量大小: {batch_size}")
    print(f"  处理限制: {limit if limit else '全量'}")
    print("=" * 60)

    # 运行 pipeline
    start_time = datetime.now().isoformat()
    try:
        run_pipeline(data_dir, output_dir, workers, batch_size, limit, logger)

        # 聚合结果
        custom_stats = FinewebEduStatsCollector.aggregate_stats(output_dir)
        end_time = datetime.now().isoformat()

        results = {
            "metadata": {
                "experiment": f"FineWeb-Edu 数据集{mode}",
                "dataset": "HuggingFaceFW/fineweb-edu",
                "start_time": start_time,
                "end_time": end_time,
            },
            "snapshot_statistics": custom_stats.get("snapshot_stats", {}),
            "score_statistics": custom_stats.get("score_stats", {}),
            "int_score_distribution": custom_stats.get("int_score_distribution", {}),
        }

        save_results(results, output_dir, logger)

        print("\n" + "=" * 60)
        print("  总结")
        print("=" * 60)
        print("  ✅ 数据集统计完成")
        print(f"  ✅ 结果文件: {output_dir}/results/fineweb_stats.json")
        print(f"  ✅ 结束时间: {end_time}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"\n❌ 执行失败: {e}")
        sys.exit(1)


def main() -> None:
    """主函数，执行命令行入口流程。"""
    parser = create_parser()
    args = parser.parse_args()

    logger = logging.getLogger("exp_001_datasets_stats")
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    if args.subcommand == "explore":
        run_explore_or_quick(args, logger, "explore")
    elif args.subcommand == "quick":
        run_explore_or_quick(args, logger, "quick")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
