"""实验 001: 数据集统计与探索。

分析数据集的统计信息（文档计数、Score分布等）。

运行命令：
    python experiments/001/exp_001_datasets_stats.py --dataset HuggingFaceFW/fineweb-edu --data-dir <path> --workers 8
    python experiments/001/exp_001_datasets_stats.py --dataset all --workers 8
"""

import argparse

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent.parent.resolve()
exp_dir = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.utils.paths import setup_experiment_paths

setup_experiment_paths(__file__)

from experiments.utils.common import get_timestamp

try:
    from .cli import create_parser, _prepare_datasets
except ImportError:
    from cli import create_parser, _prepare_datasets
from config import get_dataset_config
from pipeline import run_pipeline
from collector import DatasetStatsCollector
from io_utils import save_results
from experiments.utils.common import print_section, print_separator, print_status

# Backward compatibility aliases
_print_section = print_section
_print_separator = print_separator
_print_status = print_status

StatsDict = Dict[str, Any]


def _print_dataset_info(config, dataset_output_dir: str) -> None:
    """打印数据集信息。

    Args:
        config: 数据集配置对象。
        dataset_output_dir (str): 数据集输出目录。
    """
    _print_section("处理数据集", config.name)
    print(f"  路径: {config.path}")
    print(f"  输出目录: {dataset_output_dir}")
    _print_separator()


def _build_results_dict(
    config, custom_stats: StatsDict, start_time: str, end_time: str
) -> Dict[str, Any]:
    """构建结果字典。

    Args:
        config: 数据集配置对象。
        custom_stats (StatsDict): 自定义统计信息。
        start_time (str): 开始时间。
        end_time (str): 结束时间。

    Returns:
        Dict[str, Any]: 完整的结果字典。
    """
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


def _print_dry_run_summary(dataset_names: List[str]) -> None:
    """打印 dry run 模式配置摘要。

    Args:
        dataset_names (List[str]): 数据集名称列表。
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
    dataset_name: str, args: argparse.Namespace, dataset_output_dir: str
) -> None:
    """处理单个数据集。

    Args:
        dataset_name (str): 数据集名称。
        args (argparse.Namespace): 命令行参数。
        dataset_output_dir (str): 数据集输出目录。
    """
    config = get_dataset_config(dataset_name)
    _print_dataset_info(config, dataset_output_dir)

    os.makedirs(dataset_output_dir, exist_ok=True)

    start_time = datetime.now().isoformat()
    try:
        run_pipeline(
            config,
            dataset_output_dir,
            args.workers,
            args.batch_size,
            args.limit,
        )

        custom_stats = DatasetStatsCollector.aggregate_stats(dataset_output_dir, config)
        end_time = datetime.now().isoformat()

        if custom_stats:
            results = _build_results_dict(config, custom_stats, start_time, end_time)
            save_results(results, dataset_output_dir)

            _print_section("总结")
            print("  数据集统计完成")
            print(f"  结果文件: {dataset_output_dir}/results/dataset_stats.json")
            print(f"  结束时间: {end_time}")
            _print_separator()
            print()
        else:
            print(f"⚠️ 数据集 {config.name} 没有生成统计结果")

    except Exception as e:
        print(f"❌ 处理数据集 {config.name} 失败: {e}")
        _print_status(f"处理数据集 {config.name} 失败: {e}", "error")


def run_experiment(args: argparse.Namespace) -> None:
    """执行多数据集处理。

    Args:
        args (argparse.Namespace): 命令行参数。
    """
    dataset_names = _prepare_datasets(args)

    if args.dry_run:
        _print_dry_run_summary(dataset_names)
        return

    for dataset_name in dataset_names:
        dataset_output_dir = os.path.join(
            args.output_dir, dataset_name.replace("/", "_")
        )
        _process_single_dataset(dataset_name, args, dataset_output_dir)

    _print_section("全部完成")
    print(f"  已处理 {len(dataset_names)} 个数据集")
    _print_separator()
    print()


def main() -> None:
    """主入口函数。

    解析命令行参数并执行实验。
    """
    parser = create_parser()
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
