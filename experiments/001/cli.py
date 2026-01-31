"""命令行参数解析模块。"""

import argparse
import logging
from typing import List

from config import get_dataset_config, get_datasets_with_score


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    Returns:
        配置好的参数解析器。
    """
    parser = argparse.ArgumentParser(
        description="多数据集统计与探索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出模式")

    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        required=True,
        help="数据集名称（支持多个或 'all' 表示所有有 score 的数据集）",
    )

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
        args: 命令行参数命名空间。
        logger: 日志记录器。

    Returns:
        数据集名称列表。
    """
    if "all" in args.dataset:
        dataset_names = get_datasets_with_score()
        logger.info(f"处理所有有 score 的数据集: {dataset_names}")
    else:
        dataset_names = args.dataset
        dataset_names = [
            name
            for name in dataset_names
            if get_dataset_config(name).score_field is not None
        ]
        logger.info(f"处理数据集: {dataset_names}")

    return dataset_names
