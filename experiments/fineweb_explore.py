"""FineWeb-Edu 数据集探索脚本。

目的：
- 使用datatrove对HuggingFaceFW/fineweb-edu数据集进行全量分析
- 支持多worker并发处理和结果聚合
- 生成完整的统计报告

运行命令：
    python experiments/fineweb_explore.py --config configs/fineweb.yaml
    python experiments/fineweb_explore.py --data-dir data/datasets/HuggingFaceFW/fineweb-edu/data --output-dir outputs --workers 8 --batch-size 5000

输出：
- 完整的全量统计数据（JSON）
- 详细的分析日志
- datatrove内置统计文件
"""

# 标准库
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List

# YAML配置
import yaml

# DataTrove
from datatrove.pipeline.readers import ParquetReader

# 自定义组件
from experiments.fineweb_stats_collector import FinewebEduStatsCollector
from experiments.utils.common import (
    setup_logging,
    create_local_executor,
    get_timestamp,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件。

    Args:
        config_path: 配置文件路径

    Returns:
        解析后的配置字典
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    支持YAML配置文件和命令行参数，命令行参数会覆盖配置文件中的值。

    Returns:
        配置好的ArgumentParser实例
    """
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 数据集探索（datatrove优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python experiments/fineweb_explore.py
  python experiments/fineweb_explore.py --config configs/fineweb.yaml
  python experiments/fineweb_explore.py --config configs/fineweb.yaml --workers 16 --batch-size 10000
  python experiments/fineweb_explore.py --limit 100000 --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/fineweb.yaml",
        help="YAML配置文件路径 (默认: configs/fineweb.yaml)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据集路径 (默认: 从配置文件读取)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: 从配置文件读取)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理文档数 (默认: None=全量处理)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行worker数量 (默认: 从配置文件读取)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=None, help="批量大小 (默认: 从配置文件读取)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别 (默认: 从配置文件读取)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run模式，只打印配置而不执行",
    )

    return parser


def parquet_to_doc_adapter(
    reader, data: Dict[str, Any], source_file: str, id_in_file: int
) -> Dict[str, Any]:
    """将parquet行转换为Document所需格式。

    Args:
        reader: ParquetReader实例
        data: parquet单行数据
        source_file: 源文件路径
        id_in_file: 文件中的ID

    Returns:
        转换后的metadata字典
    """
    # Document需要的核心字段
    result = {
        "text": data.get("text"),
        "id": data.get("id"),
    }

    # 添加元数据
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


def setup_output_directories(output_dir: str) -> Dict[str, str]:
    """创建输出目录结构。

    Args:
        output_dir: 输出目录路径

    Returns:
        包含各子目录路径的字典
    """
    directories = {
        "main": output_dir,
        "logs": os.path.join(output_dir, "logs"),
        "results": os.path.join(output_dir, "results"),
    }

    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    return directories


def load_datatrove_stats(output_dir: str) -> Optional[Dict[str, Any]]:
    """加载datatrove统计结果。

    Args:
        output_dir: 输出目录

    Returns:
        合并后的datatrove统计结果，如果没有找到则返回None
    """
    try:
        stats_dir = os.path.join(output_dir, "logs", "stats")
        if not os.path.exists(stats_dir):
            return None

        # 读取所有统计文件
        stats_files = []
        for file in os.listdir(stats_dir):
            if file.endswith(".json"):
                stats_files.append(os.path.join(stats_dir, file))

        if not stats_files:
            return None

        # 初始化聚合结果
        combined_stats = {
            "doc_stats": {
                "total_documents": 0,
                "total_tokens": 0,
                "doc_len": {
                    "total": 0,
                    "mean": 0,
                    "min": 0,
                    "max": 0,
                },
            },
            "lang_stats": {},
        }

        # 读取并合并所有统计文件
        for stats_file in stats_files:
            with open(stats_file, "r", encoding="utf-8") as f:
                file_content = json.load(f)

                # datatrove stats文件是一个数组，每个元素是一个pipeline步骤的统计
                if isinstance(file_content, list):
                    for step_stat in file_content:
                        step_name = step_stat.get("name", "")
                        step_data = step_stat.get("stats", {})

                        # 从ParquetReader提取文档统计
                        if "Parquet" in step_name and isinstance(step_data, dict):
                            doc_len = step_data.get("doc_len", {})
                            doc_len_tokens = step_data.get("doc_len_tokens", {})
                            documents = step_data.get("documents", {})

                            combined_stats["doc_stats"]["total_documents"] += (
                                documents.get("total", 0)
                            )
                            combined_stats["doc_stats"]["total_tokens"] += (
                                doc_len_tokens.get("total", 0)
                            )

                            # 合并文档长度统计
                            combined_stats["doc_stats"]["doc_len"]["total"] += (
                                doc_len.get("total", 0)
                            )
                            combined_stats["doc_stats"]["doc_len"]["mean"] += (
                                doc_len.get("mean", 0)
                            )

                        # 合并语言统计
                        if "Language stats" in step_name and isinstance(
                            step_data, dict
                        ):
                            for lang, count in step_data.items():
                                if lang != "languages" and isinstance(count, dict):
                                    if lang not in combined_stats["lang_stats"]:
                                        combined_stats["lang_stats"][lang] = count.get(
                                            "total", 0
                                        )
                                    else:
                                        combined_stats["lang_stats"][lang] += count.get(
                                            "total", 0
                                        )

        # 计算平均文档长度
        if combined_stats["doc_stats"]["total_documents"] > 0:
            combined_stats["doc_stats"]["doc_len"]["mean"] /= combined_stats[
                "doc_stats"
            ]["total_documents"]

        return combined_stats

    except Exception as e:
        logging.getLogger("fineweb_explore").warning(f"无法加载datatrove统计: {e}")
        return None


def aggregate_results(
    output_dir: str, start_time: str, end_time: str, logger: logging.Logger
) -> Dict[str, Any]:
    """聚合所有统计结果生成最终报告。

    Args:
        output_dir: 输出目录
        start_time: 开始时间
        end_time: 结束时间
        logger: 日志记录器

    Returns:
        聚合后的完整统计结果
    """
    logger.info("开始聚合统计结果...")

    # 1. 加载datatrove内置统计
    datatrove_stats = load_datatrove_stats(output_dir)

    # 2. 加载自定义统计
    custom_stats = FinewebEduStatsCollector.aggregate_stats(output_dir)

    # 3. 构建最终结果
    result = {
        "metadata": {
            "experiment": "FineWeb-Edu 数据集探索（datatrove优化版）",
            "dataset": "HuggingFaceFW/fineweb-edu",
            "start_time": start_time,
            "end_time": end_time,
        },
        "global_statistics": {},
        "snapshot_statistics": {},
        "language_distribution": {},
        "domain_distribution": {},
        "score_statistics": {},
        "int_score_distribution": {},
    }

    # 整合datatrove统计
    if datatrove_stats:
        # 从datatrove统计中提取全局统计
        if "doc_stats" in datatrove_stats:
            doc_stats = datatrove_stats["doc_stats"]
            result["global_statistics"] = {
                "total_documents": doc_stats.get("docs", 0),
                "total_tokens": doc_stats.get("tokens", 0),
                "average_tokens_per_document": (
                    doc_stats.get("tokens", 0) / max(doc_stats.get("docs", 1), 1)
                ),
            }

        # 语言分布
        if "lang_stats" in datatrove_stats:
            result["language_distribution"] = datatrove_stats["lang_stats"]

    # 整合自定义统计
    if custom_stats:
        result["domain_distribution"] = custom_stats.get("domain_stats", {})
        result["snapshot_statistics"] = custom_stats.get("snapshot_stats", {})
        result["score_statistics"] = custom_stats.get("score_stats", {})
        result["int_score_distribution"] = custom_stats.get(
            "int_score_distribution", {}
        )

        # 更新元数据中的处理文件数
        if "total_docs_processed" in custom_stats:
            result["metadata"]["total_files_processed"] = custom_stats[
                "total_docs_processed"
            ]

    return result


def create_custom_pipeline(
    data_dir: str,
    output_dir: str,
    batch_size: int,
    limit: Optional[int],
    logger: logging.Logger,
) -> List:
    """创建自定义datatrove处理流水线。

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        batch_size: 批量大小
        limit: 限制文档数
        logger: 日志记录器

    Returns:
        配置好的pipeline步骤列表
    """
    logger.info("配置自定义datatrove pipeline...")

    pipeline = [
        # 1. ParquetReader - 读取parquet文件
        ParquetReader(
            data_folder=data_dir,
            glob_pattern="**/*.parquet",  # 递归搜索所有parquet文件
            batch_size=batch_size,
            limit=limit if limit is not None else -1,  # -1 表示无限制
            text_key="text",
            id_key="id",
            adapter=parquet_to_doc_adapter,
            file_progress=True,
            doc_progress=True,
        ),
        # 2. FinewebEduStatsCollector - 自定义统计（包含域名分布、快照统计、score/int_score分布）
        FinewebEduStatsCollector(output_folder=output_dir),
    ]

    logger.info(f"pipeline配置完成，共 {len(pipeline)} 个步骤")
    return pipeline


def run_pipeline(
    data_dir: str,
    output_dir: str,
    workers: int,
    batch_size: int,
    limit: Optional[int],
    logger: logging.Logger,
) -> None:
    """运行datatrove pipeline。

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        workers: worker数量
        batch_size: 批量大小
        limit: 限制文档数
        logger: 日志记录器
    """
    logger.info("开始运行datatrove pipeline...")

    # 创建pipeline
    pipeline = create_custom_pipeline(data_dir, output_dir, batch_size, limit, logger)

    # 配置executor
    executor = create_local_executor(
        pipeline=pipeline,
        workers=workers,
        logging_dir=os.path.join(output_dir, "logs"),
        skip_completed=True,  # 支持断点续传
    )

    # 运行pipeline
    try:
        logger.info(f"启动 {workers} 个worker开始处理...")
        executor.run()
        logger.info("pipeline执行完成")
    except Exception as e:
        logger.error(f"pipeline执行失败: {e}")
        raise


def save_final_results(
    results: Dict[str, Any], output_dir: str, logger: logging.Logger
) -> None:
    """保存最终统计结果。

    Args:
        results: 统计结果
        output_dir: 输出目录
        logger: 日志记录器
    """
    results_file = os.path.join(output_dir, "results", "fineweb_full_statistics.json")

    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 最终结果已保存到: {results_file}")
        print(f"\n✅ 最终结果已保存到: {results_file}")

    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        raise


def main() -> None:
    """主函数，执行datatrove数据集探索流程。"""
    # 解析命令行参数
    parser = create_parser()
    args = parser.parse_args()

    # 加载配置文件
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)

    # 合并配置：命令行参数覆盖配置文件
    # 支持嵌套配置结构（如config.data.path）和扁平结构
    data_dir = args.data_dir
    if data_dir is None:
        # 尝试从嵌套结构读取
        if "data" in config and isinstance(config["data"], dict):
            data_dir = config["data"].get(
                "path", "data/datasets/HuggingFaceFW/fineweb-edu/data"
            )
        else:
            data_dir = config.get(
                "data_dir", "data/datasets/HuggingFaceFW/fineweb-edu/data"
            )

    output_dir = args.output_dir
    if output_dir is None:
        if "output" in config and isinstance(config["output"], dict):
            output_dir = config["output"].get("dir", "outputs")
        else:
            output_dir = config.get("output_dir", "outputs")

    workers = args.workers
    if workers is None:
        if "processing" in config and isinstance(config["processing"], dict):
            workers = config["processing"].get("workers", 8)
        else:
            workers = config.get("workers", 8)

    batch_size = args.batch_size
    if batch_size is None:
        if "data" in config and isinstance(config["data"], dict):
            batch_size = config["data"].get("batch_size", 5000)
        else:
            batch_size = config.get("batch_size", 5000)

    log_level = args.log_level
    if log_level is None:
        if "output" in config and isinstance(config["output"], dict):
            log_level = config["output"].get("log_level", "INFO")
        else:
            log_level = config.get("log_level", "INFO")

    limit = args.limit
    if limit is None:
        if "processing" in config and isinstance(config["processing"], dict):
            limit = config["processing"].get("limit", None)
        else:
            limit = config.get("limit", None)

    # Dry run模式
    if args.dry_run:
        print("\n" + "=" * 60)
        print("  Dry run模式 - 配置信息")
        print("=" * 60)
        print(f"  配置文件: {args.config}")
        print(f"  数据目录: {data_dir}")
        print(f"  输出目录: {output_dir}")
        print(f"  Worker数量: {workers}")
        print(f"  批量大小: {batch_size}")
        print(f"  处理限制: {limit if limit else '全量'}")
        print(f"  日志级别: {log_level}")
        print("=" * 60)
        print("  Dry run: pipeline would be executed")
        print("=" * 60 + "\n")
        return

    # 设置输出目录
    dirs = setup_output_directories(output_dir)

    # 设置日志
    timestamp = get_timestamp()
    log_file = os.path.join(dirs["logs"], f"fineweb_explore_{timestamp}.log")
    logger = setup_logging("fineweb_explore", log_level, log_file)

    # 记录开始时间
    start_time = datetime.now().isoformat()
    logger.info("开始执行 FineWeb-Edu 数据集探索（datatrove优化版）")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Worker数量: {workers}")
    logger.info(f"批量大小: {batch_size}")
    logger.info(f"处理限制: {limit if limit else '全量'}")
    logger.info(f"日志级别: {log_level}")

    # 打印配置信息
    print("\n" + "=" * 60)
    print("  FineWeb-Edu 数据集探索（datatrove优化版）")
    print("=" * 60)
    print(f"  配置文件: {args.config}")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  Worker数量: {workers}")
    print(f"  批量大小: {batch_size}")
    print(f"  处理限制: {limit if limit else '全量'}")
    print(f"  日志级别: {log_level}")
    print(f"  开始时间: {start_time}")
    print("=" * 60)

    try:
        # 1. 运行datatrove pipeline
        run_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            workers=workers,
            batch_size=batch_size,
            limit=limit,
            logger=logger,
        )

        # 2. 聚合结果
        end_time = datetime.now().isoformat()
        results = aggregate_results(output_dir, start_time, end_time, logger)

        # 3. 保存最终结果
        save_final_results(results, output_dir, logger)

        # 4. 打印总结
        print("\n" + "=" * 60)
        print("  总结")
        print("=" * 60)
        print("  ✅ 数据集探索完成")
        print(f"  ✅ 结果文件: {output_dir}/results/fineweb_full_statistics.json")
        print(f"  ✅ 日志文件: {log_file}")
        print(f"  ✅ 结束时间: {end_time}")
        print("=" * 60 + "\n")

        logger.info("实验执行完成")

    except Exception as e:
        logger.error(f"实验执行失败: {e}")
        print(f"\n❌ 实验执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
