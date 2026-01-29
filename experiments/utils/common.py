"""实验工具模块 - 通用函数。

提供日志设置、datatrove pipeline创建、executor配置等通用功能。
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.stats import DocStats, LangStats


def get_timestamp() -> str:
    """获取当前时间戳字符串。

    Returns:
        格式化的时间戳字符串，格式为 YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(
    exp_name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """设置日志记录器。

    支持控制台和可选的文件输出。

    Args:
        exp_name: 实验名称，用作logger名称
        log_level: 日志级别，如 "INFO", "DEBUG", "WARNING", "ERROR"
        log_file: 可选的日志文件路径，如果为None则只输出到控制台

    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(exp_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    # 格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器（如果提供）
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(getattr(logging, log_level.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper()))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def create_datatrove_pipeline(
    data_dir: str,
    output_dir: str,
    include_doc_stats: bool = True,
    include_lang_stats: bool = True,
    custom_step: Optional[PipelineStep] = None,
) -> List[PipelineStep]:
    """创建datatrove处理流水线。

    包含ParquetReader和可选的统计步骤。

    Args:
        data_dir: 数据目录，包含parquet文件
        output_dir: 输出目录，用于存储统计结果
        include_doc_stats: 是否包含文档级统计（DocStats）
        include_lang_stats: 是否包含语言统计（LangStats）
        custom_step: 可选的自定义pipeline步骤，如果提供会添加到pipeline末尾

    Returns:
        配置好的pipeline步骤列表
    """
    pipeline: List[PipelineStep] = [
        # ParquetReader - 读取parquet文件
        ParquetReader(
            data_folder=data_dir,
            glob_pattern="**/*.parquet",  # 递归搜索所有parquet文件
            text_key="text",
            id_key="id",
            file_progress=True,
            doc_progress=True,
        ),
    ]

    # 可选的文档级统计
    if include_doc_stats:
        pipeline.append(
            DocStats(
                output_folder=output_dir,
            )
        )

    # 可选的语言统计
    if include_lang_stats:
        pipeline.append(
            LangStats(
                language="language",  # 使用metadata中的language字段
                output_folder=output_dir,
            )
        )

    # 可选的自定义步骤
    if custom_step is not None:
        pipeline.append(custom_step)

    return pipeline


def create_local_executor(
    pipeline: List[PipelineStep],
    workers: int,
    logging_dir: str,
    skip_completed: bool = True,
) -> LocalPipelineExecutor:
    """创建本地pipeline执行器。

    Args:
        pipeline: pipeline步骤列表
        workers: 并行worker数量
        logging_dir: 日志输出目录
        skip_completed: 是否跳过已完成的任务（支持断点续传）

    Returns:
        配置好的LocalPipelineExecutor实例
    """
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        workers=workers,
        logging_dir=logging_dir,
        skip_completed=skip_completed,
    )

    return executor
