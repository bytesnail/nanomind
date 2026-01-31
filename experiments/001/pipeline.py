"""Pipeline 创建和执行模块。"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader

from collector import DatasetStatsCollector
from config import DatasetConfig
from experiments.utils.common import create_local_executor


def parquet_to_doc_adapter(
        config: DatasetConfig,
        data: Dict[str, Any],
        source_file: str,
) -> Dict[str, Any]:
    """将 parquet 行转换为 Document 所需格式。

    Args:
        config: 数据集配置对象。
        data: parquet 行数据。
        source_file: 源文件路径。

    Returns:
        转换后的文档字典。
    """
    text_value = data.get(config.text_key)
    id_value = data.get(config.id_key) if config.id_key else None

    metadata = {}
    for key, value in data.items():
        if key != config.text_key and (config.id_key is None or key != config.id_key):
            metadata[key] = value

    if source_file:
        metadata["file_path"] = source_file

    result = {
        "text": text_value,
        "id": id_value,
        "metadata": metadata,
    }
    return result


def create_adapter(
        config: DatasetConfig,
) -> Callable[[Dict[str, Any], str], Dict[str, Any]]:
    """创建配置好的适配器函数。

    Args:
        config: 数据集配置对象。

    Returns:
        配置好的适配器函数。
    """

    def adapter(
            data: Dict[str, Any],
            source_file: str,
    ) -> Dict[str, Any]:
        """适配器函数，调用 parquet_to_doc_adapter 进行转换。"""
        return parquet_to_doc_adapter(config, data, source_file)

    return adapter


def create_stats_collector(
        config: DatasetConfig, output_folder: str
) -> DatasetStatsCollector:
    """创建配置好的统计收集器。

    Args:
        config: 数据集配置对象。
        output_folder: 输出文件夹路径。

    Returns:
        配置好的统计收集器实例。
    """
    return DatasetStatsCollector(config=config, output_folder=output_folder)


def create_pipeline(
        config: DatasetConfig,
        output_dir: str,
        batch_size: int,
        limit: Optional[int],
) -> List[PipelineStep]:
    """创建 datatrove 处理流水线。

    Args:
        config: 数据集配置对象。
        output_dir: 输出目录路径。
        batch_size: 批量大小。
        limit: 文档数量限制。

    Returns:
        datatrove 流水线步骤列表。
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

    Args:
        config: 数据集配置对象。
        output_dir: 输出目录路径。
        workers: worker 数量。
        batch_size: 批量大小。
        limit: 文档数量限制。
        logger: 日志记录器。
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
