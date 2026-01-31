"""实验工具模块 - 通用函数。"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.stats import DocStats, LangStats

from .constants import SEPARATOR_WIDTH, GB_FACTOR, MB_FACTOR

# Status emoji mapping for formatted output
_STATUS_EMOJIS = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
}


def print_separator(char: str = "=", length: int = SEPARATOR_WIDTH) -> None:
    """打印分隔线。

    Args:
        char: 分隔线字符。
        length: 分隔线长度。
    """
    print(char * length)


def print_section(
    title: str, subtitle: str = "", data: Optional[Dict[str, Any]] = None
) -> None:
    """打印带标题的部分。

    Args:
        title: 标题。
        subtitle: 副标题。
        data: 要显示的数据字典。
    """
    print_separator()
    if subtitle:
        print(f"  {title}: {subtitle}")
    else:
        print(f"  {title}")
    print_separator()
    if data:
        for key, value in data.items():
            print(f"  {key:25s}: {value}")


def print_status(message: str, status: str = "info") -> None:
    """打印带状态图标的消息。

    Args:
        message: 消息文本。
        status: 状态类型（"success", "error", "warning", "info"）。
    """
    emoji = _STATUS_EMOJIS.get(status, "")
    print(f"{emoji} {message}")


def format_bytes(size_bytes: int) -> str:
    """将字节数转换为可读格式。

    Args:
        size_bytes: 字节数。

    Returns:
        格式化后的字符串（例如："1.50 GB"）。
    """
    if size_bytes >= GB_FACTOR:
        return f"{size_bytes / GB_FACTOR:.2f} GB"
    elif size_bytes >= MB_FACTOR:
        return f"{size_bytes / MB_FACTOR:.2f} MB"
    return f"{size_bytes} bytes"


def format_percent(value: float, total: int) -> str:
    """计算百分比并格式化。

    Args:
        value: 数值。
        total: 总数。

    Returns:
        格式化后的百分比字符串（例如："75.5%"）。
    """
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def get_timestamp() -> str:
    """获取当前时间戳字符串。

    Returns:
        格式为 "YYYYMMDD_HHMMSS" 的时间戳字符串。
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_datatrove_pipeline(
    data_dir: str,
    output_dir: str,
    include_doc_stats: bool = True,
    include_lang_stats: bool = True,
    custom_step: Optional[PipelineStep] = None,
) -> List[PipelineStep]:
    """创建 datatrove 处理流水线。

    Args:
        data_dir: 数据目录。
        output_dir: 输出目录。
        include_doc_stats: 是否包含文档统计。
        include_lang_stats: 是否包含语言统计。
        custom_step: 自定义处理步骤。

    Returns:
        datatrove 流水线步骤列表。
    """
    pipeline: List[PipelineStep] = [
        ParquetReader(
            data_folder=data_dir,
            glob_pattern="**/*.parquet",
            text_key="text",
            id_key="id",
            file_progress=True,
            doc_progress=True,
        ),
    ]

    if include_doc_stats:
        pipeline.append(
            DocStats(
                output_folder=output_dir,
            )
        )

    if include_lang_stats:
        pipeline.append(
            LangStats(
                language="language",
                output_folder=output_dir,
            )
        )

    if custom_step is not None:
        pipeline.append(custom_step)

    return pipeline


def create_local_executor(
    pipeline: List[PipelineStep],
    workers: int,
    logging_dir: str,
    skip_completed: bool = True,
) -> LocalPipelineExecutor:
    """创建本地 pipeline 执行器。

    Args:
        pipeline: datatrove 流水线步骤列表。
        workers: 工作进程数。
        logging_dir: 日志目录。
        skip_completed: 是否跳过已完成的任务。

    Returns:
        本地 pipeline 执行器实例。
    """
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        workers=workers,
        logging_dir=logging_dir,
        skip_completed=skip_completed,
    )

    return executor
