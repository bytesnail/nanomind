"""FinewebEduStatsCollector API 用法示例

本脚本展示如何使用 FinewebEduStatsCollector 进行数据统计收集的快速示例。
仅处理100个文档用于快速验证API功能。
"""

from pathlib import Path
import sys
from pathlib import Path as PathLib
from typing import List

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.base import PipelineStep

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from experiments.fineweb_stats_collector import FinewebEduStatsCollector
from experiments.utils.common import create_local_executor


def create_pipeline(
    data_dir: str,
    output_dir: str,
    sample_size: int = 100,
) -> List[PipelineStep]:
    """创建处理流水线（简化版）。

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        sample_size: 采样文档数量

    Returns:
        Pipeline步骤列表
    """
    return [
        ParquetReader(
            data_folder=data_dir,
            limit=sample_size,
        ),
        FinewebEduStatsCollector(output_dir),
    ]


def run_fineweb_stats_collection() -> bool:
    """运行 Fineweb-Edu 统计收集的快速示例。

    Returns:
        bool: 成功返回True，失败返回False
    """
    print("🚀 FinewebEduStatsCollector 示例演示")

    data_dir = "data/datasets/HuggingFaceFW/fineweb-edu/data"
    output_dir = "outputs/fineweb_edu_stats"

    if not Path(data_dir).exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False

    pipeline = create_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        sample_size=100,
    )

    executor = create_local_executor(
        pipeline=pipeline,
        workers=1,
        logging_dir=f"{output_dir}/logs",
        skip_completed=False,  # 每次都重新处理，不使用缓存
    )

    try:
        executor.run()
        final_stats = FinewebEduStatsCollector.aggregate_stats(output_dir)

        if final_stats:
            print("✅ 统计收集完成!")
            print(f"📊 总文档数: {final_stats['score_stats']['total_docs']}")
            print(f"📊 总域名数: {final_stats['domain_stats']['total_domains']}")
            print(
                f"💾 结果已保存到: {output_dir}/fineweb_edu_stats/aggregated_stats.json"
            )
            return True
        return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False


if __name__ == "__main__":
    run_fineweb_stats_collection()
