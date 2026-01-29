"""测试 FinewebEduStatsCollector 统计收集功能。"""

import json
import tempfile

import pytest

from datatrove.data import Document
from experiments.fineweb_stats_collector import FinewebEduStatsCollector


@pytest.fixture
def mock_documents() -> list[Document]:
    """创建模拟文档数据用于测试。

    Returns:
        包含5个模拟文档的列表，每个文档具有不同的url、dump、score等元数据
    """
    # 域名测试数据
    domains = ["example.com", "test.org", "sample.net", "example.com", "test.org"]

    # 快照测试数据
    dumps = [
        "CC-MAIN-2021-21",
        "CC-MAIN-2021-22",
        "CC-MAIN-2021-21",
        "CC-MAIN-2021-22",
        "CC-MAIN-2021-21",
    ]

    # 分数测试数据
    scores = [1.5, 2.3, 3.7, 0.8, 4.2]
    int_scores = [1, 2, 3, 0, 4]

    mock_docs = []
    for i in range(5):
        doc = Document(
            text=f"这是第{i + 1}个测试文档的文本内容。",
            id=f"test_doc_{i + 1}",
            metadata={
                "url": f"https://{domains[i]}/page{i + 1}",
                "dump": dumps[i],
                "score": scores[i],
                "int_score": int_scores[i],
                "file_name": f"test_file_{i % 2 + 1}.parquet",
            },
        )
        mock_docs.append(doc)

    return mock_docs


@pytest.fixture
def mock_config() -> dict:
    """创建模拟配置对象。

    Returns:
        包含测试配置的字典
    """
    return {"output_folder": "", "test_mode": True}


def test_single_worker(mock_documents: list[Document], mock_config: dict) -> None:
    """测试单个worker的统计收集功能。

    Args:
        mock_documents: 模拟文档列表
        mock_config: 模拟配置字典
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建统计收集器
        collector = FinewebEduStatsCollector(temp_dir)

        # 模拟run方法的处理过程
        for doc in mock_documents:
            collector._collect_document_stats(doc)

        # 保存统计结果
        collector._save_worker_stats(rank=0)

        # 检查结果文件
        from datatrove.io import get_datafolder

        data_folder = get_datafolder(temp_dir)
        stats_file = "fineweb_edu_stats/worker_00000.json"

        # 断言：统计文件存在
        assert data_folder.isfile(stats_file), "统计文件未生成"

        with data_folder.open(stats_file, "r") as f:
            stats = json.load(f)

        # 断言：验证统计数据
        assert stats["total_docs_processed"] == 5, "处理的文档数不正确"
        assert stats["domain_stats"]["total_domains"] == 3, "域名数量不正确"
        assert len(stats["snapshot_stats"]) == 2, "快照数量不正确"
        assert stats["score_stats"]["total_docs"] == 5, "score统计文档数不正确"
        assert len(stats["int_score_distribution"]) == 5, "int_score分布不正确"


def test_multi_worker_aggregation(mock_documents: list[Document]) -> None:
    """测试多worker聚合功能。

    Args:
        mock_documents: 模拟文档列表
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # 模拟多个worker的统计文件
        for rank in range(3):
            collector = FinewebEduStatsCollector(temp_dir)

            # 为每个worker分配不同的模拟数据
            for i, doc in enumerate(mock_documents):
                # 修改metadata以区分不同的worker
                modified_doc = Document(
                    text=doc.text,
                    id=doc.id,
                    metadata={
                        **doc.metadata,
                        "url": doc.metadata["url"].replace(
                            "example.com", f"worker{rank}.example.com"
                        ),
                    },
                )
                collector._collect_document_stats(modified_doc)

            collector._save_worker_stats(rank=rank)

        # 聚合统计结果
        aggregated = FinewebEduStatsCollector.aggregate_stats(temp_dir)

        # 断言：聚合成功
        assert aggregated is not None, "聚合失败"

        # 断言：验证聚合结果
        assert aggregated["domain_stats"]["total_domains"] == 5, "总域名数不正确"
        assert len(aggregated["snapshot_stats"]) == 2, "快照数量不正确"
        assert aggregated["score_stats"]["total_docs"] == 15, "总文档数不正确"
        assert abs(aggregated["score_stats"]["mean"] - 2.5) < 0.1, "Score均值不正确"
        assert len(aggregated["int_score_distribution"]) == 5, "int_score分布不正确"


def test_data_structure_format(mock_documents: list[Document]) -> None:
    """测试输出数据格式是否符合要求。

    Args:
        mock_documents: 模拟文档列表
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        collector = FinewebEduStatsCollector(temp_dir)

        for doc in mock_documents:
            collector._collect_document_stats(doc)

        collector._save_worker_stats(rank=0)

        # 获取单个worker的统计
        from datatrove.io import get_datafolder

        data_folder = get_datafolder(temp_dir)
        stats_file = "fineweb_edu_stats/worker_00000.json"

        with data_folder.open(stats_file, "r") as f:
            stats = json.load(f)

        # 检查必需的字段
        required_fields = [
            "domain_stats",
            "snapshot_stats",
            "score_stats",
            "int_score_distribution",
        ]

        for field in required_fields:
            assert field in stats, f"缺少必需字段: {field}"

        # 检查domain_stats结构
        assert "total_domains" in stats["domain_stats"], (
            "domain_stats缺少total_domains字段"
        )
        assert "top_1000" in stats["domain_stats"], "domain_stats缺少top_1000字段"

        # 检查score_stats结构
        score_stats = stats["score_stats"]
        required_score_fields = ["mean", "median", "std", "min", "max", "total_docs"]
        for field in required_score_fields:
            assert field in score_stats, f"score_stats缺少{field}字段"

        # 检查聚合格式
        aggregated = FinewebEduStatsCollector.aggregate_stats(temp_dir)
        assert aggregated is not None, "聚合格式检查失败"
