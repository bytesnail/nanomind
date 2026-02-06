"""FineWeb-Edu 数据适配器模块。

提供将 FineWeb-Edu 原始数据转换为 Datatrove Document 对象的适配器函数，
同时进行字段筛选，只保留必要的字段。
"""

from typing import Any


def fineweb_adapter(
    self, raw_dict: dict, source_file: str, id_in_file: int
) -> dict[str, Any]:
    """将 FineWeb-Edu 原始数据转换为 dict，只保留必要字段。

    从原始 parquet 数据中提取 id、text、score 和 dump 字段，
    其中 dump 字段用于提取 CC-MAIN 批次名称。
    注意：Datatrove Document 类只支持 text, id, media, metadata 字段，
    因此 score 暂时放在 metadata 中，最终输出时通过 _default_adapter 提升到顶层。

    Args:
        self: Reader 实例（Datatrove 传入）
        raw_dict: 原始 parquet 行数据（包含 10 个字段）
        source_file: 源文件路径
        id_in_file: 在文件中的 ID

    Returns:
        dict: 包含 text, id, metadata.score, metadata.cc_main 的字典

    Raises:
        ValueError: 当 text 或 id 字段缺失时
    """
    text = raw_dict.get("text", "")
    doc_id = raw_dict.get("id", "")

    if not text:
        raise ValueError("Missing required field: text")
    if not doc_id:
        raise ValueError("Missing required field: id")

    dump = raw_dict.get("dump", "")
    cc_main = dump if dump.startswith("CC-MAIN-") else "unknown"
    score = raw_dict.get("score", 0.0)

    return {
        "text": text,
        "id": doc_id,
        "metadata": {
            "score": score,
            "cc_main": cc_main,
        },
    }


def fineweb_adapter_safe(
    self, raw_dict: dict, source_file: str, id_in_file: int
) -> dict[str, Any] | None:
    """安全版本的 fineweb_adapter，遇到异常数据返回 None。

    Args:
        self: Reader 实例
        raw_dict: 原始 parquet 行数据
        source_file: 源文件路径
        id_in_file: 在文件中的 ID

    Returns:
        dict | None: 转换后的字典，如果数据无效则返回 None
    """
    try:
        return fineweb_adapter(self, raw_dict, source_file, id_in_file)
    except (ValueError, TypeError, KeyError):
        return None
