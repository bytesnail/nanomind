"""CC-MAIN 路径写入器模块。

根据文档的 cc_main 元数据，将文档写入对应的 CC-MAIN 批次子目录。
"""

from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.writers import ParquetWriter


class CCMainPathWriter(ParquetWriter):
    """CC-MAIN 路径写入器：根据 cc_main 元数据构建输出路径。

    继承标准 ParquetWriter，重写 _get_output_filename 方法实现动态路径。
    输出路径格式：{cc_main}/{rank}.parquet
    """

    def __init__(
        self,
        output_folder: str,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "zstd",
        max_file_size: int = 512 * 1024 * 1024,
    ):
        """初始化 CC-MAIN 路径写入器。

        Args:
            output_folder: 输出文件夹路径
            compression: 压缩格式，默认 zstd
            max_file_size: 单个输出文件最大大小（字节），默认 512MB
        """
        super().__init__(
            output_folder=output_folder,
            compression=compression,
            max_file_size=max_file_size,
        )

    def _get_output_filename(
        self, document: Document, rank: int | str = 0, **kwargs
    ) -> str:
        """获取输出文件名，根据 cc_main 元数据构建路径。

        Args:
            document: 文档对象
            rank: worker 编号
            **kwargs: 额外参数

        Returns:
            str: 输出文件路径
        """
        cc_main = document.metadata.get("cc_main", "unknown")
        rank_str = (
            str(int(rank)).zfill(5) if isinstance(rank, int) else str(rank).zfill(5)
        )
        return f"{cc_main}/{rank_str}.parquet"

    def _default_adapter(self, document: Document) -> dict:
        """重写默认 adapter，将 score 从 metadata 提升到顶层。

        Args:
            document: 文档对象

        Returns:
            dict: 要写入的数据字典，包含 id, text, score 三个顶层字段
        """
        return {
            "text": document.text,
            "id": document.id,
            "score": document.metadata.get("score"),
        }
