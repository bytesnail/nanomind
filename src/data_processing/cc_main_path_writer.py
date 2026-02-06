"""CC-MAIN 路径写入器。"""

from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.writers import ParquetWriter


class CCMainPathWriter(ParquetWriter):
    def __init__(
        self,
        output_folder: str,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "zstd",
        max_file_size: int = 512 * 1024 * 1024,
    ):
        super().__init__(
            output_folder=output_folder,
            compression=compression,
            max_file_size=max_file_size,
        )

    def _get_output_filename(self, document: Document, rank: int | str = 0, **_) -> str:
        cc_main = document.metadata.get("cc_main", "unknown")
        rank_str = str(int(rank) if isinstance(rank, int) else rank).zfill(5)
        return f"{cc_main}/{rank_str}.parquet"

    def _default_adapter(self, document: Document) -> dict:
        return {
            "text": document.text,
            "id": document.id,
            "score": document.metadata.get("score"),
        }
