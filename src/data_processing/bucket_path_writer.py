from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from datatrove.pipeline.base import PipelineStep

from .bucket_config import BucketConfig


class BucketPathWriter(PipelineStep):
    def __init__(
        self,
        output_dir: str,
        buckets: list[BucketConfig],
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "zstd",
        max_file_size: int = 512 * 1024 * 1024,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.max_file_size = max_file_size
        self._states: dict[str, dict] = {
            b.name: {"buffer": [], "counter": 0, "size": 0} for b in buckets
        }
        self._rank = 0

        for b in buckets:
            (self.output_dir / b.name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _estimate_row_size(text: str, doc_id: str) -> int:
        return len(text) * 2 + len(doc_id) * 2 + 32

    def _flush_bucket(self, name: str) -> None:
        state = self._states[name]
        buffer = state["buffer"]
        if not buffer:
            return

        filepath = (
            self.output_dir / name / f"{self._rank:05d}_{state['counter']:05d}.parquet"
        )
        pq.write_table(
            pa.table({k: [d[k] for d in buffer] for k in ("text", "id", "score")}),
            filepath,
            compression=self.compression,
        )

        state["counter"] += 1
        state["buffer"] = []
        state["size"] = 0

    def _flush_all(self) -> None:
        for name in self._states:
            self._flush_bucket(name)

    def run(self, data, rank: int = 0, world_size: int = 1) -> None:
        self._rank = rank

        for doc in data:
            name = doc.metadata.get("__target_bucket")
            if name not in self._states:
                self.stat_update("missing_bucket_tag", value=1)
                continue

            row_size = self._estimate_row_size(doc.text, doc.id)
            state = self._states[name]
            if state["size"] + row_size > self.max_file_size and state["buffer"]:
                self._flush_bucket(name)

            state["buffer"].append(
                {"text": doc.text, "id": doc.id, "score": doc.metadata.get("score")}
            )
            state["size"] += row_size
            self.stat_update(f"written_{name}", value=1)

        self._flush_all()

    close = _flush_all
