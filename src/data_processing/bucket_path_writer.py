from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datatrove.pipeline.base import PipelineStep

from .bucket_config import BucketConfig
from .config_loader import DEFAULT_COMPRESSION, DEFAULT_MAX_FILE_SIZE, Compression

ROW_OVERHEAD_BYTES = 32


class BucketPathWriter(PipelineStep):
    name = "Bucket Writer"
    type = "💾 - WRITER"

    def __init__(
        self,
        output_dir: str,
        buckets: list[BucketConfig],
        compression: Compression = DEFAULT_COMPRESSION,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
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
        text_bytes = len(text.encode("utf-8"))
        id_bytes = len(doc_id.encode("utf-8"))
        return text_bytes + id_bytes + ROW_OVERHEAD_BYTES

    def _flush_bucket(self, name: str) -> None:
        state = self._states[name]
        buffer = state["buffer"]
        if not buffer:
            return

        filepath = (
            self.output_dir / name / f"{self._rank:05d}_{state['counter']:05d}.parquet"
        )

        texts = [doc["text"] for doc in buffer]
        ids = [doc["id"] for doc in buffer]
        scores = [doc["score"] for doc in buffer]

        table = pa.table({"text": texts, "id": ids, "score": scores})
        pq.write_table(table, filepath, compression=self.compression)

        state["counter"] += 1
        state["buffer"] = []
        state["size"] = 0

    def _flush_all(self) -> None:
        for name in self._states:
            self._flush_bucket(name)

    def close(self) -> None:
        self._flush_all()

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
