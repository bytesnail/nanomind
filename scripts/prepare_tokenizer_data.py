#!/usr/bin/env python3
"""基于 Datatrove 的 Tokenizer 训练数据准备脚本 (优化版 v4).

核心优化:
1. 两遍处理 - 第一遍计算采样索引，第二遍流式读取
2. 内存 O(target_count * 16 bytes) - 只存储索引，不存储文档内容
3. 正确利用 Datatrove 的并行架构 - 采样在 pipeline 外部完成
4. 针对 16 核/250GB/400MB/s 配置优化默认参数

v4 更新:
- 输出文件名格式: {dataset_name}-{bucket_name}-{counter:05d}-rank-{rank:05d}.parquet
- id 列格式: {完整路径}#{index}，与 fineweb_adapter 保持一致
- 移除 prefix 和 dataset_source 参数

设计原则:
- 采样计算与数据读取分离
- 不在 pipeline 内部累积文档对象
- 确定性采样保证可重复性
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import heapq
import json
import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from tqdm import tqdm

# 启用 jemalloc 以解决 Linux ptmalloc2 内存泄漏问题
# 参考: https://github.com/huggingface/datatrove/issues/347
_JEMALLOC_PATH = "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
if os.path.exists(_JEMALLOC_PATH) and "LD_PRELOAD" not in os.environ:
    os.environ.setdefault("LD_PRELOAD", _JEMALLOC_PATH)

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/tokenizer_data.yaml")

CPU_COUNT = os.cpu_count() or 16
DEFAULT_WORKERS = min(16, CPU_COUNT)
DEFAULT_TASKS = -1
DEFAULT_MAX_ROWS = 500_000
DEFAULT_BATCH_SIZE = 50_000
COMPRESSION = "zstd"
RANDOMIZE_START_DURATION = 5

# GitHub Code 数据集：扩展名到语言名称的映射
LANGUAGE_EXTENSIONS: dict[str, str] = {
    # C
    ".c": "c",
    ".h": "c",
    # C++
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hxx": "cpp",
    # Python
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    # Rust
    ".rs": "rust",
    # HTML
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    # CSS
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".less": "css",
    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    # Markdown
    ".md": "markdown",
    ".markdown": "markdown",
    ".mkd": "markdown",
    # JSON
    ".json": "json",
    ".jsonc": "json",
    ".jsonl": "json",
    # XML
    ".xml": "xml",
    ".xsl": "xml",
    ".xslt": "xml",
    ".svg": "xml",
    ".wsdl": "xml",
    # TOML
    ".toml": "toml",
}

# 允许的扩展名集合（从 LANGUAGE_EXTENSIONS 派生）
ALLOWED_LANGUAGES: set[str] = set(LANGUAGE_EXTENSIONS.keys())


@dataclass
class SamplingConfig:
    name: str
    source: Path
    samples: int
    buckets: dict[str, int] = field(default_factory=dict)
    stars_filter: dict[str, int] = field(default_factory=dict)

    def get_all_counts(self) -> dict[str, int]:
        if self.buckets:
            return self.buckets
        if self.stars_filter:
            return self.stars_filter
        return {}


@dataclass
class TokenizerDataConfig:
    datasets: dict[str, SamplingConfig]
    random_seed: int
    output_format: str
    output_dir: Path


@dataclass
class SamplingInfo:
    total_requested: int
    total_sampled: int
    sources: dict[str, dict[str, Any]]
    random_seed: int


def load_config(config_path: Path) -> TokenizerDataConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    datasets = {}
    for key, cfg in raw.get("datasets", {}).items():
        buckets = cfg.get("buckets", {})
        stars_filter = cfg.get("stars_filter", {})

        datasets[key] = SamplingConfig(
            name=cfg["name"],
            source=Path(cfg["source"]),
            samples=cfg["samples"],
            buckets={str(k): int(v["count"]) for k, v in buckets.items()}
            if buckets
            else {},
            stars_filter={str(k): int(v["count"]) for k, v in stars_filter.items()}
            if stars_filter
            else {},
        )

    return TokenizerDataConfig(
        datasets=datasets,
        random_seed=raw.get("random_seed", 42),
        output_format=raw.get("output_format", "parquet"),
        output_dir=Path(raw.get("output_dir", "data/datasets/nanomind_tokenizer")),
    )


def compute_doc_hash(doc_id: str, seed: int) -> int:
    """计算文档的确定性哈希值 (64 位)."""
    data = f"{seed}_{doc_id}".encode()
    return int.from_bytes(
        hashlib.md5(data, usedforsecurity=False).digest()[:8],
        "big",
    )


def determine_text_column(dataset_name: str) -> str:
    if "github" in dataset_name.lower():
        return "content"
    return "text"


def create_row_index_adapter(
    dataset_name: str,
    bucket_name: str,
    text_key: str = "text",
) -> Callable[..., dict]:
    """创建行索引适配器。

    id 格式统一为: {dataset_name}/{bucket_name}/{filename}.parquet#{row_idx}
    例如: fineweb_edu_en/4.0/00000.parquet#123
    """

    def adapter(reader: Any, data: dict, path: str, id_in_file: int | str) -> dict:
        metadata = data.pop("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                pass
        if not isinstance(metadata, dict):
            metadata = {"metadata": metadata}

        metadata["row_idx"] = id_in_file

        # parquet_path 使用相对路径，格式如：data/datasets/fineweb/en/4.0/00000.parquet
        bucket_abs_path = Path(reader.data_folder.path)
        try:
            bucket_rel_path = bucket_abs_path.relative_to(Path.cwd())
        except ValueError:
            bucket_rel_path = bucket_abs_path
        rel_path = str(bucket_rel_path / path)
        metadata["parquet_path"] = rel_path

        # file_path 保留原始数据中的 file_path 字段值（主要针对 github_code 数据集）
        if "file_path" in data:
            metadata["file_path"] = data.pop("file_path")

        # 统一生成 id 格式: {dataset_name}/{bucket_name}/{filename}.parquet#{row_idx}
        filename = Path(path).name
        doc_id = f"{dataset_name}/{bucket_name}/{filename}#{id_in_file}"

        return {
            "text": data.pop(text_key, ""),
            "id": doc_id,
            "media": data.pop("media", []),
            "metadata": data | metadata,
        }

    return adapter


def find_bucket_dir(files: list[Path], bucket_name: str) -> Path:
    bucket_dir = files[0].parent
    while bucket_dir.name != bucket_name and bucket_dir.parent != bucket_dir:
        bucket_dir = bucket_dir.parent
    return bucket_dir


def calculate_tasks(tasks: int, workers: int, item_count: int | None = None) -> int:
    if tasks > 0:
        return tasks
    if item_count is not None and item_count > 0:
        return max(1, min(workers, item_count // 10000))
    return max(1, workers)


def get_file_extension(file_path: str) -> str | None:
    """从文件路径获取扩展名."""
    if not file_path:
        return None
    ext = Path(file_path).suffix.lower()
    return ext if ext else None


def _add_to_heap(
    max_heap: list[tuple[int, int, int]],
    doc_hash: int,
    file_idx: int,
    row_idx: int,
    target_count: int,
) -> None:
    """将文档添加到最大堆中（维护最小的target_count个元素）."""
    if len(max_heap) < target_count:
        heapq.heappush(max_heap, (-doc_hash, file_idx, row_idx))
    elif doc_hash < -max_heap[0][0]:
        heapq.heapreplace(max_heap, (-doc_hash, file_idx, row_idx))


def _build_result_from_heap(
    max_heap: list[tuple[int, int, int]], file_list: list[Path]
) -> dict[Path, set[int]]:
    """从最大堆构建采样结果字典."""
    result: dict[Path, set[int]] = {}
    for _, file_idx, row_idx in max_heap:
        result.setdefault(file_list[file_idx], set()).add(row_idx)
    return result


def precompute_sampling_indices(
    files: list[Path],
    bucket_name: str,
    seed: int,
    target_count: int,
    allowed_extensions: set[str] | None = None,
) -> dict[Path, set[int]]:
    """预计算采样索引。

    Args:
        files: Parquet文件列表
        bucket_name: 桶名称
        seed: 随机种子
        target_count: 目标采样数量
        allowed_extensions: 如果提供，只采样匹配这些扩展名的文档
            注意：此时需要读取文件内容，根据 doc.metadata["file_path"] 过滤
    """
    if not files:
        return {}

    # 如果需要按扩展名过滤，则需要读取文件内容
    if allowed_extensions:
        return _precompute_with_content_filter(
            files, bucket_name, seed, target_count, allowed_extensions
        )

    # 标准流程：只读取元数据
    max_heap: list[tuple[int, int, int]] = []
    file_list = list(files)
    total_scanned = 0

    logger.info(f"  [{bucket_name}] 预计算采样索引 (目标: {target_count:,})")

    for file_idx, fp in enumerate(
        tqdm(file_list, desc=f"哈希计算 {bucket_name}", leave=False)
    ):
        try:
            num_rows = pq.read_metadata(fp).num_rows
            base_doc_id = f"{bucket_name}#{fp}#"

            for row_idx in range(num_rows):
                doc_id = f"{base_doc_id}{row_idx}"
                doc_hash = compute_doc_hash(doc_id, seed)
                total_scanned += 1
                _add_to_heap(max_heap, doc_hash, file_idx, row_idx, target_count)

        except (OSError, pa.ArrowInvalid) as e:
            logger.warning(f"处理文件失败 {fp}: {e}")

    result = _build_result_from_heap(max_heap, file_list)
    logger.info(
        f"  [{bucket_name}] 扫描 {total_scanned:,} 个文档，选中 {sum(len(v) for v in result.values()):,} 个"
    )
    return result


def _precompute_with_content_filter(
    files: list[Path],
    bucket_name: str,
    seed: int,
    target_count: int,
    allowed_extensions: set[str],
) -> dict[Path, set[int]]:
    """预计算采样索引，根据内容过滤（用于github_code）。"""
    file_list = list(files)
    max_heap: list[tuple[int, int, int]] = []
    total_scanned = 0
    total_filtered = 0

    logger.info(
        f"  [{bucket_name}] 预计算采样索引 (目标: {target_count:,}, "
        f"语言过滤: {len(allowed_extensions)} 种扩展名)"
    )

    for file_idx, fp in enumerate(
        tqdm(file_list, desc=f"哈希计算+过滤 {bucket_name}", leave=False)
    ):
        try:
            table = pq.read_table(fp, columns=["file_path"])
            file_paths = table["file_path"].to_pylist()
            base_doc_id = f"{bucket_name}#{fp}#"

            for row_idx, file_path in enumerate(file_paths):
                ext = get_file_extension(str(file_path))
                if not ext or ext not in allowed_extensions:
                    total_filtered += 1
                    continue

                doc_id = f"{base_doc_id}{row_idx}"
                doc_hash = compute_doc_hash(doc_id, seed)
                total_scanned += 1
                _add_to_heap(max_heap, doc_hash, file_idx, row_idx, target_count)

        except (OSError, pa.ArrowInvalid) as e:
            logger.warning(f"处理文件失败 {fp}: {e}")

    result = _build_result_from_heap(max_heap, file_list)
    logger.info(
        f"  [{bucket_name}] 扫描 {total_scanned:,} 个文档，"
        f"过滤 {total_filtered:,} 个，选中 {sum(len(v) for v in result.values()):,} 个"
    )
    return result


def _convert_to_row_set(value: set[int] | list[int] | str | Any) -> set[int]:
    """将各种类型转换为行索引集合."""
    if isinstance(value, set):
        return value
    if isinstance(value, list):
        return set(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, set):
                return parsed
            if hasattr(parsed, "__iter__"):
                return set(parsed)
        except (ValueError, SyntaxError):
            pass
    return set()


class IndexFilter(PipelineStep):
    name = "Index Filter"
    type = "🎯 - FILTER"

    def __init__(
        self, indices: dict[Path, set[int]] | dict[str, set[int] | list[int] | str]
    ):
        super().__init__()
        self.indices: dict[str, set[int]] = {
            str(k): _convert_to_row_set(v) for k, v in indices.items()
        }

    def run(
        self,
        data: Iterator[Document],
        rank: int = 0,
        world_size: int = 1,  # noqa: ARG002
    ) -> Iterator[Document]:
        for doc in data:
            # 使用 parquet_path 匹配索引（adapter中已设置）
            parquet_path = doc.metadata.get("parquet_path", "")
            row_idx = doc.metadata.get("row_idx")

            if parquet_path in self.indices and row_idx in self.indices[parquet_path]:
                self.stat_update("passed", value=1)
                yield doc
            else:
                self.stat_update("filtered", value=1)


class LanguageTagger(PipelineStep):
    """根据文件扩展名过滤并标记编程语言。

    只对 github_code 数据集生效，为通过的文档添加 language 字段。
    """

    name = "Language Tagger"
    type = "🏷️ - LANG"

    def __init__(self, allowed_extensions: set[str]):
        super().__init__()
        self.allowed_extensions = allowed_extensions

    def run(
        self,
        data: Iterator[Document],
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Document]:
        for doc in data:
            original_file_path = doc.metadata.get("file_path", "")
            ext = get_file_extension(original_file_path)

            if ext and ext in self.allowed_extensions:
                doc.metadata["language"] = LANGUAGE_EXTENSIONS.get(ext, "unknown")
                self.stat_update("tagged", value=1)
                yield doc
            else:
                self.stat_update("filtered", value=1)


class SourceTagger(PipelineStep):
    name = "Source Tagger"
    type = "🏷️ - TAGGER"

    def __init__(self, dataset_name: str, bucket_name: str):
        super().__init__()
        self.dataset_name = dataset_name
        self.bucket_name = bucket_name

    def run(
        self,
        data: Iterator[Document],
        rank: int = 0,
        world_size: int = 1,  # noqa: ARG002
    ) -> Iterator[Document]:
        for doc in data:
            doc.metadata["source_dataset"] = self.dataset_name
            doc.metadata["source_bucket"] = self.bucket_name
            self.stat_update("tagged", value=1)
            yield doc


class TokenizerDataWriter(PipelineStep):
    name = "Tokenizer Data Writer"
    type = "💾 - WRITER"

    def __init__(
        self,
        output_dir: str,
        dataset_name: str = "default",
        bucket_name: str = "default",
        max_rows_per_file: int = DEFAULT_MAX_ROWS,
        buffer_size: int = DEFAULT_BATCH_SIZE,
        compression: str = COMPRESSION,
        include_language: bool = False,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.bucket_name = bucket_name
        self.max_rows_per_file = max_rows_per_file
        self.buffer_size = buffer_size
        self.compression = compression
        self.include_language = include_language

        self._buffer: list[dict] = []
        self._batch_counter = 0
        self._rows_in_current_file = 0
        self._total_written = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_batch(self, batch: list[dict], rank: int = 0) -> None:
        if not batch:
            return

        table_data = {
            "id": [doc["id"] for doc in batch],
            "text": [doc["text"] for doc in batch],
            "source_dataset": [doc["source_dataset"] for doc in batch],
            "source_bucket": [doc["source_bucket"] for doc in batch],
        }

        if self.include_language:
            table_data["language"] = [doc.get("language", "unknown") for doc in batch]

        table = pa.table(table_data)

        filename = f"{self.dataset_name}-{self.bucket_name}-{self._batch_counter:05d}-rank-{rank:05d}.parquet"
        output_path = self.output_dir / filename

        pq.write_table(table, output_path, compression=self.compression)

        self._rows_in_current_file += len(batch)
        self._total_written += len(batch)
        self.stat_update("rows_written", value=len(batch))
        self._batch_counter += 1

    def run(
        self,
        data: Iterator[Document],
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        batch: list[dict] = []
        batch_count = 0

        for doc in data:
            batch_item = {
                "id": doc.id,
                "text": doc.text,
                "source_dataset": doc.metadata.get("source_dataset", "unknown"),
                "source_bucket": doc.metadata.get("source_bucket", "unknown"),
            }

            if self.include_language:
                batch_item["language"] = doc.metadata.get("language", "unknown")

            batch.append(batch_item)

            if len(batch) >= self.buffer_size:
                self._write_batch(batch, rank=rank)
                batch = []
                batch_count += 1
                if batch_count % 10 == 0:
                    gc.collect()

        if batch:
            self._write_batch(batch, rank=rank)

        gc.collect()
        logger.info(
            f"写入完成: {self._total_written:,} 行到 {self._batch_counter} 个文件"
        )

    def get_total_written(self) -> int:
        return self._total_written


def get_parquet_files(source_dir: Path, bucket_name: str) -> list[Path]:
    bucket_dir = source_dir / bucket_name
    if not bucket_dir.exists():
        logger.warning(f"桶目录不存在: {bucket_dir}")
        return []

    files = sorted(bucket_dir.rglob("*.parquet"))
    logger.info(f"  [{bucket_name}] 找到 {len(files)} 个文件")
    return files


def _count_parquet_rows(file_iter: list[Path] | Iterator[Path]) -> int:
    """统计parquet文件行数（通用函数）."""
    total = 0
    for fp in file_iter:
        try:
            total += pq.read_metadata(fp).num_rows
        except Exception as e:
            logger.warning(f"无法读取 {fp} 元数据: {e}")
    return total


def count_total_rows_fast(files: list[Path]) -> int:
    """快速统计所有文件的行数（使用元数据，不读取数据）."""
    return _count_parquet_rows(files)


def count_written_rows(output_dir: Path, dataset_name: str, bucket_name: str) -> int:
    pattern = f"{dataset_name}-{bucket_name}-*.parquet"
    return _count_parquet_rows(list(output_dir.glob(pattern)))


def print_sample_texts(
    output_dir: Path,
    dataset_name: str,
    bucket_name: str,
    max_samples: int = 2,
    max_text_length: int = 80,
) -> None:
    """打印样本内容的所有字段（一行一条，text字段截断）.

    Args:
        output_dir: 输出目录
        dataset_name: 数据集名称
        bucket_name: 桶名称
        max_samples: 最大样本数
        max_text_length: text字段最大显示长度（默认80字符）
    """
    pattern = f"{dataset_name}-{bucket_name}-*.parquet"
    files = list(output_dir.glob(pattern))

    if not files:
        return

    samples_printed = 0
    for fp in sorted(files):
        if samples_printed >= max_samples:
            break
        try:
            table = pq.read_table(fp)

            for row_idx in range(min(max_samples - samples_printed, table.num_rows)):
                row_data = {
                    col: table[col][row_idx].as_py() for col in table.column_names
                }

                # 格式化每个字段
                formatted_parts = []
                for col in ["id", "source_dataset", "source_bucket", "text"]:
                    if col in row_data:
                        val = row_data[col]
                        if isinstance(val, str):
                            # 去除换行符
                            val = val.replace("\n", " ").replace("\r", " ")
                            # text字段截断
                            if col == "text" and len(val) > max_text_length:
                                val = val[:max_text_length] + "..."
                        formatted_parts.append(f"{col}={val}")

                logger.info(
                    f"        样本 {samples_printed + 1}: {' | '.join(formatted_parts)}"
                )
                samples_printed += 1

        except Exception as e:
            logger.warning(f"无法读取样本 {fp}: {e}")


def process_bucket_streaming(
    files: list[Path],
    bucket_name: str,
    target_count: int,
    seed: int,
    dataset_name: str,
    output_dir: Path,
    workers: int = DEFAULT_WORKERS,
    tasks: int = DEFAULT_TASKS,
    max_rows_per_file: int = DEFAULT_MAX_ROWS,
    buffer_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """处理单个桶的数据，使用采样模式。

    对于github_code数据集，会先根据file_path过滤编程语言，再进行采样。
    """
    if not files:
        logger.warning(f"桶 {bucket_name} 无数据文件")
        return 0

    text_column = determine_text_column(dataset_name)

    total_rows = count_total_rows_fast(files)
    logger.info(f"  [{bucket_name}] 总行数: {total_rows:,}, 目标: {target_count:,}")

    if total_rows == 0:
        return 0

    _process_sampled(
        files=files,
        bucket_name=bucket_name,
        target_count=target_count,
        seed=seed,
        dataset_name=dataset_name,
        output_dir=output_dir,
        workers=workers,
        tasks=tasks,
        max_rows_per_file=max_rows_per_file,
        buffer_size=buffer_size,
        text_column=text_column,
    )

    actual_written = count_written_rows(output_dir, dataset_name, bucket_name)
    logger.info(f"  [{bucket_name}] 实际写入: {actual_written:,} 行")
    return actual_written


def _process_sampled(
    files: list[Path],
    bucket_name: str,
    target_count: int,
    seed: int,
    dataset_name: str,
    output_dir: Path,
    workers: int,
    tasks: int,
    max_rows_per_file: int,
    buffer_size: int,
    text_column: str,
) -> None:
    """采样处理流程。

    对于github_code数据集：
    1. 先读取文件内容，根据file_path过滤出目标编程语言的文档
    2. 对过滤后的文档计算采样索引
    3. 按索引采样并输出

    对于其他数据集：
    1. 直接计算采样索引
    2. 按索引采样并输出
    """
    # 对github_code数据集，先过滤语言再进行采样
    if "github" in dataset_name.lower():
        logger.info(f"  [{bucket_name}] github_code: 先过滤语言再采样")
        indices = precompute_sampling_indices(
            files=files,
            bucket_name=bucket_name,
            seed=seed,
            target_count=target_count,
            allowed_extensions=ALLOWED_LANGUAGES,
        )
    else:
        indices = precompute_sampling_indices(
            files=files,
            bucket_name=bucket_name,
            seed=seed,
            target_count=target_count,
        )

    if not indices:
        logger.warning(f"  [{bucket_name}] 采样索引为空")
        return

    selected_count = sum(len(v) for v in indices.values())
    logger.info(f"  [{bucket_name}] 已选择 {selected_count:,} 个索引，开始流式读取...")

    bucket_dir = find_bucket_dir(files, bucket_name)
    row_idx_adapter = create_row_index_adapter(dataset_name, bucket_name, text_column)

    pipeline: list[PipelineStep] = [
        ParquetReader(
            data_folder=str(bucket_dir),
            glob_pattern="**/*.parquet",
            text_key=text_column,
            adapter=row_idx_adapter,
        ),
        IndexFilter(indices=indices),
    ]

    # github_code数据集：添加语言标记（过滤已在采样前完成）
    if "github" in dataset_name.lower():
        pipeline.append(LanguageTagger(allowed_extensions=ALLOWED_LANGUAGES))

    pipeline.extend(
        [
            SourceTagger(dataset_name=dataset_name, bucket_name=bucket_name),
            TokenizerDataWriter(
                output_dir=str(output_dir),
                dataset_name=dataset_name,
                bucket_name=bucket_name,
                max_rows_per_file=max_rows_per_file,
                buffer_size=min(selected_count, buffer_size),
                include_language="github" in dataset_name.lower(),
            ),
        ]
    )

    actual_tasks = calculate_tasks(tasks, workers, selected_count)

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=actual_tasks,
        workers=workers,
        logging_dir=str(output_dir / "logs" / dataset_name / bucket_name),
        randomize_start_duration=RANDOMIZE_START_DURATION,
    )

    executor.run()


def process_dataset(
    source_key: str,
    config: SamplingConfig,
    seed: int,
    output_dir: Path,
    workers: int,
    tasks: int,
    max_rows_per_file: int,
    buffer_size: int,
) -> dict[str, Any]:
    """处理单个数据集的所有桶."""
    logger.info(f"处理数据集 [{source_key}]: {config.name}")
    logger.info(f"  源目录: {config.source}")

    bucket_stats = {}
    counts = config.get_all_counts()

    for bucket_name, target_count in counts.items():
        files = get_parquet_files(config.source, bucket_name)

        if not files:
            logger.warning(f"  [{bucket_name}] 无数据文件")
            bucket_stats[bucket_name] = {
                "requested": target_count,
                "sampled": 0,
            }
            continue

        sampled = process_bucket_streaming(
            files=files,
            bucket_name=bucket_name,
            target_count=target_count,
            seed=seed,
            dataset_name=config.name,
            output_dir=output_dir,
            workers=workers,
            tasks=tasks,
            max_rows_per_file=max_rows_per_file,
            buffer_size=buffer_size,
        )

        bucket_stats[bucket_name] = {
            "requested": target_count,
            "sampled": sampled,
        }

    total_requested = sum(s["requested"] for s in bucket_stats.values())
    total_sampled = sum(s["sampled"] for s in bucket_stats.values())

    logger.info(
        f"  [{config.name}] 总计: 请求 {total_requested:,}, 实际 {total_sampled:,}"
    )

    return {
        "name": config.name,
        "source": str(config.source),
        "requested": total_requested,
        "sampled": total_sampled,
        "buckets": bucket_stats,
    }


def save_sampling_info(info: SamplingInfo, output_dir: Path) -> Path:
    """保存采样信息到 JSON 文件."""
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = output_dir / "sampling_info.json"

    data = {
        "total_requested": info.total_requested,
        "total_sampled": info.total_sampled,
        "random_seed": info.random_seed,
        "sources": info.sources,
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"采样信息已保存: {info_path}")
    return info_path


def _log_section(title: str, width: int = 60) -> None:
    """记录带分隔线的章节标题."""
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def _log_config_info(
    config: TokenizerDataConfig,
    workers: int,
    tasks: int,
    max_rows: int,
    buffer_size: int,
) -> None:
    """记录配置信息."""
    _log_section("准备 Tokenizer 训练数据 (Datatrove 优化版)")
    logger.info(f"配置文件: {CONFIG_PATH}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"随机种子: {config.random_seed}")
    logger.info(f"Workers: {workers} (优化: 减少进程切换)")
    logger.info(f"Tasks: {tasks} (优化: 1:1匹配workers)")
    logger.info(f"每文件最大行数: {max_rows:,}")
    logger.info(f"写入缓冲区: {buffer_size:,} (优化: 匹配400MB/s磁盘)")
    logger.info(f"数据集数量: {len(config.datasets)}")


def prepare_tokenizer_data(
    workers: int = DEFAULT_WORKERS,
    tasks: int = DEFAULT_TASKS,
    max_rows_per_file: int = DEFAULT_MAX_ROWS,
    buffer_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """主函数：准备 tokenizer 训练数据."""
    try:
        config = load_config(CONFIG_PATH)
        _log_config_info(config, workers, tasks, max_rows_per_file, buffer_size)

        source_stats = {}

        for source_key, dataset_config in config.datasets.items():
            print()
            stats = process_dataset(
                source_key=source_key,
                config=dataset_config,
                seed=config.random_seed,
                output_dir=config.output_dir,
                workers=workers,
                tasks=tasks,
                max_rows_per_file=max_rows_per_file,
                buffer_size=buffer_size,
            )
            source_stats[source_key] = stats

            logger.info(f"  [{source_key}] 已完成 {stats['sampled']:,} 个样本")

        total_requested = sum(s["requested"] for s in source_stats.values())
        total_sampled = sum(s["sampled"] for s in source_stats.values())

        # 保存采样信息
        info = SamplingInfo(
            total_requested=total_requested,
            total_sampled=total_sampled,
            sources=source_stats,
            random_seed=config.random_seed,
        )
        save_sampling_info(info, config.output_dir)

        # 输出最终报告
        print()
        _log_section("处理完成")
        logger.info(f"总请求样本: {total_requested:,}")
        logger.info(f"总采样样本: {total_sampled:,}")
        logger.info(f"采样率: {total_sampled / total_requested:.1%}")
        logger.info(f"输出目录: {config.output_dir}")

        print()
        logger.info("各数据源采样详情:")
        total_actual_written = 0
        for source_key, stats in source_stats.items():
            dataset_name = stats["name"]
            logger.info(f"  [{source_key}] {dataset_name}")

            dataset_actual = 0
            if stats["buckets"]:
                for bucket_name, bucket_stats in stats["buckets"].items():
                    actual_written = count_written_rows(
                        config.output_dir, dataset_name, bucket_name
                    )
                    dataset_actual += actual_written
                    logger.info(
                        f"      {bucket_name}: 目标={bucket_stats['requested']:,}, "
                        f"采样={bucket_stats['sampled']:,}, 实际={actual_written:,}"
                    )
                    # 打印样本内容（每个桶最多2条，text字段截断）
                    if actual_written > 0:
                        print_sample_texts(
                            config.output_dir,
                            dataset_name,
                            bucket_name,
                            max_samples=2,
                            max_text_length=200,
                        )

            total_actual_written += dataset_actual
            logger.info(
                f"    汇总: 请求={stats['requested']:,}, 采样={stats['sampled']:,}, "
                f"实际={dataset_actual:,}"
            )

        logger.info(
            f"总实际写入: {total_actual_written:,} (与采样总数 {total_sampled:,} 对比)"
        )

        return 0

    except Exception as e:
        logger.exception(f"处理失败: {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="准备 Tokenizer 训练数据 - 基于 Datatrove 的优化版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置 (针对16核/250GB/400MB/s优化)
  python scripts/prepare_tokenizer_data.py

  # 调整workers数量 (建议不超过16以避免进程切换开销)
  python scripts/prepare_tokenizer_data.py --workers 8

  # 调整缓冲区大小以优化IO性能
  python scripts/prepare_tokenizer_data.py --buffer-size 100000
        """,
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行度（进程数）。建议不超过16。 (默认: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=int,
        default=DEFAULT_TASKS,
        help=f"任务数。建议等于workers。 (默认: {DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"每个输出文件的最大行数 (默认: {DEFAULT_MAX_ROWS})",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"写入缓冲区大小 (默认: {DEFAULT_BATCH_SIZE})",
    )

    args = parser.parse_args()

    return prepare_tokenizer_data(
        workers=args.workers,
        tasks=args.tasks,
        max_rows_per_file=args.max_rows,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    sys.exit(main())
