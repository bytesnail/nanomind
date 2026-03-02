# AGENTS.md - data_processing

数据处理核心模块：FineWeb-Edu 评分分桶 + Tokenizer 数据采样流水线。

## OVERVIEW

基于 Datatrove 的高性能并行处理框架，提供评分过滤、确定性采样、多桶写入。

## STRUCTURE

```
src/data_processing/
├── __init__.py              # 导出 18 个 API
├── config_loader.py         # YAML 加载 + 默认常量 (LRU 缓存)
├── bucket_config.py         # BucketConfig frozen dataclass + 二分查找
├── score_filter.py          # ScoreFilter: MD5 哈希确定性采样
├── bucket_path_writer.py    # BucketPathWriter: 多桶并行 Parquet
├── parquet_merger.py        # 合并小文件到目标大小
├── validation.py            # Parquet 验证 + 统计报告
└── fineweb_edu/             # FineWeb-Edu 专用流水线 [AGENTS.md]
```

## WHERE TO LOOK

| 任务 | 文件 | 关键函数 |
|------|------|----------|
| 配置加载 | `config_loader.py` | `get_dataset_configs()`, `get_processing_config()` |
| 评分桶 | `bucket_config.py` | `BucketConfig`, `find_bucket_for_score()` |
| 确定性采样 | `score_filter.py` | `ScoreFilter` (MD5 hash) |
| 多桶写入 | `bucket_path_writer.py` | `BucketPathWriter` |
| 文件合并 | `parquet_merger.py` | `merge_all_buckets()` |
| 验证 | `validation.py` | `validate_bucket()`, `print_report()` |

## PUBLIC API

```python
from src.data_processing import (
    # 配置
    BucketConfig, Compression,
    find_bucket_for_score, get_all_bucket_configs,
    # PipelineSteps
    BucketPathWriter, ScoreFilter,
    # 处理
    fineweb_adapter, normalize_score,
    process_all_datasets, process_single_dataset,
    # 工具
    merge_all_buckets, merge_bucket_files,
    validate_all_buckets, validate_bucket, validate_file, print_report,
)
```

## DEFAULTS (config_loader.py)

```python
DEFAULT_WORKERS = 8
DEFAULT_TASKS = 8
DEFAULT_RANDOM_SEED = 42
DEFAULT_COMPRESSION = "zstd"
DEFAULT_MAX_FILE_SIZE = 512 * 1024 * 1024  # 512MB
```

## ARCHITECTURE

```
config/*.yaml
    ↓ (LRU cached)
config_loader.py → DEFAULT_* constants
    ↓
bucket_config.py → BucketConfig dataclass
    ↓
PipelineSteps (ScoreFilter, BucketPathWriter)
    ↓
fineweb_edu/reorganizer.py → process_single_dataset
```

## KEY PATTERNS

### LRU 缓存 (6 处)
```python
@lru_cache(maxsize=10)
def _load_yaml(path: Path) -> dict: ...

@lru_cache(maxsize=1)
def get_dataset_configs() -> dict: ...
```

### PipelineStep 继承
```python
from datatrove.pipeline.base import PipelineStep

class ScoreFilter(PipelineStep):
    name = "Score Filter"
    type = "🎯 - FILTER"
    def __call__(self, doc: Document) -> Document | None: ...
```

### 确定性采样 (MD5)
```python
h = int.from_bytes(
    hashlib.md5(f"{seed}_{doc_id}".encode(), usedforsecurity=False).digest()[:8], "big"
)
return h / (2**64) < sampling_rate
```

## ANTI-PATTERNS

| 禁止 | 替代 |
|------|------|
| 模块顶层加载配置 | `@lru_cache` 延迟加载 |
| 假设配置字段存在 | `.get(key, default)` |
| 循环内创建临时对象 | 预分配或复用 |
| 存储完整 Path 对象 | 存储整数索引 |
| 共享 `logging_dir` | 每数据集独立目录 |

## TESTING

```bash
pytest tests/test_bucket_config.py       # BucketConfig
pytest tests/test_score_filter.py        # 采样算法
pytest tests/test_bucket_path_writer.py  # Writer
pytest tests/test_parquet_merger.py      # 合并
```

Fixtures: `tests/conftest.py`
