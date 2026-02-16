# AGENTS.md - data_processing

数据处理模块：FineWeb-Edu 数据集质量评分分桶重组流水线。

## OVERVIEW

提供数据集预处理和重组功能，基于 Datatrove 构建高性能并行处理流水线。

## STRUCTURE

```
src/data_processing/
├── __init__.py              # 导出 18 个公开 API
├── config_loader.py         # YAML 配置加载 (LRU 缓存)
├── bucket_config.py         # BucketConfig 数据类 + 二分查找
├── score_filter.py          # ScoreFilter PipelineStep
├── bucket_path_writer.py    # BucketPathWriter PipelineStep
├── parquet_merger.py        # 文件合并工具
├── validation.py            # 数据验证
├── README.md                # 模块文档
└── fineweb_edu/             # FineWeb-Edu 专用子模块 [AGENTS.md]
```

## WHERE TO LOOK

| 任务 | 文件 | 函数/类 |
|------|------|---------|
| 配置加载 | `config_loader.py` | `get_dataset_configs()`, `get_processing_config()` |
| 评分桶 | `bucket_config.py` | `BucketConfig`, `find_bucket_for_score()` |
| 过滤采样 | `score_filter.py` | `ScoreFilter` |
| 多桶写入 | `bucket_path_writer.py` | `BucketPathWriter` |
| 文件合并 | `parquet_merger.py` | `merge_all_buckets()` |
| 数据验证 | `validation.py` | `validate_bucket()`, `print_report()` |

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

## PATTERNS

### LRU 缓存
所有配置加载函数使用 `@lru_cache` 缓存：
- `_load_yaml(path)` — YAML 文件缓存
- `get_dataset_configs()` — 数据集配置缓存
- `get_bucket_configs_for_dataset(key)` — 桶配置缓存

### PipelineStep 继承
```python
from datatrove.pipeline.base import PipelineStep

class ScoreFilter(PipelineStep):
    def __init__(self, buckets: list[BucketConfig], random_seed: int): ...
    def __call__(self, doc: Document) -> Document | None: ...
```

### 相对导入
```python
from ..config_loader import get_dataset_configs
from ..bucket_config import BucketConfig
```

## ANTI-PATTERNS

| 禁止 | 替代 |
|------|------|
| 模块顶层加载配置 | 使用 `@lru_cache` 延迟加载 |
| 假设配置字段存在 | 使用 `.get(key, default)` |
| 循环内创建临时对象 | 预分配或复用 |
| 存储完整 Path 对象 | 存储整数索引 |

## CONFIGURATION

配置文件位于 `config/`:
- `dataset.yaml` — 数据集定义、评分桶、归一化
- `processing.yaml` — workers, compression, 文件大小
- `paths.yaml` — 路径配置

## TESTING

```bash
pytest tests/test_bucket_config.py      # BucketConfig 测试
pytest tests/test_score_filter.py       # ScoreFilter 测试
pytest tests/test_bucket_path_writer.py # Writer 测试
pytest tests/test_parquet_merger.py     # 合并测试
```
