# AGENTS.md - fineweb_edu

FineWeb-Edu 数据集处理流水线入口。

## OVERVIEW

实现 FineWeb-Edu 数据集的质量评分分桶重组，支持多语言数据集。

## STRUCTURE

```
fineweb_edu/
├── __init__.py      # 导出 9 个公开 API
├── __main__.py      # CLI 入口: python -m src.data_processing.fineweb_edu
├── adapters.py      # 数据适配器: fineweb_adapter, normalize_score
└── reorganizer.py   # 核心流水线: process_all_datasets, process_single_dataset
```

## WHERE TO LOOK

| 任务 | 文件 | 函数 |
|------|------|------|
| CLI 入口 | `__main__.py` | `main()` |
| 批量处理 | `reorganizer.py` | `process_all_datasets()` |
| 单数据集 | `reorganizer.py` | `process_single_dataset()` |
| 创建流水线 | `reorganizer.py` | `create_pipeline()` |
| 数据适配 | `adapters.py` | `fineweb_adapter()` |
| 分数归一化 | `adapters.py` | `normalize_score()` |

## PIPELINE FLOW

```
ParquetReader(adapter=fineweb_adapter)
    │  → 添加 metadata["score"], metadata["row_idx"]
    ↓
ScoreFilter(buckets, random_seed)
    │  → 二分查找桶 + 确定性采样
    ↓
BucketPathWriter(buckets, compression, max_size)
    │  → 多桶并行 Parquet 写入
    ↓
merge_all_buckets(output_dir, target_size)
    └  → 合并小文件到目标大小
```

## PUBLIC API

```python
from src.data_processing.fineweb_edu import (
    create_pipeline,
    fineweb_adapter,
    get_default_config,
    main,
    normalize_score,
    process_all_datasets,
    process_single_dataset,
    setup_logging,
)
```

## CLI USAGE

```bash
# 处理所有配置的数据集
python -m src.data_processing.fineweb_edu

# 输出日志位于
data/datasets/fineweb/logs/multi_bucket_en/processing.log
data/datasets/fineweb/logs/multi_bucket_zh/processing.log
```

## ADAPTER

`fineweb_adapter(doc)` 处理原始 Parquet 文档：
1. 提取 `score` 字段到 `metadata["score"]`
2. 添加 `metadata["row_idx"]` 用于采样
3. 应用 `normalize_score()` 归一化（中文数据集）

## LOGGING

```python
# 日志目录命名
log_name = f"multi_bucket_{output_dir.name}"  # 如 multi_bucket_en

# 日志路径
output_dir.parent / "logs" / log_name / "processing.log"
```

## MULTI-DATASET CONFIG

`config/dataset.yaml`:
```yaml
datasets:
  en:
    score_normalization: {enabled: false}
    buckets: [{name: "2.5", min_score: 2.5, sampling_rate: 0.25}, ...]
  zh:
    score_normalization: {enabled: true, multiplier: 5.0}
    buckets: [{name: "2.5", min_score: 2.5, sampling_rate: 0.40}, ...]
```

## ANTI-PATTERNS

| 禁止 | 原因 |
|------|------|
| 共享 `logging_dir` | 导致 Datatrove 任务跳过 |
| 负数/空路径生成 ID | 会导致 IndexFilter 失效 |
| 忽略输入目录检查 | 应先 `input_dir.exists()` |

## TESTING

```bash
pytest tests/test_fineweb_reorganizer.py  # 流水线测试
pytest tests/test_adapters.py              # 适配器测试
```
