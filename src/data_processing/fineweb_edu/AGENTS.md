# AGENTS.md - fineweb_edu

**Generated:** 2026-03-04
**Commit:** f5aee16

FineWeb-Edu 数据处理流水线：评分分桶重组 + 多语言支持。

## OVERVIEW

实现 FineWeb-Edu 多语言数据集的质量分层与确定性采样，输出分桶 Parquet 文件。

## STRUCTURE

```
fineweb_edu/
├── __init__.py      # 导出 8 个 API
├── __main__.py      # CLI: python -m src.data_processing.fineweb_edu
├── adapters.py      # fineweb_adapter, normalize_score
└── reorganizer.py   # process_all_datasets, create_pipeline
```

## WHERE TO LOOK

| 任务 | 文件 | 函数 |
|------|------|------|
| CLI 入口 | `__main__.py` | `main()` |
| 批量处理 | `reorganizer.py` | `process_all_datasets()` |
| 单数据集 | `reorganizer.py` | `process_single_dataset()` |
| 流水线构建 | `reorganizer.py` | `create_pipeline()` |
| 数据适配 | `adapters.py` | `fineweb_adapter()` |
| 分数归一化 | `adapters.py` | `normalize_score()` |

## PIPELINE FLOW

```
ParquetReader(adapter=fineweb_adapter)
    │  → metadata["score"], metadata["row_idx"]
    ↓
ScoreFilter(buckets, random_seed)
    │  → 二分查找 + MD5 哈希采样
    ↓
BucketPathWriter(buckets, compression, max_size)
    │  → 多桶并行写入
    ↓
merge_all_buckets(output_dir, target_size)
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
python -m src.data_processing.fineweb_edu    # 处理所有数据集
# 日志: data/datasets/fineweb/logs/multi_bucket_{lang}/processing.log
```

## ADAPTER BEHAVIOR

`fineweb_adapter(doc)`:
1. 提取 `score` → `metadata["score"]` (默认 0.0)
2. 添加 `metadata["row_idx"]` = id_in_file (采样用)
3. 应用 `normalize_score()` (中文 × 5.0)
4. 缺失 text → 返回 None (过滤)

## CONFIG STRUCTURE

`config/dataset.yaml`:
```yaml
datasets:
  {lang}:
    score_normalization: {enabled: bool, multiplier: float}
    input_dir: path
    output_dir: path
    buckets:
      - {name: "2.5", min_score: 2.5, max_score: 3.0, sampling_rate: 0.25}
```

## LOGGING

```python
log_name = f"multi_bucket_{output_dir.name}"
log_path = output_dir.parent / "logs" / log_name / "processing.log"
```

## ANTI-PATTERNS

| 禁止 | 原因 |
|------|------|
| 共享 `logging_dir` | Datatrove 跳过任务 |
| 负数/空路径 ID | IndexFilter 失效 |
| 跳过 `input_dir.exists()` | 静默失败 |

## TESTING

```bash
pytest tests/test_fineweb_reorganizer.py  # 流水线
pytest tests/test_adapters.py              # 适配器
```

## KNOWLEDGE BASE

> 📚 **按需查阅**: `docs/KNOWLEDGE_BASE.md` 包含 FineWeb-Edu 处理相关的经验教训、Datatrove 框架使用技巧、性能优化建议和常见问题解决方案。在流水线调试或优化时可参考对应章节。
