# FineWeb-Edu 数据处理模块

FineWeb-Edu 数据集质量评分分桶重组的数据处理流水线。

## 模块结构

```
src/data_processing/
├── __init__.py              # 模块入口，导出 18 个公开 API
├── config_loader.py         # YAML 配置加载器 (LRU 缓存)
├── bucket_config.py         # BucketConfig 数据类 + 二分查找
├── score_filter.py          # ScoreFilter PipelineStep
├── bucket_path_writer.py    # BucketPathWriter PipelineStep
├── parquet_merger.py        # Parquet 文件合并工具
├── validation.py            # 数据验证 + 报告生成
└── fineweb_edu/             # FineWeb-Edu 专用子模块
    ├── __init__.py          # 子模块入口
    ├── __main__.py          # CLI 入口
    ├── adapters.py          # 数据适配器
    └── reorganizer.py       # 处理流水线
```

## 公开 API

```python
from src.data_processing import (
    # 配置
    BucketConfig,
    Compression,
    find_bucket_for_score,
    find_bucket_in_sorted,
    get_all_bucket_configs,
    # PipelineSteps
    BucketPathWriter,
    ScoreFilter,
    # 处理函数
    fineweb_adapter,
    normalize_score,
    process_all_datasets,
    process_single_dataset,
    # 工具
    merge_all_buckets,
    merge_bucket_files,
    print_report,
    validate_all_buckets,
    validate_bucket,
    validate_file,
)
```

## 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `BucketConfig` | `bucket_config.py` | 评分桶配置数据类 |
| `ScoreFilter` | `score_filter.py` | 评分过滤 + 确定性采样 |
| `BucketPathWriter` | `bucket_path_writer.py` | 多桶并行 Parquet 写入 |
| `merge_all_buckets` | `parquet_merger.py` | 合并所有桶的文件 |
| `fineweb_adapter` | `adapters.py` | 数据适配器函数 |
| `normalize_score` | `adapters.py` | 分数归一化 |
| `process_all_datasets` | `reorganizer.py` | 处理所有配置的数据集 |
| `process_single_dataset` | `reorganizer.py` | 处理单个数据集 |

## CLI 使用

```bash
# 处理所有配置的数据集
python -m src.data_processing.fineweb_edu

# 试运行（小规模测试）
python scripts/trial_run.py

# 试运行指定数据集
python scripts/trial_run.py --dataset zh

# 验证输出
python scripts/validate_output.py --all

# 验证单个目录
python scripts/validate_output.py --input data/datasets/fineweb/en --dataset en
```

## 使用示例

### 处理所有数据集

```python
from src.data_processing import process_all_datasets

results = process_all_datasets()
print(results)  # {'en': ['2.5', '3.0', '3.5', '4.0'], 'zh': [...]}
```

### 处理单个数据集

```python
from pathlib import Path
from src.data_processing import process_single_dataset, get_all_bucket_configs

buckets = get_all_bucket_configs("en")

result = process_single_dataset(
    input_dir=Path("data/datasets/HuggingFaceFW/fineweb-edu"),
    output_dir=Path("data/datasets/fineweb/en"),
    buckets=buckets,
    workers=16,
    tasks=32,
    random_seed=42,
)
```

### 查找评分对应的桶

```python
from src.data_processing import find_bucket_for_score

bucket = find_bucket_for_score(3.2, "en")
print(bucket.name)           # "3.0"
print(bucket.sampling_rate)  # 0.5
```

## 配置文件

| 文件 | 说明 |
|------|------|
| `config/dataset.yaml` | 数据集定义（评分桶、归一化配置、路径） |
| `config/processing.yaml` | 处理参数（workers、compression、文件大小） |
| `config/paths.yaml` | 路径配置 |

## 相关文档

- [设计文档](../../docs/fineweb_edu_data_reorganization_design.md) - 架构设计、扩展指南
- [项目 README](../../README.md) - 项目总体说明
