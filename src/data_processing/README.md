# FineWeb-Edu 数据处理模块

基于 Datatrove 的高性能数据处理流水线，实现 FineWeb-Edu 数据集的质量评分分桶重组。

## 模块结构

```
src/data_processing/
├── __init__.py              # 模块入口，导出公开 API
├── bucket_config.py         # 评分桶配置
├── score_filter.py          # 评分过滤与确定性采样
├── bucket_path_writer.py    # 多桶并行 Parquet 写入
├── parquet_merger.py        # Parquet 文件合并
├── validation.py            # 数据验证与报告
├── config_loader.py         # YAML 配置加载
└── fineweb_edu/             # FineWeb-Edu 专用流水线
    ├── __init__.py
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
    get_all_bucket_configs,
    # Pipeline 组件
    ScoreFilter,
    BucketPathWriter,
    # 处理函数
    fineweb_adapter,
    normalize_score,
    process_all_datasets,
    process_single_dataset,
    # 工具函数
    merge_all_buckets,
    merge_bucket_files,
    validate_all_buckets,
    validate_bucket,
    validate_file,
    print_report,
)
```

## 核心组件

| 组件 | 说明 |
|------|------|
| `BucketConfig` | 评分桶配置数据类（名称、分数范围、采样率） |
| `ScoreFilter` | 评分过滤 PipelineStep，支持 MD5 哈希确定性采样 |
| `BucketPathWriter` | 多桶并行 Parquet 写入 PipelineStep |
| `merge_all_buckets` | 合并各桶的小文件到目标大小 |
| `fineweb_adapter` | FineWeb-Edu 数据适配器 |
| `normalize_score` | 分数归一化（中文数据集 ×5.0） |

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

配置示例 (`config/dataset.yaml`):

```yaml
datasets:
  en:
    score_normalization:
      enabled: false
    input_dir: "data/datasets/HuggingFaceFW/fineweb-edu"
    output_dir: "data/datasets/fineweb/en"
    buckets:
      - name: "2.5"
        min_score: 2.5
        max_score: 3.0
        sampling_rate: 0.25
      - name: "4.0"
        min_score: 4.0
        sampling_rate: 1.0
```

## CLI 使用

```bash
# 处理所有配置的数据集
python -m src.data_processing.fineweb_edu

# 试运行（小规模测试）
python scripts/trial_run.py --dataset zh

# 验证输出
python scripts/validate_output.py --all
```

## 相关文档

- [项目 README](../../README.md) - 项目总体说明与快速开始
- [设计文档](../../docs/fineweb_edu_data_reorganization_design.md) - 系统架构与扩展指南
- [AGENTS.md](../../AGENTS.md) - 开发规范
