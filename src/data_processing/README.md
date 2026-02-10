# FineWeb-Edu 数据处理模块

FineWeb-Edu 数据集质量评分分桶重组的数据处理流水线。

## 模块结构

```
src/data_processing/
├── __init__.py              # 模块导出
├── bucket_config.py         # 评分桶配置管理
├── bucket_path_writer.py    # 多桶并行 Parquet 写入器
├── config_loader.py         # YAML 配置加载器
├── parquet_merger.py        # Parquet 文件合并工具
├── score_filter.py          # 评分过滤 + 确定性采样
└── fineweb_edu/             # FineWeb-Edu 专用子模块
    ├── __init__.py          # 子模块导出
    ├── __main__.py          # CLI 入口
    ├── adapters.py          # 数据适配器
    └── reorganizer.py       # 数据处理流水线
```

## 核心组件

### 通用组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `BucketConfig` | `bucket_config.py` | 评分桶配置数据类 |
| `ScoreFilter` | `score_filter.py` | 评分过滤 + 确定性采样 |
| `BucketPathWriter` | `bucket_path_writer.py` | 多桶并行写入器 |
| `merge_all_buckets` | `parquet_merger.py` | 合并所有桶的文件 |

### FineWeb-Edu 专用组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `fineweb_adapter` | `adapters.py` | 数据适配器函数 |
| `process_all_datasets` | `reorganizer.py` | 处理所有配置的数据集 |
| `process_single_dataset` | `reorganizer.py` | 处理单个数据集 |

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
from src.data_processing.bucket_config import find_bucket_for_score

bucket = find_bucket_for_score(3.2, "en")
print(bucket.name)           # "3.0"
print(bucket.sampling_rate)  # 0.5
```

## CLI 使用

```bash
# 处理所有配置的数据集
python -m src.data_processing.fineweb_edu

# 验证输出
python scripts/validate_output.py --input data/datasets/fineweb/en

# 试运行
python scripts/trial_run.py
```

### 环境变量

```bash
export FINEWEB_LOG_DIR="custom/logs"
export FINEWEB_TRIAL_INPUT_DIR="data/test_input"
export FINEWEB_TRIAL_OUTPUT_DIR="data/test_output"

python -m src.data_processing.fineweb_edu
```

## 配置系统

配置文件位于 `config/` 目录：

- `dataset.yaml` - 数据集定义（评分桶、归一化配置、路径）
- `processing.yaml` - 处理参数（workers、compression、文件大小）
- `paths.yaml` - 路径配置（支持环境变量覆盖）

## 模块导入

### 从根模块导入（推荐）

```python
from src.data_processing import (
    BucketConfig,
    BucketPathWriter,
    ScoreFilter,
    find_bucket_for_score,
    get_all_bucket_configs,
    merge_all_buckets,
    fineweb_adapter,
    process_all_datasets,
    process_single_dataset,
)
```

### 从子模块导入

```python
from src.data_processing.fineweb_edu import (
    fineweb_adapter,
    process_all_datasets,
    process_single_dataset,
)

from src.data_processing.bucket_config import BucketConfig
from src.data_processing.score_filter import ScoreFilter
```

## 相关文档

- [完整设计文档](../../docs/fineweb_edu_data_reorganization_design.md) - 架构设计、扩展指南
- [项目 README](../../README.md) - 项目总体说明
