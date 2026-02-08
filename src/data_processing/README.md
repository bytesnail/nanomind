# FineWeb-Edu 数据处理模块

将 FineWeb-Edu 数据集按质量评分分桶重组，支持多语言、分层采样和高性能并行处理。

## 核心组件

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `adapters.py` | 数据适配器 | `fineweb_adapter()`, `normalize_score()` |
| `bucket_config.py` | 评分桶配置管理 | `BucketConfig`, `find_bucket_for_score()` |
| `score_filter.py` | 评分过滤 + 确定性采样 | `ScoreFilter` |
| `bucket_path_writer.py` | 多桶并行 Parquet 写入器 | `BucketPathWriter` |
| `config_loader.py` | YAML 配置加载器 | `get_dataset_configs()`, `get_processing_config()` |
| `fineweb_reorganizer.py` | CLI 主入口 | `process_all_datasets()`, `create_pipeline()` |

## 处理流程

```
输入 Parquet 文件
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ ParquetReader │ --> │ ScoreFilter  │ --> │BucketPathWriter│
│   数据读取    │     │ 评分过滤+采样 │     │   分桶写入    │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  - 字段筛选           - 区间匹配           - 多桶并行
  - ID 生成            - 确定性采样          - 自动分片
  - 评分归一化          - 统计追踪            - 压缩输出
```

## 性能优化

- **单次读取**: 输入数据集只被读取一次，避免重复 I/O（相比传统方式 I/O 减少约 75%）
- **多桶并行**: 一次处理所有评分桶，大幅提升性能
- **确定性采样**: 使用 MD5 哈希确保采样结果可复现
- **流式处理**: 基于生成器的流水线，避免全量加载到内存

## 评分桶配置

评分桶配置位于 `config/dataset.yaml`：

### 英文数据集（HuggingFaceFW/fineweb-edu）

| 桶名称 | 评分区间 | 采样率 | 说明 |
|--------|----------|--------|------|
| 2.5 | 2.5 ≤ score < 3.0 | 25% | 中低质量数据 |
| 3.0 | 3.0 ≤ score < 3.5 | 50% | 中等质量数据 |
| 3.5 | 3.5 ≤ score < 4.0 | 80% | 高质量数据 |
| 4.0 | score ≥ 4.0 | 100% | 顶级质量数据 |

### 中文数据集（Fineweb-Edu-Chinese-V2.1）

中文数据集使用归一化评分（0.0-1.0），系统自动转换为原始评分（×5）：

| 桶名称 | 评分区间 | 采样率 | 归一化范围 |
|--------|----------|--------|-----------|
| 2.5 | 2.5 ≤ score < 3.0 | 40% | 0.50-0.60 |
| 3.0 | 3.0 ≤ score < 3.5 | 60% | 0.60-0.70 |
| 3.5 | 3.5 ≤ score < 4.0 | 90% | 0.70-0.80 |
| 4.0 | score ≥ 4.0 | 100% | ≥ 0.80 |

**区间规则**: 采用**左闭右开**区间 `[min_score, max_score)`，最后一个桶无上界。

## API 使用示例

### 基本使用

```python
from pathlib import Path
from src.data_processing.fineweb_reorganizer import process_all_datasets

# 处理所有配置的数据集
results = process_all_datasets()
print(results)  # {'en': ['2.5', '3.0', '3.5', '4.0'], 'zh': ['2.5', '3.0', '3.5', '4.0']}
```

### 单数据集处理

```python
from pathlib import Path
from src.data_processing.fineweb_reorganizer import process_single_dataset
from src.data_processing.bucket_config import get_all_bucket_configs

# 获取评分桶配置
buckets = get_all_bucket_configs("en")

# 处理单个数据集
result = process_single_dataset(
    input_dir=Path("data/datasets/HuggingFaceFW/fineweb-edu"),
    output_dir=Path("data/datasets/fineweb/en"),
    buckets=buckets,
    workers=16,
    tasks=32,
    random_seed=42,
)
print(result)  # ['2.5', '3.0', '3.5', '4.0']
```

### 查找评分对应的桶

```python
from src.data_processing.bucket_config import find_bucket_for_score

bucket = find_bucket_for_score(3.2, "en")
print(bucket.name)  # "3.0"
print(bucket.sampling_rate)  # 0.5
```

### 使用数据适配器

```python
from src.data_processing.adapters import fineweb_adapter

raw_data = {
    "text": "Example document text",
    "score": 3.5,
    "dump": "CC-MAIN-2024-10"
}

result = fineweb_adapter(None, raw_data, "fineweb-edu/data/train.parquet", 0)
print(result)
# {
#     "text": "Example document text",
#     "id": "data/train.parquet#0",
#     "metadata": {"score": 3.5, "cc_main": "CC-MAIN-2024-10"}
# }
```

## CLI 使用方法

```bash
# 处理所有配置的数据集
python -m src.data_processing.fineweb_reorganizer

# 验证输出
python scripts/validate_output.py --input data/datasets/fineweb/en
```

### 环境变量

使用环境变量覆盖配置：

```bash
export FINEWEB_LOG_DIR="custom/logs"
export FINEWEB_TRIAL_INPUT_DIR="data/test_input"
export FINEWEB_TRIAL_OUTPUT_DIR="data/test_output"

python -m src.data_processing.fineweb_reorganizer
```

## 配置系统

### 配置文件结构

```
config/
├── dataset.yaml       # 数据集定义（评分桶、归一化配置、路径）
├── processing.yaml    # 处理参数（workers、compression、文件大小）
└── paths.yaml         # 路径配置（支持环境变量覆盖）
```

### 配置加载示例

```python
from src.data_processing.config_loader import (
    get_dataset_configs,
    get_processing_config,
    get_paths_config,
)

# 获取所有数据集配置
datasets = get_dataset_configs()

# 获取处理参数
processing = get_processing_config()
workers = processing.get("workers", 8)

# 获取路径配置（自动应用环境变量覆盖）
paths = get_paths_config()
log_dir = paths.get("log_dir")
```

## 模块导出

```python
from src.data_processing import (
    BucketConfig,           # 评分桶配置类
    BucketPathWriter,       # 多桶写入器
    Compression,            # 压缩类型字面量
    ScoreFilter,            # 评分过滤器
    find_bucket_for_score,  # 查找评分对应的桶
    fineweb_adapter,        # 数据适配器函数
    get_all_bucket_configs, # 获取所有评分桶配置
    normalize_score,        # 评分归一化函数
)
```

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_score_filter.py -v
pytest tests/test_bucket_config.py -v
pytest tests/test_adapters.py -v
pytest tests/test_bucket_path_writer.py -v
pytest tests/test_fineweb_reorganizer.py -v
```

## 相关文档

- [完整设计文档](../../docs/fineweb_edu_data_reorganization_design.md) - 架构设计、算法详解、性能优化（含中文数据集评分分析）
- [项目 README](../../README.md) - 项目总体说明
