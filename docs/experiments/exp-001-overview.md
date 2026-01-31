# 实验 001：数据集统计与探索

## 概述

实验 001 使用 Datatrove 框架对多个数据集进行统计分析，包括文档计数、Score 分布、文本长度分布等。

**目的**：快速了解数据集的基本统计信息，为后续实验提供数据基础。

---

## 实验结构

```
experiments/001/
├── exp_001_datasets_stats.py      # 主实验脚本
├── cli.py                        # 命令行接口
├── config.py                     # 数据集配置（dataclass）
├── pipeline.py                   # Datatrove 处理流水线
├── collector.py                  # 统计收集器
├── stats_utils.py                # 统计工具函数
├── io_utils.py                  # I/O 工具
├── __init__.py
└── __main__.py
```

---

## 运行方式

### 基本命令

```bash
# 查看帮助
python -m experiments.001 --help

# 运行单个数据集
python -m experiments.001 explore \
    --dataset HuggingFaceFW/fineweb-edu \
    --data-dir data/datasets/HuggingFaceFW/fineweb-edu/data/ \
    --workers 8

# 运行所有有 score 字段的数据集
python -m experiments.001 explore --dataset all --workers 8

# 详细模式
python -m experiments.001 explore \
    --dataset HuggingFaceFW/fineweb-edu \
    --data-dir data/datasets/HuggingFaceFW/fineweb-edu/data/ \
    --workers 8 \
    --verbose
```

### 命令行参数

| 参数 | 简写 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| `--dataset` | - | str+ | ✅ | - | 数据集名称（支持多个或 'all'） |
| `--data-dir` | - | str | - | - | 数据目录（由配置自动填充） |
| `--output-dir` | - | str | - | `outputs/exp_001_datasets_stats` | 输出目录 |
| `--workers` | - | int | - | `8` | Worker 数量 |
| `--batch-size` | - | int | - | `5000` | 批量大小 |
| `--limit` | - | int | - | - | 限制文档数 |
| `--dry-run` | - | flag | - | `False` | 演示模式 |
| `--verbose` | -v | flag | - | `False` | 详细输出模式 |

---

## 支持的数据集

### HuggingFaceFW/fineweb-edu

**路径**: `data/datasets/HuggingFaceFW/fineweb-edu/data/`

**字段**:
- `text`: 文本内容
- `id`: 文档 ID
- `dump`: 数据集分组（metadata）
- `score`: 质量分数
- `int_score`: 整数分数

**统计项**:
- 文档总数
- Score 分布（min, max, mean, median, std）
- Score 分桶统计
- 文本长度分布
- Dump 级别统计

### opencsg/Fineweb-Edu-Chinese-V2.1

**路径**: `data/datasets/opencsg/Fineweb-Edu-Chinese-V2.1/`

**字段**:
- `text`: 文本内容
- `score`: 质量分数

**统计项**:
- 文档总数
- Score 分布
- 文本长度分布
- 目录级别统计

### HuggingFaceTB/finemath

**路径**: `data/datasets/HuggingFaceTB/finemath/`

**字段**:
- `text`: 文本内容
- `snapshot_type`: 快照类型（metadata）
- `score`: 质量分数
- `int_score`: 整数分数

**统计项**:
- 文档总数
- Score 分布
- 文本长度分布
- Snapshot type 统计

### nvidia/Nemotron-CC-Math-v1

**路径**: `data/datasets/nvidia/Nemotron-CC-Math-v1/`

**字段**:
- `text`: 文本内容
- `id`: 文档 ID
- `metadata.finemath_scores`: Score 字段（嵌套）
- `metadata.finemath_int_scores`: int_score 字段（嵌套）

**统计项**:
- 文档总数
- Score 分布（从嵌套字段提取）
- 文本长度分布

### nick007x/github-code-2025

**路径**: `data/datasets/nick007x/github-code-2025/`

**字段**:
- `content`: 代码内容（对应 text_key）
- `repo_id`: 仓库 ID（metadata）

**统计项**:
- 文档总数
- 代码长度分布
- 仓库级别统计

**注意**: 此数据集没有 score 字段，将跳过 Score 相关统计。

---

## 配置管理

### DatasetConfig 类

```python
@dataclass
class DatasetConfig:
    """数据集配置类。"""
    name: str                              # 数据集名称
    path: str                              # 数据路径
    text_key: str = "text"                 # 文本字段名
    id_key: Optional[str] = "id"            # ID 字段名
    group_field: Optional[str] = None         # 分组字段（metadata）
    group_by: Optional[str] = None           # 分组策略
    score_field: Optional[str] = "score"     # Score 字段名或路径
    int_score_field: Optional[str] = None     # int_score 字段名
    glob_pattern: str = "**/*.parquet"      # 文件匹配模式
```

### 数据集注册表

所有支持的数据集在 `DATASET_CONFIGS` 字典中注册：

```python
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "HuggingFaceFW/fineweb-edu": DatasetConfig(
        name="HuggingFaceFW/fineweb-edu",
        path="data/datasets/HuggingFaceFW/fineweb-edu/data/",
        text_key="text",
        id_key="id",
        group_field="dump",
        score_field="score",
        int_score_field="int_score",
    ),
    # ...
}
```

---

## 处理流水线

### 流水线组件

1. **ParquetReader**: 读取 Parquet 数据文件
2. **DatasetStatsCollector**: 收集统计信息
3. **结果聚合**: 聚合所有 worker 的统计结果

### 流水线创建

```python
def create_pipeline(
    config: DatasetConfig,
    output_dir: str,
    batch_size: int,
    limit: Optional[int]
) -> List[PipelineStep]:
    """创建 datatrove 处理流水线。"""
    return [
        ParquetReader(
            data_folder=config.path,
            glob_pattern=config.glob_pattern,
            batch_size=batch_size,
            limit=limit if limit else -1,
            text_key=config.text_key,
            adapter=create_adapter(config),
        ),
        create_stats_collector(config, output_dir),
    ]
```

---

## 输出结果

### 输出目录结构

```
outputs/exp_001_datasets_stats/
├── HuggingFaceFW_fineweb-edu/
│   ├── summary.json           # 汇总统计
│   ├── score_distribution.json  # Score 分布
│   ├── text_length_distribution.json  # 文本长度分布
│   └── metadata/
│       └── dump_stats.json   # Dump 级别统计
├── opencsg_Fineweb-Edu-Chinese-V2.1/
│   └── ...
└── ...
```

### 汇总统计（summary.json）

```json
{
    "dataset_name": "HuggingFaceFW/fineweb-edu",
    "total_documents": 1000000,
    "total_files": 100,
    "processing_time_seconds": 120.5,
    "statistics": {
        "score": {
            "min": 0.0,
            "max": 1.0,
            "mean": 0.75,
            "median": 0.8,
            "std": 0.15,
            "count": 1000000
        },
        "text_length": {
            "min": 10,
            "max": 100000,
            "mean": 5000,
            "median": 3000,
            "std": 8000
        }
    }
}
```

---

## 示例输出

```bash
$ python -m experiments.001 explore --dataset HuggingFaceFW/fineweb-edu --workers 8

================================================================================
数据集统计与探索
================================================================================

数据集: HuggingFaceFW/fineweb-edu
数据路径: data/datasets/HuggingFaceFW/fineweb-edu/data/
Worker 数量: 8
批量大小: 5000

正在创建数据处理流水线...
开始处理数据...

[INFO] Worker 0: 处理中...
[INFO] Worker 1: 处理中...
...

================================================================================
统计结果
================================================================================
数据集: HuggingFaceFW/fineweb-edu
总文档数: 1,000,000
总文件数: 100
处理时间: 120.5 秒

Score 统计:
  最小值: 0.0
  最大值: 1.0
  平均值: 0.75
  中位数: 0.8
  标准差: 0.15

文本长度统计:
  最小值: 10
  最大值: 100,000
  平均值: 5,000
  中位数: 3,000

输出目录: outputs/exp_001_datasets_stats/HuggingFaceFW_fineweb-edu/
```

---

## 故障排查

### 问题 1: 数据集不存在

**错误**: `ValueError: Unknown dataset: xxx`

**解决方案**:
- 检查数据集名称是否正确
- 查看支持的数据集列表
- 添加新数据集配置到 `config.py`

### 问题 2: 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小 `--batch-size`
- 减少并行 worker 数量（`--workers`）
- 使用 `--limit` 限制处理的文档数

### 问题 3: 数据路径错误

**错误**: `FileNotFoundError: data/datasets/xxx`

**解决方案**:
- 确认数据已下载到正确路径
- 检查 `--data-dir` 参数
- 更新 `config.py` 中的路径配置

---

## 相关文档

- [项目结构](project-structure.md) - 实验目录组织
- [实验管理](management.md) - 实验追踪和对比
- [开始实验](getting-started.md) - 创建新实验
- [Fineweb 统计实验](fineweb_stats.md) - 详细的 Fineweb 统计分析

---

## 下一步

- 分析统计结果，了解数据集特征
- 基于 Score 筛选高质量数据
- 选择适合的数据集进行后续训练实验
