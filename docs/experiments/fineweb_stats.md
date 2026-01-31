# FineWeb-Edu 统计实验文档

## 概述

FineWeb-Edu 统计实验提供了对 HuggingFaceFW/fineweb-edu 数据集的完整分析和统计收集功能。该实验使用 DataTrove 框架进行高效的数据处理，支持多 worker 并发处理和结果聚合。

> **注意**：FineWeb-Edu 统计功能已整合到 `exp_001` 数据集统计实验中。本文档详细介绍 FineWeb-Edu 数据集的统计分析和结果解读。完整的实验说明请参考 [exp-001 概览](exp-001-overview.md)。

## 项目结构

### 实验文件（exp_001）

FineWeb-Edu 统计作为 exp_001 的一个支持数据集，使用以下文件：

```
experiments/001/
├── exp_001_datasets_stats.py    # 主实验脚本
├── cli.py                        # 命令行接口
├── config.py                     # 数据集配置（包含 fineweb-edu 配置）
├── pipeline.py                   # Datatrove 处理流水线
├── collector.py                  # 统计收集器
└── stats_utils.py                # 统计工具函数

experiments/utils/
├── common.py                     # 通用工具（日志、格式化等）
├── paths.py                      # 路径处理
└── constants.py                 # 常量定义
```

### 输出结构

```
outputs/
├── fineweb/                     # 完整探索输出
│   ├── logs/                    # 日志文件
│   │   └── stats/               # datatrove 内置统计
│   ├── results/                 # 最终结果
│   │   └── fineweb_full_statistics.json
│   └── fineweb_edu_stats/       # 自定义统计
│       ├── worker_*.json        # 各 worker 统计
│       └── aggregated_stats.json  # 聚合统计
└── fineweb_quick/               # 快速统计输出
```

## 功能特性

### 1. 数据集探索 (fineweb_explore.py)

使用 DataTrove 框架进行全量数据分析：

- **多 worker 并发处理**: 支持并行处理，提高效率
- **断点续传**: 支持跳过已完成的任务
- **灵活配置**: 支持 YAML 配置文件和命令行参数
- **完整统计**: 收集文档、语言、域名、快照、score 等多种统计信息

### 2. 自定义统计收集器 (fineweb_stats_collector.py)

实现 DataTrove PipelineStep，收集以下统计信息：

- **域名分布**: Top-1000 域名和总域名数
- **快照统计**: 每个快照的文档数和文件数
- **score 分布**: mean、median、std、min、max、分位数
- **int_score 分布**: 计数分布

### 3. 命令行接口

实验使用 `python -m experiments.001` 命令运行，支持以下参数：

- `--dataset`: 指定数据集名称（如 `HuggingFaceFW/fineweb-edu`）
- `--data-dir`: 数据目录（默认从配置读取）
- `--workers`: Worker 数量（默认 8）
- `--batch-size`: 批量大小（默认 5000）
- `--limit`: 限制处理的文档数（可选）
- `--verbose`: 详细输出模式（可选）

## 使用方法

### 运行 FineWeb-Edu 统计

```bash
# 完整数据集统计
python -m experiments.001 explore --dataset HuggingFaceFW/fineweb-edu --workers 8

# 快速统计（限制文档数）
python -m experiments.001 explore --dataset HuggingFaceFW/fineweb-edu --workers 8 --limit 10000

# 详细输出模式
python -m experiments.001 explore --dataset HuggingFaceFW/fineweb-edu --workers 8 --verbose
```

### 查看帮助

```bash
python -m experiments.001 --help
```


## 配置说明

FineWeb-Edu 数据集配置存储在 `experiments/001/config.py` 中：

```python
@dataclass
class DatasetConfig:
    """数据集配置类。"""
    name: str
    path: str
    text_key: str = "text"
    id_key: Optional[str] = "id"
    group_field: Optional[str] = None
    score_field: Optional[str] = "score"
    int_score_field: Optional[str] = None
    glob_pattern: str = "**/*.parquet"

DATASET_CONFIGS = {
    "HuggingFaceFW/fineweb-edu": DatasetConfig(
        name="HuggingFaceFW/fineweb-edu",
        path="data/datasets/HuggingFaceFW/fineweb-edu/data/",
        text_key="text",
        id_key="id",
        group_field="dump",
        score_field="score",
        int_score_field="int_score",
        glob_pattern="**/*.parquet",
    ),
    # ... 其他数据集配置
}
```

## 命令行参数

主要参数：

- `--dataset`: 数据集名称（必需），如 `HuggingFaceFW/fineweb-edu`
- `--data-dir`: 数据目录（可选，默认从配置读取）
- `--output-dir`: 输出目录（可选，默认 `outputs/exp_001_datasets_stats`）
- `--workers`: 并行 worker 数量（默认 8）
- `--batch-size`: 批量大小（默认 5000）
- `--limit`: 限制处理的文档数（可选）
- `--verbose`, `-v`: 详细输出模式（可选）
- `--dry-run`: 演示模式，仅打印配置不实际执行（可选）
- `--log-level`: 日志级别（默认: 从配置文件读取）

### quick 子命令参数

- `--limit`: 限制处理文档数（默认: 10000）
- `--data-dir`: 数据集路径（默认: 从配置文件读取）
- `--output-dir`: 输出目录（默认: 从配置文件读取）
- `--workers`: 并行 worker 数量（默认: 从配置文件读取）
- `--batch-size`: 批量大小（默认: 从配置文件读取）

## 输出结果

### 最终统计文件 (fineweb_full_statistics.json)

```json
{
  "metadata": {
    "experiment": "FineWeb-Edu 数据集探索（datatrove优化版）",
    "dataset": "HuggingFaceFW/fineweb-edu",
    "start_time": "2026-01-29T00:00:00",
    "end_time": "2026-01-29T01:00:00"
  },
  "global_statistics": {
    "total_documents": 1000000,
    "total_tokens": 5000000000,
    "average_tokens_per_document": 5000
  },
  "snapshot_statistics": {
    "CC-MAIN-2021-21": {
      "doc_count": 500000,
      "file_count": 1000
    }
  },
  "language_distribution": {
    "en": 900000,
    "es": 50000,
    "fr": 30000
  },
  "domain_distribution": {
    "total_domains": 50000,
    "top_1000": {
      "example.com": 10000,
      "test.org": 8000
    }
  },
  "score_statistics": {
    "mean": 2.5,
    "median": 2.3,
    "std": 1.2,
    "min": 0.0,
    "max": 5.0,
    "percentiles": {
      "10%": 1.0,
      "50%": 2.3,
      "90%": 4.0
    },
    "total_docs": 1000000
  },
  "int_score_distribution": {
    "1": 500000,
    "2": 300000,
    "3": 150000,
    "4": 40000,
    "5": 10000
  }
}
```

## 测试

运行测试验证功能：

```bash
# 运行所有测试
pytest tests/experiments/test_fineweb_stats.py -v

# 运行特定测试
pytest tests/experiments/test_fineweb_stats.py::test_single_worker -v

# 查看测试覆盖率
pytest tests/experiments/test_fineweb_stats.py --cov
```

### 测试覆盖内容

1. **单 worker 统计收集**: 验证单个 worker 的统计收集功能
2. **多 worker 聚合**: 验证多 worker 统计结果聚合
3. **数据结构格式**: 验证输出数据格式是否符合要求

## 最佳实践

### 1. 选择合适的配置

- **完整探索**: 使用 `fineweb.yaml`，处理全量数据
- **快速验证**: 使用 `fineweb_quick.yaml`，处理小样本数据
- **调试**: 使用 `--verbose` 和 `--dry-run` 参数

### 2. 性能优化

- 根据 CPU 核心数调整 `--workers` 参数
- 根据 GPU 内存调整 `--batch-size` 参数
- 使用 `--limit` 参数快速验证流程

### 3. 结果分析

- 查看最终统计文件 `fineweb_full_statistics.json`
- 检查日志文件了解处理过程
- 分析 datatrove 内置统计和自定义统计

## 故障排查

### 常见问题

1. **数据目录不存在**
   - 确认数据路径正确
   - 检查数据是否已下载

2. **内存不足**
   - 减小 `--batch-size`
   - 减少 `--workers` 数量
   - 使用 `--limit` 限制处理文档数

3. **统计聚合失败**
   - 检查日志文件了解错误原因
   - 确认所有 worker 统计文件已生成

### 调试技巧

```bash
# 使用详细日志
python -m experiments.001 explore --dataset HuggingFaceFW/fineweb-edu --workers 8 --verbose

# 使用 dry run 查看配置
python -m experiments.001 explore --dataset HuggingFaceFW/fineweb-edu --dry-run

# 查看帮助信息
python -m experiments.001 --help
```

## 相关文档

- [exp-001 概览](exp-001-overview.md) - 数据集统计实验完整说明
- [项目结构](project-structure.md) - 实验目录组织
- [实验管理](management.md) - 实验追踪和对比
- [环境初始化](../environment/setup.md) - 配置开发环境
- [开始实验](getting-started.md) - 创建新实验

## 版本历史

- **v1.0** (2026-01-29): 初始版本，统一命令行入口，重构实验脚本
