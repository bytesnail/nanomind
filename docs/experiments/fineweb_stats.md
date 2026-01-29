# FineWeb-Edu 统计实验文档

## 概述

FineWeb-Edu 统计实验提供了对 HuggingFaceFW/fineweb-edu 数据集的完整分析和统计收集功能。该实验使用 DataTrove 框架进行高效的数据处理，支持多 worker 并发处理和结果聚合。

## 项目结构

### 核心文件

```
experiments/
├── fineweb_explore.py          # 主探索脚本（datatrove 优化版）
├── fineweb_stats_collector.py  # 自定义统计收集器 PipelineStep
├── fineweb_demo.py              # 快速演示脚本（100个文档）
└── run_fineweb.py               # 统一命令行入口

configs/
├── fineweb.yaml                 # 完整探索配置
└── fineweb_quick.yaml           # 快速统计配置

tests/experiments/
└── test_fineweb_stats.py        # 统计收集器测试
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

### 3. 统一命令行入口 (run_fineweb.py)

提供统一的命令行接口，支持以下子命令：

- `explore`: 完整数据集探索
- `quick`: 快速统计（限制文档数）
- `demo`: 运行演示（100 个文档）
- `test`: 运行测试

## 使用方法

### 完整数据集探索

```bash
# 使用默认配置
python experiments/run_fineweb.py explore

# 指定配置文件
python experiments/run_fineweb.py explore --config configs/fineweb.yaml

# 自定义参数
python experiments/run_fineweb.py explore --workers 16 --batch-size 10000
```

### 快速统计

```bash
# 使用默认配置（限制 10000 个文档）
python experiments/run_fineweb.py quick

# 自定义限制文档数
python experiments/run_fineweb.py quick --limit 5000
```

### 运行演示

```bash
# 快速演示（处理 100 个文档）
python experiments/run_fineweb.py demo
```

### 运行测试

```bash
# 运行所有测试
python experiments/run_fineweb.py test

# 详细输出
python experiments/run_fineweb.py test -- -v
```

### 直接调用脚本

```bash
# 直接调用探索脚本
python experiments/fineweb_explore.py --config configs/fineweb.yaml

# 直接调用演示脚本
python experiments/fineweb_demo.py
```

## 配置说明

### 完整探索配置 (configs/fineweb.yaml)

```yaml
# 数据路径和批处理
data:
  path: "data/datasets/HuggingFaceFW/fineweb-edu/data"
  batch_size: 5000

# 处理配置
processing:
  workers: 8
  limit: null  # null 表示全量处理

# 输出配置
output:
  dir: "outputs/fineweb"
  log_level: "INFO"

# 统计收集选项
stats:
  collect_domain: true
  collect_score: true
  collect_int_score: true
  collect_snapshot: true
  include_doc_stats: true
  include_lang_stats: true
```

### 快速统计配置 (configs/fineweb_quick.yaml)

```yaml
# 数据路径和批处理
data:
  path: "data/datasets/HuggingFaceFW/fineweb-edu/data"
  batch_size: 1000

# 处理配置
processing:
  workers: 2
  limit: 10000  # 仅处理 10k 文档

# 输出配置
output:
  dir: "outputs/fineweb_quick"
  log_level: "INFO"

# 统计收集选项
stats:
  collect_domain: true
  collect_score: true
  collect_int_score: false
  collect_snapshot: false
  include_doc_stats: true
  include_lang_stats: true
```

## 命令行参数

### 全局参数

- `--config`: 配置文件路径（默认: `configs/fineweb.yaml`）
- `--verbose`, `-v`: 详细输出模式（DEBUG 日志级别）
- `--dry-run`: 演示模式，仅打印配置不实际执行

### explore 子命令参数

- `--data-dir`: 数据集路径（默认: 从配置文件读取）
- `--output-dir`: 输出目录（默认: 从配置文件读取）
- `--workers`: 并行 worker 数量（默认: 从配置文件读取）
- `--batch-size`: 批量大小（默认: 从配置文件读取）
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
python experiments/run_fineweb.py explore --verbose

# 使用 dry run 查看配置
python experiments/run_fineweb.py explore --dry-run

# 运行测试验证功能
python experiments/run_fineweb.py test
```

## 相关文档

- [项目文档](README.md)
- [环境初始化](../environment/setup.md)
- [实验管理](management.md)
- [项目结构](project-structure.md)

## 版本历史

- **v1.0** (2026-01-29): 初始版本，统一命令行入口，重构实验脚本
