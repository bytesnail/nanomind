# FineWeb-Edu 数据集质量评分分桶重组系统设计文档

> **版本**: 2.0  
> **最后更新**: 2026-03-04  
> **目标数据集**: HuggingFaceFW/fineweb-edu (英文), opencsg/Fineweb-Edu-Chinese-V2.1 (中文)

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [核心组件](#3-核心组件)
4. [配置系统](#4-配置系统)
5. [性能优化](#5-性能优化)
6. [多语言支持](#6-多语言支持)
7. [测试策略](#7-测试策略)
8. [扩展指南](#8-扩展指南)

---

## 1. 项目概述

### 1.1 设计目标

实现 FineWeb-Edu 数据集按质量评分（`score` 字段）进行分桶重组，通过分层采样策略控制低质量数据的保留比例。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **单次读取多桶处理** | 单遍读取输入数据，同时处理所有评分桶，I/O 效率提升约 75% |
| **确定性采样** | 基于 MD5 哈希的伪随机采样，确保结果可复现 |
| **多语言支持** | 自动识别英文原版（1.0-5.0分）和中文版本（归一化0.0-1.0分） |
| **灵活配置** | YAML 配置文件 |
| **模块化架构** | 基于 Datatrove Pipeline 的组件化设计 |

### 1.3 评分桶策略

配置见 [`config/dataset.yaml`](../../config/dataset.yaml)。

#### 英文数据集（HuggingFaceFW/fineweb-edu）

| 质量评分区间 | 桶名称 | 采样率 | 数据质量等级 |
|-------------|--------|--------|-------------|
| 2.5 ≤ score < 3.0 | 2.5 | 25% | 中低质量 |
| 3.0 ≤ score < 3.5 | 3.0 | 50% | 中等质量 |
| 3.5 ≤ score < 4.0 | 3.5 | 80% | 高质量 |
| score ≥ 4.0 | 4.0 | 100% | 顶级质量 |

#### 中文数据集（Fineweb-Edu-Chinese-V2.1）

中文数据集使用归一化评分（0.0-1.0），系统自动转换为 0.0-5.0 范围：

| 质量评分区间 | 桶名称 | 采样率 | 归一化范围 |
|-------------|--------|--------|-----------|
| 2.5 ≤ score < 3.0 | 2.5 | 40% | 0.50-0.60 |
| 3.0 ≤ score < 3.5 | 3.0 | 60% | 0.60-0.70 |
| 3.5 ≤ score < 4.0 | 3.5 | 90% | 0.70-0.80 |
| score ≥ 4.0 | 4.0 | 100% | ≥ 0.80 |

**转换公式**: `score = normalized_score × 5`

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    FineWeb-Edu Reorganizer                      │
├─────────────────────────────────────────────────────────────────┤
│  配置文件层 (config/)                                             │
│  ├── dataset.yaml        # 数据集定义                            │
│  ├── processing.yaml     # 处理参数                              │
│  └── paths.yaml          # 路径配置                              │
├─────────────────────────────────────────────────────────────────┤
│  数据处理层 (src/data_processing/)                                │
│  ├── fineweb_edu/              # FineWeb-Edu 专用子模块           │
│  │   ├── adapters.py          # 数据适配器                       │
│  │   └── reorganizer.py       # CLI 主入口                       │
│  ├── bucket_config.py          # 评分桶配置管理                   │
│  ├── score_filter.py           # 评分过滤 + 确定性采样            │
│  ├── bucket_path_writer.py     # 多桶并行写入器                   │
│  ├── parquet_merger.py         # Parquet 文件合并工具             │
│  ├── validation.py             # 数据验证                        │
│  └── config_loader.py          # 配置加载器                      │
├─────────────────────────────────────────────────────────────────┤
│  常量定义 (src/constants.py)                                      │
│  └── 项目级常量（特殊token、语言扩展名等）                         │
├─────────────────────────────────────────────────────────────────┤
│  工具脚本层 (scripts/)                                            │
│  ├── trial_run.py              # 试运行脚本                      │
│  ├── validate_output.py        # 输出验证工具                    │
│  └── utils.py                  # 工具函数                        │
├─────────────────────────────────────────────────────────────────┤
│  测试层 (tests/)                                                  │
│  └── 9个单元测试文件                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据处理流水线

```
输入 Parquet 文件
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ ParquetReader │ ──> │ ScoreFilter  │ ──> │BucketPathWriter│
│   数据读取    │     │ 评分过滤+采样 │     │   分桶写入    │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
   - 字段筛选           - 区间匹配           - 多桶并行
   - ID 生成            - 确定性采样          - 自动分片
   - 评分归一化          - 统计追踪            - 压缩输出
       │                    │                    │
       ▼                    ▼                    ▼
     提取 text,           匹配评分桶           按桶写入
     score, dump          应用采样率           Parquet
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │  ParquetMerger  │
                                        │  合并小文件     │
                                        └─────────────────┘
                                                  │
                                                  ▼
                                           {00000,...}.parquet
```

### 2.3 输出目录结构

```
data/datasets/fineweb/
├── en/                              # 英文数据集输出
│   ├── 2.5/
│   │   ├── 00000.parquet
│   │   └── ...
│   ├── 3.0/
│   ├── 3.5/
│   └── 4.0/
└── zh/                              # 中文数据集输出
    ├── 2.5/
    ├── 3.0/
    ├── 3.5/
    └── 4.0/
```

---

## 3. 核心组件

### 3.1 数据适配器 (`fineweb_edu/adapters.py`)

**职责**: 将原始数据转换为统一格式。

**处理流程**:
1. **字段筛选**: 提取 `text`, `score`, `dump`
2. **ID 生成**: 创建唯一文档标识符 `{source}#{索引}`
3. **评分归一化**: 根据配置自动转换评分（中文数据集 ×5）

**输出格式**:
```python
{
    "text": str,
    "id": str,  # 格式: "{相对路径}#{索引}"
    "metadata": {
        "score": float,      # 归一化后的质量评分
        "cc_main": str,      # Common Crawl 源标识
    }
}
```

**数据集自动识别**:
```python
def _get_dataset_config_for_source(reader: Any | None = None) -> dict[str, Any]:
    """根据数据读取器自动识别数据集类型"""
    all_configs = get_dataset_configs()
    data_folder_path = str(reader.data_folder.path)
    
    for config in all_configs.values():
        input_dir = config.get("input_dir", "")
        if input_dir and input_dir in data_folder_path:
            return config
    return {}
```

### 3.2 评分过滤器 (`score_filter.py`)

继承自 Datatrove `PipelineStep`，实现评分过滤和分层采样。

#### 核心算法

**1. 评分区间匹配（二分查找）** [`score_filter.py:61`](../../src/data_processing/score_filter.py:61)

```python
def find_bucket_in_sorted(score, buckets):
    """O(log n) 复杂度二分查找"""
    left, right = 0, len(buckets)
    while left < right:
        mid = (left + right) // 2
        bucket = buckets[mid]
        if score < bucket.min_score:
            right = mid
        elif bucket.max_score is not None and score >= bucket.max_score:
            left = mid + 1
        else:
            return bucket
    return None
```

**2. 确定性采样** [`score_filter.py:82`](../../src/data_processing/score_filter.py:82)

```python
def _should_sample(self, doc_id: str, rate: float) -> bool:
    """基于 MD5 哈希的确定性采样"""
    if rate >= 1.0:
        return True
    data = f"{self.random_seed}_{doc_id}".encode()
    h = int.from_bytes(
        hashlib.md5(data, usedforsecurity=False).digest()[:8], "big"
    )
    return h / (2**64) < rate
```

### 3.3 桶路径写入器 (`bucket_path_writer.py`)

实现多桶并行写入，支持大文件自动分片。

**内存缓冲策略**:
```python
self._states = {
    bucket_name: {
        "buffer": [],      # 文档缓冲列表
        "counter": 0,      # 文件计数器
        "size": 0,         # 当前缓冲字节数
    }
    for bucket_name in buckets
}
```

**自动分片机制**:
```python
# 当缓冲大小超过阈值时触发写入
if state["size"] + row_size > max_file_size and state["buffer"]:
    self._flush_bucket(bucket_name)

# 文件命名: {rank:05d}_{counter:05d}.parquet
```

### 3.4 评分桶配置 (`bucket_config.py`)

**BucketConfig 数据类** [`bucket_config.py:7`](../../src/data_processing/bucket_config.py:7):

```python
@dataclass(frozen=True)
class BucketConfig:
    name: str              # 桶名称（如 "3.0"）
    min_score: float       # 最小评分（包含）
    max_score: float | None  # 最大评分（不包含）
    sampling_rate: float   # 采样率 0.0-1.0

    def contains(self, score: float) -> bool:
        """检查评分是否在当前桶的区间内（左闭右开）"""
        if score < self.min_score:
            return False
        return self.max_score is None or score < self.max_score
```

**区间定义规则**:
- 采用**左闭右开**区间：`[min_score, max_score)`
- 最后一个桶 `max_score=None`，表示无上界
- 桶列表按 `min_score` 排序，支持二分查找

### 3.5 配置加载器 (`config_loader.py`)

**核心函数** [`config_loader.py:19`](../../src/data_processing/config_loader.py:19):

```python
@lru_cache(maxsize=10)
def _load_yaml(name: str) -> dict[str, Any]:
    """YAML 文件读取结果缓存"""
    ...

@lru_cache(maxsize=1)
def get_dataset_config_dict() -> dict[str, Any]:
    """获取完整的数据集配置字典"""
    return _load_yaml("dataset")

@lru_cache(maxsize=1)
def get_processing_config() -> dict[str, Any]:
    """获取处理参数配置"""
    return _load_yaml("processing")

@lru_cache(maxsize=1)
def get_paths_config() -> dict[str, Any]:
    """获取路径配置"""
    return _load_yaml("paths")
```

### 3.6 Parquet 文件合并器 (`parquet_merger.py`)

**职责**: 将多个小文件合并成指定大小（默认 10GB）的文件。

**合并策略**:

1. **单文件优化**: 桶内只有一个文件时，直接重命名为 `00000.parquet`
2. **流式处理**: 批量并行读取，边读边写
3. **桶间串行**: 多个桶按顺序处理，避免内存峰值

**核心函数** [`parquet_merger.py`](../../src/data_processing/parquet_merger.py):

```python
def merge_bucket_files(
    bucket_dir: Path,
    target_file_size: int,  # 默认 10GB
    compression: Compression = "zstd",
    remove_source: bool = True,
    max_workers: int | None = None,
) -> list[Path]:
    """将桶内的小文件合并成指定大小的大文件"""

def merge_all_buckets(
    output_dir: Path,
    target_file_size: int,
    compression: Compression = "zstd",
    remove_source: bool = True,
    max_workers: int | None = None,
) -> dict[str, list[Path]]:
    """串行合并所有桶的文件"""
```

---

## 4. 配置系统

### 4.1 数据集配置 (`config/dataset.yaml`)

```yaml
datasets:
  en:
    name: "fineweb_edu_en"
    score_normalization:
      enabled: false
    input_dir: "data/datasets/HuggingFaceFW/fineweb-edu"
    output_dir: "data/datasets/fineweb/en"
    buckets:
      - name: "2.5"
        min_score: 2.5
        max_score: 3.0
        sampling_rate: 0.25
      - name: "3.0"
        min_score: 3.0
        max_score: 3.5
        sampling_rate: 0.50
      - name: "3.5"
        min_score: 3.5
        max_score: 4.0
        sampling_rate: 0.80
      - name: "4.0"
        min_score: 4.0
        sampling_rate: 1.0

  zh:
    name: "fineweb_edu_zh"
    score_normalization:
      enabled: true
      multiplier: 5.0
    input_dir: "data/datasets/opencsg/Fineweb-Edu-Chinese-V2.1"
    output_dir: "data/datasets/fineweb/zh"
    buckets:
      - name: "2.5"
        min_score: 2.5
        max_score: 3.0
        sampling_rate: 0.40
      - name: "3.0"
        min_score: 3.0
        max_score: 3.5
        sampling_rate: 0.60
      - name: "3.5"
        min_score: 3.5
        max_score: 4.0
        sampling_rate: 0.90
      - name: "4.0"
        min_score: 4.0
        sampling_rate: 1.0
```

**自动识别机制**: 系统通过匹配 `input_dir` 路径来识别数据集类型。

### 4.2 处理参数配置 (`config/processing.yaml`)

```yaml
workers: 32                       # 本地并行工作进程数
tasks: 2500                       # Datatrove 任务数
random_seed: 42                   # 采样随机种子
compression: "zstd"               # Parquet 压缩格式
max_file_size_bytes: 10737418240  # 输出文件大小限制（10GB）

logging:
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  level: "INFO"

trial:                            # 试运行参数
  max_files: 5
  max_rows: 2000
  max_file_size_bytes: 134217728  # 128MB
```

### 4.3 路径配置 (`config/paths.yaml`)

```yaml
log_dir: "logs/fineweb_processing"
trial_input_dir: "data/datasets/test_fineweb_input"
trial_output_dir: "data/datasets/test_fineweb_output"
```

---

## 5. 性能优化

### 5.1 单次读取多桶处理

**传统方式 vs 优化方式**:

```
传统方式（N个桶需要N次读取）:
读取数据 → 过滤桶1 → 写入桶1
读取数据 → 过滤桶2 → 写入桶2
...

优化方式（单次读取）:
读取数据
      │
      ▼
评分过滤（同时检查所有桶）
      │
      ├── 匹配桶1 (30% 采样) --> 写入桶1
      ├── 匹配桶2 (60% 采样) --> 写入桶2
      ├── 匹配桶3 (80% 采样) --> 写入桶3
      └── 匹配桶4 (100% 采样) -> 写入桶4
```

**性能提升**: 对于 4 个桶，I/O 量减少约 75%。

### 5.2 Datatrove 并行架构

```
LocalPipelineExecutor
├── tasks: 2500           # 任务总数（数据分片）
├── workers: 32           # 本地并行进程数
└── pipeline:
    ├── ParquetReader     # 并行读取
    ├── ScoreFilter       # 并行过滤
    └── BucketPathWriter  # 并行写入
```

### 5.3 内存与文件优化

#### Pipeline 阶段优化

- **流式处理**: 基于生成器的流水线，避免全量加载
- **缓冲写入**: `BucketPathWriter` 按文件大小阈值批量写入
- **对象复用**: 使用 `@lru_cache` 缓存配置对象

#### 文件合并阶段优化 (`ParquetMerger`)

**优化策略**:

| 优化项 | 实现方式 | 效果 |
|--------|---------|------|
| **单文件短路** | 只有一个文件时直接 `rename` | 零 I/O 开销 |
| **批量并行读取** | 每批读取 `max_workers * 2` 个文件 | I/O 并行化 |
| **流式写入** | 使用 `ParquetWriter` 边读边写 | 内存占用稳定 |
| **并行删除** | 使用 `ThreadPoolExecutor` 批量删除 | 清理加速 |
| **桶间串行** | 多个桶按顺序处理 | 避免内存峰值 |

**内存占用估算**:
```
峰值内存 ≈ batch_size × 平均文件大小
         = (max_workers × 2) × 500KB
         = 64 × 500KB = 32MB (max_workers=32 时)
```

---

## 6. 多语言支持

### 6.1 评分差异处理

| 数据集 | 存储格式 | 处理方式 |
|--------|----------|----------|
| HuggingFaceFW/fineweb-edu | 原始值 (1.0-5.0) | 直接使用 |
| Fineweb-Edu-Chinese-V2.1 | 归一化 (0.0-1.0) | 自动 ×5 转换 |

### 6.2 中文数据集评分详解

**转换公式**: `score = normalized_score × 5`

**数据集预过滤**:
仅保留归一化评分 ≥ 0.50（转换后评分 ≥ 2.5）的样本：

| 文件夹 | 归一化范围 | 转换后评分 |
|--------|-----------|-----------|
| `2_3/` | 0.50-0.60 | 2.5-3.0 |
| `3_4/` | 0.60-0.80 | 3.0-4.0 |
| `4_5/` | 0.80-0.94 | 4.0-4.70 |

### 6.3 自动检测机制

```python
def _get_dataset_config_for_source(reader: Any | None = None) -> dict[str, Any]:
    """根据数据读取器自动识别数据集类型"""
    all_configs = get_dataset_configs()
    data_folder_path = str(reader.data_folder.path)
    
    for config in all_configs.values():
        input_dir = config.get("input_dir", "")
        if input_dir and input_dir in data_folder_path:
            return config
    
    return {}
```

---

## 7. 测试策略

### 7.1 单元测试覆盖

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_adapters.py` | ID 生成、评分归一化、字段提取 |
| `test_bucket_config.py` | 评分桶区间匹配、配置加载 |
| `test_score_filter.py` | 过滤逻辑、采样算法、边界条件 |
| `test_bucket_path_writer.py` | 多桶写入、文件分片、路由逻辑 |
| `test_fineweb_reorganizer.py` | 流水线组装、配置加载、集成测试 |
| `test_parquet_merger.py` | 文件合并、大小控制、压缩 |
| `test_prepare_tokenizer_data.py` | Tokenizer数据准备、采样 |

### 7.2 关键测试场景

**1. 区间边界测试**（左闭右开）:
```python
(2.5, "2.5")    # 边界包含
(2.9, "2.5")    # 区间内
(3.0, "3.0")    # 边界不包含在前一桶
```

**2. 确定性采样测试**:
```python
# 相同种子应产生相同结果
f1 = ScoreFilter(buckets, random_seed=42)
f2 = ScoreFilter(buckets, random_seed=42)
assert sample_results(f1) == sample_results(f2)
```

### 7.3 试运行脚本

```bash
# 创建测试数据并运行完整流程
python scripts/trial_run.py

# 试运行指定数据集
python scripts/trial_run.py --dataset zh

# 分析采样准确性
python scripts/trial_run.py --analyze-sampling
```

---

## 8. 扩展指南

### 8.1 添加新的评分桶

编辑 [`config/dataset.yaml`](../../config/dataset.yaml)，在对应数据集下添加新的桶定义：

```yaml
buckets:
  - name: "2.8"           # 新桶名称
    min_score: 2.8        # 最小评分（包含）
    max_score: 3.0        # 最大评分（不包含）
    sampling_rate: 0.4    # 采样率 40%
```

**注意事项**:
- 确保新区间不与现有区间重叠
- 保持桶按 `min_score` 升序排列
- 最后一个桶可将 `max_score` 设为 `null` 表示无上界

### 8.2 添加新的数据集

**步骤 1**: 准备数据，下载到 `data/datasets/`

**步骤 2**: 配置数据集 ([`config/dataset.yaml`](../../config/dataset.yaml)):

```yaml
datasets:
  new_lang:
    name: "fineweb_edu_new"
    score_normalization:
      enabled: true
      multiplier: 5.0
    input_dir: "data/datasets/your-dataset"
    output_dir: "data/datasets/fineweb/new_lang"
    buckets:
      - name: "2.5"
        min_score: 2.5
        max_score: 3.0
        sampling_rate: 0.25
```

**步骤 3**: 运行处理:
```bash
python -m src.data_processing.fineweb_edu
```

### 8.3 添加新的数据处理子模块

要为新的数据集创建处理子模块，参考以下结构：

```
src/data_processing/
└── your_dataset/            # 新的数据集子模块
    ├── __init__.py          # 导出子模块 API
    ├── __main__.py          # CLI 入口（可选）
    ├── adapters.py          # 数据集适配器（必须）
    └── processor.py         # 处理逻辑（可选）
```

在 `adapters.py` 中实现：
- `normalize_score()` 函数处理评分转换
- `_get_dataset_config_for_source()` 函数自动识别数据集
- 适配器函数转换原始数据格式

---

## 附录

### 项目常量 ([`src/constants.py`](../../src/constants.py))

```python
# Tokenizer 特殊 tokens
SPECIAL_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<think>", "</think>"
]

# GitHub Code 语言扩展名映射
LANGUAGE_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".c": "c", ".cpp": "cpp", ".rs": "rust",
    # ... 共18种扩展名
}
```

### 相关文档

- [Tokenizer 训练设计](tokenizer_training_design.md)
- [项目知识库](KNOWLEDGE_BASE.md)
- [FineWeb-Edu 论文](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Datatrove 文档](https://github.com/huggingface/datatrove)

### 数据许可

- **FineWeb-Edu 数据集**: ODC-BY 1.0 License
- **项目代码**: MIT License

---

*最后更新: 2026-03-04*
