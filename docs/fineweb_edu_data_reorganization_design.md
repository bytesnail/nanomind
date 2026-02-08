# FineWeb-Edu 数据集质量评分分桶重组系统设计文档

> **版本**: 1.0  
> **最后更新**: 2025年2月  
> **目标数据集**: HuggingFaceFW/fineweb-edu (英文), opencsg/Fineweb-Edu-Chinese-V2.1 (中文)

## 1. 项目概述

### 1.1 设计目标

本项目实现 FineWeb-Edu 数据集按质量评分（`score` 字段）进行分桶重组，通过分层采样策略控制低质量数据的保留比例，从而构建高质量的训练数据集。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **多语言支持** | 同时支持英文原版（1.0-5.0分）和中文版本（归一化0.0-1.0分） |
| **一次读取多桶处理** | 单遍读取输入数据，同时处理所有评分桶，避免重复 I/O |
| **确定性采样** | 基于 MD5 哈希的伪随机采样，确保结果可复现 |
| **灵活配置** | YAML 配置文件 + 环境变量覆盖机制 |
| **可扩展架构** | 基于 Datatrove Pipeline 的模块化设计 |

### 1.3 评分桶策略

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

**转换公式**: `original_score = normalized_score × 5`

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FineWeb-Edu Reorganizer                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  配置文件层 (config/)                                                         │
│  ├── dataset.yaml        # 数据集定义（输入路径、评分桶、归一化配置）            │
│  ├── processing.yaml     # 处理参数（workers、压缩、文件大小）                 │
│  └── paths.yaml          # 路径配置（支持环境变量覆盖）                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  数据处理层 (src/data_processing/)                                            │
│  ├── fineweb_reorganizer.py    # CLI 主入口，多数据集协调                      │
│  ├── adapters.py               # 数据适配器（ID生成、评分归一化）              │
│  ├── score_filter.py           # 评分过滤 + 确定性采样                         │
│  ├── bucket_path_writer.py     # 多桶并行写入器                               │
│  ├── bucket_config.py          # 评分桶配置管理                               │
│  └── config_loader.py          # 配置加载器（缓存 + 环境变量）                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  工具脚本层 (scripts/)                                                        │
│  ├── trial_run.py              # 试运行脚本（创建测试数据 + 验证）              │
│  └── validate_output.py        # 输出验证工具                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  测试层 (tests/)                                                              │
│  └── 单元测试覆盖所有核心组件                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据处理流水线

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

### 2.3 输出目录结构

```
data/datasets/fineweb/
├── en/                              # 英文数据集输出
│   ├── 2.5/
│   │   ├── 00000_00000.parquet     # {rank}_{counter}.parquet
│   │   ├── 00000_00001.parquet
│   │   └── ...
│   ├── 3.0/
│   ├── 3.5/
│   └── 4.0/
│
└── zh/                              # 中文数据集输出
    ├── 2.5/
    ├── 3.0/
    ├── 3.5/
    └── 4.0/
```

---

## 3. 核心组件详解

### 3.1 数据适配器 (`adapters.py`)

#### 职责
- 字段筛选：从原始数据中提取 `text`, `score`, `dump`
- ID 生成：创建唯一文档标识符
- 评分归一化：支持中文数据集的评分转换

#### ID 生成规则

```python
{相对路径}#{索引}

# 示例
"data/datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2024-10/train.parquet#42"
      ↓ 提取相对路径（从 root_marker 后开始）
"data/CC-MAIN-2024-10/train.parquet#42"
```

#### 评分归一化

```python
def normalize_score(raw_score, normalization_config):
    """
    支持两种模式：
    1. 无归一化: 英文数据集直接使用原始评分
    2. 乘数归一化: 中文数据集 score × 5
    """
    if normalization_config and normalization_config.get("enabled"):
        multiplier = normalization_config.get("multiplier", 1.0)
        return raw_score * multiplier
    return raw_score
```

### 3.2 评分过滤器 (`score_filter.py`)

继承自 `Datatrove PipelineStep`，实现评分过滤和分层采样。

#### 核心算法

**1. 评分区间匹配（二分查找）**

```python
def _find_bucket_in_sorted(score, buckets):
    """在已排序的桶列表中二分查找，O(log n) 复杂度"""
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

**2. 确定性采样**

```python
def _should_sample(self, doc_id: str, rate: float) -> bool:
    """
    基于 MD5 哈希的确定性采样
    - 相同的 (random_seed, doc_id) 总是产生相同结果
    - 不依赖随机数生成器状态
    """
    if rate >= 1.0:
        return True
    data = f"{self.random_seed}_{doc_id}".encode()
    hash_bytes = hashlib.md5(data, usedforsecurity=False).digest()[:8]
    hash_int = int.from_bytes(hash_bytes, "big")
    return hash_int / (2**64) < rate
```

**3. 统计追踪**

过滤器自动追踪以下指标：
- `missing_score`: 缺少评分字段的文档数
- `filtered_out`: 不在任何桶范围内的文档数
- `kept_{bucket}`: 各桶保留的文档数
- `sampled_out_{bucket}`: 各桶被采样过滤的文档数

### 3.3 桶路径写入器 (`bucket_path_writer.py`)

实现多桶并行写入，支持大文件自动分片。

#### 内存缓冲策略

```python
# 每个桶独立维护写入状态
self._states = {
    bucket_name: {
        "buffer": [],      # 文档缓冲列表
        "counter": 0,      # 文件计数器
        "size": 0,         # 当前缓冲字节数
    }
    for bucket_name in buckets
}
```

#### 自动分片机制

```python
# 当缓冲大小超过阈值时触发写入
if state["size"] + row_size > max_file_size and state["buffer"]:
    self._flush_bucket(bucket_name)

# 文件命名: {rank:05d}_{counter:05d}.parquet
# 示例: 00000_00000.parquet, 00000_00001.parquet
```

### 3.4 评分桶配置 (`bucket_config.py`)

#### BucketConfig 数据类

```python
@dataclass(frozen=True)
class BucketConfig:
    name: str              # 桶名称（如 "3.0"）
    min_score: float       # 最小评分（包含）
    max_score: float       # 最大评分（不包含），None 表示无上界
    sampling_rate: float   # 采样率 0.0-1.0
```

**区间定义规则**:
- 采用**左闭右开**区间：`[min_score, max_score)`
- 最后一个桶 `max_score=None`，表示无上界
- 桶列表按 `min_score` 排序，支持二分查找

### 3.5 配置加载器 (`config_loader.py`)

#### 配置层级

1. **默认配置**: 代码中硬编码的默认值
2. **YAML 配置**: `config/` 目录下的配置文件
3. **环境变量**: `FINEWEB_{KEY}` 格式覆盖路径配置

#### 缓存机制

```python
@lru_cache(maxsize=10)
def _load_yaml(name: str) -> dict:
    """YAML 文件读取结果缓存，避免重复 IO"""
    ...
```

#### 环境变量覆盖

```python
def get_paths_config() -> dict:
    paths = dict(_load_yaml("paths"))
    for key in paths:
        if env_var := os.getenv(f"FINEWEB_{key.upper()}"):
            paths[key] = env_var  # 环境变量优先级最高
    return paths
```

---

## 4. 配置系统

### 4.1 数据集配置 (`config/dataset.yaml`)

```yaml
datasets:
  en:                               # 数据集标识符
    name: "fineweb_edu_en"         # 数据集名称
    root_marker: "fineweb-edu"     # 路径提取标记
    score_normalization:           # 评分归一化配置
      enabled: false               # 英文版无需归一化
    input_dir: "data/datasets/..." # 输入路径
    output_dir: "data/datasets/..."# 输出路径
    buckets:                       # 评分桶定义
      - name: "2.5"
        min_score: 2.5
        max_score: 3.0
        sampling_rate: 0.25
      # ... 更多桶

  zh:                              # 中文数据集
    root_marker: "Fineweb-Edu-Chinese-V2.1"
    score_normalization:
      enabled: true
      multiplier: 5.0              # 归一化评分 × 5
    # ...
```

### 4.2 处理参数配置 (`config/processing.yaml`)

```yaml
workers: 32                       # 本地并行工作进程数
tasks: 2500                       # Datatrove 任务数（控制并行度）
random_seed: 42                   # 采样随机种子
compression: "zstd"               # Parquet 压缩格式
max_file_size_bytes: 2147483648   # 输出文件大小限制（2GB）

logging:
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  level: "INFO"

trial:                            # 试运行参数
  max_files: 5
  max_rows: 2000
  max_file_size_bytes: 134217728  # 128MB
```

### 4.3 环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `FINEWEB_LOG_DIR` | 日志目录 | `logs/fineweb_processing` |
| `FINEWEB_TRIAL_INPUT_DIR` | 试运行输入目录 | `data/test_input` |
| `FINEWEB_TRIAL_OUTPUT_DIR` | 试运行输出目录 | `data/test_output` |

---

## 5. 性能优化

### 5.1 单次读取多桶处理

**传统方式的问题**:
```
对于 N 个桶，需要读取输入数据 N 次
- 读取数据 → 过滤桶1 → 写入桶1
- 读取数据 → 过滤桶2 → 写入桶2
- ...
```

**优化后的方式**:
```
单次读取数据
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

**调优建议**:
- `tasks` 应该大于 `workers`，确保负载均衡
- 对于 I/O 密集型任务，可适当增加 `workers`
- 每个 worker 处理 `tasks / workers` 个任务

### 5.3 内存优化

- **流式处理**: 基于生成器的流水线，避免全量加载
- **缓冲写入**: 按文件大小阈值批量写入，减少小文件
- **对象复用**: 使用 `@lru_cache` 缓存配置对象

---

## 6. 多语言支持设计

### 6.1 评分差异

| 数据集 | 原始评分范围 | 存储格式 | 处理方式 |
|--------|-------------|----------|----------|
| HuggingFaceFW/fineweb-edu | 1.0-5.0 | 原始值 | 直接使用 |
| Fineweb-Edu-Chinese-V2.1 | 0.0-5.0 | 归一化 0.0-1.0 | 自动 ×5 转换 |

### 6.2 中文数据集评分详解

#### 问题背景

中文数据集 `opencsg/Fineweb-Edu-Chinese-V2.1` 中的 `score` 字段存储为归一化浮点数（0.0-1.0），而非直观的 1.0-5.0 质量评分。这是因为数据集使用 BERT 分类器进行质量评估，输出被归一化到 0-1 范围。

#### 核心发现

- **存储格式**: 归一化浮点数 0.0-1.0（原始评分 0.0-5.0 经 BERT 分类器归一化）
- **映射公式**: `original_score = normalized_score × 5`
- **官方验证**: [OpenCSG 回复](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu-v2/discussions/2) *"multiply the score by 5"*

数据集经过质量过滤，仅保留原始评分 ≥ 2.5 的样本：

| 文件夹 | 归一化范围 | 原始评分 |
|--------|-----------|---------|
| `2_3/` | 0.50-0.60 | 2.5-3.0 |
| `3_4/` | 0.60-0.80 | 3.0-4.0 |
| `4_5/` | 0.80-0.94 | 4.0-4.70 |

#### 常见误区

| ❌ 错误理解 | ✅ 正确理解 |
|------------|-----------|
| score 是 1-5 分 | 实际为归一化值 0-1 |
| 最高是 5.0 分 | 实际最高约 4.70 分 |
| 使用 `(score-1)/4` 转换 | 正确公式是 `score × 5` |

### 6.3 自动检测机制

```python
def _get_dataset_config_for_source(source: str) -> dict:
    """根据文件路径自动识别数据集类型"""
    datasets = get_dataset_configs()
    for lang, config in datasets.items():
        marker = config.get("root_marker", "")
        if marker and marker in source:
            return config  # 返回对应数据集配置
    return datasets.get("en", {})  # 默认返回英文配置
```

### 6.4 配置隔离

每个数据集在 `dataset.yaml` 中有独立配置：
- 独立的 `root_marker` 用于路径识别
- 独立的 `score_normalization` 配置
- 独立的评分桶定义和采样率（中文和英文可使用不同的采样策略）

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

### 7.2 关键测试场景

**1. 区间边界测试**
```python
# 验证左闭右开区间
(2.5, "2.5")    # 边界包含
(2.9, "2.5")    # 区间内
(3.0, "3.0")    # 边界不包含在前一桶
```

**2. 确定性采样测试**
```python
# 相同种子应产生相同结果
f1 = ScoreFilter(buckets, random_seed=42)
f2 = ScoreFilter(buckets, random_seed=42)
assert sample_results(f1) == sample_results(f2)

# 不同种子应产生不同结果
f3 = ScoreFilter(buckets, random_seed=24)
assert sample_results(f1) != sample_results(f3)
```

**3. 采样率准确性测试**
```python
# 验证实际采样率与配置采样率误差 < 5%
actual_rate = output_count / input_count
assert abs(actual_rate - expected_rate) / expected_rate < 0.05
```

### 7.3 试运行脚本 (`trial_run.py`)

提供端到端测试能力：
1. 从原始数据创建小规模测试数据集
2. 执行完整处理流程
3. 验证输出结果
4. 分析采样准确性

```bash
# 创建测试数据并运行完整流程
python scripts/trial_run.py

# 分析采样准确性
python scripts/trial_run.py --analyze-sampling
```

---

## 8. 使用指南

### 8.1 快速开始

```bash
# 1. 安装依赖
uv pip install -r requirements.txt

# 2. 处理所有配置的数据集
python -m src.data_processing.fineweb_reorganizer

# 3. 验证输出
python scripts/validate_output.py --input data/datasets/fineweb/en
```

### 8.2 命令行参数

当前实现通过配置文件管理参数，支持运行时覆盖：

```bash
# 使用环境变量覆盖路径
export FINEWEB_LOG_DIR="custom/logs"
python -m src.data_processing.fineweb_reorganizer

# 修改配置文件后重新运行
# 编辑 config/processing.yaml 调整 workers 和 tasks
```

### 8.3 生产环境运行

```bash
# 使用 time 统计运行时间
time python -m src.data_processing.fineweb_reorganizer

# 处理完成后自动验证
python -m src.data_processing.fineweb_reorganizer && \
  python scripts/validate_output.py --input data/datasets/fineweb/en
```

### 8.4 添加新数据集

1. **准备数据**: 下载数据集到 `data/datasets/`

2. **配置数据集** (`config/dataset.yaml`):
```yaml
datasets:
  new_lang:
    name: "fineweb_edu_new"
    root_marker: "dataset-root-folder"
    score_normalization:
      enabled: true  # 或 false
      multiplier: 5.0
    input_dir: "data/datasets/..."
    output_dir: "data/datasets/fineweb/new_lang"
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
```

3. **运行处理**:
```bash
python -m src.data_processing.fineweb_reorganizer
```

---

## 9. 扩展与维护

### 9.1 添加新的评分桶

编辑 `config/dataset.yaml`，在对应数据集下添加新的桶定义：

```yaml
buckets:
  - name: "2.8"           # 新桶名称
    min_score: 2.8        # 最小评分（包含）
    max_score: 3.0        # 最大评分（不包含）
    sampling_rate: 0.4    # 采样率 40%
```

**注意事项**:
- 确保新区间不与现有区间重叠
- 保持桶按 `min_score` 升序排列（代码会自动排序）
- 最后一个桶可将 `max_score` 设为 `null` 表示无上界

### 9.2 修改采样算法

采样逻辑位于 `score_filter.py` 的 `_should_sample` 方法：

```python
def _should_sample(self, doc_id: str, rate: float) -> bool:
    # 自定义采样逻辑
    # 返回 True 保留文档，False 过滤文档
    pass
```

### 9.3 添加新的输出格式

继承 `Datatrove PipelineStep` 实现新的写入器：

```python
from datatrove.pipeline.base import PipelineStep

class CustomWriter(PipelineStep):
    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            # 自定义写入逻辑
            pass
```

然后在 `fineweb_reorganizer.py` 的 `create_pipeline` 中替换 `BucketPathWriter`。

---

## 10. 参考

### 10.1 相关文档

- [FineWeb-Edu 论文](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Datatrove 文档](https://github.com/huggingface/datatrove)
- [OpenCSG 中文数据集说明](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu-v2)

### 10.2 数据许可

FineWeb-Edu 数据集基于 **ODC-BY 1.0** 许可证发布。

---

## 附录 A: 配置文件完整示例

### A.1 dataset.yaml

```yaml
datasets:
  en:
    name: "fineweb_edu_en"
    root_marker: "fineweb-edu"
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
    root_marker: "Fineweb-Edu-Chinese-V2.1"
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

### A.2 processing.yaml

```yaml
workers: 32
tasks: 2500
random_seed: 42
compression: "zstd"
max_file_size_bytes: 2147483648

logging:
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  level: "INFO"

trial:
  max_files: 5
  max_rows: 2000
  max_file_size_bytes: 134217728
```

### A.3 paths.yaml

```yaml
log_dir: "logs/fineweb_processing"
trial_input_dir: "data/datasets/test_fineweb_input"
trial_output_dir: "data/datasets/test_fineweb_output"
```
