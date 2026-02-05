# FineWeb-Edu 数据集质量评分分桶重组计划文档

> **文档状态**: ✅ 已最终定稿  
> **创建日期**: 2026-02-05  
> **目标数据集**: HuggingFaceFW/fineweb-edu  
> **输出目录**: `data/datasets/fineweb/`

---

## 1. 项目背景与目标

### 1.1 背景

FineWeb-Edu 是一个高质量的教育类网页文本数据集，包含约 1.3T tokens（score≥3 版本）或 5.4T tokens（score≥2 版本）。数据集使用基于 LLaMA3-70B-Instruct 的教育价值分类器进行评分，分数范围为 0-5 分。

当前数据集按 Common Crawl 批次（CC-MAIN-YYYY-WW）组织，每个批次包含多个 parquet 文件。这种组织方式不利于按质量评分进行快速筛选和访问。

### 1.2 目标

将 FineWeb-Edu 数据集按照质量评分（`score` 字段）进行分桶重组，**仅保留 `id/text/score` 三个核心字段**，并按照以下采样策略进行数据筛选：

| 质量评分区间 | 采样率 | 说明 |
|-------------|--------|------|
| < 2.8 | **0%**（丢弃） | 低质量数据 |
| 2.8 - 3.0 | **30%** | 中低质量数据，采样保留 |
| 3.0 - 3.5 | **60%** | 中等质量数据，采样保留 |
| 3.5 - 4.0 | **80%** | 高质量数据，采样保留 |
| ≥ 4.0 | **100%** | 顶级质量数据，全部保留 |

生成新的目录结构：

```
data/datasets/fineweb/
└── en/
    └── {score_threshold}/
        └── CC-MAIN-xxxx-xx/
            └── xxxxxxxx.parquet
```

其中 `score_threshold` 表示该目录下数据的评分区间标识（采用"向上包含"策略，每个目录仅包含特定评分区间）。

**字段筛选说明**：
- 新数据集仅包含：`id`（唯一标识）、`text`（文本内容）、`score`（质量评分）
- 去除字段：`dump`、`url`、`file_path`、`language`、`language_score`、`token_count`、`int_score`

**采样策略说明**：
- 通过分层采样，优先保留高质量数据
- 预估最终数据量约为原始数据的 **35%**（约 600GB - 1TB）
- 大幅降低存储需求，同时保证数据质量

---

## 2. 数据集现状分析

### 2.1 数据结构

**原始字段**（来自 parquet 文件）：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `text` | string | 文本内容 |
| `id` | string | 唯一标识符（UUID） |
| `dump` | string | Common Crawl 批次（如 CC-MAIN-2021-21） |
| `url` | string | 原始网页 URL |
| `file_path` | string | S3 文件路径 |
| `language` | string | 语言代码（固定为 'en'） |
| `language_score` | double | 语言识别置信度 |
| `token_count` | int64 | token 数量 |
| `score` | double | **教育价值评分（0-5分）** |
| `int_score` | int64 | 整数化评分（0-5） |

### 2.2 评分分布特征

基于样本文件（CC-MAIN-2021-21/train-00000-of-00018.parquet，766,891 条记录）分析：

**基本统计**：
- 最小值：2.5156
- 最大值：5.2188
- 均值：3.0024
- 标准差：0.3962

**百分位数**：

| 百分位 | 分数 |
|--------|------|
| 1% | 2.5156 |
| 5% | 2.5469 |
| 10% | 2.5781 |
| 25% | 2.6875 |
| 50% | 2.9062 |
| 75% | 3.2344 |
| 90% | 3.5781 |
| 95% | 3.7812 |
| 99% | 4.1250 |

**分布特点**：
- 数据高度集中在 2.5-3.5 分区间（约 70%）
- 4.0 分以上数据稀少（<2%）
- 5.0 分以上数据极罕见（<0.01%）

### 2.3 数据规模

- **总大小**: 4.2 TB
- **CC-MAIN 批次数量**: 110 个
- **Parquet 文件数量**: 2,410 个
- **可用磁盘空间**: 3.5 TB

**关键约束与解决方案**: 

原始数据集 4.2TB > 可用磁盘空间 3.5TB，但通过**字段筛选 + 分层采样**，可以显著减少数据体积：

| 处理步骤 | 效果 | 预估数据量 |
|---------|------|-----------|
| 原始数据 | 10 个字段 | 4.2 TB |
| 字段筛选 | 仅保留 3 个字段 | ~3.5 TB |
| 分层采样 | 按质量评分采样 | **~600GB - 1TB** |

**采样策略效果分析**（基于样本数据估算）：

| 评分区间 | 原始占比 | 采样率 | 采样后占比 | 预估数据量 |
|---------|---------|--------|-----------|-----------|
| < 2.8 | ~35% | 0% | 0% | 0 GB |
| 2.8 - 3.0 | ~20% | 30% | ~6% | ~250 GB |
| 3.0 - 3.5 | ~35% | 60% | ~21% | ~880 GB |
| 3.5 - 4.0 | ~8% | 80% | ~6.4% | ~270 GB |
| ≥ 4.0 | ~2% | 100% | ~2% | ~85 GB |
| **合计** | **100%** | - | **~35%** | **~1.5 TB（未压缩估算）** |

*注：~1.5 TB 为未压缩的理论估算值。实际使用 zstd 压缩后约为 600GB - 1TB。实际数据量可能因原始数据分布和压缩率有所差异。*

通过字段筛选和分层采样，新数据集预估 **~600GB - 1TB**，完全在 3.5TB 磁盘空间内，且有充足余量用于临时处理和冗余。

---

## 3. 分桶策略分析

### 3.1 分桶策略设计

基于采样策略（丢弃 <2.8 数据，2.8-3.0 采样 30%，3.0-3.5 采样 60%，3.5-4.0 采样 80%，≥4.0 保留 100%），设计以下分桶策略：

#### 核心分桶（基于采样后的数据，采用"向上包含"策略）

| 目录 | 评分区间 | 采样率 | 说明 | 预估占比 |
|------|---------|--------|------|---------|
| `2.8/` | 2.8 ≤ score < 3.0 | 30% | 中低质量数据 | ~6% |
| `3.0/` | 3.0 ≤ score < 3.5 | 60% | 中等质量数据 | ~21% |
| `3.5/` | 3.5 ≤ score < 4.0 | 80% | 高质量数据 | ~6.4% |
| `4.0/` | score ≥ 4.0 | 100% | 顶级质量数据 | ~2% |

**分桶逻辑说明**（"向上包含"策略，数据不重叠）：

| 目录 | 评分区间 | 边界处理 |
|------|---------|---------|
| `2.8/` | 2.8 ≤ score < 3.0 | 包含 2.8，不包含 3.0 |
| `3.0/` | 3.0 ≤ score < 3.5 | 包含 3.0，不包含 3.5 |
| `3.5/` | 3.5 ≤ score < 4.0 | 包含 3.5，不包含 4.0 |
| `4.0/` | score ≥ 4.0 | 包含 4.0 及以上 |

- `2.8/`：**仅**包含 2.8-3.0 区间数据（30%采样）
- `3.0/`：**仅**包含 3.0-3.5 区间数据（60%采样）
- `3.5/`：**仅**包含 3.5-4.0 区间数据（80%采样）
- `4.0/`：**仅**包含 ≥4.0 区间数据（100%保留）

**数据组合方式**：
- 需要 2.8+ 全部数据：加载 2.8/ + 3.0/ + 3.5/ + 4.0/
- 需要 3.0+ 数据：加载 3.0/ + 3.5/ + 4.0/
- 需要 3.5+ 数据：加载 3.5/ + 4.0/
- 需要单一质量层级：加载对应目录即可

#### 采样策略实现

```python
# 采样逻辑伪代码
def apply_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    按评分区间应用分层采样
    """
    sampled_dfs = []
    
    # < 2.8: 丢弃
    df_28_30 = df[(df['score'] >= 2.8) & (df['score'] < 3.0)]
    sampled_dfs.append(df_28_30.sample(frac=0.3, random_state=42))
    
    # 3.0 - 3.5: 60% 采样
    df_30_35 = df[(df['score'] >= 3.0) & (df['score'] < 3.5)]
    sampled_dfs.append(df_30_35.sample(frac=0.6, random_state=42))
    
    # 3.5 - 4.0: 80% 采样
    df_35_40 = df[(df['score'] >= 3.5) & (df['score'] < 4.0)]
    sampled_dfs.append(df_35_40.sample(frac=0.8, random_state=42))
    
    # >= 4.0: 100% 保留
    df_40_plus = df[df['score'] >= 4.0]
    sampled_dfs.append(df_40_plus)
    
    return pd.concat(sampled_dfs, ignore_index=True)
```

### 3.2 分桶策略总结

**采用方案**：**4 个核心分桶（2.8, 3.0, 3.5, 4.0）**

**设计理由**：
1. **与采样策略对齐**：以 2.8 为起点（丢弃更低质量数据）
2. **层级清晰**：每个桶包含特定评分区间的数据
3. **实用性强**：4.0/ 目录可直接用于高质量训练，2.8/ 用于大规模预训练
4. **数据量合理**：预估总数据量 ~600GB - 1TB
5. **存储优化**：采用"向上包含"策略，各目录数据互不重叠，避免重复存储

---

## 4. 目标目录结构设计

### 4.1 推荐结构

```
data/datasets/fineweb/
└── en/                                    # 语言目录（预留多语言扩展）
    ├── 4.0/                               # 4.0 ≤ score（100%保留，顶级质量）
    │   └── CC-MAIN-2021-21/
    │       └── train-00000-of-00018.parquet
    ├── 3.5/                               # 3.5 ≤ score < 4.0（80%采样）
    │   └── ...
    ├── 3.0/                               # 3.0 ≤ score < 3.5（60%采样）
    │   └── ...
    └── 2.8/                               # 2.8 ≤ score < 3.0（30%采样）
        └── ...
```

### 4.2 设计说明

1. **语言目录（`en/`）**：FineWeb-Edu 当前只有英文数据，但预留多语言扩展能力
2. **评分目录（`X.X/`）**：目录名表示该目录下数据的评分区间下限
3. **"向上包含"策略**：各目录数据**互不重叠**，每个目录仅包含特定评分区间
4. **采样说明**：每个目录内的数据已经过采样处理，不同评分区间有不同的采样率
5. **数据独立性**：每个目录包含独立的 parquet 文件，可直接使用或组合使用

### 4.3 各目录数据内容说明

| 目录 | 评分区间 | 采样率 | 预估数据量 | 适用场景 |
|------|---------|--------|-----------|---------|
| `2.8/` | 2.8 ≤ score < 3.0 | 30% | ~250 GB（~80GB 压缩后） | 中低质量数据 |
| `3.0/` | 3.0 ≤ score < 3.5 | 60% | ~880 GB（~280GB 压缩后） | 中等质量数据 |
| `3.5/` | 3.5 ≤ score < 4.0 | 80% | ~270 GB（~90GB 压缩后） | 高质量数据 |
| `4.0/` | score ≥ 4.0 | 100% | ~85 GB（~30GB 压缩后） | 顶级质量数据 |

*注：压缩后大小按 zstd 压缩率约 3:1 估算*

**数据组合建议**：

| 目标数据 | 需要加载的目录 | 总预估大小（压缩后） |
|---------|---------------|-------------------|
| 2.8+ 全部数据 | 2.8/ + 3.0/ + 3.5/ + 4.0/ | ~600GB - 1TB |
| 3.0+ 数据 | 3.0/ + 3.5/ + 4.0/ | ~500GB - 800GB |
| 3.5+ 数据 | 3.5/ + 4.0/ | ~150GB - 300GB |
| 仅顶级数据 | 4.0/ | ~50GB - 100GB |

---

## 5. 技术实现方案（基于 Datatrove）

### 5.1 为什么选择 Datatrove

**Datatrove** 是 Hugging Face 开发的大规模数据处理库，专为文本数据预训练设计，具有以下优势：

| 特性 | 说明 |
|------|------|
| **高效 I/O** | 基于内存映射和流式处理，适合 TB 级数据 |
| **并行处理** | 内置多进程/多线程支持，自动利用多核 CPU |
| **管道化设计** | 通过 `Pipeline` 组合多个处理步骤，代码清晰可维护 |
| **数据格式支持** | 原生支持 Parquet、JSONL、ZST 等格式 |
| **容错机制** | 支持断点续传和错误恢复 |
| **与 HF 生态集成** | 无缝对接 Hugging Face datasets 和模型训练流程 |

### 5.2 Datatrove 核心概念

```python
from datatrove.pipeline import Pipeline
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.filters import LambdaFilter
from datatrove.data import Document

# Datatrove 处理流程
pipeline = Pipeline(
    # 1. 读取阶段
    ParquetReader(
        data_folder="source/path",
        columns=["id", "text", "score"],  # 仅读取需要的列
    ),
    
    # 2. 处理阶段（自定义过滤器、转换器）
    CustomFilter(),  # 丢弃 score < 2.8 的数据
    StratifiedSampler(),  # 分层采样
    
    # 3. 写入阶段
    ParquetWriter(
        output_folder="target/path",
        compression="zstd",
    ),
)

# 执行管道
pipeline.run()
```

### 5.3 实现架构

采用 **Datatrove Pipeline** 实现整个处理流程：

```
┌─────────────────────────────────────────────────────────────┐
│                    Datatrove Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  1. ParquetReader                                            │
│     ├── 读取: data/datasets/HuggingFaceFW/fineweb-edu/data   │
│     ├── 列投影: ["id", "text", "score"]                       │
│     └── 输出: Document 对象流                                │
├─────────────────────────────────────────────────────────────┤
│  2. ScoreFilter (自定义)                                     │
│     ├── 过滤: score < 2.8 的数据                             │
│     └── 输出: 符合条件的 Document 流                         │
├─────────────────────────────────────────────────────────────┤
│  3. StratifiedSampler (自定义)                               │
│     ├── 2.8-3.0: 30% 采样                                    │
│     ├── 3.0-3.5: 60% 采样                                    │
│     ├── 3.5-4.0: 80% 采样                                    │
│     └── ≥4.0: 100% 保留                                      │
├─────────────────────────────────────────────────────────────┤
│  4. MultiBucketWriter (自定义)                               │
│     ├── 按评分区间分发到不同桶                               │
│     ├── 2.8/: 2.8-3.0 数据                                   │
│     ├── 3.0/: 3.0-3.5 数据                                   │
│     ├── 3.5/: 3.5-4.0 数据                                   │
│     └── 4.0/: ≥4.0 数据                                      │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 核心组件设计

#### 5.4.1 自定义 Filter：ScoreFilter

```python
from datatrove.pipeline.filters import BaseFilter
from datatrove.data import Document

class ScoreFilter(BaseFilter):
    """过滤 score < 2.8 的数据"""
    
    def __init__(self, min_score: float = 2.8):
        super().__init__()
        self.min_score = min_score
    
    def filter(self, doc: Document) -> bool:
        """返回 True 表示保留，False 表示丢弃"""
        score = doc.metadata.get("score", 0)
        return score >= self.min_score
```

#### 5.4.2 自定义 Sampler：StratifiedSampler

```python
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document
import random

class StratifiedSampler(PipelineStep):
    """
    分层采样器
    按评分区间应用不同采样率
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__()
        self.random = random.Random(random_seed)
        self.sampling_rates = {
            (2.8, 3.0): 0.30,
            (3.0, 3.5): 0.60,
            (3.5, 4.0): 0.80,
            (4.0, float('inf')): 1.0,
        }
    
    def __call__(self, doc: Document) -> Document | None:
        """
        根据评分决定是否采样保留
        返回 Document 表示保留，返回 None 表示丢弃
        """
        score = doc.metadata.get("score", 0)
        
        # 确定评分区间和采样率
        for (low, high), rate in self.sampling_rates.items():
            if low <= score < high:
                # 按概率采样
                if self.random.random() < rate:
                    return doc
                else:
                    return None
        
        return None  # 不在任何区间内，丢弃
```

#### 5.4.3 自定义 Writer：MultiBucketWriter

```python
from datatrove.pipeline.writers import ParquetWriter
from datatrove.data import Document
from pathlib import Path

class MultiBucketWriter:
    """
    多桶写入器
    按评分区间将数据写入不同目录，保持原始 CC-MAIN 结构
    """
    
    def __init__(self, base_output_path: Path):
        self.base_output_path = base_output_path
        self.writers = {}
    
    def _get_writer(self, bucket: str, cc_main: str):
        """获取或创建对应桶和 CC-MAIN 批次的 writer"""
        key = f"{bucket}/{cc_main}"
        if key not in self.writers:
            bucket_path = self.base_output_path / "en" / bucket / cc_main
            self.writers[key] = ParquetWriter(
                output_folder=str(bucket_path),
                compression="zstd",
            )
        return self.writers[key]
    
    def __call__(self, doc: Document):
        """将文档写入对应评分桶"""
        score = doc.metadata.get("score", 0)
        
        # 确定目标桶（左闭右开区间）
        if 2.8 <= score < 3.0:
            bucket = "2.8"
        elif 3.0 <= score < 3.5:
            bucket = "3.0"
        elif 3.5 <= score < 4.0:
            bucket = "3.5"
        elif score >= 4.0:
            bucket = "4.0"
        else:
            return  # 不写入任何桶（score < 2.8）
        
        # 从原始文件路径提取 CC-MAIN 批次信息
        # doc.metadata["file_path"] 包含原始 parquet 文件路径
        original_path = doc.metadata.get("file_path", "")
        cc_main = self._extract_cc_main(original_path)
        
        # 获取对应 writer 并写入
        writer = self._get_writer(bucket, cc_main)
        writer(doc)
    
    def _extract_cc_main(self, file_path: str) -> str:
        """从文件路径提取 CC-MAIN 批次名称"""
        # 路径格式: .../data/CC-MAIN-2021-21/train-00000-of-00018.parquet
        import re
        match = re.search(r'CC-MAIN-\d{4}-\d{2}', file_path)
        return match.group(0) if match else "unknown"
```

#### 5.4.4 完整 Pipeline 组装

```python
from datatrove.pipeline import Pipeline
from datatrove.pipeline.readers import ParquetReader
from datatrove.executor import LocalPipelineExecutor
from pathlib import Path

def create_fineweb_pipeline(
    source_dir: Path = Path("data/datasets/HuggingFaceFW/fineweb-edu/data"),
    target_dir: Path = Path("data/datasets/fineweb"),
    random_seed: int = 42,
) -> Pipeline:
    """
    创建 FineWeb-Edu 数据处理 Pipeline
    """
    
    # 1. 读取阶段
    reader = ParquetReader(
        data_folder=str(source_dir),
        columns=["id", "text", "score"],  # 列投影，仅读取需要的字段
        glob_pattern="**/*.parquet",
    )
    
    # 2. 过滤阶段
    score_filter = ScoreFilter(min_score=2.8)
    
    # 3. 采样阶段
    sampler = StratifiedSampler(random_seed=random_seed)
    
    # 4. 写入阶段（多桶）
    writer = MultiBucketWriter(base_output_path=target_dir)
    
    # 组装 Pipeline
    pipeline = Pipeline(
        reader,
        score_filter,
        sampler,
        writer,
    )
    
    return pipeline

# 执行 Pipeline
if __name__ == "__main__":
    pipeline = create_fineweb_pipeline()
    
    # 使用 LocalPipelineExecutor 执行
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        workers=8,  # 并行工作进程数
        logging_dir="logs/fineweb_processing",
    )
    
    executor.run()
```

### 5.5 Datatrove 优势在本项目中的体现

| 项目需求 | Datatrove 解决方案 |
|---------|-------------------|
| **字段筛选** | `ParquetReader(columns=[...])` 列投影 |
| **分层采样** | 自定义 `StratifiedSampler` 组件 |
| **多桶输出** | 自定义 `MultiBucketWriter` 组件 |
| **并行处理** | `LocalPipelineExecutor(workers=N)` |
| **进度跟踪** | 内置 tqdm 进度条和日志记录 |
| **断点续传** | 自动保存处理状态，支持中断恢复 |
| **内存优化** | 流式处理，避免一次性加载大文件 |

### 5.6 依赖安装

```bash
# datatrove 已包含在项目依赖中
# pyproject.toml 中已配置: datatrove[all]>=0.8.0

# 如需单独安装
uv add datatrove[all] --no-sync
```

### 5.7 处理步骤详解

使用 Datatrove 的处理流程分为以下阶段：

| 阶段 | Datatrove 组件 | 说明 |
|------|---------------|------|
| 1. 扫描 | `ParquetReader` | 自动扫描所有 parquet 文件 |
| 2. 读取 | `ParquetReader(columns=[...])` | 列投影，仅读取 id/text/score |
| 3. 过滤 | `ScoreFilter` (自定义) | 丢弃 score < 2.8 的数据 |
| 4. 采样 | `StratifiedSampler` (自定义) | 按评分区间分层采样 |
| 5. 写入 | `MultiBucketWriter` (自定义) | 按评分区间分桶写入 |
| 6. 验证 | 独立验证脚本 | 验证输出完整性和采样比例 |

**详细步骤**：

1. **扫描阶段**
   - `ParquetReader` 自动遍历 `HuggingFaceFW/fineweb-edu/data/` 下所有 parquet 文件
   - 支持 glob 模式匹配和递归扫描

2. **读取与筛选阶段**
   - `ParquetReader(columns=["id", "text", "score"])` 进行列投影
   - 仅读取需要的三个字段，减少内存占用
   - Datatrove 自动处理文件分块和流式读取

3. **过滤阶段**
   - 自定义 `ScoreFilter` 组件过滤 score < 2.8 的数据
   - 返回 `False` 的数据会被自动丢弃

4. **分层采样阶段**
   - 自定义 `StratifiedSampler` 组件按评分区间采样
   - 每个 Document 独立判断是否保留
   - 使用固定随机种子保证可复现性

5. **分桶写入阶段**（"向上包含"策略，数据不重叠）
   - 自定义 `MultiBucketWriter` 按评分区间分发数据
   - 自动创建目录结构：`{target_dir}/en/{threshold}/{CC-MAIN-xxx}/`
   - 各评分桶数据：
     - `2.8/`：仅 2.8-3.0 区间数据（30%采样）
     - `3.0/`：仅 3.0-3.5 区间数据（60%采样）
     - `3.5/`：仅 3.5-4.0 区间数据（80%采样）
     - `4.0/`：仅 ≥4.0 区间数据（100%保留）
   - 使用 zstd 压缩

6. **验证阶段**
   - 验证输出文件完整性
   - 抽样验证数据内容和采样比例
   - 生成统计报告

### 5.8 关键问题与解决方案

#### 问题1：如何高效读取和写入大规模 parquet 文件？

**方案**：
- 使用 PyArrow 的列投影功能，仅读取需要的三列
- 采用分块读取（chunked reading）避免内存溢出
- 使用多进程并行处理多个文件
- 写入时使用适当的压缩算法（snappy 平衡速度，zstd 平衡压缩率）

```python
# 示例代码
import pyarrow.parquet as pq
import pyarrow as pa

# 读取指定列
table = pq.read_table(
    file_path,
    columns=["id", "text", "score"]
)

# 分块读取（适用于超大文件）
parquet_file = pq.ParquetFile(file_path)
for batch in parquet_file.iter_batches(
    batch_size=10000,
    columns=["id", "text", "score"]
):
    # 处理批次
    pass
```

#### 问题2：如何保证采样的可复现性？

**方案**：
- 使用固定的随机种子（random_state）
- 记录采样配置和随机种子到元数据文件
- 支持通过配置文件调整采样率

```python
# 采样配置持久化
sampling_config = {
    "random_state": 42,
    "sampling_rates": {
        "2.8-3.0": 0.30,
        "3.0-3.5": 0.60,
        "3.5-4.0": 0.80,
        ">=4.0": 1.0
    }
}
# 保存到 metadata.json
```

#### 问题3：如何处理不同评分桶的数据重叠？

**方案**：采用**"向上包含"策略**（已确认）

每个评分桶只包含该区间内的数据，数据不重复：

| 目录 | 评分区间 | 采样率 | 数据范围 |
|------|---------|--------|---------|
| `2.8/` | 2.8 ≤ score < 3.0 | 30% | 仅 2.8-3.0 区间数据 |
| `3.0/` | 3.0 ≤ score < 3.5 | 60% | 仅 3.0-3.5 区间数据 |
| `3.5/` | 3.5 ≤ score < 4.0 | 80% | 仅 3.5-4.0 区间数据 |
| `4.0/` | score ≥ 4.0 | 100% | 仅 ≥4.0 区间数据 |

**优势**：
- 各目录数据**互不重叠**，总数据量最小化
- 可根据需要灵活组合不同质量层级的数据
- 避免重复存储，节省磁盘空间

**使用方式**：
- 需要 2.8+ 全部数据：加载 2.8/, 3.0/, 3.5/, 4.0/ 四个目录
- 需要 3.0+ 数据：加载 3.0/, 3.5/, 4.0/ 三个目录
- 需要单一质量层级：加载对应目录即可

#### 问题4：如何处理增量更新？

**方案**：
- 记录已处理的文件列表和校验和
- 增量处理时跳过已处理文件
- 提供 `--force` 选项强制重新处理

#### 问题5：磁盘空间监控

**方案**：
- 处理前检查可用空间
- 定期报告磁盘使用情况
- 预留 20% 安全余量

---

## 6. 风险评估与缓解措施

### 6.1 风险识别

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 磁盘空间不足 | 低 | 高 | 字段筛选后数据量可控，监控磁盘使用 |
| 处理时间过长 | 高 | 中 | 并行处理，进度显示，断点续传 |
| 数据丢失或损坏 | 低 | 高 | 验证脚本，抽样检查，校验和比对 |
| 内存溢出 | 中 | 高 | 分块读取，控制并行度 |
| 评分阈值选择不当 | 中 | 中 | 提供可配置阈值，预留调整空间 |

### 6.2 缓解措施详解

1. **磁盘空间监控**
   - 处理前检查可用空间
   - 处理中定期报告磁盘使用
   - Datatrove 自动监控和报告资源使用情况

2. **并行处理**
   - 使用 `LocalPipelineExecutor(workers=N)` 并行处理
   - Datatrove 自动管理多进程，根据 CPU 核心数优化并行度
   - 支持动态调整并发数

3. **断点续传**
   - Datatrove 内置状态管理和断点续传机制
   - 自动记录处理进度到日志目录
   - 支持从中断点恢复，避免重复处理

4. **验证机制**
   - 处理完成后验证输出文件完整性
   - 抽样验证数据可读性和采样比例
   - 生成详细的统计报告

---

## 7. 决策记录

> 本节记录所有已确认的决策事项，作为项目实施的基础。

### 7.1 已确认决策汇总

| 决策项 | 最终选择 | 说明 |
|--------|----------|------|
| **分桶策略** | 基于采样策略的分桶 | 2.8, 3.0, 3.5, 4.0 四个评分桶 |
| **数据重叠策略** | "向上包含"策略 | 各目录数据互不重叠 |
| **实现框架** | Datatrove | Hugging Face 数据处理库 |
| **数据范围** | **全部处理** | 110 个 CC-MAIN 批次，4.2TB |
| **采样策略** | 分层采样 | 0%/30%/60%/80%/100% |
| **随机种子** | 42 | 保证可复现性 |
| **输出目录** | `data/datasets/fineweb/` | 按评分分桶存储 |
| **保留字段** | id, text, score | 仅3个核心字段 |
| **压缩算法** | zstd | 平衡压缩率和速度 |
| **并行度** | 自动调整 | Datatrove 自动管理 |

### 7.2 决策依据

1. **分桶策略**：与采样策略完全对齐，以2.8为起点丢弃低质量数据
2. **数据重叠策略**："向上包含"避免数据重复，节省存储空间
3. **实现框架**：Datatrove专为大规模文本数据处理设计，内置并行和容错
4. **数据范围**：全部处理以获得完整的高质量数据集
5. **采样策略**：优先保留高质量数据（≥4.0保留100%，<2.8丢弃）

---

## 8. 后续工作计划

待本计划文档评审通过后，将按以下步骤实施：

### 阶段1：Datatrove 基础实现（预计 1-2 天）

1. 创建项目目录结构（`src/`）
2. 实现 `ScoreFilter`：过滤 score < 2.8 的数据
3. 实现 `StratifiedSampler`：按评分区间分层采样
4. 实现 `MultiBucketWriter`：按评分区间分桶写入
5. 组装完整 Pipeline 并测试

### 阶段2：优化与完善（预计 1 天）

1. 配置 `LocalPipelineExecutor` 并行参数
2. 添加进度显示和日志记录（Datatrove 内置）
3. 测试断点续传功能
4. 添加验证和统计报告功能
5. 实现 CLI 接口

### 阶段3：测试与文档（预计 0.5-1 天）

1. 进行小规模测试（选择 1-2 个 CC-MAIN 批次）
2. 验证采样比例和输出完整性
3. 编写使用文档
4. 进行全量数据处理

### 阶段4：验证与交付（预计 0.5 天）

1. 验证输出目录结构
2. 验证 parquet 文件完整性
3. 验证采样比例符合预期
4. 生成最终统计报告
5. 交付使用说明

---

## 9. 附录

### 9.1 参考资源

- FineWeb-Edu 官方文档：https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- FineWeb-Edu 论文：https://arxiv.org/abs/2406.17557
- HuggingFace datasets 库文档：https://huggingface.co/docs/datasets
- **Datatrove 官方文档**：https://github.com/huggingface/datatrove
- **Datatrove API 参考**：https://github.com/huggingface/datatrove/tree/main/src/datatrove
- **Datatrove 示例**：https://github.com/huggingface/datatrove/tree/main/examples

### 9.2 术语表

| 术语 | 说明 |
|------|------|
| CC-MAIN | Common Crawl 的快照批次命名格式（CC-MAIN-YYYY-WW） |
| Parquet | 列式存储格式，适合大规模数据分析 |
| Score | FineWeb-Edu 的教育价值评分（0-5分） |
| Token | 文本经过 tokenizer 处理后的基本单位 |
| **Datatrove** | Hugging Face 开发的大规模文本数据处理库 |
| **Pipeline** | 数据处理管道，由多个处理步骤串联组成 |
| **Document** | Datatrove 中的数据单元，包含文本和元数据 |
| **Filter** | 数据过滤器，用于筛选符合条件的文档 |
| **Executor** | 管道执行器，负责调度和执行处理流程 |

### 9.3 数据统计摘要

| 指标 | 数值 | 说明 |
|------|------|------|
| **原始数据集总大小** | 4.2 TB | 包含 10 个字段 |
| **CC-MAIN 批次数量** | 110 个 | - |
| **Parquet 文件数量** | 2,410 个 | - |
| **可用磁盘空间** | 3.5 TB | - |
| **Score 范围** | 2.5156 - 5.2188 | - |
| **Score 均值** | 3.0024 | - |
| **字段筛选后预估大小** | ~3.5 TB | 仅保留 id/text/score 三个字段 |
| **分层采样后预估大小** | **~600GB - 1TB** | 应用采样策略后 |
| **数据缩减比例** | **~75-85%** | 字段筛选 + 分层采样 + 丢弃低分数据 |

### 9.4 字段说明

**保留字段**（新数据集）：

| 字段名 | 类型 | 说明 | 必要性 |
|--------|------|------|--------|
| `id` | string | 唯一标识符 | 用于数据追踪和去重 |
| `text` | string | 文本内容 | 核心训练数据 |
| `score` | double | 教育价值评分 | 质量控制和筛选 |

**去除字段**：

| 字段名 | 去除原因 |
|--------|----------|
| `dump` | 可通过文件路径推断 |
| `url` | 训练时不需要 |
| `file_path` | 可通过文件路径推断 |
| `language` | 固定为 'en'，无变化 |
| `language_score` | 已筛选为高质量英文数据 |
| `token_count` | 训练时可重新计算 |
| `int_score` | 可由 score 推导 |

---

## 10. 文档状态

### 10.1 评审结论

| 检查项 | 状态 | 说明 |
|--------|------|------|
| **需求完整性** | ✅ 通过 | 所有功能需求已明确 |
| **技术可行性** | ✅ 通过 | Datatrove 框架支持所有需求 |
| **数据量评估** | ✅ 通过 | 预估 ~600GB - 1TB，在磁盘容量范围内 |
| **风险评估** | ✅ 通过 | 已识别风险并制定缓解措施 |
| **决策对齐** | ✅ 通过 | 所有决策事项已与用户确认 |

### 10.2 当前状态

**🟢 计划文档已最终定稿，等待编码实施指令**

### 10.3 下一步行动

待您确认后，将立即开始编码实施。

**实施顺序**：
1. 阶段1：Datatrove 基础实现（1-2天）
2. 阶段2：优化与完善（1天）
3. 阶段3：测试与文档（0.5-1天）
4. 阶段4：验证与交付（0.5天）

---

**文档结束**
