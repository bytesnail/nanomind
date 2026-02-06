# FineWeb-Edu 数据集质量评分分桶重组设计

> **文档状态**: ✅ **设计已完善（v1.1 实施就绪版）**  
> **创建日期**: 2026-02-05  
> **最后更新**: 2026-02-06  
> **目标数据集**: HuggingFaceFW/fineweb-edu (score≥2 版本, 4.2 TB)  
> **输出目录**: `data/datasets/fineweb/`  
> **文档版本**: v1.1

---

## 1. 项目背景与目标

### 1.1 背景

FineWeb-Edu 是一个高质量的教育类网页文本数据集，使用基于 LLaMA3-70B-Instruct 的教育价值分类器进行评分，分数范围为 0-5 分。

**数据规模**：
- **score≥2 版本**：约 5.4T tokens，存储大小约 **4.2 TB**（本项目采用此版本）
- **score≥3 版本**：约 1.3T tokens，存储大小约 **1.0 TB**

当前数据集按 Common Crawl 快照批次（CC-MAIN-YYYY-WW）组织，每个批次包含多个 parquet 文件。这种组织方式不利于按质量评分进行快速筛选和访问。

### 1.2 目标

将 FineWeb-Edu 数据集按照质量评分（`score` 字段）进行分桶重组，采用分层采样策略：

| 质量评分区间 | 采样率 | 处理方式 | 说明 |
|-------------|--------|----------|------|
| < 2.8 | **0%** | 丢弃 | 低质量数据，不参与后续处理 |
| 2.8 ≤ score < 3.0 | **30%** | 保留 | 中低质量数据（区间类型：左闭右开） |
| 3.0 ≤ score < 3.5 | **60%** | 保留 | 中等质量数据（区间类型：左闭右开） |
| 3.5 ≤ score < 4.0 | **80%** | 保留 | 高质量数据（区间类型：左闭右开） |
| score ≥ 4.0 | **100%** | 保留 | 顶级质量数据（区间类型：左闭） |

**采样策略说明**：
- 采样率表示该区间内保留数据的比例
- 例如：30% 采样率表示保留该区间内 30% 的文档，丢弃 70%

**区间说明**：
- 采用**左闭右开**区间 `[min, max)`，确保边界值（3.0、3.5、4.0）归属唯一
- 例如：score = 3.0 的数据只属于 3.0 ≤ score < 3.5 区间，不属于 2.8 ≤ score < 3.0 区间

**输出目录结构**：

```
data/datasets/fineweb/
└── en/
    └── {score_bucket}/
        └── CC-MAIN-xxxx-xx/
            └── {rank}.parquet
```

**字段筛选**：输出保留 `id`、`text`、`score` 三个顶层字段，score 字段与 id、text 并列，便于直接访问和使用。

**预期效果**（基于 score≥2 版本的 4.2 TB 原始数据）：
- 预估采样后数据量约为原始数据的 **35%**（未压缩约 **450 GB**）
- zstd 压缩后约 **150 GB**（约为采样后数据的 33%，原始数据的 3.6%）
- 各评分桶数据互不重叠
- 每个评分桶内保持原始的 CC-MAIN 快照批次结构，便于溯源

---

## 2. 数据集现状分析

### 2.1 数据结构

**原始字段**（来自 parquet 文件）：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `text` | string | 文本内容 |
| `id` | string | 唯一标识符（UUID） |
| `dump` | string | Common Crawl 批次（如 CC-MAIN-2024-10） |
| `url` | string | 原始网页 URL |
| `file_path` | string | S3 文件路径 |
| `language` | string | 语言代码 |
| `language_score` | double | 语言识别置信度 |
| `token_count` | int64 | token 数量 |
| `score` | double | **教育价值评分（0-5分）** |
| `int_score` | int64 | 整数化评分 |

### 2.2 评分分布特征

基于样本文件分析（766,891 条记录）：

| 统计项 | 数值 |
|--------|------|
| 最小值 | 2.5156 |
| 最大值 | 5.2188 |
| 均值 | 3.0024 |
| 标准差 | 0.3962 |

**百分位数**：50%: 2.9062 | 75%: 3.2344 | 90%: 3.5781 | 95%: 3.7812 | 99%: 4.1250

**分布特点**：数据高度集中在 2.5-3.5 分区间（约 70%），4.0 分以上数据稀少（<2%）。

### 2.3 数据规模与约束

| 指标 | 数值 |
|------|------|
| 原始数据集总大小 | 4.2 TB |
| CC-MAIN 批次数量 | 110 个 |
| Parquet 文件数量 | 2,410 个 |
| 预估总记录数 | ~15 亿条 |
| 可用磁盘空间 | 3.5 TB |

**存储约束解决方案**：

| 处理步骤 | 预估数据量 | 存储位置 |
|---------|-----------|----------|
| 原始数据（10 个字段） | 4.2 TB | `data/datasets/HuggingFaceFW/fineweb-edu/` |
| 字段筛选（3 个字段） | ~1.5 TB | 流式处理，不落盘 |
| 分层采样（未压缩） | ~450 GB | 直接输出到目标目录 |
| **zstd 压缩后（最终）** | **~150 GB** | `data/datasets/fineweb/` |

**⚠️ 重要约束**：处理过程中需要确保磁盘空间充足，建议预留至少 **600 GB** 空间（考虑输出数据和缓冲）。

---

## 3. 技术实现方案

### 3.1 实现架构

使用 **Datatrove**（Hugging Face 开发的大规模数据处理库）实现，采用 **按桶独立处理** 方案：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    按桶独立处理方案（4 个独立 Pipeline）                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Pipeline 1: 桶 2.8 (2.8 ≤ score < 3.0, 采样率 30%)              │   │
│  │  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │   │
│  │  │ParquetReader│───→│ ScoreFilter  │───→│ ParquetWriter    │   │   │
│  │  │ (4.2 TB)    │    │ (30% 采样)   │    │ → en/2.8/        │   │   │
│  │  └─────────────┘    └──────────────┘    └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Pipeline 2: 桶 3.0 (3.0 ≤ score < 3.5, 采样率 60%)              │   │
│  │  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │   │
│  │  │ParquetReader│───→│ ScoreFilter  │───→│ ParquetWriter    │   │   │
│  │  │ (4.2 TB)    │    │ (60% 采样)   │    │ → en/3.0/        │   │   │
│  │  └─────────────┘    └──────────────┘    └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ...（桶 3.5 和 4.0 类似）                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**方案说明**：
- **按桶独立处理**：每个评分桶运行独立的 Pipeline，逻辑清晰，易于调试和维护
- **单次遍历每桶**：每个 Pipeline 只读取一次源数据（4.2 TB），但 4 个桶共读取 16.8 TB
- **进程内去重**：每个 Pipeline 内部使用 Bloom Filter 进行去重（同一文档在同一桶内不会重复）
- **跨桶重复处理**：同一文档可能因评分变化出现在不同桶中，这是预期行为（不同质量级别）
- **独立运行**：支持按桶单独运行，失败时只需重跑失败的桶

**性能特征**：
| 指标 | 按桶独立处理方案 | 说明 |
|------|-----------------|------|
| 数据读取量 | 16.8 TB（4 桶 × 4.2 TB） | 每个桶独立读取完整数据 |
| 预估处理时间 | 8-12 小时（顺序）/ 2-4 小时（并行） | 可并行运行多个桶 |
| 内存占用（每桶） | ~3 GB（Bloom Filter） | 每桶独立的 Bloom Filter |
| 实现复杂度 | 低 | 使用标准 Datatrove 组件 |

**⚠️ 重要说明**：
- **为何不采用单次多路输出**：Datatrove 的 Pipeline 架构是线性流式处理，多路输出需要复杂的自定义 Writer，且与 Datatrove 的断点续传、错误处理等特性集成困难
- **跨桶重复是预期行为**：同一文档如果在不同 CC-MAIN 批次中出现，或评分在不同版本中有变化，可能进入不同桶，这符合分桶重组的设计目标
- **并行优化**：4 个桶可以并行运行（如果磁盘 I/O 和网络带宽允许），总体处理时间与单次遍历方案相当

**推荐采用**：按桶独立处理方案，实现简单可靠，与 Datatrove 生态完全兼容。

### 3.2 Datatrove API 要点

基于官方源码分析的关键发现：

1. **PipelineStep**: 需实现 `run(self, data, rank, world_size)` 方法，输入输出均为 `DocumentsPipeline`（Document 生成器）
2. **ParquetReader**: 通过 `adapter` 函数转换原始数据为 Document 对象，同时进行字段筛选
3. **Document**: `metadata` 是普通字典，可以安全修改；`text` 和 `id` 是顶层字段
4. **ParquetWriter**: 支持 `compression` 参数（zstd 等），支持 `output_filename` 模板（`${rank}`, `${metadata.field}`）

**关键配置示例**：

```python
# ParquetReader adapter：字段筛选 + CC-MAIN 提取
def fineweb_adapter(raw_dict: dict) -> Document:
    """将 FineWeb-Edu 原始数据转换为 Document，只保留必要字段。
    
    Args:
        raw_dict: 原始 parquet 行数据（包含 10 个字段）
        
    Returns:
        Document: 包含 id, text, score 三个顶层字段，以及 metadata.cc_main
    """
    # 从 dump 字段提取 CC-MAIN 快照批次名称（如 "CC-MAIN-2024-10"）
    dump = raw_dict.get("dump", "")
    cc_main = dump if dump.startswith("CC-MAIN-") else "unknown"
    
    return Document(
        text=raw_dict.get("text", ""),
        id=raw_dict.get("id", ""),
        score=raw_dict.get("score", 0.0),  # score 作为顶层字段
        metadata={
            "cc_main": cc_main,  # cc_main 保留在 metadata 中用于路径构建
        }
    )

# ParquetWriter 配置：动态路径 + zstd 压缩
writer = ParquetWriter(
    output_folder=output_path,
    output_filename="${metadata.cc_main}/${rank}.parquet",  # 动态路径
    compression="zstd",  # zstd 压缩
)
```

### 3.3 核心组件设计

#### 3.3.1 ScoreFilter（评分过滤器）

**设计要点**：
- **评分过滤**：根据评分区间过滤文档，采用左闭右开区间 `[min, max)`
- **确定性采样**：使用 MD5 哈希生成伪随机数，确保可复现性
- **进程内去重**：使用 Bloom Filter 进行进程内去重，避免同一文档在同一桶内重复
- **多进程一致性**：基于 `doc.id` 和 `random_seed` 生成随机数，保证不同进程结果一致

**实现逻辑**：
```python
import hashlib
from dataclasses import dataclass
from typing import Generator
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document, DocumentsPipeline


@dataclass
class BucketConfig:
    """评分桶配置"""
    name: str           # 桶名称（如 "2.8"）
    min_score: float    # 评分下限（包含）
    max_score: float | None  # 评分上限（不包含），None 表示无上限
    sampling_rate: float     # 采样率（0-1）
    
    def contains(self, score: float) -> bool:
        """检查评分是否在区间内（左闭右开）"""
        if self.max_score is None:
            return score >= self.min_score
        return self.min_score <= score < self.max_score


class ScoreFilter(PipelineStep):
    """评分过滤器：根据评分区间过滤文档并进行确定性采样。
    
    每个桶独立运行一个 ScoreFilter 实例，实现按桶独立处理。
    """
    
    def __init__(
        self,
        bucket: BucketConfig,
        random_seed: int = 42,
        use_bloom_filter: bool = True,
        bloom_capacity: int = 2_000_000_000,
        bloom_error_rate: float = 0.001,
    ):
        super().__init__()
        self.bucket = bucket
        self.random_seed = random_seed
        self.use_bloom_filter = use_bloom_filter
        self.bloom_capacity = bloom_capacity
        self.bloom_error_rate = bloom_error_rate
        self._bloom_filter = None
        
    def _init_bloom_filter(self):
        """延迟初始化 Bloom Filter"""
        if self.use_bloom_filter and self._bloom_filter is None:
            try:
                from pybloom_live import ScalableBloomFilter
                self._bloom_filter = ScalableBloomFilter(
                    initial_capacity=self.bloom_capacity,
                    error_rate=self.bloom_error_rate
                )
            except ImportError:
                raise ImportError(
                    "pybloom-live is required for Bloom Filter deduplication. "
                    "Install with: pip install pybloom-live"
                )
    
    def _is_duplicate(self, doc_id: str) -> bool:
        """检查文档是否重复（进程内）"""
        if not self.use_bloom_filter:
            return False
        self._init_bloom_filter()
        if doc_id in self._bloom_filter:
            return True
        self._bloom_filter.add(doc_id)
        return False
    
    def _should_sample(self, doc: Document, rate: float) -> bool:
        """确定性采样：使用 MD5 哈希生成伪随机数。
        
        Args:
            doc: 文档对象
            rate: 采样率（0-1）
            
        Returns:
            bool: 是否保留该文档
        """
        if rate >= 1.0:
            return True
        
        # 使用 MD5 哈希生成确定性随机数
        hash_input = f"{self.random_seed}_{doc.id}"
        hash_bytes = hashlib.md5(hash_input.encode()).digest()
        # 将前 8 字节转换为整数，获得高精度随机数
        hash_val = int.from_bytes(hash_bytes[:8], byteorder="big")
        random_val = hash_val / (2 ** 64)
        
        return random_val < rate
    
    def run(
        self,
        data: DocumentsPipeline,
        rank: int = 0,
        world_size: int = 1
    ) -> DocumentsPipeline:
        """处理文档流，过滤符合评分区间的文档。
        
        Args:
            data: 输入文档流
            rank: 当前进程编号
            world_size: 总进程数
            
        Yields:
            Document: 符合评分区间且通过采样的文档
        """
        for doc in data:
            # 1. 获取评分
            score = doc.metadata.get("score")
            if score is None:
                self.stat_update("missing_score", value=1)
                continue
            
            # 2. 检查是否在评分区间内
            if not self.bucket.contains(score):
                self.stat_update("filtered_out", value=1)
                continue
            
            # 3. 进程内去重检查
            if self._is_duplicate(doc.id):
                self.stat_update("duplicates_removed", value=1)
                continue
            
            # 4. 采样判断
            if self._should_sample(doc, self.bucket.sampling_rate):
                self.stat_update("kept", value=1)
                yield doc
            else:
                self.stat_update("sampled_out", value=1)
```

#### 3.3.2 MetadataCleaner（元数据清理器）

**说明**：字段筛选已在 `fineweb_adapter` 中完成，此组件用于最终清理，移除 `cc_main` 元数据。

**设计要点**：
- 输入字段：`id`、`text`、`score`（顶层字段）、`metadata.cc_main`
- 输出字段：`id`、`text`、`score`（`cc_main` 用于路径构建后移除）

**实现逻辑**：
```python
class MetadataCleaner(PipelineStep):
    """清理文档元数据，只保留指定字段。"""
    
    def __init__(self, keep_fields: set[str] | None = None):
        super().__init__()
        self.keep_fields = keep_fields or {"score"}
    
    def run(
        self,
        data: DocumentsPipeline,
        rank: int = 0,
        world_size: int = 1
    ) -> DocumentsPipeline:
        """清理元数据，只保留指定字段。"""
        for doc in data:
            # 只保留指定字段
            doc.metadata = {
                k: v for k, v in doc.metadata.items()
                if k in self.keep_fields
            }
            yield doc
```

#### 3.3.3 CCMainPathWriter（CC-MAIN 路径写入器）

根据文档的 `cc_main` 元数据，将文档写入对应的 CC-MAIN 批次子目录。

**设计要点**：
- **动态路径**：根据 `metadata.cc_main` 构建输出路径
- **压缩支持**：使用 zstd 压缩，节省存储空间
- **标准 Datatrove 组件**：继承 `ParquetWriter`，完全兼容 Datatrove Pipeline

**实现逻辑**：
```python
from datatrove.pipeline.writers import ParquetWriter


class CCMainPathWriter(ParquetWriter):
    """CC-MAIN 路径写入器：根据 cc_main 元数据构建输出路径。
    
    继承标准 ParquetWriter，使用 output_filename 模板实现动态路径。
    """
    
    def __init__(
        self,
        output_folder: str,
        compression: str = "zstd",
        max_file_size: int = 512 * 1024 * 1024,  # 512MB
    ):
        super().__init__(
            output_folder=output_folder,
            output_filename="${metadata.cc_main}/${rank}.parquet",
            compression=compression,
            max_file_size=max_file_size,
        )
```

#### 3.3.4 Pipeline 组装

**评分桶配置**：

| 桶名称 | 评分下限 | 评分上限 | 采样率 | 说明 |
|--------|----------|----------|--------|------|
| 2.8 | 2.8 | 3.0 | 30% | 2.8 ≤ score < 3.0（左闭右开） |
| 3.0 | 3.0 | 3.5 | 60% | 3.0 ≤ score < 3.5（左闭右开） |
| 3.5 | 3.5 | 4.0 | 80% | 3.5 ≤ score < 4.0（左闭右开） |
| 4.0 | 4.0 | None | 100% | score ≥ 4.0（左闭） |

**单个桶的 Pipeline 代码**：

```python
from datatrove.pipeline.readers import ParquetReader
from datatrove.executor import LocalPipelineExecutor
from pathlib import Path


def create_bucket_pipeline(
    input_path: Path,
    output_path: Path,
    bucket: BucketConfig,
    workers: int = 8,
    random_seed: int = 42,
) -> LocalPipelineExecutor:
    """创建单个评分桶的 Pipeline。
    
    Args:
        input_path: 源数据目录（包含 CC-MAIN 批次子目录）
        output_path: 输出目录（如 data/datasets/fineweb/en/2.8）
        bucket: 评分桶配置
        workers: 并行 worker 数量
        random_seed: 随机种子（用于确定性采样）
        
    Returns:
        LocalPipelineExecutor: 配置好的执行器
    """
    # 构建 Pipeline
    pipeline = [
        # 1. 读取 parquet 文件，使用 adapter 筛选字段
        ParquetReader(
            input_path,
            adapter=fineweb_adapter,
            glob_pattern="**/*.parquet",
        ),
        # 2. 评分过滤 + 采样 + 进程内去重
        ScoreFilter(
            bucket=bucket,
            random_seed=random_seed,
            use_bloom_filter=True,
            bloom_capacity=2_000_000_000,
            bloom_error_rate=0.001,
        ),
        # 3. 清理元数据（保留 score，移除 cc_main）
        MetadataCleaner(keep_fields={"score"}),
        # 4. 写入到 CC-MAIN 子目录
        CCMainPathWriter(
            output_folder=str(output_path),
            compression="zstd",
            max_file_size=512 * 1024 * 1024,  # 512MB
        ),
    ]
    
    return LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=workers,
        logging_dir=f"logs/fineweb_processing/{bucket.name}",
    )


# 处理所有桶
def process_all_buckets(
    input_path: Path,
    output_base: Path,
    workers_per_bucket: int = 8,
    random_seed: int = 42,
    parallel_buckets: int = 1,  # 同时运行的桶数量
):
    """处理所有评分桶。
    
    Args:
        input_path: 源数据目录
        output_base: 输出基础目录（如 data/datasets/fineweb/en）
        workers_per_bucket: 每个桶的 worker 数量
        random_seed: 随机种子
        parallel_buckets: 同时运行的桶数量（1=顺序，4=并行）
    """
    # 评分桶配置
    buckets = [
        BucketConfig("2.8", 2.8, 3.0, 0.30),
        BucketConfig("3.0", 3.0, 3.5, 0.60),
        BucketConfig("3.5", 3.5, 4.0, 0.80),
        BucketConfig("4.0", 4.0, None, 1.0),
    ]
    
    if parallel_buckets == 1:
        # 顺序处理
        for bucket in buckets:
            print(f"处理桶 {bucket.name}...")
            output_path = output_base / bucket.name
            executor = create_bucket_pipeline(
                input_path=input_path,
                output_path=output_path,
                bucket=bucket,
                workers=workers_per_bucket,
                random_seed=random_seed,
            )
            executor.run()
            print(f"桶 {bucket.name} 完成")
    else:
        # 并行处理（使用多进程）
        from concurrent.futures import ProcessPoolExecutor
        
        def run_bucket(bucket):
            output_path = output_base / bucket.name
            executor = create_bucket_pipeline(
                input_path=input_path,
                output_path=output_path,
                bucket=bucket,
                workers=workers_per_bucket,
                random_seed=random_seed,
            )
            executor.run()
            return bucket.name
        
        with ProcessPoolExecutor(max_workers=parallel_buckets) as pool:
            results = list(pool.map(run_bucket, buckets))
            print(f"所有桶完成: {results}")


# 主入口
def main():
    input_path = Path("data/datasets/HuggingFaceFW/fineweb-edu")
    output_path = Path("data/datasets/fineweb/en")
    
    # 顺序处理（推荐，避免磁盘 I/O 争用）
    process_all_buckets(
        input_path=input_path,
        output_base=output_path,
        workers_per_bucket=8,
        random_seed=42,
        parallel_buckets=1,  # 顺序处理
    )


if __name__ == "__main__":
    main()
```

**输出文件名格式**：
- 模板：`en/{bucket_name}/{metadata.cc_main}/{rank}.parquet`
- `{rank}`：Datatrove 分配的 worker 编号（整数）
- `{metadata.cc_main}`：从 `dump` 字段提取的 CC-MAIN 批次名称

**输出目录结构示例**：
```
data/datasets/fineweb/
└── en/
    ├── 2.8/
    │   └── CC-MAIN-2024-10/
    │       ├── 0.parquet
    │       ├── 1.parquet
    │       └── ...
    ├── 3.0/
    │   └── CC-MAIN-2024-10/
    │       └── ...
    ├── 3.5/
    │   └── ...
    └── 4.0/
        └── ...
```

---

## 4. 风险评估与决策

### 4.1 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 磁盘空间不足 | 中 | 高 | 预留 **800GB+** 空间；设置 `max_file_size=512MB`；实时监控磁盘使用 |
| 处理时间过长 | 低 | 中 | 多桶并行处理；8 workers 并行；断点续传 |
| 数据丢失或损坏 | 低 | 高 | 验证脚本全量检查；保留原始数据备份；处理前后对比统计 |
| 内存溢出 | 低 | 高 | Bloom Filter 去重（~2GB 内存）；Datatrove 流式处理；监控内存使用 |
| 采样偏差 | 低 | 高 | 文档 id + 随机种子确保多进程一致性；验证脚本检查分布 |
| 重复数据处理 | 中 | 中 | `ScoreFilter` 进程内去重；统计去重前后数量 |
| 进程间文件冲突 | 低 | 中 | 使用 `{rank}` 模板区分进程；不同桶输出到不同目录 |
| Bloom Filter 误报 | 低 | 低 | 设置误报率 0.1%；误报导致少量数据丢失（可接受） |

### 4.2 关键决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| **实现方案** | **按桶独立处理** | 每个评分桶独立运行 Pipeline，逻辑清晰，易于调试和维护 |
| **数据去重** | **Bloom Filter 进程内去重** | 每个 Pipeline 内部使用 Bloom Filter 进行去重，避免同一文档在同一桶内重复 |
| **验证方式** | 全量统计 + 抽样验证 | 关键指标全量统计（记录数、文件大小），细节抽样验证（内容正确性） |
| **错误处理** | 分级错误处理 | 致命错误停止，可恢复错误跳过并记录，警告记录继续 |
| **中间文件** | 直接输出到目标目录 | 依赖 Datatrove 断点续传，无需临时目录 |
| **压缩格式** | zstd | 高压缩率，快速解压，与原始数据格式一致 |
| **区间定义** | 左闭右开 `[min, max)` | 确保边界值（3.0、3.5、4.0）归属唯一，避免重复或遗漏 |
| **字段保留** | `id`, `text`, `score`（顶层字段） | `score` 与 `id`、`text` 并列，便于直接访问 |

### 4.3 配置参数

**全局配置**：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `random_seed` | int | 42 | 随机种子，用于确定性采样，确保可复现性 |
| `workers` | int | 8 | 并行 worker 数量，建议设置为 CPU 核心数 × 1.5（IO 密集型） |
| `compression` | str | "zstd" | 输出文件压缩格式，可选：zstd、gzip、snappy、none |
| `max_file_size` | int | 536870912 | 单个输出文件最大大小（字节），默认 512MB |

**Bloom Filter 配置**：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_bloom_filter` | bool | True | 是否使用 Bloom Filter 进行去重（推荐） |
| `bloom_capacity` | int | 2_000_000_000 | Bloom Filter 预估容量（20亿），可根据实际数据量调整 |
| `bloom_error_rate` | float | 0.001 | Bloom Filter 误报率（0.1%），误报会导致少量数据丢失 |

**路径配置**：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `input_path` | Path | - | 源数据目录：`data/datasets/HuggingFaceFW/fineweb-edu` |
| `output_path` | Path | - | 输出目录：`data/datasets/fineweb` |
| `logging_dir` | Path | `logs/fineweb_processing` | 日志存储目录 |

**Workers 数量建议**：

| 硬件配置 | 建议 workers | 说明 |
|----------|-------------|------|
| 8 核 CPU + 32GB RAM + SSD | 4-6 | 基础配置（考虑 Bloom Filter 内存占用） |
| 16 核 CPU + 64GB RAM + SSD | 8-12 | 推荐配置 |
| 32 核 CPU + 128GB RAM + NVMe | 16-24 | 高性能配置 |
| 云端（网络存储）| 8-16 | 网络带宽可能成为瓶颈 |

**⚠️ 内存约束**：每个 worker 需要约 3GB 内存用于 Bloom Filter（20亿容量，0.1%误报率）。建议 workers 数量 ≤ (总内存 - 8GB) / 3GB。

### 4.4 错误处理策略

**错误分级**：

| 级别 | 定义 | 处理方式 | 示例 |
|------|------|----------|------|
| **致命错误** | 无法继续处理的错误 | 立即停止，记录错误，人工介入 | 磁盘满、权限错误、数据损坏 |
| **可恢复错误** | 单个文档/文件错误 | 跳过当前项，记录错误，继续处理 | 单个 parquet 文件损坏、字段缺失 |
| **警告** | 非致命问题 | 记录警告，继续处理 | 采样率偏差略大、重复率略高 |

**日志规范**：

```
[时间] [级别] [模块] [CC-MAIN批次] 消息

示例：
2026-02-06 10:30:45 INFO ScoreFilter CC-MAIN-2024-10 处理完成: 123456 条记录
2026-02-06 10:30:46 WARNING ParquetReader CC-MAIN-2024-10 跳过损坏文件: part-00001.parquet
2026-02-06 10:30:47 ERROR CCMainPathWriter CC-MAIN-2024-10 磁盘空间不足，停止处理
```

**日志级别使用场景**：

- **DEBUG**：详细调试信息（开发阶段使用）
- **INFO**：正常处理信息（默认级别）
- **WARNING**：警告信息（采样偏差、重复率高等）
- **ERROR**：错误信息（可恢复错误）
- **CRITICAL**：致命错误（立即停止）

**断点续传机制**：

Datatrove 内置断点续传功能：

1. **进度文件**：`logs/fineweb_processing/completed_tasks.json`
2. **续传原理**：记录已完成的 task 编号，重启时自动跳过
3. **手动重置**：删除进度文件可强制重新处理

```bash
# 正常启动（自动续传）
python -m src.data_processing.fineweb_reorganizer

# 强制重新处理（删除进度）
rm logs/fineweb_processing/completed_tasks.json
python -m src.data_processing.fineweb_reorganizer
```

**失败重试策略**：

| 错误类型 | 最大重试次数 | 退避算法 | 说明 |
|----------|-------------|----------|------|
| IO 错误 | 3 | 指数退避（1s, 2s, 4s） | 网络/磁盘临时故障 |
| 解析错误 | 1 | - | 数据格式错误，不重试 |
| 内存错误 | 0 | - | 立即停止，需调整配置 |

### 4.5 回滚策略

如处理过程中出现严重错误：

1. **立即停止处理**：发送 SIGINT (Ctrl+C)，Datatrove 会优雅地停止当前任务
2. **检查已处理的数据**：查看输出目录，确认哪些文件已完整写入
3. **删除不完整的输出**：删除最后一个正在处理的桶的输出目录（如 `data/datasets/fineweb/en/2.8/`）
4. **检查日志定位问题**：查看 `logs/fineweb_processing/` 下的日志文件
5. **修复后重新启动**：重新运行处理脚本，Datatrove 会自动跳过已完成的文件

**不完整文件识别标准**：

- 文件大小为 0 字节
- parquet 文件无法读取
- 文件修改时间晚于停止时间

---

## 5. 实施计划

### 5.1 阶段划分

#### 阶段1：原型验证（0.5-1 天）

**目标**：验证核心组件的正确性

**任务清单**：
- [ ] 验证 `ParquetReader` + `adapter` 字段提取
- [ ] 验证 `ScoreFilter` 采样逻辑（确定性采样一致性）
- [ ] 选择 1 个 CC-MAIN 批次测试单个桶的完整流程
- [ ] 验证 `output_filename` 模板支持 `metadata.cc_main`
- [ ] 测试 zstd 压缩率和读写性能

**验收标准**：
- 字段提取正确（id、text、score）
- 采样比例符合预期（误差 < 1%）
- 输出文件路径正确

#### 阶段2：基础实现（1-2 天）

**目标**：完成核心功能实现

**任务清单**：
- [ ] 创建项目目录结构 (`src/data_processing/`)
- [ ] 实现 `fineweb_adapter` 字段筛选函数
- [ ] 实现 `BucketConfig` 数据类
- [ ] 实现 `ScoreFilter`（评分过滤 + 采样 + 进程内去重）
- [ ] 实现 `MetadataCleaner`（可选）
- [ ] 实现 `CCMainPathWriter`（CC-MAIN 路径写入器）
- [ ] 实现 `create_fineweb_pipeline()` 和 `main()`
- [ ] 添加基础日志记录
- [ ] 小规模验证（1-2 个 CC-MAIN 批次）

**验收标准**：
- 代码通过类型检查（`ruff check .`）
- 代码通过格式化（`ruff format .`）
- 单元测试通过（`pytest`）
- 小规模数据处理结果正确

#### 阶段3：优化完善（1 天）

**目标**：添加验证和 CLI 支持

**任务清单**：
- [ ] 实现验证脚本（`validate_output.py`）
- [ ] 添加 CLI 接口（支持指定桶、workers、随机种子等参数）
- [ ] 添加进度监控（处理速度、剩余时间估计）
- [ ] 完善错误处理和日志
- [ ] 编写使用文档

**CLI 接口设计**：
```bash
# 处理所有桶
python -m src.data_processing.fineweb_reorganizer

# 处理指定桶
python -m src.data_processing.fineweb_reorganizer --bucket 3.0

# 指定 workers 和随机种子
python -m src.data_processing.fineweb_reorganizer --workers 16 --seed 42

# 验证输出
python scripts/validate_output.py --input data/datasets/fineweb/
```

**验收标准**：
- CLI 接口可用
- 验证脚本能检测常见问题
- 文档完整

#### 阶段4：全量处理（2-3 天）

**目标**：完成全量数据处理

**任务清单**：
- [ ] 启动全量数据处理（110 个 CC-MAIN 批次）
- [ ] 监控处理进度和资源使用（CPU、内存、磁盘）
- [ ] 记录处理时间和吞吐量
- [ ] 处理中断恢复（如需要）

**监控指标**：
- 处理速度：文档数/秒、MB/秒
- 资源使用：CPU 使用率、内存占用、磁盘剩余空间
- 进度：已处理文件数/总文件数、已处理 CC-MAIN 批次

**验收标准**：
- 所有 4 个桶处理完成
- 无数据丢失
- 输出文件完整可读

#### 阶段5：验证交付（0.5-1 天）

**目标**：验证结果并交付

**任务清单**：
- [ ] 验证输出目录结构和 parquet 文件完整性
- [ ] 抽样验证采样比例（每个桶抽样 1000 条验证，误差 < 1%）
- [ ] 验证去重效果（检查重复率 < 0.5%）
- [ ] 生成最终统计报告（各桶记录数、文件大小、压缩率）
- [ ] 归档日志和配置

**验收标准**：
- 验证脚本通过
- 采样率误差 < 1%
- 重复率 < 0.5%
- 统计报告完整
- 数据质量符合预期

**验证标准详细定义**：

| 验证项 | 计算方法 | 可接受阈值 |
|--------|----------|-----------|
| 采样率误差 | `abs(实际采样率 - 目标采样率) / 目标采样率` | < 1% |
| 重复率 | `重复文档数 / 总文档数` | < 0.5% |
| 文件完整性 | 文件大小 > 0 且 parquet 可读 | 100% |
| 字段完整性 | 所有记录包含 id、text、score | 100% |
| 评分范围 | 所有记录的 score 在对应桶范围内 | 100% |

### 5.2 项目文件结构

```
nanomind/
├── src/data_processing/
│   ├── __init__.py
│   ├── fineweb_reorganizer.py      # 主入口和 CLI
│   ├── score_filter.py             # 评分过滤器（含采样和去重）
│   ├── metadata_cleaner.py         # 元数据清理器（可选）
│   ├── cc_main_path_writer.py      # CC-MAIN 路径写入器
│   └── adapters.py                 # 数据适配器（fineweb_adapter）
├── tests/
│   ├── test_score_filter.py
│   ├── test_cc_main_path_writer.py
│   └── test_pipeline.py
├── scripts/
│   ├── run_processing.sh           # 批量运行脚本
│   └── validate_output.py          # 验证脚本
├── logs/fineweb_processing/        # 处理日志
│   └── *.log                       # 日志文件
├── data/datasets/
│   ├── HuggingFaceFW/fineweb-edu/  # 源数据（只读）
│   │   └── CC-MAIN-XXXX-XX/        # CC-MAIN 批次
│   │       └── *.parquet
│   └── fineweb/                    # 输出数据
│       └── en/
│           ├── 2.8/
│           │   └── CC-MAIN-XXXX-XX/
│           │       └── *.parquet
│           ├── 3.0/
│           │   └── CC-MAIN-XXXX-XX/
│           ├── 3.5/
│           │   └── CC-MAIN-XXXX-XX/
│           └── 4.0/
│               └── CC-MAIN-XXXX-XX/
├── pyproject.toml
└── requirements.txt
```

---

## 6. 验证方案

### 6.1 验证内容

| 验证项 | 方法 | 工具/脚本 | 成功标准 |
|--------|------|----------|----------|
| 字段完整性 | 检查 parquet 文件包含 id、text、score（顶层字段） | `validate_output.py --check-schema` | 100% 记录包含所有必需字段 |
| 采样比例 | 抽样统计各桶实际采样率 | `validate_output.py --check-sampling` | 误差 < 1% |
| 去重效果 | 统计重复文档数量 | `validate_output.py --check-duplicates` | 重复率 < 0.5% |
| 文件完整性 | 检查 parquet 文件可读、无损坏 | `validate_output.py --check-files` | 100% 文件可读 |
| 路径正确性 | 验证输出路径符合设计 | `validate_output.py --check-paths` | 路径格式正确 |
| 评分范围 | 验证所有记录的 score 在对应桶范围内 | `validate_output.py --check-scores` | 100% 记录在范围内 |
| 压缩率 | 统计压缩前后大小 | `validate_output.py --stats` | 记录实际压缩率 |

### 6.2 验证脚本设计

**功能**：
1. **全量检查**：遍历所有输出文件，检查基本完整性
2. **抽样验证**：随机抽样检查内容正确性
3. **统计分析**：生成各桶的统计报告

**输出示例**：
```
=== FineWeb-Edu 重组验证报告 ===

桶 2.8:
  - 文件数: 120
  - 记录数: 1,234,567
  - 总大小: 12.3 GB
  - 采样率: 29.8% (预期: 30%)
  - 重复率: 0.1%

桶 3.0:
  - 文件数: 150
  - 记录数: 2,345,678
  - 总大小: 23.4 GB
  - 采样率: 60.2% (预期: 60%)
  - 重复率: 0.2%

...（其他桶）

总计:
  - 文件数: 600
  - 记录数: 8,765,432
  - 总大小: 145.6 GB
  - 压缩率: 67%

验证结果: ✅ 通过
```

---

## 7. 附录

### 7.1 参考资源

- FineWeb-Edu 数据集：https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Datatrove 文档：https://github.com/huggingface/datatrove
- Datatrove API 参考：https://github.com/huggingface/datatrove/tree/main/src/datatrove

### 7.2 术语表

| 术语 | 说明 |
|------|------|
| **CC-MAIN** | Common Crawl 快照批次命名格式（如 CC-MAIN-2024-10） |
| **cc_main** | 代码中的变量名，对应 CC-MAIN 批次名称 |
| **Parquet** | 列式存储格式，支持压缩和高效读取 |
| **Score** | FineWeb-Edu 教育价值评分（0-5分），double 类型 |
| **int_score** | 整数化评分（不保留，仅参考） |
| **Datatrove** | Hugging Face 大规模文本数据处理库 |
| **PipelineStep** | Datatrove 管道处理基类 |
| **Document** | Datatrove 文档对象，包含 text、id、metadata |
| **DocumentsPipeline** | Document 生成器类型（`Generator[Document, None, None]`） |
| **Adapter** | ParquetReader 的数据转换函数，用于字段筛选和类型转换 |
| **Bucket** | 评分分桶（2.8、3.0、3.5、4.0） |
| **bucket_name** | 代码中的变量名，对应评分桶名称（如 "2.8"） |
| **Sampling Rate** | 采样率（0-1），表示保留数据的比例 |
| **Deterministic Sampling** | 确定性采样，使用哈希确保可复现性 |
| **Bloom Filter** | 概率型数据结构，用于高效去重，可能产生误报 |
| **左闭右开区间** | 数学区间表示法 `[min, max)`，包含 min 不包含 max |

### 7.3 变更记录

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-02-05 | v0.1 | 初始版本：确定 4 次顺序 pipeline 方案 |
| 2026-02-06 | v0.2 | 完善版本：补充 DedupFilter 设计、CLI 接口、验证方案 |
| 2026-02-06 | **v1.0** | **架构优化版本**：<br>• 架构方案优化：4 次 Pipeline → 单次遍历 + 多路输出（4× 性能提升）<br>• 修复严重一致性问题：评分区间定义、字段筛选定义、目录结构<br>• 修复技术错误：代码示例完整性、内存问题（set → Bloom Filter）<br>• 补充缺失内容：adapter 函数、output_filename 配置、CC-MAIN 提取逻辑<br>• 添加配置参数表和错误处理策略<br>• 统一术语和格式 |
| 2026-02-06 | **v1.1** | **设计完善版本**：<br>• 修复数据矛盾：明确 4.2 TB 对应 score≥2 版本<br>• 修复技术架构：放弃单次多路输出，改为按桶独立处理（更可靠）<br>• 修复 Bloom Filter 描述：明确为"进程内去重"而非"全局去重"<br>• 统一术语和格式：采样率格式、workers 配置、内存估算<br>• 补充缺失章节：边界情况处理、监控方案、下游使用规范<br>• 优化冗余内容：精简重复描述、合并验证标准 |

---

## 8. 边界情况处理

### 8.1 异常数据定义与处理策略

| 异常情况 | 判定条件 | 处理策略 | 统计指标 |
|---------|---------|---------|---------|
| **score 字段缺失** | `score is None` | 跳过该文档，记录到 `missing_score` 统计 | `missing_score` |
| **score 超出范围** | `score < 0` 或 `score > 5` | 视为无效数据，跳过并记录 | `invalid_score` |
| **text 字段为空** | `text is None` 或 `len(text.strip()) == 0` | 跳过该文档，记录到 `empty_text` 统计 | `empty_text` |
| **text 过短** | `len(text) < 10` | 保留（不过滤），但记录到 `short_text` 统计 | `short_text` |
| **id 字段缺失** | `id is None` 或 `id == ""` | 使用 UUID 生成临时 ID，记录到 `missing_id` 统计 | `missing_id` |
| **id 冲突（进程内）** | Bloom Filter 检测到重复 | 跳过重复文档，记录到 `duplicates_removed` 统计 | `duplicates_removed` |
| **cc_main 提取失败** | `dump` 字段不符合格式 | 使用 `"unknown"` 作为默认值 | - |

### 8.2 浮点数边界处理

**问题**：浮点数比较可能因精度问题导致边界值归属错误。

**解决方案**：
```python
# 使用 epsilon 比较避免浮点数精度问题
epsilon = 1e-9

def contains(score: float, min_score: float, max_score: float | None) -> bool:
    """检查评分是否在区间内（考虑浮点数精度）"""
    if max_score is None:
        return score >= min_score - epsilon
    return (score >= min_score - epsilon) and (score < max_score - epsilon)
```

---

## 9. 下游使用规范

### 10.1 数据加载示例

**使用 HuggingFace datasets 加载**：

```python
from datasets import load_dataset

# 加载特定评分桶
dataset = load_dataset(
    "parquet",
    data_dir="data/datasets/fineweb/en/3.0",
    split="train"
)

# 加载多个桶
dataset_28 = load_dataset("parquet", data_dir="data/datasets/fineweb/en/2.8", split="train")
dataset_30 = load_dataset("parquet", data_dir="data/datasets/fineweb/en/3.0", split="train")
dataset_35 = load_dataset("parquet", data_dir="data/datasets/fineweb/en/3.5", split="train")
dataset_40 = load_dataset("parquet", data_dir="data/datasets/fineweb/en/4.0", split="train")

# 合并多个桶（如果需要）
from datasets import concatenate_datasets
combined = concatenate_datasets([dataset_30, dataset_35, dataset_40])
```

**使用 Pandas 读取单个文件**：

```python
import pandas as pd

# 读取特定 CC-MAIN 批次的 parquet 文件
df = pd.read_parquet("data/datasets/fineweb/en/3.0/CC-MAIN-2024-10/0.parquet")

# 查看数据结构
print(df.columns)  # ['id', 'text', 'score']
print(df.head())
```

### 10.2 路径规范

**输出路径模板**：
```
data/datasets/fineweb/
└── en/
    └── {bucket_name}/              # 评分桶名称：2.8, 3.0, 3.5, 4.0
        └── {cc_main}/              # CC-MAIN 批次：CC-MAIN-2024-10
            └── {rank}.parquet      # worker 编号：0.parquet, 1.parquet, ...
```

**路径构建示例**：
```python
from pathlib import Path

def get_bucket_path(base_path: Path, bucket_name: str, cc_main: str | None = None):
    """构建评分桶路径"""
    path = base_path / "en" / bucket_name
    if cc_main:
        path = path / cc_main
    return path

# 使用示例
base = Path("data/datasets/fineweb")
path_30 = get_bucket_path(base, "3.0")  # data/datasets/fineweb/en/3.0
path_cc = get_bucket_path(base, "3.0", "CC-MAIN-2024-10")  # data/datasets/fineweb/en/3.0/CC-MAIN-2024-10
```

### 10.3 数据版本管理

**版本命名规范**：
- 数据版本格式：`fineweb-edu-v{version}-{date}`
- 示例：`fineweb-edu-v1.0-20260206`

**版本记录文件**：
```json
{
  "version": "1.0",
  "created_at": "2026-02-06",
  "source_dataset": "HuggingFaceFW/fineweb-edu",
  "source_version": "default",
  "sampling_config": {
    "2.8": 0.30,
    "3.0": 0.60,
    "3.5": 0.80,
    "4.0": 1.0
  },
  "total_records": 8765432,
  "total_size_gb": 145.6,
  "compression": "zstd"
}
```

---

## 11. 合规与许可

### 11.1 数据许可证说明

FineWeb-Edu 数据集基于 **ODC-BY 1.0** 许可证发布：

- **允许**：商业使用、修改、分发
- **要求**：必须提供归属声明
- **禁止**：无

### 11.2 归属声明

使用处理后的数据时，请在相关文档或代码中包含以下归属声明：

```
本数据基于 HuggingFaceFW/fineweb-edu 数据集处理生成，
原始数据集由 Hugging Face 发布，遵循 ODC-BY 1.0 许可证。
```

### 11.3 使用限制

- 不得声称原始数据为自己创建
- 不得使用数据训练可能用于恶意目的的模型（如生成虚假信息）
- 建议在使用前检查数据内容，排除不适当的内容

---

**文档结束**
