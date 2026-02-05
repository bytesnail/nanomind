# FineWeb-Edu 数据集质量评分分桶重组设计

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

将 FineWeb-Edu 数据集按照质量评分（`score` 字段）进行分桶重组，采用分层采样策略：

| 质量评分区间 | 采样率 | 说明 |
|-------------|--------|------|
| < 2.8 | **0%**（丢弃） | 低质量数据 |
| 2.8 ≤ score < 3.0 | **30%** | 中低质量数据 |
| 3.0 ≤ score < 3.5 | **60%** | 中等质量数据 |
| 3.5 ≤ score < 4.0 | **80%** | 高质量数据 |
| score ≥ 4.0 | **100%** | 顶级质量数据 |

**输出目录结构**：

```
data/datasets/fineweb/
└── en/
    └── {score_bucket}/
        └── CC-MAIN-xxxx-xx/
            └── xxxxxxxx.parquet
```

**字段筛选**：输出仅保留 `id`、`text`、`score`（处理时需临时保留 `file_path` 用于提取 CC-MAIN 批次信息）。

**预期效果**：
- 预估最终数据量约为原始数据的 **35%**（zstd 压缩后约 150 GB）
- 采用"区间分桶"策略，各目录数据互不重叠

---

## 2. 数据集现状分析

### 2.1 数据结构

**原始字段**（来自 parquet 文件）：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `text` | string | 文本内容 |
| `id` | string | 唯一标识符（UUID） |
| `dump` | string | Common Crawl 批次 |
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

**百分位数**：

| 百分位 | 分数 |
|--------|------|
| 50% | 2.9062 |
| 75% | 3.2344 |
| 90% | 3.5781 |
| 95% | 3.7812 |
| 99% | 4.1250 |

**分布特点**：数据高度集中在 2.5-3.5 分区间（约 70%），4.0 分以上数据稀少（<2%）。

### 2.3 数据规模与约束

| 指标 | 数值 |
|------|------|
| 原始数据集总大小 | 4.2 TB |
| CC-MAIN 批次数量 | 110 个 |
| Parquet 文件数量 | 2,410 个 |
| 可用磁盘空间 | 3.5 TB |

**存储约束解决方案**：

| 处理步骤 | 预估数据量 |
|---------|-----------|
| 原始数据（10 个字段） | 4.2 TB |
| 字段筛选（3 个字段） | ~1.3 TB |
| 分层采样（未压缩） | ~445 GB |
| zstd 压缩后 | ~150 GB |

---

## 3. 技术实现方案

### 3.1 实现架构

使用 **Datatrove**（Hugging Face 开发的大规模数据处理库）实现：

```
┌─────────────────────────────────────────────────────────────┐
│                    Datatrove Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  1. ParquetReader                                            │
│     ├── 读取: data/datasets/HuggingFaceFW/fineweb-edu/data   │
│     ├── 列投影: ["id", "text", "score", "file_path"]          │
│     └── 输出: Document 对象流                                │
├─────────────────────────────────────────────────────────────┤
│  2. StratifiedSampler (自定义)                               │
│     ├── 过滤: score < 2.8 的数据（自动丢弃）                 │
│     ├── 2.8-3.0: 30% 采样                                    │
│     ├── 3.0-3.5: 60% 采样                                    │
│     ├── 3.5-4.0: 80% 采样                                    │
│     └── ≥4.0: 100% 保留                                      │
├─────────────────────────────────────────────────────────────┤
│  3. MultiBucketWriter (自定义)                               │
│     ├── 按评分区间分发到不同桶                               │
│     ├── 2.8/: 2.8-3.0 数据                                   │
│     ├── 3.0/: 3.0-3.5 数据                                   │
│     ├── 3.5/: 3.5-4.0 数据                                   │
│     └── 4.0/: ≥4.0 数据                                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件设计

#### 3.2.1 StratifiedSampler

```python
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document
import hashlib

class StratifiedSampler(PipelineStep):
    """分层采样器：按评分区间应用不同采样率（左闭右开区间）"""
    
    def __init__(self, random_seed: int = 42):
        super().__init__()
        self.random_seed = random_seed
        self.sampling_rates = [
            (2.8, 3.0, 0.30),
            (3.0, 3.5, 0.60),
            (3.5, 4.0, 0.80),
            (4.0, float('inf'), 1.0),
        ]
    
    def __call__(self, doc: Document) -> Document | None:
        """根据评分决定是否采样保留。返回 Document 表示保留，None 表示丢弃。"""
        score = doc.metadata.get("score", 0)
        
        for low, high, rate in self.sampling_rates:
            if low <= score < high:
                # 使用确定性哈希确保可复现性
                hash_input = f"{self.random_seed}_{doc.id}_{low}_{high}"
                hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                random_val = (hash_val % 10000) / 10000.0
                
                return doc if random_val < rate else None
        
        return None  # score < 2.8，丢弃
```

**设计要点**：使用 MD5 哈希生成确定性随机数，每个文档独立采样，确保多进程环境下结果一致。

#### 3.2.2 MultiBucketWriter

```python
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter
from datatrove.data import Document
from pathlib import Path
import re

class MultiBucketWriter(PipelineStep):
    """多桶写入器：按评分区间将数据写入不同目录，保持原始 CC-MAIN 结构"""
    
    def __init__(self, base_output_path: Path, compression: str = "zstd", **kwargs):
        super().__init__()
        self.base_output_path = Path(base_output_path)
        self.compression = compression
        self.kwargs = kwargs
        self.writers = {}
        self.cc_main_pattern = re.compile(r'CC-MAIN-\d{4}-\d{2}')
    
    def _get_bucket(self, score: float) -> str | None:
        """根据评分确定目标桶"""
        if 2.8 <= score < 3.0:
            return "2.8"
        elif 3.0 <= score < 3.5:
            return "3.0"
        elif 3.5 <= score < 4.0:
            return "3.5"
        elif score >= 4.0:
            return "4.0"
        return None
    
    def _extract_cc_main(self, file_path: str) -> str:
        """从文件路径提取 CC-MAIN 批次名称"""
        match = self.cc_main_pattern.search(file_path)
        return match.group(0) if match else "unknown"
    
    def __call__(self, doc: Document) -> Document:
        """将文档写入对应评分桶"""
        score = doc.metadata.get("score", 0)
        bucket = self._get_bucket(score)
        
        if bucket is None:
            return doc  # score < 2.8，跳过写入
        
        original_path = doc.metadata.get("file_path", "")
        cc_main = self._extract_cc_main(original_path)
        
        key = f"{bucket}/{cc_main}"
        if key not in self.writers:
            bucket_path = self.base_output_path / "en" / bucket / cc_main
            bucket_path.mkdir(parents=True, exist_ok=True)
            self.writers[key] = ParquetWriter(
                output_folder=str(bucket_path),
                compression=self.compression,
                **self.kwargs
            )
        
        self.writers[key](doc)
        return doc
    
    def close(self):
        """关闭所有 writer 释放资源"""
        for writer in self.writers.values():
            if hasattr(writer, 'close'):
                writer.close()
        self.writers.clear()
```

**设计要点**：动态创建 `ParquetWriter` 实例管理各桶写入，提供 `close()` 方法用于资源释放。

#### 3.2.3 Pipeline 组装

```python
from datatrove.pipeline import Pipeline
from datatrove.pipeline.readers import ParquetReader
from datatrove.executor import LocalPipelineExecutor
from pathlib import Path

def create_fineweb_pipeline(
    source_dir: Path = Path("data/datasets/HuggingFaceFW/fineweb-edu"),
    target_dir: Path = Path("data/datasets/fineweb"),
    random_seed: int = 42,
) -> tuple[Pipeline, MultiBucketWriter]:
    """创建 FineWeb-Edu 数据处理 Pipeline"""
    
    reader = ParquetReader(
        data_folder=str(source_dir),
        columns=["id", "text", "score", "file_path"],
        glob_pattern="**/*.parquet",
    )
    
    sampler = StratifiedSampler(random_seed=random_seed)
    writer = MultiBucketWriter(base_output_path=target_dir)
    
    pipeline = Pipeline(reader, sampler, writer)
    
    return pipeline, writer

# 执行 Pipeline
if __name__ == "__main__":
    pipeline, writer = create_fineweb_pipeline()
    
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        workers=8,
        logging_dir="logs/fineweb_processing",
    )
    
    try:
        executor.run()
    finally:
        writer.close()  # 确保资源释放
```

### 3.3 Datatrove 优势

| 项目需求 | Datatrove 解决方案 |
|---------|-------------------|
| 字段筛选 | `ParquetReader(columns=[...])` 列投影 |
| 分层采样 | 自定义 `StratifiedSampler` 组件 |
| 多桶输出 | 自定义 `MultiBucketWriter` 组件 |
| 并行处理 | `LocalPipelineExecutor(workers=N)` |
| 进度跟踪 | 内置 tqdm 进度条和日志记录 |
| 断点续传 | 自动保存处理状态 |
| 内存优化 | 流式处理 |
| 采样一致性 | 文档 id + 随机种子确保多进程一致性 |

---

## 4. 风险评估与缓解措施

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 磁盘空间不足 | 低 | 高 | 处理后约150 GB，预留充足空间 |
| 处理时间过长 | 中 | 中 | 并行处理（8-16 workers），断点续传 |
| 数据丢失或损坏 | 低 | 高 | 验证脚本，抽样检查 |
| 内存溢出 | 低 | 高 | Datatrove 流式处理 |
| 采样偏差 | 低 | 高 | 文档id+随机种子确保多进程一致性 |

---

## 5. 决策记录

| 决策项 | 最终选择 | 说明 |
|--------|----------|------|
| **分桶策略** | 4 个评分桶 | 2.8, 3.0, 3.5, 4.0（左闭右开区间） |
| **实现框架** | Datatrove | Hugging Face 数据处理库 |
| **数据范围** | 全部处理 | 110 个 CC-MAIN 批次 |
| **采样策略** | 分层采样 | 0%/30%/60%/80%/100% |
| **随机种子** | 42 | 保证可复现性 |
| **输出目录** | `data/datasets/fineweb/` | 按评分分桶存储 |
| **保留字段** | id, text, score | 处理时需保留 file_path |
| **压缩算法** | zstd | 平衡压缩率和速度 |
| **并行度** | 8-16 workers | 根据 CPU 核心数调整 |

---

## 6. 后续工作计划

### 阶段1：基础实现（1-2 天）

1. 创建项目目录结构
2. 实现 `StratifiedSampler`：按评分区间分层采样
3. 实现 `MultiBucketWriter`：按评分区间分桶写入
4. 组装 Pipeline 并测试
5. 小规模验证（1-2 个 CC-MAIN 批次）

### 阶段2：优化完善（1 天）

1. 配置 `LocalPipelineExecutor` 并行参数
2. 测试断点续传功能
3. 实现验证脚本
4. 实现 CLI 接口

### 阶段3：全量处理（2-3 天）

1. 启动全量数据处理（110 个 CC-MAIN 批次）
2. 监控处理进度和资源使用
3. 处理中断恢复（如需要）

### 阶段4：验证交付（0.5-1 天）

1. 验证输出目录结构
2. 验证 parquet 文件完整性
3. 抽样验证采样比例
4. 生成最终统计报告

---

## 7. 附录

### 7.1 参考资源

- FineWeb-Edu 数据集：https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Datatrove 文档：https://github.com/huggingface/datatrove

### 7.2 术语表

| 术语 | 说明 |
|------|------|
| CC-MAIN | Common Crawl 快照批次命名格式 |
| Parquet | 列式存储格式 |
| Score | FineWeb-Edu 教育价值评分（0-5分） |
| Datatrove | Hugging Face 大规模文本数据处理库 |
| 区间分桶 | 按评分区间分目录存储，数据互不重叠 |
| 确定性采样 | 使用文档 id + 随机种子确保采样可复现 |

---

**文档结束**
