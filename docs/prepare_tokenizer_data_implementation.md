# prepare_tokenizer_data.py 实现文档

> **文件路径**: `scripts/prepare_tokenizer_data.py`  
> **总代码行数**: 904 行  
> **最后更新**: 2026-02-14  
> **作者**: chenkun

---

## 1. 概述

### 1.1 功能定位

`prepare_tokenizer_data.py` 是一个**多数据源流式采样工具**，用于从多个预处理的 Parquet 数据集（FineWeb-EN、FineWeb-ZH、GitHub Code、Nemotron-CC-Math）中按配置比例采样，生成用于训练 64K BPE Tokenizer 的 40M 样本数据。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **确定性采样** | 基于 MD5 哈希，相同种子产生相同结果 |
| **真正的流式处理** | 边读取边写入，不累积全部数据到内存 |
| **多数据源支持** | 支持 buckets/stars_filter 两种分桶策略 |
| **并发 IO 优化** | 文件统计阶段使用 2× workers 并发 |
| **分文件输出** | 自动分片为多个 Parquet 文件 |

---

## 2. 架构设计

### 2.1 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                     prepare_tokenizer_data.py                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Config Load │───▶│ Dataset Loop │───▶│ Bucket Loop  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ YAML Parser  │    │ Parallel     │    │ Stream       │      │
│  │ Validation   │    │ File Stats   │    │ Top-K Select │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              StreamingParquetWriter                      │  │
│  │     (流式写入，buffer_size=5000, max_rows=500k)          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Output: data/datasets/nanomind_tokenizer/              │  │
│  │  ├── train-{idx:05d}-of-{total:05d}.parquet             │  │
│  │  └── sampling_info.json                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心类图

```
┌─────────────────────────┐
│   DocHash (NamedTuple)  │
├─────────────────────────┤
│ hash_value: int         │
│ doc_id: str             │
│ file_path: Path         │
│ row_index: int          │
└─────────────────────────┘
           │
           │ 用于 Top-K 选择
           ▼
┌─────────────────────────┐
│ SamplingConfig          │
├─────────────────────────┤
│ name: str               │
│ source: Path            │
│ samples: int            │
│ buckets: dict           │
│ stars_filter: dict      │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│StreamingParquetWriter   │
├─────────────────────────┤
│ - write(sample)         │
│ - close() -> [Path]     │
│ - __enter__/__exit__    │
└─────────────────────────┘
```

---

## 3. 实现演进历史

### 3.1 版本演进

| 版本 | Commit | 日期 | 主要改进 |
|------|--------|------|----------|
| **v1.0** | `edd2a78` | 2026-02-13 | 初始实现，串行处理，全量加载 |
| **v2.0** | `fa34b79` | 2026-02-13 | 添加并发处理，提取 `compute_doc_hash`，优化 Parquet 写入 |
| **v3.0** | `a57df8d` | 2026-02-14 | 引入 `StreamingParquetWriter`，重构为流式写入，添加 `parallel_map` 工具函数 |
| **v4.0** | `723636b` | 2026-02-14 | **真正的流式处理**，使用 `ParquetFile.iter_batches()`，内存优化 |

### 3.2 关键变更详解

#### v1.0 → v2.0: 并发化改造

**问题**: 串行处理大文件太慢  
**解决**: 引入 `ThreadPoolExecutor` 并行处理文件

```python
# 新增核心函数
- compute_doc_hash(doc_id, seed) -> int  # 提取可测试的哈希计算
- count_bucket_samples_parallel()        # 并行统计行数
- sample_from_bucket() 重构              # 分阶段并行计算
```

#### v2.0 → v3.0: 流式写入

**问题**: 40M 样本全部加载到内存导致 OOM  
**解决**: 边采样边写入

```python
# 新增 StreamingParquetWriter 类
class StreamingParquetWriter:
    def write(self, sample: SampleDoc) -> None  # 缓冲写入
    def close(self) -> list[Path]               # 刷盘并重命名

# 重构主流程
- prepare_tokenizer_data() 不再累积 all_samples 列表
- process_dataset() 传入 writer 参数直接写入
```

#### v3.0 → v4.0: 真正的流式读取

**问题**: `pq.read_table()` 仍会将整个文件加载到内存  
**解决**: 使用 `ParquetFile.iter_batches()` 实现真正的流式

```python
# 核心改进
- stream_file_rows()  # 使用 iter_batches() 的生成器
- select_top_k_document_hashes()  # 流式 Top-K 选择（只保留索引，不保留数据）
- 内存优化：堆中存储 (int, int, int) 而非 DocHash 对象
```

---

## 4. 核心算法详解

### 4.1 确定性哈希采样

```python
def compute_doc_hash(doc_id: str, seed: int) -> int:
    """计算文档的确定性哈希值。"""
    data = f"{seed}_{doc_id}".encode()
    return int.from_bytes(
        hashlib.md5(data, usedforsecurity=False).digest()[:8],
        "big",
    )
```

**设计要点**:
- 使用 `seed` + `doc_id` 确保不同运行产生不同但确定的结果
- MD5 取前 8 字节作为 64 位无符号整数
- `usedforsecurity=False` 避免安全警告（MD5 不用于加密场景）

### 4.2 流式 Top-K 选择

```python
def select_top_k_document_hashes(files, bucket_name, seed, target_count):
    """流式选择哈希值最小的前 target_count 个文档。"""
    max_heap = []  # 存储 (-hash, file_idx, row_idx)
    
    for file_idx, fp in enumerate(files):
        for row_idx in range(num_rows):
            doc_id = f"{bucket_name}#{fp.name}#{row_idx}"
            doc_hash = compute_doc_hash(doc_id, seed)
            
            if len(max_heap) < target_count:
                heapq.heappush(max_heap, (-doc_hash, file_idx, row_idx))
            elif doc_hash < -max_heap[0][0]:
                heapq.heapreplace(max_heap, (-doc_hash, file_idx, row_idx))
    
    # 构建文件到索引的映射
    file_to_indices = {}
    for _, file_idx, row_idx in max_heap:
        fp = file_list[file_idx]
        file_to_indices.setdefault(fp, set()).add(row_idx)
    
    # 显式释放大对象
    del max_heap
    gc.collect()
    
    return file_to_indices
```

**内存优化策略**:
1. **索引代替对象**: 堆中存储 `(int, int, int)` 而非 `DocHash` 对象
2. **串行处理**: 避免并行导致的内存峰值
3. **及时 GC**: 完成后立即删除大对象并触发 GC

### 4.3 流式文件读取

```python
def stream_file_rows(file_path, text_column, batch_size=50000, indices=None):
    """流式读取文件的行，产生 (row_index, text) 对。"""
    with pq.ParquetFile(file_path) as pf:
        row_idx = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
            for text in batch.column(text_column).to_pylist():
                if indices is None or row_idx in indices:
                    yield row_idx, text
                row_idx += 1
```

**关键改进**:
- 使用 `ParquetFile.iter_batches()` 而非 `read_table()`
- 只读取指定列，减少 I/O
- 支持可选的 `indices` 过滤，用于读取已选中的行

---

## 5. 配置系统

### 5.1 配置文件结构

```yaml
# config/tokenizer_data.yaml
datasets:
  fineweb_en:                    # 数据集标识
    name: "fineweb_edu_en"       # 输出中的 source_dataset 值
    source: "data/datasets/fineweb/en"  # 源数据目录
    samples: 12000000            # 总样本数（仅用于参考）
    buckets:                     # 分桶配置
      4.0: {count: 5400000}
      3.5: {count: 2400000}
      3.0: {count: 2400000}
      2.5: {count: 1800000}
  
  github_code:                   # 另一种分桶方式
    name: "github_code"
    source: "data/datasets/nick007x/github-code-2025"
    samples: 12000000
    stars_filter:                # 用于 GitHub 的 stars 筛选
      above_2: {count: 10000000}
      below_2: {count: 2000000}

random_seed: 42
output_format: "parquet"
output_dir: "data/datasets/nanomind_tokenizer"
```

### 5.2 配置解析逻辑

```python
@dataclass
class SamplingConfig:
    name: str
    source: Path
    samples: int
    buckets: dict[str, int] = field(default_factory=dict)
    stars_filter: dict[str, int] = field(default_factory=dict)
    
    def get_all_counts(self) -> dict[str, int]:
        """统一获取分桶计数（支持 buckets 或 stars_filter）。"""
        if self.buckets:
            return self.buckets
        if self.stars_filter:
            return self.stars_filter
        return {}
```

---

## 6. 性能优化策略

### 6.1 内存优化

| 优化点 | 实现方式 | 效果 |
|--------|----------|------|
| **流式读取** | `ParquetFile.iter_batches()` | 避免加载整个文件 |
| **流式写入** | `StreamingParquetWriter` (buffer=5000) | 边生成边写入 |
| **索引存储** | Top-K 只存索引，不存数据 | 堆内存从 O(N) 降到 O(K) |
| **列裁剪** | `columns=[text_column]` | 只读必要列 |
| **元数据缓存** | `@lru_cache(maxsize=1024)` | 避免重复统计文件行数 |

### 6.2 并发策略

```python
# IO 密集型操作使用更高并发
DEFAULT_WORKERS = min(32, os.cpu_count() or 4)
DEFAULT_IO_WORKERS = DEFAULT_WORKERS * 2  # 统计阶段 2× workers

# 三阶段并发
1. 统计文件行数: 并行 (io_workers)
2. 计算哈希值:   串行（避免内存累积）
3. 读取数据:     流式（单文件顺序读）
```

### 6.3 默认参数调优

```python
DEFAULT_MAX_ROWS = 500_000    # 每文件 50万行，平衡文件数和单文件大小
DEFAULT_BATCH_SIZE = 50_000   # 流式读取批次，影响内存和速度
COMPRESSION = "zstd"          # 压缩率和速度的平衡
```

---

## 7. 输出格式

### 7.1 Parquet Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 文档文本内容 |
| `source_dataset` | string | 来源数据集名称（如 fineweb_edu_en） |
| `source_bucket` | string | 来源分桶标识（如 4.0, above_2） |

### 7.2 采样信息 JSON

```json
{
  "total_requested": 40000000,
  "total_sampled": 39876543,
  "random_seed": 42,
  "sources": {
    "fineweb_en": {
      "name": "fineweb_edu_en",
      "source": "data/datasets/fineweb/en",
      "requested": 12000000,
      "sampled": 11982345,
      "buckets": {
        "4.0": {"requested": 5400000, "sampled": 5392000},
        "3.5": {"requested": 2400000, "sampled": 2398000}
      }
    }
  }
}
```

---

## 8. 测试覆盖

### 8.1 测试文件

**文件**: `tests/test_prepare_tokenizer_data.py` (490 行)

### 8.2 测试类别

| 测试类 | 用例数 | 覆盖功能 |
|--------|--------|----------|
| `TestComputeDocHash` | 4 | 哈希计算确定性、种子影响、返回值范围 |
| `TestDetermineTextColumn` | 10 | GitHub 数据集使用 content 字段 |
| `TestLoadConfig` | 3 | 配置加载、stars_filter、buckets |
| `TestStreamingParquetWriter` | 5 | 流式写入、分文件、Schema 正确性 |
| `TestStreamFileRows` | 6 | 流式读取、索引筛选、边界情况 |
| `TestSelectTopKDocumentHashes` | 4 | Top-K 选择、边界情况、确定性 |

### 8.3 运行测试

```bash
pytest tests/test_prepare_tokenizer_data.py -v
pytest tests/test_prepare_tokenizer_data.py -xvs  # 详细输出
```

---

## 9. 使用指南

### 9.1 基本用法

```bash
# 使用默认配置
python scripts/prepare_tokenizer_data.py

# 指定 workers 数量
python scripts/prepare_tokenizer_data.py --workers 8

# 调整批次大小（内存敏感时减小）
python scripts/prepare_tokenizer_data.py --batch-size 10000

# 调整每文件行数
python scripts/prepare_tokenizer_data.py --max-rows 100000
```

### 9.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workers` | `min(32, cpu_count)` | 整体并行度基准 |
| `--io-workers` | `workers * 2` | IO 统计阶段并发数 |
| `--batch-size` | 50000 | 流式读取批次大小 |
| `--max-rows` | 500000 | 每个输出文件的最大行数 |

---

## 10. 注意事项与最佳实践

### 10.1 ⚠️ 重要注意事项

#### 1. 内存使用监控

虽然实现了流式处理，但仍需注意：
- **Top-K 选择阶段**: 如果 `target_count` 很大（如 1000万），堆会占用较多内存
- **Parquet 元数据**: 使用 `@lru_cache` 缓存，如果文件数超过 1024，缓存会失效重新加载

#### 2. 确定性保证

- **相同 seed + 相同输入数据 = 相同输出**
- 修改代码中的哈希算法会破坏确定性
- 文件顺序（`sorted()`）影响结果，确保文件系统返回稳定顺序

#### 3. 错误处理

- 单个文件读取失败不会中断整个流程
- 失败文件会被记录到日志，但不会被重试
- 建议定期检查日志中的警告信息

#### 4. 并发安全

- `StreamingParquetWriter.write()` 不是线程安全的
- 所有写入都在主线程串行执行
- IO 统计使用 `ThreadPoolExecutor`，但与写入阶段分离

### 10.2 性能调优建议

| 场景 | 建议 |
|------|------|
| **内存充足** | 增大 `--batch-size` 到 100k，提高吞吐量 |
| **内存受限** | 减小 `--batch-size` 到 10k，减小 `--max-rows` 到 100k |
| **高速 SSD** | 增大 `--io-workers` 到 32 |
| **网络存储** | 减小 `--io-workers` 到 4，避免连接数过多 |
| **小文件多** | 增大 `--max-rows` 减少输出文件数 |
| **大文件少** | 减小 `--max-rows` 提高并行写入粒度 |

### 10.3 常见问题排查

#### Q: 输出文件数量过多
**A**: 增大 `--max-rows` 参数（默认 50万，可增大到 100-200万）

#### Q: 内存使用过高
**A**: 
1. 减小 `--batch-size`（默认 5万，可减小到 1万）
2. 检查是否有超大桶（target_count >> 实际数据）
3. 确保 `gc.collect()` 被正常触发

#### Q: 采样结果不一致
**A**:
1. 检查 `random_seed` 是否相同
2. 检查源数据文件是否有变化
3. 检查文件排序是否稳定

#### Q: 处理速度太慢
**A**:
1. 增大 `--workers`（建议设为 CPU 核心数）
2. 确保使用本地存储而非网络存储
3. 检查磁盘 I/O 是否成为瓶颈

### 10.4 代码维护建议

1. **添加新数据源**: 在 `determine_text_column()` 中添加字段名映射
2. **修改采样算法**: 确保保持确定性，更新相关测试
3. **扩展输出字段**: 修改 `create_sample_doc()` 和 `StreamingParquetWriter._write_batch()`
4. **性能优化**: 使用 `line_profiler` 分析热点，避免过早优化

---

## 11. 依赖与兼容性

### 11.1 依赖包

```python
# 标准库
argparse, gc, hashlib, heapq, json, logging, os, sys
concurrent.futures, dataclasses, functools, pathlib, typing

# 第三方
pyarrow>=15.0.0      # Parquet 读写
pyarrow.parquet      # iter_batches 需要较新版本
pyyaml>=6.0          # 配置解析
tqdm>=4.65.0         # 进度条
```

### 11.2 Python 版本

- **最低要求**: Python 3.9+
- **推荐版本**: Python 3.13（项目主要开发版本）
- **类型注解**: 使用 `from __future__ import annotations` 支持延迟求值

---

## 12. 附录

### 12.1 相关文档

- [Tokenizer 训练设计](docs/tokenizer_training_design.md)
- [FineWeb 数据重组](docs/fineweb_edu_data_reorganization_design.md)
- [配置文件](../config/tokenizer_data.yaml)
- [单元测试](../tests/test_prepare_tokenizer_data.py)

### 12.2 Git 历史

```bash
# 查看完整修改历史
git log --all --oneline -- scripts/prepare_tokenizer_data.py

# 查看某个版本的详细修改
git show edd2a78 -- scripts/prepare_tokenizer_data.py
git show fa34b79 -- scripts/prepare_tokenizer_data.py
git show a57df8d -- scripts/prepare_tokenizer_data.py
git show 723636b -- scripts/prepare_tokenizer_data.py
```

### 12.3 代码统计

```bash
# 代码行数统计
wc -l scripts/prepare_tokenizer_data.py
cd scripts && python -c "import prepare_tokenizer_data; print(len(prepare_tokenizer_data.__doc__))"

# 函数数量统计
grep -c "^def " scripts/prepare_tokenizer_data.py
grep -c "^class " scripts/prepare_tokenizer_data.py
```

---

*文档生成时间: 2026-02-14*  
*基于 commit: 723636bdb5a2fd68a66ad1a30bf2a0685bbbf0cd*
