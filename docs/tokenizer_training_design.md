# Tokenizer 训练设计文档

> **目标**: 训练与 Qwen3-Next 兼容的 32K 词表 BPE Tokenizer  
> **训练样本**: 800K 多领域混合数据  
> **更新日期**: 2026-02-22

---

## 目录

1. [核心配置](#1-核心配置)
   - [1.1 词表结构](#11-词表结构)
   - [1.2 特殊 Token](#12-特殊-token)
2. [训练数据](#2-训练数据)
   - [2.1 数据配比](#21-数据配比)
   - [2.2 数据目录](#22-数据目录)
3. [训练流程](#3-训练流程)
   - [3.1 准备模板](#31-准备模板)
   - [3.2 数据采样](#32-数据采样)
   - [3.3 Tokenizer 训练](#33-tokenizer-训练)
4. [验证与输出](#4-验证与输出)
5. [实现清单](#5-实现清单)
6. [附录](#6-附录)

---

## 1. 核心配置

### 1.1 词表结构

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **总词表大小** | 32005 | ID 0-32004（训练时使用 `--vocab-size 32005`） |
| **BPE tokens** | 32000 | 从数据学习（ID 0-31999） |
| **特殊 tokens** | 5 | 手动定义（ID 32000-32004） |
| **算法** | BPE | Byte Pair Encoding |
| **BPE 最小词频** | 2 | min_frequency |
| **基础架构** | Qwen3-Next | 复制 pretokenizer/normalizer/decoder，不继承词表 |

### 1.2 特殊 Token

| ID | Token | 用途 | 模型配置 |
| ---- | ------- | ---- | -------- |
| 32000 | `<\|endoftext\|>` | 文本结束 | `tokenizer.pad_token` |
| 32001 | `<\|im_start\|>` | 对话开始 | 特殊标记 |
| 32002 | `<\|im_end\|>` | 对话结束 | `tokenizer.eos_token` |
| 32003 | `<\|think\|>` | 推理开始 | 特殊标记 |
| 32004 | `<\|/think\|>` | 推理结束 | 特殊标记 |

**模型配置**:
- `bos_token` = `None`
- `eos_token` = `<|im_end|>` (32002)
- `pad_token` = `<|endoftext|>` (32000)
- `unk_token` = `None`

**对话格式示例**:
```
<|im_start|>user
问题<|im_end|>
<|im_start|>assistant
<think>推理过程...</think>
答案<|im_end|>
```

---

## 2. 训练数据

### 2.1 数据配比

总计 **800K** 样本，多领域混合：

| 数据集 | 样本数 | 占比 | 明细 |
|--------|--------|------|------|
| **FineWeb-EN** | 192K | 24% | 4.0分: 96K (50%), 3.5分: 48K (25%), 3.0分: 28.8K (15%), 2.5分: 19.2K (10%) |
| **FineWeb-ZH** | 224K | 28% | 4.0分: 112K (50%), 3.5分: 56K (25%), 3.0分: 33.6K (15%), 2.5分: 22.4K (10%) |
| **GitHub Code** | 256K | 32% | ≥2 stars: 217.6K (85%), <2 stars: 38.4K (15%) |
| **Nemotron-CC-Math** | 128K | 16% | 4plus: 76.8K (60%), 4plus_MIND: 32K (25%), 3: 19.2K (15%) |

> 数据采样配置详见 [config/tokenizer_data.yaml](#tokenizer_data_yaml)。

### 2.2 数据目录

**源数据**（已预处理）：
```
data/datasets/
├── fineweb/                      # FineWeb（已分桶）
│   ├── en/{2.5,3.0,3.5,4.0}/
│   └── zh/{2.5,3.0,3.5,4.0}/
├── nick007x/github-code-2025/    # GitHub Code
│   ├── above-2-stars/
│   └── below-2-stars/
└── nvidia/Nemotron-CC-Math-v1/   # 数学数据
    ├── 3/, 4plus/, 4plus_MIND/
```

> FineWeb 分桶预处理详见 [数据重组设计文档](fineweb_edu_data_reorganization_design.md)

**训练输入**（采样脚本输出）：
```
data/datasets/nanomind_tokenizer/
├── train-{idx:05d}-rank-{rank:05d}.parquet  # 采样数据（zstd 压缩）
├── sampling_info.json                        # 采样元信息
└── logs/                                     # Datatrove 日志目录
    ├── fineweb_edu_en_4.0/
    ├── fineweb_edu_en_3.5/
    └── ...
```

**输出文件字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文档唯一标识（格式：`{完整路径}#{index}`） |
| `text` | string | 文本内容 |
| `source_dataset` | string | 数据集名称（如 `fineweb_edu_en`） |
| `source_bucket` | string | 桶名称（如 `4.0`、`above-2-stars`） |
| `language` | string | 编程语言（仅 GitHub Code 数据集有该字段） |
---

## 3. 训练流程

| 阶段 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1. 准备模板 | `prepare_template.py` | Qwen3-Next | `output/qwen3_next_tokenizer/` |
| 2. 数据采样 | `prepare_tokenizer_data.py` | 源数据目录 | `data/datasets/nanomind_tokenizer/` |
| 3. Tokenizer训练 | `train_tokenizer.py` | 采样后数据 | `output/tokenizer_32k/` |

### 3.1 准备模板（一次性）

```python
from transformers import AutoTokenizer

base = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    trust_remote_code=True
)
base.save_pretrained("output/qwen3_next_tokenizer")
```

### 3.2 数据采样

**目标**: 从各数据源按配置比例采样 800K 样本。

```bash
# 使用默认配置（针对 32 核/250GB/400MB/s 优化）
python scripts/prepare_tokenizer_data.py

# 自定义参数
python scripts/prepare_tokenizer_data.py \
    --workers 16 \
    --tasks -1 \
    --max-rows 500000 \
    --buffer-size 50000
```

**命令行参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workers, -w` | `min(16, CPU_COUNT)` | 并行进程数（建议不超过 16 以减少进程切换开销） |
| `--tasks, -t` | `-1` | 任务分片数（-1 表示自动计算，等于 workers） |
| `--max-rows` | `500000` | 每个输出文件的最大行数 |
| `--buffer-size` | `50000` | 写入缓冲区大小（匹配 400MB/s 磁盘） |

**采样配置** (`config/tokenizer_data.yaml`):

```yaml
datasets:
  fineweb_en:
    name: "fineweb_edu_en"
    source: "data/datasets/fineweb/en"
    samples: 192000
    buckets:
      4.0: {count: 96000}  # 50%
      3.5: {count: 48000}   # 25%
      3.0: {count: 28800}   # 15%
      2.5: {count: 19200}   # 10%
  fineweb_zh:
    name: "fineweb_edu_zh"
    source: "data/datasets/fineweb/zh"
    samples: 224000
    buckets:
      4.0: {count: 112000}  # 50%
      3.5: {count: 56000}   # 25%
      3.0: {count: 33600}   # 15%
      2.5: {count: 22400}   # 10%
  github_code:
    name: "github_code"
    source: "data/datasets/nick007x/github-code-2025"
    samples: 256000
    stars_filter:
      above-2-stars: {count: 217600}  # 85%
      below-2-stars: {count: 38400}   # 15%
  nemotron_math:
    name: "nemotron_cc_math"
    source: "data/datasets/nvidia/Nemotron-CC-Math-v1"
    samples: 128000
    buckets:
      4plus: {count: 76800}   # 60%
      4plus_MIND: {count: 32000}   # 25%
      3: {count: 19200}   # 15%

random_seed: 42
output_format: "parquet"
output_dir: "data/datasets/nanomind_tokenizer"
```

**配置字段说明**:

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 数据集名称，用于标记输出数据来源 |
| `source` | 是 | 源数据目录路径 |
| `samples` | 是 | 目标采样数量 |
| `buckets` | 二选一 | 质量分桶配置（FineWeb、Nemotron） |
| `stars_filter` | 二选一 | Stars 过滤配置（GitHub Code） |

#### 内存与性能优化

**数据规模**:
- FineWeb 单个桶可能包含 **~100 个文件，每文件 4GB+**（总计 400GB/桶）
- 处理 40M 样本需考虑内存峰值和 I/O 吞吐量

**核心优化策略（两遍处理）**:

脚本采用 **两遍处理架构**，将采样计算与数据读取分离：

| 阶段 | 操作 | 内存占用 |
|------|------|----------|
| **第一遍** | 预计算采样索引（只读元数据和计算哈希） | O(target_count × 16 bytes) |
| **第二遍** | 流式读取选中文档并写入 Parquet | 流式，不累积 |

**采样算法**:
- 使用确定性哈希（MD5 前 8 字节）+ 最大堆采样
- 保证相同 seed 下采样结果可重复
- 目标数 ≥ 总数 90% 时自动切换为全量处理模式

**实现要点**:

| 优化项 | 策略 |
|--------|------|
| **内存分配器** | 启用 jemalloc 解决 Linux ptmalloc2 内存泄漏（自动检测 `/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`） |
| **分块读取** | 单文件流式读取，避免全量加载 |
| **流式写入** | 边采样边写入 Parquet，不累积全部数据 |
| **并行处理** | 基于 Datatrove `LocalPipelineExecutor` 多进程并行 |
| **定期 GC** | 每处理 10 批次显式调用 `gc.collect()` 防止内存碎片累积 |

**推荐配置**:

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `workers` | `min(16, CPU_COUNT)` | 8-16 | 建议不超过 16 以减少进程切换开销 |
| `tasks` | `-1`（自动=workers） | 等于 workers | 1:1 匹配 workers 效率最优 |
| `max-rows` | `500000` | 100000-1000000 | 每文件行数，影响文件数量 |
| `buffer-size` | `50000` | 10000-100000 | 写入缓冲区，匹配 400MB/s 磁盘 |

**Pipeline 组件**:

```python
pipeline = [
    ParquetReader(...),      # 流式读取 Parquet 文件
    IndexFilter(indices),    # 过滤未选中的文档（采样模式下）
    LanguageTagger(...),     # 标记编程语言（仅 GitHub Code）
    SourceTagger(...),       # 添加来源标签
    TokenizerDataWriter(...),# 写入输出文件
]
```

**GitHub Code 语言过滤**:

针对 GitHub Code 数据集，实现了基于文件扩展名的语言过滤和标记：

| 语言 | 扩展名 |
|------|--------|
| C | `.c`, `.h` |
| C++ | `.cpp`, `.hpp`, `.cc`, `.cxx`, `.hxx` |
| Python | `.py`, `.pyw`, `.pyi` |
| Rust | `.rs` |
| HTML | `.html`, `.htm`, `.xhtml` |
| CSS | `.css`, `.scss`, `.sass`, `.less` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` |
| Markdown | `.md`, `.markdown`, `.mkd` |
| JSON | `.json`, `.jsonc`, `.jsonl` |
| XML | `.xml`, `.xsl`, `.xslt`, `.svg`, `.wsdl` |
| TOML | `.toml` |

- 只保留上述扩展名的文件，其他文件被过滤
- 通过 `LANGUAGE_EXTENSIONS` 常量定义扩展名到语言名称的映射
- 通过 `ALLOWED_LANGUAGES` 集合（从 `LANGUAGE_EXTENSIONS` 派生）控制允许的扩展名

### 3.3 Tokenizer 训练

```bash
python scripts/train_tokenizer.py \
    --data-dir data/datasets/nanomind_tokenizer \
    --template-dir output/qwen3_next_tokenizer \
    --output-dir output/tokenizer_32k \
    --vocab-size 32005 \
    --validate
```

**训练步骤**:
1. 从模板加载 pretokenizer/normalizer/decoder 配置
2. 空白初始化，在采样数据上学习 32000 个 BPE 合并规则
3. 添加 5 个特殊 token（ID 32000-32004）
4. 配置 eos/pad/bos/unk 映射

#### 训练内存优化

| 阶段 | 内存瓶颈 | 优化策略 |
|------|----------|----------|
| **数据迭代** | 40M 文本加载 | 使用生成器流式迭代，batch_size=10000 |
| **BPE 训练** | 词频统计 + 合并队列 | 使用 `tokenizers` 库的增量训练，控制并发 |
| **最终保存** | 完整词表序列化 | 直接写入磁盘，不驻留内存 |

**训练时间估算**（参考值）:
- 40M 样本 × 平均 500 tokens ≈ 20B tokens
- 32K BPE 训练：预计 4-8 小时（32 核 CPU）

---

## 4. 验证与输出

### 4.1 自动验证

训练时添加 `--validate` 参数，检查：
- 词表大小 = 32005
- 特殊 token ID 正确（32000-32004）
- 编解码一致性

### 4.2 手动验证

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("output/tokenizer_32k")

# 验证词表大小
assert tokenizer.vocab_size == 32005

# 验证特殊 token
print(tokenizer.special_tokens_map)

# 编解码测试
text = "<|im_start|>assistant\n<think>推理</think>答案<|im_end|>"
encoded = tokenizer(text, return_tensors="pt")
decoded = tokenizer.decode(encoded["input_ids"][0])
assert text == decoded
```

### 4.3 输出文件

```
output/tokenizer_32k/
├── tokenizer.json              # 词表与合并规则
├── tokenizer_config.json       # Tokenizer配置
├── special_tokens_map.json     # 特殊token映射
└── vocab.txt                   # 可读词汇表
```

---

## 5. 实现清单

| 文件 | 说明 | 依赖 |
|------|------|------|
| `config/tokenizer_data.yaml` | 数据采样配置（数据集名称、路径、采样数量） | - |
| `scripts/prepare_template.py` | 复制 Qwen3-Next 架构 | transformers |
| `scripts/prepare_tokenizer_data.py` | 多数据源采样（两遍处理、流式写入） | datatrove, pyarrow |
| `scripts/train_tokenizer.py` | BPE 训练主脚本 | tokenizers |

**依赖版本要求**:
```
tokenizers>=0.22.0
transformers>=4.40.0
datatrove>=0.8.0
pyarrow>=15.0.0
```

**prepare_tokenizer_data.py 核心模块**:

| 模块/类/常量 | 功能 |
|--------------|------|
| `LANGUAGE_EXTENSIONS` | 扩展名到语言名称的映射字典（38个扩展名） |
| `ALLOWED_LANGUAGES` | 允许的扩展名集合（从 `LANGUAGE_EXTENSIONS` 派生） |
| `SamplingConfig` | 数据集采样配置数据类 |
| `TokenizerDataConfig` | 全局配置数据类 |
| `precompute_sampling_indices()` | 第一遍：预计算采样索引 |
| `IndexFilter` | Pipeline 步骤：过滤未选中文档 |
| `LanguageTagger` | Pipeline 步骤：标记编程语言（仅 GitHub Code） |
| `SourceTagger` | Pipeline 步骤：添加来源标签 |
| `TokenizerDataWriter` | Pipeline 步骤：流式写入 Parquet |
| `process_bucket_streaming()` | 处理单个桶（自动选择全量/采样模式） |
---

## 6. 附录

### 6.1 扩展预留

如需视觉/多模态支持，可添加特殊 token（如 `<|vision_start|>`、`<|image_pad|>` 等），并相应增加 `vocab_size`。

### 6.2 相关文档

- [数据重组设计](fineweb_edu_data_reorganization_design.md)
- [FineWeb-Edu 论文](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Datatrove 文档](https://github.com/huggingface/datatrove)

---

*最后更新: 2026-02-22*
