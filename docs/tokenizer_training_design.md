# Tokenizer 训练设计文档

> **目标**: 训练与 Qwen3-Next 兼容的 36K 词表 BPE Tokenizer  
> **训练样本**: 3M 多领域混合数据  
> **更新日期**: 2026-03-03

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
| **总词表大小** | 36005 | ID 0-36004（训练时使用 `--vocab-size 36005`） |
| **BPE tokens** | 36000 | 从数据学习（ID 0-35999） |
| **特殊 tokens** | 5 | 手动定义（ID 36000-36004） |
| **算法** | BPE | Byte Pair Encoding |
| **BPE 最小词频** | 2 | min_frequency |
| **基础架构** | Qwen3-Next | 继承完整 Qwen2Tokenizer 配置，仅替换词表和特殊token |

### 1.2 特殊 Token

| ID | Token | 用途 | 模型配置 |
| ---- | ------- | ---- | -------- |
| 36000 | `<\|endoftext\|>` | 文本结束 | `tokenizer.pad_token` |
| 36001 | `<\|im_start\|>` | 对话开始 | 特殊标记 |
| 36002 | `<\|im_end\|>` | 对话结束 | `tokenizer.eos_token` |
| 36003 | `<think>` | 推理开始 | 特殊标记 |
| 36004 | `</think>` | 推理结束 | 特殊标记 |

**特殊 Token 配置**:

| 配置项 | 内容 |
|--------|------|
| `extra_special_tokens` | `[<\|im_start\|>, <\|im_end\|>, <think>, </think>]` (4个) |
| `added_tokens` | `[<\|endoftext\|>, <\|im_start\|>, <\|im_end\|>, <think>, </think>]` (5个) |

**模型属性映射**:
- `bos_token` = `None`
- `eos_token` = `<|im_end|>` (36002)
- `pad_token` = `<|endoftext|>` (36000)
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

总计 **3M** 样本，多领域混合：

| 数据集 | 样本数 | 占比 | 明细 |
|--------|--------|------|------|
| **FineWeb-EN** | 720K | 24% | 4.0分: 288K (40%), 3.5分: 180K (25%), 3.0分: 144K (20%), 2.5分: 108K (15%) |
| **FineWeb-ZH** | 1.2M | 40% | 4.0分: 480K (40%), 3.5分: 300K (25%), 3.0分: 240K (20%), 2.5分: 180K (15%) |
| **GitHub Code** | 660K | 22% | ≥2 stars: 528K (80%), <2 stars: 132K (20%) |
| **Nemotron-CC-Math** | 420K | 14% | 4plus: 210K (50%), 4plus_MIND: 105K (25%), 3: 105K (25%) |

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
├── {dataset_name}-{bucket_name}-{counter:05d}-rank-{rank:05d}.parquet
├── sampling_info.json                        # 采样元信息
└── logs/                                     # Datatrove 日志目录
    ├── fineweb_edu_en/4.0/
    ├── fineweb_edu_en/3.5/
    └── ...
```

**输出文件字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文档唯一标识（格式：`{dataset_name}/{bucket_name}/{filename}.parquet#{row_idx}`） |
| `text` | string | 文本内容 |
| `source_dataset` | string | 数据集名称（如 `fineweb_edu_en`） |
| `source_bucket` | string | 桶名称（如 `4.0`、`above-2-stars`） |
| `language` | string | 编程语言（仅 GitHub Code 数据集有该字段） |
---

## 3. 训练流程

| 阶段 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1. 准备模板 | `prepare_tokenizer_template.py` | Qwen3-Next | `output/qwen3_next_tokenizer/` |
| 2. 数据采样 | `prepare_tokenizer_data.py` | 源数据目录 | `data/datasets/nanomind_tokenizer/` |
| 3. Tokenizer训练 | `train_tokenizer.py` | 采样后数据 | `output/tokenizer_36k/` |

### 3.1 准备模板

**目标**: 从 Hugging Face 下载 Qwen3-Next tokenizer 并精简特殊 token。

```bash
python scripts/prepare_tokenizer_template.py
```

**输出两个目录**:

| 目录 | 内容 |
|------|------|
| `output/qwen3_next_tokenizer_origin/` | 原始 Qwen3-Next tokenizer（完整26个特殊token） |
| `output/qwen3_next_tokenizer/` | 精简版模板（仅5个基础特殊token） |

**精简处理**:
- `tokenizer_config.json`: `extra_special_tokens` 仅保留 `[<|im_start|>, <|im_end|>, <think>, </think>]`
- `tokenizer.json`: `added_tokens` 仅保留5个基础 token
- `chat_template.jinja`: 原样复制

### 3.2 数据采样

**目标**: 从各数据源按配置比例采样 3M 样本。

```bash
# 使用默认配置
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
| `--workers, -w` | `min(16, CPU_COUNT)` | 并行进程数（建议不超过16） |
| `--tasks, -t` | `-1` | 任务分片数（-1=自动，等于workers） |
| `--max-rows` | `500000` | 每个输出文件的最大行数 |
| `--buffer-size` | `50000` | 写入缓冲区大小 |

**采样配置** (`config/tokenizer_data.yaml`):

```yaml
datasets:
  fineweb_en:
    name: "fineweb_edu_en"
    source: "data/datasets/fineweb/en"
    samples: 720000
    buckets:
      4.0: {count: 288000}   # 40%
      3.5: {count: 180000}   # 25%
      3.0: {count: 144000}   # 20%
      2.5: {count: 108000}   # 15%
  fineweb_zh:
    name: "fineweb_edu_zh"
    source: "data/datasets/fineweb/zh"
    samples: 1200000
    buckets:
      4.0: {count: 480000}   # 40%
      3.5: {count: 300000}   # 25%
      3.0: {count: 240000}   # 20%
      2.5: {count: 180000}   # 15%
  github_code:
    name: "github_code"
    source: "data/datasets/nick007x/github-code-2025"
    samples: 660000
    stars_filter:
      above-2-stars: {count: 528000}  # 80%
      below-2-stars: {count: 132000}  # 20%
  nemotron_math:
    name: "nemotron_cc_math"
    source: "data/datasets/nvidia/Nemotron-CC-Math-v1"
    samples: 420000
    buckets:
      4plus: {count: 210000}      # 50%
      4plus_MIND: {count: 105000} # 25%
      3: {count: 105000}          # 25%

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

**两遍处理架构**:

| 阶段 | 操作 | 内存占用 |
|------|------|----------|
| **第一遍** | 预计算采样索引（只读元数据和计算哈希） | O(target_count × 16 bytes) |
| **第二遍** | 流式读取选中文档并写入 Parquet | 流式，不累积 |

**优化策略**:

| 优化项 | 策略 |
|--------|------|
| **内存分配器** | 启用 jemalloc 解决 Linux ptmalloc2 内存泄漏 |
| **分块读取** | 单文件流式读取，避免全量加载 |
| **流式写入** | 边采样边写入 Parquet，不累积 |
| **并行处理** | 基于 Datatrove `LocalPipelineExecutor` 多进程 |
| **定期 GC** | 每处理10批次显式调用 `gc.collect()` |

**推荐配置**:

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `workers` | `min(16, CPU_COUNT)` | 8-16 | 不超过16以减少进程切换 |
| `tasks` | `-1`（自动=workers） | 等于 workers | 1:1 匹配效率最优 |
| `max-rows` | `500000` | 100000-1000000 | 每文件行数 |
| `buffer-size` | `50000` | 10000-100000 | 匹配 400MB/s 磁盘 |

**Pipeline 组件**:

```python
pipeline = [
    ParquetReader(...),      # 流式读取 Parquet
    IndexFilter(indices),    # 过滤未选中的文档
    LanguageTagger(...),     # 标记编程语言（仅 GitHub Code）
    SourceTagger(...),       # 添加来源标签
    TokenizerDataWriter(...),# 写入输出文件
]
```

**GitHub Code 语言过滤**:

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

### 3.3 Tokenizer 训练

**训练方法**: 使用 `train_new_from_iterator` 自动继承模板配置。

```bash
python scripts/train_tokenizer.py \
    --data-dir data/datasets/nanomind_tokenizer \
    --template-dir output/qwen3_next_tokenizer \
    --output-dir output/tokenizer_36k \
    --vocab-size 36005 \
    --validate
```

**命令行参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir, -d` | `data/datasets/nanomind_tokenizer` | 训练数据目录 |
| `--template-dir, -t` | `output/qwen3_next_tokenizer` | 模板 tokenizer 目录 |
| `--output-dir, -o` | `output/tokenizer_36k` | 输出目录 |
| `--vocab-size, -v` | `36005` | 目标词表大小（36000 BPE + 5 特殊token） |
| `--batch-size, -b` | `10000` | Parquet 读取批次大小 |
| `--validate` | `True` | 是否执行验证 |

**训练步骤**:

1. **加载模板 Tokenizer**
   ```python
   template = AutoTokenizer.from_pretrained(
       "output/qwen3_next_tokenizer",
       trust_remote_code=True
   )
   ```

2. **使用 `train_new_from_iterator` 训练**
   ```python
   new_tokenizer = template.train_new_from_iterator(
       text_iterator(),
       vocab_size=36005,  # 36000 BPE + 5个特殊token
   )
   ```

3. **后处理**: 自动调整 vocab ID 映射，特殊 token 放到词表末尾

**关键说明**:

| 特性 | `train_new_from_iterator` 优势 |
|------|-------------------------------|
| 配置继承 | 自动继承模板的 normalizer、pre_tokenizer、decoder 等 |
| 特殊token | 自动处理 ID 映射 |
| 错误风险 | 低（自动处理配置项） |

#### 训练内存优化

| 阶段 | 内存瓶颈 | 优化策略 |
|------|----------|----------|
| **数据迭代** | 3M 文本加载 | 生成器流式迭代，batch_size=10000 |
| **BPE 训练** | 词频统计 + 合并队列 | 内部优化 |
| **最终保存** | 完整词表序列化 | 直接写入磁盘 |

**训练时间估算**:
- 3M 样本 × 平均 500 tokens ≈ 1.5B tokens
- 36K BPE 训练：预计 4-8 小时（32 核 CPU）

---

## 4. 验证与输出

### 4.1 自动验证

训练时添加 `--validate` 参数，检查：

**基础验证**:
- 词表大小 = 36005（36000 BPE + 5 特殊token）
- 特殊 token ID 正确（36000-36004）
- 编解码一致性

**特殊 Token 配置验证**:
- `added_tokens` 包含 5 个 token
- `extra_special_tokens` 包含 4 个 token（不含 `<|endoftext|>`）

### 4.2 手动验证

```python
from transformers import AutoTokenizer

# 加载训练的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("output/tokenizer_36k")

# 验证词表大小
assert len(tokenizer) == 36005

# 验证属性映射
assert tokenizer.eos_token == "<|im_end|>"
assert tokenizer.pad_token == "<|endoftext|>"

# 编解码测试
text = "<|im_start|>assistant\n<think>推理...</think>答案<|im_end|>"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
assert tokenizer.encode(decoded) == encoded
```

### 4.3 输出文件

**输出目录结构**：

```
output/tokenizer_36k/
├── tokenizer.json              # 词表与合并规则
├── tokenizer_config.json       # Tokenizer配置
└── chat_template.jinja         # 对话模板（从模板复制）
```

**文件说明**:

| 文件 | 来源 | 说明 |
|------|------|------|
| `tokenizer.json` | 训练生成 | 36K BPE词表 + 5个特殊token + 继承的配置 |
| `tokenizer_config.json` | 训练生成 | `extra_special_tokens`（4个）、模型属性映射 |
| `chat_template.jinja` | 模板复制 | 对话格式模板 |

---

## 5. 实现清单

| 文件 | 说明 | 依赖 |
|------|------|------|
| `config/tokenizer_data.yaml` | 数据采样配置 | - |
| `scripts/prepare_tokenizer_template.py` | 下载并精简 Qwen3-Next 模板 | transformers |
| `scripts/prepare_tokenizer_data.py` | 多数据源采样（两遍处理） | datatrove, pyarrow |
| `scripts/train_tokenizer.py` | BPE 训练主脚本 | tokenizers, transformers |

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
| `LANGUAGE_EXTENSIONS` | 扩展名到语言名称的映射（38个扩展名） |
| `ALLOWED_LANGUAGES` | 允许的扩展名集合 |
| `SamplingConfig` | 数据集采样配置数据类 |
| `TokenizerDataConfig` | 全局配置数据类 |
| `precompute_sampling_indices()` | 第一遍：预计算采样索引 |
| `IndexFilter` | Pipeline 步骤：过滤未选中文档 |
| `LanguageTagger` | Pipeline 步骤：标记编程语言 |
| `SourceTagger` | Pipeline 步骤：添加来源标签 |
| `TokenizerDataWriter` | Pipeline 步骤：流式写入 Parquet |
| `process_bucket_streaming()` | 处理单个桶（采样模式） |

**train_tokenizer.py 核心功能**:

| 函数 | 功能 |
|------|------|
| `load_template_tokenizer()` | 加载模板 tokenizer |
| `create_text_iterator()` | 创建文本迭代器 |
| `train_tokenizer_with_iterator()` | 使用 `train_new_from_iterator` 训练 |
| `postprocess_tokenizer_files()` | 后处理：调整 vocab ID 映射 |
| `validate_tokenizer()` | 执行完整验证 |

---

## 6. 附录

### 6.1 模板中的特殊 Token（Qwen3-Next 原始）

| 类型 | Token | ID | 说明 |
|------|-------|-----|------|
| 基础 | `<\|endoftext\|>` | 151643 | 文本结束 / pad token |
| 对话 | `<\|im_start\|>` | 151644 | 对话开始 |
| 对话 | `<\|im_end\|>` | 151645 | 对话结束 / eos token |
| 视觉 | `<\|vision_start\|>` ~ `<\|video_pad\|>` | 151652-151656 | 视觉输入相关 |
| FIM | `<\|fim_prefix\|>` ~ `<\|fim_suffix\|>` | 151659-151661 | 代码填充 |
| 工具 | `<tool_call>`, `</tool_call>` | 151657-151658 | 工具调用 |
| 推理 | `<think>`, `</think>` | 151667-151668 | 思考过程 |

### 6.2 训练后的特殊 Token

| ID | Token | 用途 | 来源 |
|----|-------|------|------|
| 36000 | `<\|endoftext\|>` | 文本结束 / pad | 模板继承 |
| 36001 | `<\|im_start\|>` | 对话开始 | 模板继承 |
| 36002 | `<\|im_end\|>` | 对话结束 / eos | 模板继承 |
| 36003 | `<think>` | 推理开始 | 模板继承 |
| 36004 | `</think>` | 推理结束 | 模板继承 |

**已移除**: 所有视觉/多模态、FIM、工具调用相关 token。

**如需扩展**: 添加视觉/多模态支持时，可从 `output/qwen3_next_tokenizer_origin/` 恢复相应 token 并增加 `vocab_size`。

### 6.3 相关文档

- [数据重组设计](fineweb_edu_data_reorganization_design.md)
- [FineWeb-Edu 论文](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Datatrove 文档](https://github.com/huggingface/datatrove)

---

*最后更新: 2026-03-03*
