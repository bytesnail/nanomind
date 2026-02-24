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
| **基础架构** | Qwen3-Next | 继承完整 Qwen2Tokenizer 配置，仅替换词表和特殊token |

### 1.2 特殊 Token

| ID | Token | 用途 | 模型配置 |
| ---- | ------- | ---- | -------- |
| 32000 | `<\|endoftext\|>` | 文本结束 | `tokenizer.pad_token` |
| 32001 | `<\|im_start\|>` | 对话开始 | 特殊标记 |
| 32002 | `<\|im_end\|>` | 对话结束 | `tokenizer.eos_token` |
| 32003 | `<think>` | 推理开始 | 特殊标记 |
| 32004 | `</think>` | 推理结束 | 特殊标记 |

**特殊 Token 配置**:

| 配置项 | 内容 |
|--------|------|
| `extra_special_tokens` | `[<\|im_start\|>, <\|im_end\|>, <think>, </think>]` (4个) |
| `added_tokens` | `[<\|endoftext\|>, <\|im_start\|>, <\|im_end\|>, <think>, </think>]` (5个) |

**说明**:
- `extra_special_tokens` 仅保留对话和推理相关的4个特殊token
- `added_tokens` 包含5个token，比 `extra_special_tokens` 多一个 `<|endoftext|>` (ID 32000)
- 已移除模板中原有的视觉/多模态相关token

**模型属性映射**:
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

1. **加载模板 Tokenizer**
   ```python
   from transformers import AutoTokenizer
   
   # 加载完整的 Qwen2Tokenizer 作为模板
   template = AutoTokenizer.from_pretrained(
       "output/qwen3_next_tokenizer",
       trust_remote_code=True
   )
   ```

2. **提取模板配置（除词表外）**
   - 从 `tokenizer.json` 提取：
     - `normalizer`: NFC 标准化规则
     - `pre_tokenizer`: ByteLevel + Regex Split 组合
     - `post_processor`: ByteLevel 后处理
     - `decoder`: ByteLevel 解码器
   - 从 `tokenizer_config.json` 提取：
     - `bos_token`, `eos_token`, `pad_token`, `unk_token` 的映射逻辑
     - `model_max_length`, `clean_up_tokenization_spaces` 等元配置

3. **在采样数据上训练新词表**
   - 使用 `tokenizers` 库创建空白 BPE trainer
   - 在 800K 采样数据上学习 32000 个 BPE 合并规则
   - 保持与模板相同的 pretokenizer 行为，确保兼容性

4. **构建新的 Tokenizer 配置**
   - 将训练得到的 32K 词表与模板的 normalizer/pretokenizer/decoder 组合
   - 配置 `added_tokens`: 5 个特殊 token（ID 32000-32004）
     - `<|endoftext|>` (32000), `<|im_start|>` (32001), `<|im_end|>` (32002)
     - `<think>` (32003), `</think>` (32004)
   - 配置 `extra_special_tokens`: 4 个 token（不含 `<|endoftext|>`）
     - `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`
   - 保持与模板一致的 `eos_token`/`pad_token` 等模型属性映射
   - 将训练得到的 32K 词表与模板的 normalizer/pretokenizer/decoder 组合
   - 添加 5 个特殊 token（ID 32000-32004）到 `added_tokens`
   - 配置 `extra_special_tokens` 仅包含推理相关 token
   - 保持与模板一致的 `eos_token`/`pad_token` 等模型属性映射

5. **保存并验证**
   - 保存为与模板完全兼容的目录结构
   - 验证所有配置项与模板一致（除词表和特殊token外）

**关键说明**:

- 不同于传统的"空白初始化"，本方案**继承完整的 Qwen2Tokenizer 架构**，仅替换：
  - 不同于传统的"空白初始化"，本方案**继承完整的 Qwen2Tokenizer 架构**，仅替换：
  - `model.vocab`: 从数据新训练的 32K BPE 词表
  - `added_tokens`: 5 个特殊 token（ID 32000-32004）
  - `extra_special_tokens`: 4 个 token（`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`），移除视觉/多模态相关 token
  - `added_tokens`: 精简后的特殊 token 集合
  - `extra_special_tokens`: 移除视觉/多模态相关 token，保留推理相关 token

- **模板继承的组件**（保持不变）：
  | 组件 | 来源 | 说明 |
  |------|------|------|
  | `normalizer` | 模板 | NFC Unicode 标准化 |
  | `pre_tokenizer` | 模板 | ByteLevel + Regex Split |
  | `post_processor` | 模板 | ByteLevel 后处理 |
  | `decoder` | 模板 | ByteLevel 解码器 |
  | 模型属性映射 | 模板 | `eos_token`→`<|im_end|>`, `pad_token`→`<|endoftext|>` |
  | `model_max_length` | 模板 | 1010000 |
  | `clean_up_tokenization_spaces` | 模板 | false |

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

**基础验证**:
- 词表大小 = 32005（32000 BPE + 5 特殊token）
- 特殊 token ID 正确（32000-32004）
- 编解码一致性

**特殊 Token 配置验证**:
- `added_tokens` 包含 5 个 token：`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`
- `extra_special_tokens` 包含 4 个 token：`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`（不含 `<|endoftext|>`）
- `extra_special_tokens` 不包含视觉/多模态相关 token

**模板一致性验证**:
- `normalizer` 配置与模板一致（NFC）
- `pre_tokenizer` 配置与模板一致（ByteLevel + Regex Split）
- `decoder` 配置与模板一致（ByteLevel）
- 模型属性映射与模板一致：
  - `eos_token` = `<|im_end|>`
  - `pad_token` = `<|endoftext|>`
  - `bos_token` = `None`
  - `unk_token` = `None`
- 词表大小 = 32005（32000 BPE + 5 特殊token）
- 特殊 token ID 正确（32000-32004）
- 编解码一致性

**模板一致性验证**:
- `normalizer` 配置与模板一致（NFC）
- `pre_tokenizer` 配置与模板一致（ByteLevel + Regex Split）
- `decoder` 配置与模板一致（ByteLevel）
- 模型属性映射与模板一致：
  - `eos_token` = `<|im_end|>`
  - `pad_token` = `<|endoftext|>`
  - `bos_token` = `None`
  - `unk_token` = `None`
- `extra_special_tokens` 仅包含推理相关token（无视觉/多模态token）

### 4.2 手动验证

```python
from transformers import AutoTokenizer
import json

# 加载训练的 tokenizer 和模板
tokenizer = AutoTokenizer.from_pretrained("output/tokenizer_32k")
template = AutoTokenizer.from_pretrained(
    "output/qwen3_next_tokenizer",
    trust_remote_code=True
)

# 1. 验证词表大小
assert tokenizer.vocab_size == 32005, "词表大小应为 32005"

# 2. 验证特殊 token
print("特殊 token 映射:")
print(tokenizer.special_tokens_map)

# 验证关键属性与模板一致
assert tokenizer.eos_token == template.eos_token, "eos_token 应与模板一致"
assert tokenizer.pad_token == template.pad_token, "pad_token 应与模板一致"
assert tokenizer.bos_token == template.bos_token, "bos_token 应与模板一致"
assert tokenizer.unk_token == template.unk_token, "unk_token 应与模板一致"

# 3. 验证 added_tokens 和 extra_special_tokens
with open("output/tokenizer_32k/tokenizer.json") as f:
    tokenizer_json = json.load(f)
with open("output/tokenizer_32k/tokenizer_config.json") as f:
    config = json.load(f)

# 验证 added_tokens (5个)
expected_added = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
actual_added = [t["content"] for t in tokenizer_json.get("added_tokens", [])]
assert set(actual_added) == set(expected_added), f"added_tokens 应为 {expected_added}, 实际是 {actual_added}"
print(f"✓ added_tokens: {actual_added}")

# 验证 extra_special_tokens (4个，不含<|endoftext|>)
expected_special = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
actual_special = config.get("extra_special_tokens", [])
assert set(actual_special) == set(expected_special), f"extra_special_tokens 应为 {expected_special}, 实际是 {actual_special}"
print(f"✓ extra_special_tokens: {actual_special}")

# 检查没有视觉/多模态相关的token
vision_tokens = ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", 
                 "<|image_pad|>", "<|video_pad|>"]
for vt in vision_tokens:
    assert vt not in actual_special, f"extra_special_tokens 不应包含视觉token: {vt}"
with open("output/tokenizer_32k/tokenizer_config.json") as f:
    config = json.load(f)
    
expected_special = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
actual_special = config.get("extra_special_tokens", [])
print(f"extra_special_tokens: {actual_special}")

# 检查没有视觉/多模态相关的token
vision_tokens = ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", 
                 "<|image_pad|>", "<|video_pad|>"]
for vt in vision_tokens:
    assert vt not in actual_special, f"不应包含视觉token: {vt}"

# 4. 编解码测试
text = "<|im_start|>assistant\n<think>推理过程...</think>答案<|im_end|>"
encoded = tokenizer(text, return_tensors="pt")
decoded = tokenizer.decode(encoded["input_ids"][0])
assert text == decoded, "编解码应一致"

# 5. 验证 normalizer/pretokenizer 行为与模板一致
test_text = "Café\nHello  World"
template_ids = template.encode(test_text)
trained_ids = tokenizer.encode(test_text)
# 注意：词表不同，但 pretokenizer 行为应一致（测试前几个token的切分）

print("✓ 所有验证通过！")
```


### 4.3 输出文件

**输出目录结构**（与 `output/qwen3_next_tokenizer/` 模板完全一致）：

```
output/tokenizer_32k/
├── tokenizer.json              # 词表与合并规则（包含新的32K BPE词表）
├── tokenizer_config.json       # Tokenizer配置（特殊token映射等）
└── chat_template.jinja         # 对话模板（从模板复制）
```

**文件说明**:

| 文件 | 来源 | 说明 |
|------|------|------|
| `tokenizer.json` | 训练生成 + 模板配置 | 包含：①新训练的32K BPE词表 ②从模板继承的normalizer/pretokenizer/decoder/post_processor ③`added_tokens`：5个特殊token（`<\|endoftext\|>`, `<\|im_start\|>`, `<\|im_end\|>`, `<think>`, `</think>`） |
| `tokenizer_config.json` | 训练生成 | 包含：`extra_special_tokens`（4个：不含`<\|endoftext\|>`，仅`<\|im_start\|>`, `<\|im_end\|>`, `<think>`, `</think>`）、`eos_token`/`pad_token`等模型属性映射 |
| `chat_template.jinja` | 模板复制 | 从 `output/qwen3_next_tokenizer/` 原样复制，保持对话格式兼容 |

**与模板的差异**:
- `tokenizer.json` 中的 `model.vocab`：模板原始词表 → 新训练的32K词表
- `tokenizer.json` 中的 `added_tokens`：模板原始特殊token → 5个新特殊token（`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`）
- `tokenizer_config.json` 中的 `extra_special_tokens`：模板完整列表 → 4个token（`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`，不含`<|endoftext|>`，移除视觉/多模态相关token）
- `tokenizer.json` 中的 `model.vocab`：模板原始词表 → 新训练的32K词表
- `tokenizer.json` 中的 `added_tokens`：模板原始特殊token → 新的5个特殊token
- `tokenizer_config.json` 中的 `extra_special_tokens`：模板完整列表 → 精简后的推理相关token
---

## 5. 实现清单

| 文件 | 说明 | 依赖 |
|------|------|------|
| `config/tokenizer_data.yaml` | 数据采样配置（数据集名称、路径、采样数量） | - |
| `scripts/prepare_template.py` | 复制 Qwen3-Next 架构 | transformers |
| `scripts/prepare_tokenizer_data.py` | 多数据源采样（两遍处理、流式写入） | datatrove, pyarrow |
| `scripts/train_tokenizer.py` | BPE 训练主脚本（基于模板继承） | tokenizers, transformers |

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

**train_tokenizer.py 核心功能**:

| 函数/类 | 功能 |
|---------|------|
| `load_template_tokenizer()` | 从模板目录加载完整的 Qwen2Tokenizer |
| `extract_template_config()` | 提取模板的 normalizer/pretokenizer/decoder 配置 |
| `train_bpe_vocab()` | 在采样数据上训练 32K BPE 词表 |
| `build_tokenizer_from_template()` | 将新词表与模板配置组合，创建完整 tokenizer |
| `configure_special_tokens()` | 配置特殊 token：`added_tokens`（5个）和 `extra_special_tokens`（4个，不含`<\|endoftext\|>`） |
| `verify_template_consistency()` | 验证输出与模板的一致性（除词表外） |
| `main()` | CLI 入口，支持 --template-dir, --vocab-size, --validate 等参数 |
---

## 6. 附录

### 6.1 特殊 Token 说明

**模板中的特殊 Token（Qwen3-Next）**:

模板 `Qwen/Qwen3-Next-80B-A3B-Instruct` 包含以下特殊 token：

| 类型 | Token | ID | 说明 |
|------|-------|-----|------|
| 基础 | `<\|endoftext\|>` | 151643 | 文本结束 / pad token |
| 对话 | `<\|im_start\|>` | 151644 | 对话开始 |
| 对话 | `<\|im_end\|>` | 151645 | 对话结束 / eos token |
| 视觉 | `<\|object_ref_start\|>` | 151646 | 对象引用开始 |
| 视觉 | `<\|object_ref_end\|>` | 151647 | 对象引用结束 |
| 视觉 | `<\|box_start\|>` | 151648 | 边界框开始 |
| 视觉 | `<\|box_end\|>` | 151649 | 边界框结束 |
| 视觉 | `<\|quad_start\|>` | 151650 | 四边形开始 |
| 视觉 | `<\|quad_end\|>` | 151651 | 四边形结束 |
| 视觉 | `<\|vision_start\|>` | 151652 | 视觉输入开始 |
| 视觉 | `<\|vision_end\|>` | 151653 | 视觉输入结束 |
| 视觉 | `<\|vision_pad\|>` | 151654 | 视觉填充 |
| 视觉 | `<\|image_pad\|>` | 151655 | 图像填充 |
| 视觉 | `<\|video_pad\|>` | 151656 | 视频填充 |
| FIM | `<\|fim_prefix\|>` | 151659 | 代码填充前缀 |
| FIM | `<\|fim_middle\|>` | 151660 | 代码填充中间 |
| FIM | `<\|fim_suffix\|>` | 151661 | 代码填充后缀 |
| 工具 | `<tool_call>` | 151657 | 工具调用开始 |
| 工具 | `</tool_call>` | 151658 | 工具调用结束 |
| 推理 | `<think>` | 151667 | 思考开始 |
| 推理 | `</think>` | 151668 | 思考结束 |

**特殊 Token 配置对比**:

| 配置项 | 模板 (Qwen3-Next) | 训练后 (本 tokenizer) |
|--------|-------------------|----------------------|
| `added_tokens` | 26个（含视觉、FIM、工具、推理等） | 5个：`<\|endoftext\|>`, `<\|im_start\|>`, `<\|im_end\|>`, `<think>`, `</think>` |
| `extra_special_tokens` | 12个（含视觉、多模态） | 4个：`<\|im_start\|>`, `<\|im_end\|>`, `<think>`, `</think>`（不含`<\|endoftext\|>`） |

**说明**:
- `extra_special_tokens` 是 `added_tokens` 的子集（不包含`<|endoftext|>`）
- 已移除：视觉相关（`<|vision_*|>`, `<|image_pad|>`, `<|video_pad|>`）、对象引用（`<|object_ref_*|>`, `<|box_*|>`, `<|quad_*|>`）、FIM（`<|fim_*|>`）、工具调用（`<tool_call>`）

**如需扩展**: 添加视觉/多模态支持时，可从模板恢复相应 token 并增加 `vocab_size`。

| ID | Token | 用途 | 来源 |
|----|-------|------|------|
| 32000 | `<\|endoftext\|>` | 文本结束 / pad token | 模板继承 |
| 32001 | `<\|im_start\|>` | 对话开始 | 模板继承 |
| 32002 | `<\|im_end\|>` | 对话结束 / eos token | 模板继承 |
| 32003 | `<think>` | 推理开始 | 新增 |
| 32004 | `</think>` | 推理结束 | 新增 |

**移除的 Token**: 所有视觉相关（`<|vision_*|>`, `<|image_pad|>`, `<|video_pad|>`）、对象引用（`<|object_ref_*|>`, `<|box_*|>`, `<|quad_*|>`）、FIM（`<|fim_*|>`）和工具调用（`<tool_call>`）相关的 token 已移除。

**如需扩展**: 添加视觉/多模态支持时，可从模板恢复相应 token 并增加 `vocab_size`。



### 6.2 相关文档

- [数据重组设计](fineweb_edu_data_reorganization_design.md)
- [FineWeb-Edu 论文](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Datatrove 文档](https://github.com/huggingface/datatrove)

---

*最后更新: 2026-02-24*
