# Tokenizer 训练设计文档

> **目标**: 训练与 Qwen3-Next 兼容的 36K 词表 BPE Tokenizer  
> **训练样本**: 3M 多领域混合数据  
> **更新日期**: 2026-03-04

---

## 目录

1. [核心配置](#1-核心配置)
2. [训练数据](#2-训练数据)
3. [训练流程](#3-训练流程)
4. [验证与输出](#4-验证与输出)
5. [实现清单](#5-实现清单)
6. [附录](#6-附录)

---

## 1. 核心配置

### 1.1 词表结构

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **总词表大小** | 36005 | 36000 BPE + 5 特殊 token |
| **BPE tokens** | 36000 | ID 0-35999，从数据学习 |
| **特殊 tokens** | 5 | ID 36000-36004，手动定义 |
| **算法** | BPE | Byte Pair Encoding |
| **基础架构** | Qwen3-Next | 继承 Qwen2Tokenizer 配置 |

### 1.2 特殊 Token

特殊 token 定义在 [`src/constants.py`](../../src/constants.py):

```python
SPECIAL_TOKENS = [
    "<|endoftext|>",   # ID 36000 - padding
    "<|im_start|>",    # ID 36001 - 对话开始
    "<|im_end|>",      # ID 36002 - 对话结束 / eos
    "<think>",         # ID 36003 - 推理开始
    "</think>",        # ID 36004 - 推理结束
]
```

**模型属性映射**:
- `pad_token` = `<|endoftext|>` (36000)
- `eos_token` = `<|im_end|>` (36002)
- `bos_token` = `None`
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

总计 **3M** 样本，配置见 [`config/tokenizer_data.yaml`](../../config/tokenizer_data.yaml):

| 数据集 | 样本数 | 占比 | 明细 |
|--------|--------|------|------|
| **FineWeb-EN** | 720K | 24% | 4.0: 288K (40%), 3.5: 180K (25%), 3.0: 144K (20%), 2.5: 108K (15%) |
| **FineWeb-ZH** | 1.2M | 40% | 4.0: 480K (40%), 3.5: 300K (25%), 3.0: 240K (20%), 2.5: 180K (15%) |
| **GitHub Code** | 660K | 22% | ≥2 stars: 528K (80%), <2 stars: 132K (20%) |
| **Nemotron Math** | 420K | 14% | 4plus: 210K (50%), 4plus_MIND: 105K (25%), 3: 105K (25%) |

### 2.2 数据目录

**源数据**:
```
data/datasets/
├── fineweb/en/{2.5,3.0,3.5,4.0}/       # FineWeb-EN 分桶
├── fineweb/zh/{2.5,3.0,3.5,4.0}/       # FineWeb-ZH 分桶
├── nick007x/github-code-2025/           # GitHub Code
└── nvidia/Nemotron-CC-Math-v1/          # 数学数据
```

**训练输入**:
```
data/datasets/nanomind_tokenizer/
├── {dataset}-{bucket}-{counter}-rank-{rank}.parquet
└── sampling_info.json
```

**输出文件字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | `{dataset}/{bucket}/{filename}#{row_idx}` |
| `text` | string | 文本内容 |
| `source_dataset` | string | 数据集名称 |
| `source_bucket` | string | 桶名称 |
| `language` | string | 仅 GitHub Code 有 |

---

## 3. 训练流程

| 阶段 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1. 准备模板 | `prepare_tokenizer_template.py` | Qwen3-Next | `output/qwen3_next_tokenizer/` |
| 2. 数据采样 | `prepare_tokenizer_data.py` | 源数据 | `data/datasets/nanomind_tokenizer/` |
| 3. Tokenizer训练 | `train_tokenizer.py` | 采样数据 | `output/tokenizer_36k/` |

### 3.1 准备模板

```bash
python scripts/prepare_tokenizer_template.py
```

**配置** ([`config/tokenizer.yaml`](../../config/tokenizer.yaml)):

```yaml
template:
  model: "Qwen/Qwen3-Next-80B-A3B-Instruct"
  output_dir: "output/qwen3_next_tokenizer_origin"
  modified_dir: "output/qwen3_next_tokenizer"
```

**输出**:
- `output/qwen3_next_tokenizer_origin/` - 原始模板（26个特殊token）
- `output/qwen3_next_tokenizer/` - 精简版（仅5个基础token）

**精简逻辑**:
- 保留 `BASE_SPECIAL_TOKENS` + `THINK_TOKENS` (共5个)
- 移除视觉、FIM、工具调用相关token

### 3.2 数据采样

```bash
python scripts/prepare_tokenizer_data.py [--workers 16] [--tasks -1]
```

**配置** ([`config/tokenizer_data.yaml`](../../config/tokenizer_data.yaml) + [`config/tokenizer.yaml`](../../config/tokenizer.yaml)):

```yaml
# tokenizer_data.yaml - 数据配比
datasets:
  fineweb_en:
    name: "fineweb_edu_en"
    source: "data/datasets/fineweb/en"
    samples: 720000
    buckets:
      4.0: {count: 288000}
      ...

# tokenizer.yaml - 处理参数
preparation:
  workers: 16
  tasks: -1
  max_rows_per_file: 500000
  buffer_size: 50000
  compression: "zstd"
```

**两遍处理架构**:

| 阶段 | 操作 | 内存占用 |
|------|------|----------|
| **第一遍** | 预计算采样索引（哈希+最大堆） | O(target_count × 16 bytes) |
| **第二遍** | 流式读取选中行并写入 | 流式，不累积 |

**Pipeline 组件**:
```
ParquetReader → IndexFilter → LanguageTagger → SourceTagger → TokenizerDataWriter
```

**GitHub Code 语言过滤** ([`src/constants.py`](../../src/constants.py)):

```python
LANGUAGE_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".c": "c", ".cpp": "cpp", ".rs": "rust",
    # ... 共18种扩展名
}
ALLOWED_LANGUAGE_EXTENSIONS = frozenset(LANGUAGE_EXTENSIONS)
```

### 3.3 Tokenizer 训练

```bash
python scripts/train_tokenizer.py [--validate]
```

**配置** ([`config/tokenizer.yaml`](../../config/tokenizer.yaml)):

```yaml
training:
  vocab_size: 36005
  data_dir: "data/datasets/nanomind_tokenizer"
  output_dir: "output/tokenizer_36k"
  batch_size: 10000
```

**训练步骤**:
1. 加载模板: `AutoTokenizer.from_pretrained(template_dir)`
2. 创建迭代器: 流式读取 Parquet
3. 训练: `template.train_new_from_iterator(text_iterator, vocab_size=36005)`
4. 后处理: 重新映射 vocab ID，调整 added_tokens
5. 验证: 检查词表大小、特殊token、编解码一致性

**后处理逻辑**:
- vocab 中过滤特殊token，重新从0编号
- added_tokens 从 vocab 最大值后连续编号

---

## 4. 验证与输出

### 4.1 自动验证

训练时添加 `--validate`，检查:
- 词表大小 = 36005
- added_tokens 包含5个特殊token
- extra_special_tokens 包含4个（不含`<|endoftext|>`）
- 编解码一致性

### 4.2 输出文件

```
output/tokenizer_36k/
├── tokenizer.json          # 词表与合并规则
├── tokenizer_config.json   # Tokenizer配置
└── chat_template.jinja     # 对话模板
```

---

## 5. 实现清单

| 文件 | 说明 |
|------|------|
| `config/tokenizer.yaml` | Tokenizer训练配置 |
| `config/tokenizer_data.yaml` | 数据采样配置 |
| `src/constants.py` | 特殊token、语言扩展名常量 |
| `scripts/prepare_tokenizer_template.py` | 模板准备 |
| `scripts/prepare_tokenizer_data.py` | 数据采样 |
| `scripts/train_tokenizer.py` | BPE训练 |

**核心类/函数**:

| 名称 | 位置 | 功能 |
|------|------|------|
| `SPECIAL_TOKENS` | `src/constants.py:9` | 5个特殊token列表 |
| `LANGUAGE_EXTENSIONS` | `src/constants.py:14` | 语言扩展名映射 |
| `IndexFilter` | `prepare_tokenizer_data.py:373` | 索引过滤PipelineStep |
| `LanguageTagger` | `prepare_tokenizer_data.py:403` | 语言标记PipelineStep |
| `SourceTagger` | `prepare_tokenizer_data.py:434` | 来源标记PipelineStep |
| `TokenizerDataWriter` | `prepare_tokenizer_data.py:456` | Parquet写入PipelineStep |
| `postprocess_tokenizer_files` | `train_tokenizer.py:221` | 后处理函数 |
| `validate_tokenizer` | `train_tokenizer.py:384` | 验证函数 |

---

## 6. 附录

### 6.1 配置依赖关系

```
scripts/prepare_tokenizer_data.py
├── config/tokenizer_data.yaml    # 数据配比
├── config/tokenizer.yaml         # workers, buffer_size
└── src/constants.py              # LANGUAGE_EXTENSIONS

scripts/train_tokenizer.py
├── config/tokenizer.yaml         # vocab_size, paths
└── src/constants.py              # SPECIAL_TOKENS
```

### 6.2 相关文档

- [FineWeb-Edu 数据重组设计](fineweb_edu_data_reorganization_design.md)
- [数据处理模块 API](../src/data_processing/README.md)
- [项目知识库](KNOWLEDGE_BASE.md)

---

*最后更新: 2026-03-04*
