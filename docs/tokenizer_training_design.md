# Tokenizer 训练设计文档

> **目标**: 训练与 Qwen3-Next 兼容的 64K 词表 BPE Tokenizer  
> **训练样本**: 40M 多领域混合数据  
> **更新日期**: 2026-02-13

---

## 1. 核心配置

### 1.1 词表结构

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **总词表大小** | 64005 | ID 0-64004（训练时使用 `--vocab-size 64005`） |
| **BPE tokens** | 64000 | 从数据学习（ID 0-63999） |
| **特殊 tokens** | 5 | 手动定义（ID 64000-64004） |
| **算法** | BPE | Byte Pair Encoding |
| **BPE 最小词频** | 2 | min_frequency |
| **基础架构** | Qwen3-Next | 仅复制配置（pretokenizer/normalizer/decoder），不继承词表 |

### 1.2 特殊 Token

| ID | Token | 用途 | 模型配置 |
|----|-------|------|----------|
| 64000 | `<|endoftext|>` | 文本结束 | `tokenizer.pad_token` |
| 64001 | `<|im_start|>` | 对话开始 | 特殊标记 |
| 64002 | `<|im_end|>` | 对话结束 | `tokenizer.eos_token` |
| 64003 | `<|think|>` | 推理开始 | 特殊标记 |
| 64004 | `<|/think|>` | 推理结束 | 特殊标记 |

> **注意**: `bos_token` 和 `unk_token` 设为 `None`

**对话格式示例**:
```
<|im_start|>user
问题<|im_end|>
<|im_start|>assistant
<|think|>推理过程...<|/think|>
答案<|im_end|>
```

---

## 2. 训练数据

### 2.1 数据配比（40M 总计）

| 数据集 | 样本数 | 占比 | 明细 |
|--------|--------|------|------|
| **FineWeb-EN** | 12.0M | 30% | 4.0分(5.4M), 3.5分(2.4M), 3.0分(2.4M), 2.5分(1.8M) |
| **FineWeb-ZH** | 10.0M | 25% | 4.0分(4.5M), 3.5分(2.0M), 3.0分(2.0M), 2.5分(1.5M) |
| **GitHub Code** | 12.0M | 30% | ≥2 stars(10M), <2 stars(2M) |
| **Nemotron-CC-Math** | 6.0M | 15% | 4plus(3.0M), 4plus_MIND(1.92M), 3(1.08M) |

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

**训练输入**（采样脚本输出）：
```
data/datasets/nanomind_tokenizer/
├── train-{idx:05d}-of-{total:05d}.parquet
└── sampling_info.json            # 采样元信息（种子、配比、统计）
```

> FineWeb 分桶预处理详见 [数据重组设计文档](fineweb_edu_data_reorganization_design.md)

---

## 3. 训练流程

| 阶段 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1. 准备模板 | `prepare_template.py` | Qwen3-Next | `output/qwen3_next_tokenizer/` |
| 2. 数据采样 | `prepare_tokenizer_data.py` | 源数据目录 | `data/datasets/nanomind_tokenizer/` |
| 3. Tokenizer训练 | `train_tokenizer.py` | 采样后数据 | `output/tokenizer_64k/` |

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

**目标**: 从各数据源按配置比例采样，每个桶独立打乱后随机取样。

```bash
python scripts/prepare_tokenizer_data.py \
    --config config/tokenizer_data.yaml \
    --output-dir data/datasets/nanomind_tokenizer
```

**采样配置** (`config/tokenizer_data.yaml`):
```yaml
datasets:
  fineweb_en:
    source: "data/datasets/fineweb/en"
    samples: 12000000
    buckets:
      4.0: {count: 5400000}
      3.5: {count: 2400000}
      3.0: {count: 2400000}
      2.5: {count: 1800000}

  fineweb_zh:
    source: "data/datasets/fineweb/zh"
    samples: 10000000
    buckets:
      4.0: {count: 4500000}
      3.5: {count: 2000000}
      3.0: {count: 2000000}
      2.5: {count: 1500000}

  github_code:
    source: "data/datasets/nick007x/github-code-2025"
    samples: 12000000
    stars_filter:
      above_2: {count: 10000000}
      below_2: {count: 2000000}

  nemotron_math:
    source: "data/datasets/nvidia/Nemotron-CC-Math-v1"
    samples: 6000000
    buckets:
      4plus: {count: 3000000}
      4plus_MIND: {count: 1920000}
      3: {count: 1080000}

random_seed: 42
output_format: parquet
```

**实现要点**:
- 基于 Datatrove 框架的 `PipelineStep` 实现
- 每个桶独立处理：读取 → 打乱 → 采样指定数量
- 使用 `LocalPipelineExecutor` 并行处理

### 3.3 Tokenizer 训练

```bash
python scripts/train_tokenizer.py \
    --data-dir data/datasets/nanomind_tokenizer \
    --template-dir output/qwen3_next_tokenizer \
    --output-dir output/tokenizer_64k \
    --vocab-size 64005 \
    --validate
```

**训练步骤**:
1. 从模板加载 pretokenizer/normalizer/decoder 配置
2. 空白初始化，在采样数据上学习 64000 个 BPE 合并规则
3. 添加 5 个特殊 token（ID 64000-64004）
4. 配置 eos/pad/bos/unk 映射（eos=64002, pad=64000, bos/unk=None）

---

## 4. 验证与输出

### 4.1 自动验证

训练时添加 `--validate` 参数，自动检查：
- 词表大小 = 64005
- 特殊 token 存在且 ID 正确（64000-64004）
- 编解码一致性

### 4.2 手动验证

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("output/tokenizer_64k")

# 验证词表大小
assert tokenizer.vocab_size == 64005

# 验证特殊 token
print(tokenizer.special_tokens_map)

# 编解码测试
text = "<|im_start|>assistant\n<|think|>推理<|/think|>答案<|im_end|>"
encoded = tokenizer(text, return_tensors="pt")
decoded = tokenizer.decode(encoded["input_ids"][0])
assert text == decoded
```

### 4.3 输出文件

```
output/tokenizer_64k/
├── tokenizer.json              # 词表与合并规则
├── tokenizer_config.json       # Tokenizer配置
├── special_tokens_map.json     # 特殊token映射
├── training_info.json          # 训练元信息
└── vocab.txt                   # 可读词汇表
```

---

## 5. 实现清单

### 5.1 文件清单

| 文件 | 状态 | 说明 | 技术栈 |
|------|------|------|--------|
| `config/tokenizer_data.yaml` | 已设计 | 数据采样配置 | YAML |
| `scripts/prepare_template.py` | 待实现 | 复制 Qwen3-Next 架构 | transformers |
| `scripts/prepare_tokenizer_data.py` | 待实现 | 多数据源采样 | datatrove |
| `scripts/train_tokenizer.py` | 待实现 | BPE 训练主脚本 | tokenizers |

### 5.2 依赖

```
tokenizers>=0.22.0
transformers>=4.40.0
datatrove>=0.8.0
```

### 5.3 扩展预留

如需视觉/多模态支持，可添加特殊 token（如 `<|vision_start|>`、`<|image_pad|>` 等），并相应增加 `vocab_size`。

---

*相关文档: [数据重组设计](fineweb_edu_data_reorganization_design.md)*
