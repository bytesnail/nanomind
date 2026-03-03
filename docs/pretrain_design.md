# nanomind 预训练项目设计文档

## 文档信息

| 属性 | 值 |
|------|-----|
| **文档版本** | v1.0 |
| **创建日期** | 2026-03-04 |
| **模型名称** | nanomind |
| **总参数量** | ~1.26B (20层, 32路由专家 + 1共享专家/层) |
| **激活参数** | ~363M (3路由专家 + 1共享专家/token) |
| **激活占比** | ~28.8% |
| **基础架构** | Qwen3-Next-MoE (高稀疏度 MoE + 混合注意力) |
| **Tokenizer** | 36K BPE (output/tokenizer_36k) |
| **硬件平台** | 2× RTX 2080 Ti (Turing, FP16, 22GB) |

---

## 目录

1. [概述](#一概述)
2. [模型架构设计](#二模型架构设计)
3. [Tokenizer 配置](#三tokenizer-配置)
4. [数据预处理设计](#四数据预处理设计)
5. [预训练配置](#五预训练配置)
6. [训练基础设施](#六训练基础设施)
7. [硬件资源规划](#七硬件资源规划)
8. [实施路线图](#八实施路线图)

---

## 一、概述

### 1.1 项目目标

本项目旨在基于 Qwen3-Next 架构，使用 `modular_transformers` 特性，训练一个约 **1.26B 总参数、~360M 激活参数**（激活占比~28.8%）的轻量级 MoE 大语言模型 **nanomind**。该模型将使用项目自研的 36K BPE Tokenizer，并在精心筛选的多语言语料上进行预训练。

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **架构兼容** | 完全兼容 Qwen3-Next 架构，可复用其生态工具 |
| **资源友好** | 适配 2×22GB GPU 的消费级硬件配置 |
| **渐进迭代** | 先跑通流程，再逐步优化数据配比和超参数 |
| **可观测性** | 全程使用 WandB 跟踪训练过程 |

### 1.3 技术栈

| 组件 | 技术选择 |
|------|----------|
| 模型框架 | Hugging Face Transformers + modular_transformers |
| 训练加速 | DeepSpeed ZeRO-2/3 + Hugging Face Accelerate |
| 数据流水线 | Datatrove (复用现有基础设施) |
| 实验跟踪 | Weights & Biases (WandB) |
| 配置管理 | YAML + Hydra (可选) |

---

## 二、模型架构设计

### 2.1 参考架构分析

基于 **Qwen3-Next** (qwen3_next) 官方配置和 Hugging Face transformers 实现：

**Qwen3-Next 核心特性：**
- **混合注意力架构**: Linear Attention (Gated DeltaNet) + Full Attention (Gated Attention) 3:1 交替
- **MoE 结构**: 高稀疏度混合专家 (Mixture of Experts)
- **核心机制**: Gated DeltaNet + Gated Attention 混合架构
- **位置编码**: RoPE (Rotary Position Embedding), partial_rotary_factor=0.25
- **激活函数**: SwiGLU (SiLU + Gating)
- **归一化**: RMSNorm (Root Mean Square Layer Normalization)
- **注意力**: Grouped Query Attention (GQA), head_dim 固定 256
- **长上下文**: 支持 256K 上下文，可扩展至 1M tokens
- **多 Token 预测**: Multi-Token Prediction (MTP) 机制

### 2.2 Qwen3-Next 参数缩放规律

参考 **Qwen3-Next** 系列模型配置：

**Qwen3-Next-80B-A3B 详细配置:**

| 参数 | 配置值 | 说明 |
|------|--------|------|
| 总参数量 | 80B | 总存储参数 |
| 激活参数 | 3B | 每 token 激活参数 |
| 层数 | 48 | Transformer 层数 |
| 隐藏维度 | 2048 | hidden_size |
| 注意力头 (Q) | 16 | GQA Query 头数 |
| 注意力头 (KV) | 2 | GQA KV 头数 |
| 头维度 | 256 | head_dim (固定) |
| RoPE 维度 | 64 | partial_rotary_factor=0.25 |

**混合布局:**
```
12 组 × [3层 (Gated DeltaNet -> MoE) + 1层 (Gated Attention -> MoE)]
= 12 × 4 = 48 层
```

**Gated DeltaNet 配置:**
- 线性注意力 V 头: 32 个
- 线性注意力 QK 头: 16 个
- 头维度: 128

**MoE 配置:**
- 专家总数: 512
- 激活专家数: 10 (路由专家) + 1 (共享专家)
- 专家中间维度: 512
- 稀疏度: 3B/80B = 3.75%

**关键发现：**
```
1. head_dim 固定为 256 (Gated Attention), 128 (Gated DeltaNet)
2. 混合架构: 3:1 的 DeltaNet:Attention 比例
3. 高稀疏度 MoE: 仅激活 3-4% 的参数
4. partial_rotary_factor = 0.25 (RoPE 只应用于 25% 维度)
5. rope_theta = 10,000,000 (支持长上下文)
6. 训练数据: 15T tokens
```

### 2.3 nanomind MoE 架构配置

目标：**~1.26B 总参数，~360M 激活参数，激活占比~28.5%**，基于 **Qwen3-Next (qwen3_next)** MoE 架构。

**基于 Qwen3-Next-80B-A3B 的缩放设计：**

| 参数 | Qwen3-Next-80B-A3B | **nanomind-1.2B-MoE (目标)** | 缩放比例 |
|------|-------------------|---------------------------|----------|
| 总参数量 | 80B | **~1.26B** | 1/63 |
| 激活参数 | 3B | **~360M** | 1/8 |
| 激活占比 | 3.75% | **~28.5%** | 更高稀疏度 |
| 隐藏维度 | 2048 | **1152** | ~1/1.8 |
| 层数 | 48 | **20** | 5/12 |
| 注意力头 (Q) | 16 | **8** | 1/2 |
| 注意力头 (KV) | 2 | **2** | 保持 |
| 头维度 | 256 | **256** | 保持 |
| 路由专家总数 | 512 | **32** | 1/16 |
| 激活专家 | 10 + 1 | **3 + 1** | ~1/3 |
| 专家中间维度 | 512 | **448** | ~7/8 |

**推荐配置 (config/model/nanomind_1b_moe.yaml):**

```yaml
model:
  # 核心架构参数 - 基于 Qwen3-Next MoE 架构
  model_type: "qwen3_next_moe"   # 使用 Qwen3-Next MoE 架构类
  architectures: ["Qwen3NextMoeForCausalLM"]
  
  # 词表配置
  vocab_size: 36005              # 36K BPE + 5 特殊 token
  
  # 基础架构
  hidden_size: 1152              # d_model
  num_hidden_layers: 20          # n_layers (20层，3:1混合注意力)
  
  # Gated Attention 配置
  num_attention_heads: 8         # Q 头数
  num_key_value_heads: 2         # KV 头数 (GQA 4:1)
  head_dim: 256                  # 头维度 (固定 256)
  attention_bias: false
  attention_dropout: 0.0
  attn_output_gate: true         # Gated Attention 门控
  
  # Gated DeltaNet 配置
  linear_num_key_heads: 16       # 线性注意力 QK 头
  linear_num_value_heads: 16     # 线性注意力 V 头
  linear_key_head_dim: 128       # 线性注意力头维度
  linear_value_head_dim: 128
  linear_conv_kernel_dim: 4      # 卷积核维度
  
  # 混合布局配置
  layer_types:                   # 3:1 DeltaNet:Attention 交替
    - "linear_attention"
    - "linear_attention"
    - "linear_attention"
    - "full_attention"
    # ... 重复 5 次，共 20 层 (15层 DeltaNet + 5层 Attention)
  full_attention_interval: 4     # 每 4 层一个 Full Attention
  
  # MoE 配置 - 激活占比目标 ~28.5%
  num_experts: 32                # 每层路由专家数 (32个)
  num_experts_per_tok: 3         # 每token激活路由专家数 (3个)
  num_shared_experts: 1          # 共享专家数 (1个，始终激活)
  moe_intermediate_size: 448     # 专家 FFN 中间维度 (从512调整到448)
  router_aux_loss_coef: 0.001    # 负载均衡损失系数
  
  # FFN 激活
  hidden_act: "silu"             # SwiGLU
  
  # 位置编码 - RoPE
  max_position_embeddings: 4096       # 初始训练长度 (逐步扩展到 32K+)
  rope_theta: 10000000.0              # Qwen3-Next 使用 10M
  partial_rotary_factor: 0.25         # 25% 维度应用 RoPE
  rope_parameters:
    rope_type: "default"
    mrope_interleaved: true
    mrope_section: [11, 11, 10]
    
  # 归一化
  rms_norm_eps: 1.0e-6
  
  # Multi-Token Prediction (MTP)
  mtp_num_hidden_layers: 1
  mtp_use_dedicated_embeddings: false
  
  # 初始化
  initializer_range: 0.02
  
  # 其他
  tie_word_embeddings: true      # 绑定输入输出 embedding
  use_cache: true
  
  # 训练配置
  torch_dtype: "bfloat16"
```

**参数量估算 (详细计算):**

```python
# nanomind-1.7B-MoE 参数量计算
config = {
    "vocab_size": 36005,
    "hidden_size": 1024,
    "num_layers": 24,
    "num_heads": 8,
    "num_kv_heads": 2,
    "head_dim": 256,
    "num_experts": 64,
    "num_experts_per_tok": 6,
    "num_shared_experts": 1,
    "moe_intermediate_size": 512,
}

# 1. Embedding 层
embed_params = 36005 * 1024 = 36.9M

# 2. 每层共享参数 (Attention + Norm)
# Gated Attention:
# - Q proj: 1024 × (8 × 256) = 2.05M
# - K proj: 1024 × (2 × 256) = 0.52M
# - V proj: 1024 × (2 × 256) = 0.52M
# - O proj: (8 × 256) × 1024 = 2.05M
attn_params = 2.05 + 0.52 + 0.52 + 2.05 = 5.14M

# Gated DeltaNet (简化估算):
# - QK: 1024 × (16 × 128) × 2 = 4.19M
# - V: 1024 × (16 × 128) = 2.10M
# - Output: (16 × 128) × 1024 = 2.10M
delta_net_params = 4.19 + 2.10 + 2.10 = 8.39M

# Layer Norm: 2 × 1024 = 0.002M
norm_params = 0.002M

# 每层共享参数 (交替使用 attn 或 delta_net)
avg_attn_params = (5.14 + 8.39) / 2 = 6.77M

# 3. MoE 专家参数
# 每个专家: 3 × moe_intermediate_size × hidden_size
expert_params = 3 * 512 * 1024 = 1.57M

# 所有专家 (64 个)
all_experts_params = 64 * 1.57M = 100.6M

# 共享专家 (1 个)
shared_expert_params = 1.57M

# 4. 总参数量计算
total_params = (
    embed_params +                           # 36.9M
    24 * avg_attn_params +                   # 24 × 6.77M = 162.5M
    24 * all_experts_params +                # 24 × 100.6M = 2.41B (错误! 专家跨层共享)
    # 修正: MoE 专家是跨层独立的，每层有自己的专家
    # 实际上应该是: 每层有 64 个专家
    # 重新计算:
)

# 正确的 MoE 参数量计算:
# - 每层有独立的 64 个专家
# - 每层专家总参数: 64 × 1.57M = 100.6M
# - 24 层专家总参数: 24 × 100.6M = 2.41B (太大了!)

# 实际上，对于小模型，我们应该减少专家数量或层数
# 调整方案: 每层 16 个专家，共 24 层
# - 每层专家: 16 × 1.57M = 25.1M
# - 24 层: 24 × 25.1M = 603M

# 或者: 保持 64 专家，但减少层数到 12
# - 12 层 × 100.6M = 1.2B (仍然太大)

# 更合理的方案: 共享专家跨层，每层 16 个路由专家
# - 共享专家 (跨层): 1.57M
# - 每层路由专家: 16 × 1.57M = 25.1M
# - 24 层路由专家: 24 × 25.1M = 603M
# - Attention/DeltaNet (24层): 24 × 6.77M = 162.5M
# - Embedding: 36.9M × 2 = 73.8M (tied)
# - 总计: 603 + 162.5 + 73.8 + 1.57 ≈ 841M

# 为了达到 1B，可以:
# - 增加 hidden_size 到 1152
# - 或增加专家数量到 20/层
```

**优化后的推荐配置 (20层, 激活占比 ~28.5%):**

```yaml
# 调整后达到 ~1.26B 总参数，~28.5% 激活占比
hidden_size: 1152              # d_model
num_hidden_layers: 20          # 20层 (15层 DeltaNet + 5层 Attention)
num_experts_per_layer: 32      # 每层 32 个路由专家
num_experts_per_tok: 3         # 每 token 激活 3 个路由专家
num_shared_experts: 1          # 1 个共享专家 (始终激活)
moe_intermediate_size: 448     # 专家 FFN 中间维度 (从512调整到448)
```

**参数量明细 (~1.26B):**

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Embedding (tied) | 41.5M | 3.3% |
| Gated Attention (5层) | 36.1M | 2.9% |
| Gated DeltaNet (15层) | 161.5M | 12.8% |
| MoE 专家 (20层 × 33专家) | ~1,023M | 81.2% |
| LayerNorm + 其他 | 0.1M | <0.1% |
| **总计** | **~1,262M (~1.26B)** | 100% |
| **激活参数** (固定 + 20层 × 4专家) | **~363M** | **28.8%** |

**训练优势：**
- **适中规模**: 1.26B 总参数，更接近目标 1.2B
- **高效稀疏**: 28.8% 激活占比，计算成本更低
- **层数优化**: 20层 (15 DeltaNet + 5 Attention)，保持3:1比例
- **推理更快**: 每 token 只激活 4 个专家 (3路由+1共享)，推理效率高

---

## 三、Tokenizer 配置

### 3.1 Tokenizer 规格

项目已训练完成的 Tokenizer (`output/tokenizer_36k`)：

```yaml
tokenizer:
  vocab_size: 36005
  # 词表组成:
  # - 36000 BPE tokens
  # - 5 特殊 tokens
  
  special_tokens:
    - "<|endoftext|>"    # ID: 36000, 用途: padding, 文本结束
    - "<|im_start|>"     # ID: 36001, 用途: 对话开始
    - "<|im_end|>"       # ID: 36002, 用途: 对话结束
    - "<think>"          # ID: 36003, 用途: 思考开始 (CoT)
    - "</think>"         # ID: 36004, 用途: 思考结束 (CoT)
```

### 3.2 与模型的集成

```python
# 模型配置中指定 tokenizer 相关参数
config = Qwen3Config(
    vocab_size=36005,
    pad_token_id=36000,
    eos_token_id=36000,  # 或使用其他 token
    bos_token_id=None,   # Qwen3 不使用 BOS
    # ... 其他参数
)
```

---

## 四、数据预处理设计

### 4.1 数据源

**可用数据集 (data/datasets/):**

| 数据集 | 路径 | 类型 | 质量分桶 |
|--------|------|------|----------|
| FineWeb-EN | fineweb/en | 英文教育文本 | 2.5/3.0/3.5/4.0 |
| FineWeb-ZH | fineweb/zh | 中文教育文本 | 2.5/3.0/3.5/4.0 |
| GitHub Code | nick007x/github-code-2025 | 代码 | stars 过滤 |
| Nemotron Math | nvidia/Nemotron-CC-Math-v1 | 数学文本 | quality 分数 |

### 4.2 两层分桶策略

#### 第一层：Token 长度分桶

将文档按 Token 长度划分到不同桶中：

| 长度桶 | 范围 | 用途 |
|--------|------|------|
| short | (0, 512] | **预训练初期使用** |
| medium | (512, 1024] | 预训练中期 |
| long | (1024, 2048] | 预训练后期 |
| xl | (2048, 4096] | 长上下文训练 |

**分桶实现逻辑：**

```python
def get_length_bucket(token_count: int) -> str:
    """根据 token 数量返回长度分桶名称"""
    if token_count <= 512:
        return "short"
    elif token_count <= 1024:
        return "medium"
    elif token_count <= 2048:
        return "long"
    elif token_count <= 4096:
        return "xl"
    else:
        return "xxl"  # 超长文档，可能需要截断

# Token 计数使用 output/tokenizer_36k
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("output/tokenizer_36k")
token_count = len(tokenizer.encode(text, add_special_tokens=False))
```

#### 第二层：质量分桶

复用现有质量评分体系：

| 数据集 | 质量指标 | 分桶策略 |
|--------|----------|----------|
| FineWeb | edu_score | 2.5/3.0/3.5/4.0 |
| GitHub Code | stars | above-2-stars / below-2-stars |
| Nemotron Math | quality_score | 3 / 4plus / 4plus_MIND |

### 4.3 输出目录结构

```
data/datasets/nanomind_pretrain/
├── short/                      # ≤512 tokens
│   ├── fineweb_en/
│   │   ├── 2.5/
│   │   ├── 3.0/
│   │   ├── 3.5/
│   │   └── 4.0/
│   ├── fineweb_zh/
│   ├── github_code/
│   └── nemotron_math/
├── medium/                     # 513-1024 tokens
│   └── ...
├── long/                       # 1025-2048 tokens
│   └── ...
└── xl/                         # 2049-4096 tokens
    └── ...
```

### 4.4 数据处理流水线

**阶段 1: Token 长度计算**

```python
# scripts/calculate_token_lengths.py
from transformers import AutoTokenizer
import pyarrow.parquet as pq
from pathlib import Path

def process_dataset(input_dir: Path, output_dir: Path, tokenizer_path: str):
    """
    读取 Parquet 文件，计算每个文档的 token 长度，
    输出带 token_count 列的新 Parquet
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    for parquet_file in input_dir.glob("*.parquet"):
        # 流式读取
        with pq.ParquetFile(parquet_file) as pf:
            for batch in pf.iter_batches(batch_size=10000, columns=["text"]):
                texts = batch.column("text").to_pylist()
                token_counts = [
                    len(tokenizer.encode(text, add_special_tokens=False))
                    for text in texts
                ]
                # 写入新文件...
```

**阶段 2: 两层分桶聚合**

```python
# scripts/bucket_documents.py
def assign_buckets(input_dir: Path, output_base: Path):
    """
    根据 token_count 和质量分数，将文档分配到对应桶
    """
    # 1. 读取带 token_count 的 Parquet
    # 2. 按 (length_bucket, quality_bucket) 分组
    # 3. 输出到对应目录
```

**阶段 3: 数据验证**

```python
# scripts/validate_pretrain_data.py
def validate_bucket(bucket_dir: Path):
    """
    验证每个桶的数据：
    - Token 长度分布是否符合预期
    - 文档数量是否充足
    - 质量分数分布是否合理
    """
```

### 4.5 初期数据配比 (Phase 1)

**预训练第一阶段**（使用 ≤512 tokens 数据）：

复用 Tokenizer 训练时的配比：

| 数据源 | 文档数 | 占比 | 质量配比 |
|--------|--------|------|----------|
| FineWeb-EN | 720K | 24% | 4.0(40%)/3.5(25%)/3.0(20%)/2.5(15%) |
| FineWeb-ZH | 1.2M | 40% | 4.0(40%)/3.5(25%)/3.0(20%)/2.5(15%) |
| GitHub Code | 660K | 22% | above-2-stars(80%)/below-2-stars(20%) |
| Nemotron Math | 420K | 14% | 4plus(50%)/4plus_MIND(25%)/3(25%) |
| **总计** | **3M** | **100%** | - |

**Token 数量估算：**

```
平均文档长度: ~300 tokens (≤512 桶)
总 Token 数: 3M × 300 = 900M tokens
```

### 4.6 后续扩展计划 (Phase 2+)

| 阶段 | 数据范围 | Token 数 | 目标 |
|------|----------|----------|------|
| Phase 1 | ≤512 | ~1B | 跑通流程 |
| Phase 2 | ≤1024 | ~3B | 增加中长文本 |
| Phase 3 | ≤4096 | ~10B | Chinchilla optimal |

---

## 五、预训练配置

### 5.1 Chinchilla Scaling Law

对于 1B 参数模型，Chinchilla optimal 训练需要约 **20B tokens**。

考虑到硬件限制和渐进迭代策略：

| 阶段 | Tokens | 说明 |
|------|--------|------|
| 保守起步 | 1-3B | 比 optimal 少 1 个数量级，验证流程 |
| 标准训练 | 10-20B | 接近 Chinchilla optimal |
| 过训练 | 30B+ | 小型模型通常可受益于更多数据 |

### 5.2 训练超参数 (适配 ZeRO-3)

**基础配置 (config/training/base.yaml):**

```yaml
training:
  # 训练周期
  num_train_epochs: 1
  max_steps: -1  # 按 epochs 计算
  
  # 批次大小 (适配 ZeRO-3 小显存)
  per_device_train_batch_size: 2      # ZeRO-3 下单卡 batch size 减半
  gradient_accumulation_steps: 16      # 增加梯度累积保持 global batch
  total_batch_size: 64                # 2 × 2 GPUs × 16 = 64 (global)
  
  # 序列长度
  max_seq_length: 512                 # Phase 1
  
  # 优化器
  learning_rate: 3.0e-4               # 1B 模型典型 lr
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1.0e-8
  max_grad_norm: 1.0
  
  # 学习率调度
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.01                  # 1% warmup
  
  # 精度 (⚠️ RTX 2080 Ti 不支持 BF16, 使用 FP16)
  bf16: false                         # 2080 Ti 不支持 BF16
  fp16: true                          # 使用 FP16 + Tensor Cores
  
  # DeepSpeed 特定配置
  deepspeed: "config/deepspeed/zero3.json"  # 使用 ZeRO-3
  
  # 日志与保存
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  
  # 评估
  evaluation_strategy: "steps"
  
  # 其他
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  remove_unused_columns: false
  
  # 显存优化
  gradient_checkpointing: true        # 梯度检查点，节省激活值显存
  optim: "adamw_torch"                # 使用 Torch AdamW
```

**不同阶段的配置变体：**

```yaml
# config/training/phase1_short.yaml
extends: base
max_seq_length: 512
data_path: "data/datasets/nanomind_pretrain/short"

# config/training/phase2_medium.yaml
extends: base
max_seq_length: 1024
data_path: "data/datasets/nanomind_pretrain/medium"
per_device_train_batch_size: 2  # 减少 batch size
gradient_accumulation_steps: 16  # 保持 global batch size
```

### 5.3 DeepSpeed 配置 (推荐 ZeRO-3)

考虑到 2×22GB 显存限制，**强烈推荐使用 ZeRO-3** 配合 CPU Offload，可显著降低单卡显存占用。

**ZeRO-3 + CPU Offload 配置 (推荐用于 2×22GB):**

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto"
}
```

**ZeRO-3 显存优化说明：**

| 配置 | ZeRO-2 | ZeRO-3 | 节省 |
|------|--------|--------|------|
| 单卡模型参数 | 全部 | 1/N | ~50% (2卡) |
| 优化器状态 | CPU offload | CPU offload | ~80% |
| 梯度 | 分布式 | 分布式 | ~50% |
| **单卡总显存** | ~12GB | **~6-8GB** | **~40%** |

**ZeRO-2 配置 (备用，如果 ZeRO-3 速度太慢):**

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "wall_clock_breakdown": false
}
```

### 5.4 Accelerate 配置 (ZeRO-3)

**配置文件 (config/accelerate/deepspeed_zero3.yaml):**

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 8
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu          # ZeRO-3: 参数也 offload 到 CPU
  zero3_init_flag: true              # ZeRO-3 初始化
  zero3_save_16bit_model: true       # 保存 16-bit 模型
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
gpu_ids: 0,1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

**启动命令:**

```bash
# 使用 ZeRO-3 配置启动
accelerate launch \
  --config_file config/accelerate/deepspeed_zero3.yaml \
  scripts/train_nanomind.py \
  --config config/training/phase1_short.yaml
```

### 5.5 训练脚本

**主训练脚本 (scripts/train_nanomind.py):**

```python
#!/usr/bin/env python3
"""
nanomind 预训练脚本
Usage:
    accelerate launch --config_file config/accelerate/deepspeed_zero2.yaml \
        scripts/train_nanomind.py \
        --config config/training/phase1_short.yaml
"""

import argparse
import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
import wandb

from src.data_processing import load_config

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default="output/nanomind_pretrain")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 初始化 Accelerator
    accelerator = Accelerator()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("output/tokenizer_36k")
    
    # 创建/加载模型配置
    model_config = AutoConfig.for_model(
        "qwen3",
        vocab_size=36005,
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=4096,
        max_position_embeddings=4096,
        # ... 其他参数
    )
    
    # 初始化模型
    model = AutoModelForCausalLM.from_config(model_config)
    
    # 加载数据集
    dataset = load_from_disk(config["data_path"])
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言建模
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        # ... 从 config 加载
    )
    
    # 初始化 WandB
    if accelerator.is_main_process:
        wandb.init(
            project="nanomind-pretrain",
            name=f"nanomind-1b-{config['phase']}",
            config=config,
        )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=data_collator,
    )
    
    # 训练
    trainer.train()
    
    # 保存最终模型
    if accelerator.is_main_process:
        trainer.save_model(args.output_dir / "final")


if __name__ == "__main__":
    main()
```

---

## 六、训练基础设施

### 6.1 启动命令

**单节点多卡训练：**

```bash
# 使用 Accelerate + DeepSpeed
accelerate launch \
    --config_file config/accelerate/deepspeed_zero2.yaml \
    scripts/train_nanomind.py \
    --config config/training/phase1_short.yaml \
    --output_dir output/nanomind_pretrain/phase1
```

**直接使用 Transformers Trainer：**

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    scripts/train_nanomind.py \
    --deepspeed config/deepspeed/zero2.json \
    --config config/training/phase1_short.yaml
```

### 6.2 WandB 集成

**实验跟踪配置：**

```yaml
# config/wandb/default.yaml
wandb:
  project: "nanomind-pretrain"
  entity: null  # 你的 wandb 用户名/team
  tags:
    - "nanomind"
    - "1b"
    - "phase1"
  notes: "nanomind 1B model pretraining"
```

**跟踪指标：**

- **训练指标**: loss, learning_rate, grad_norm, throughput (tokens/s)
- **验证指标**: eval_loss, perplexity
- **硬件指标**: GPU 显存、温度、利用率 (通过 WandB 自动收集)
- **自定义指标**: 各数据源 loss、序列长度分布

### 6.3 断点续训

```python
# 在 TrainingArguments 中启用
training_args = TrainingArguments(
    resume_from_checkpoint=True,  # 自动检测最新 checkpoint
    # 或指定路径
    # resume_from_checkpoint="output/nanomind_pretrain/phase1/checkpoint-1000",
)
```

---

## 七、硬件资源规划

### 7.1 硬件规格

| 组件 | 规格 | 备注 |
|------|------|------|
| CPU | 16核 32线程 | 数据加载、预处理 |
| 内存 | 250GB | 充足，可支持 CPU offload |
| GPU | 2× NVIDIA 2080 Ti | 定制 22GB 显存版 |
| 存储 | 待确认 | 需容纳数据集 + 模型 checkpoint |

### 7.2 RTX 2080 Ti 功能支持分析

**显卡规格:**
- **架构**: Turing (Compute Capability 7.5)
- **Tensor Cores**: 第一代 Tensor Cores (240 个)
- **显存**: 22GB (定制版) / 标准版 11GB

**数据类型支持矩阵:**

| 数据类型 | 软件支持 | 硬件加速 | 实际性能 | 备注 |
|----------|----------|----------|----------|------|
| FP32 | ✅ 支持 | ✅ CUDA Cores | 基准 | 基准精度 |
| **FP16** | **✅ 支持** | **✅ Tensor Cores** | **~2-3x 加速** | **推荐用于训练** |
| BF16 | ⚠️ 可运行 | ❌ **无硬件加速** | **比 FP32 慢** | 会回退到 FP32 执行 |
| INT8 | ✅ 支持 | ✅ Tensor Cores | ~4x 加速 | 推理量化 |
| TF32 | ❌ 不支持 | ❌ 无 | - | Ampere+ 特性 |
| FP8 | ❌ 不支持 | ❌ 无 | - | Hopper/Ada 特性 |

**⚠️ BF16 重要说明:**
```
虽然 PyTorch 允许在 RTX 2080 Ti 上使用 BF16 数据类型，但 Turing 架构
没有 BF16 Tensor Cores。BF16 计算会回退到 FP32 CUDA Cores 执行，
导致:
- 速度比 FP16 慢 2-3 倍
- 内存占用与 FP32 相同
- 没有性能优势

因此强烈建议使用 FP16 而非 BF16。
```

**⚠️ 关键限制与影响:**

```yaml
hardware_constraints:
  # 1. BF16 不支持 (最重要!)
  bf16_support: false
  impact: "必须使用 FP16 代替 BF16"
  solution: "使用 fp16: true 替代 bf16: true"
  
  # 2. FP16 稳定性注意事项
  fp16_limitations:
    - "动态范围比 BF16 小 (5.96e-8 vs 1.18e-38)"
    - "可能出现梯度下溢 (underflow)"
    - "需要 Gradient Scaler 稳定训练"
  
  # 3. DeepSpeed 兼容性
  deepspeed_support:
    zero2: "✅ 完全支持"
    zero3: "✅ 完全支持"
    offload: "✅ 支持 CPU offload"
    
  # 4. Flash Attention 支持
  flash_attention: "⚠️ 仅支持 Flash Attention 1.x"
  note: "Flash Attention 2.x 需要 Ampere+"
  
  # 5. FP8 支持
  fp8_support: "❌ 不支持"
  fp8_note: "FP8 需要 Hopper (H100) 或 Ada Lovelace (RTX 40系列) 架构"
```

**训练配置调整 (针对 2080 Ti):**

```yaml
# config/training/phase1_short.yaml
# 关键修改: FP16 替代 BF16

training:
  # ❌ 禁用 BF16 (2080 Ti 不支持)
  bf16: false
  
  # ✅ 启用 FP16 (使用 Tensor Cores)
  fp16: true
  fp16_opt_level: "O1"           # 混合精度级别
  fp16_backend: "auto"           # 自动选择 backend
  fp16_full_eval: false          # 评估也用 FP16
  
  # FP16 稳定性优化
  gradient_accumulation_steps: 16  # 增加累积步数，稳定梯度
  max_grad_norm: 1.0              # 梯度裁剪
  
  # 学习率调整 (FP16 可能需要更保守的 lr)
  learning_rate: 2.0e-4           # 比 BF16 略低
  
  # Loss Scaling (FP16 必需)
  fp16_loss_scale: "dynamic"      # 动态 loss scaling
  fp16_initial_loss_scale: 2^16   # 初始 scale
  fp16_min_loss_scale: 1.0        # 最小 scale
```

**DeepSpeed 配置 (FP16 版本):**

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": false
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

**FP16 vs BF16 性能对比:**

| 指标 | BF16 (Ampere+) | FP16 (Turing) | 影响 |
|------|----------------|---------------|------|
| 训练速度 | 基准 | ~90-95% | 轻微 slowdown |
| 内存占用 | 相同 | 相同 | 无差异 |
| 数值稳定性 | 更好 | 需调参 | 需 gradient scaler |
| 收敛难度 | 容易 | 中等 | 可能需要调整 lr |

**推荐实践:**

```python
# 训练脚本中的 FP16 配置
from torch.cuda.amp import autocast, GradScaler

# 初始化 gradient scaler (FP16 必需)
scaler = GradScaler()

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()
    
    # 使用 autocast 自动处理 FP16
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    
    # 梯度裁剪 (FP16 推荐)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()
```

### 7.2 显存占用估算 (ZeRO-3)

**nanomind-1B-MoE 模型状态 (BF16):**

| 组件 | 总参数量 | 单卡显存 (ZeRO-3) |
|------|----------|-------------------|
| 模型参数 | 1.09B | ~1.1 GB (1/N) |
| 梯度 | 1.09B | ~1.1 GB (1/N) |
| Adam 优化器状态 | 2.18B | CPU Offload |
| **单卡总计** | - | **~2.2 GB** |

**激活值 (Activation) 估算：**

```
激活值 ≈ batch_size × seq_len × hidden_size × layers × 常量
        ≈ 2 × 512 × 1152 × 24 × 34 ≈ 0.9 GB (MoE 激活参数小)
```

**单卡总显存需求 (ZeRO-3 + Offload):**

```
模型参数 (1.1GB) + 梯度 (1.1GB) + 激活值 (0.9GB) + 开销 (~1GB) ≈ 4.1 GB
```

**对比总结:**

| 配置 | 单卡显存需求 | 适合场景 |
|------|-------------|----------|
| ZeRO-2 | ~7-8GB | 速度优先 |
| **ZeRO-3 (推荐)** | **~4-5GB** | **显存受限** |
| ZeRO-3 + Offload | ~3-4GB | 极致显存优化 |

**系统预留规划：**

```yaml
resource_limits:
  gpu_memory_fraction: 0.80    # 每卡使用 80% (17.6GB / 22GB) - 保守预留
  cpu_memory_fraction: 0.90    # 系统内存使用 90% (225GB / 250GB)
  system_reserve:
    gpu: 4.4GB per GPU         # 预留更多显存，防止系统卡顿
    cpu: 25GB                  # 留给 OS 和其他进程
```

### 7.3 训练速度估算

**理论计算：**

```
单卡吞吐量: ~50,000 tokens/s (2080 Ti, 1B model, BF16)
双卡总吞吐量: ~100,000 tokens/s

1B tokens 训练时间: 1B / 100K = 10,000s ≈ 2.8 hours
10B tokens 训练时间: 10B / 100K = 100,000s ≈ 28 hours
```

**实际估算：**

| 数据规模 | 预计时间 | 备注 |
|----------|----------|------|
| 1B tokens | ~4-6 小时 | Phase 1 验证 |
| 3B tokens | ~12-18 小时 | 短文本完整训练 |
| 10B tokens | ~40-60 小时 | 接近 Chinchilla |

### 7.4 资源监控

```python
# scripts/monitor_resources.py
import psutil
import pynvml

def log_system_stats():
    """记录系统资源使用情况"""
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # GPU
    pynvml.nvmlInit()
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        print(f"GPU {i}: {info.used / 1e9:.2f}GB / {info.total / 1e9:.2f}GB "
              f"({util.gpu}% util)")
    
    print(f"CPU: {cpu_percent}%, Memory: {memory.percent}%")
```

---

## 八、实施路线图

### 8.1 阶段规划

#### Phase 0: 基础设施准备 (1-2 天)

- [ ] 创建模型配置文件
- [ ] 实现数据分桶脚本
- [ ] 配置 DeepSpeed + Accelerate
- [ ] 搭建 WandB 项目
- [ ] 编写训练脚本框架

#### Phase 1: 数据预处理 (2-3 天)

- [ ] 运行 Token 长度计算
- [ ] 执行两层分桶聚合
- [ ] 验证输出数据质量
- [ ] 生成 Phase 1 (≤512 tokens) 数据集

#### Phase 2: 预训练流程验证 (1-2 天)

- [ ] 小规模试验（1% 数据）
- [ ] 验证显存占用符合预期
- [ ] 验证 checkpoint 保存/加载
- [ ] 验证 WandB 指标记录

#### Phase 3: 正式训练 (1-2 周)

- [ ] Phase 1: ≤512 tokens, 1-3B tokens
- [ ] Phase 2: ≤1024 tokens, 3-5B tokens
- [ ] Phase 3: ≤4096 tokens, 10B+ tokens (可选)

#### Phase 4: 评估与迭代 (持续)

- [ ] 下游任务评估
- [ ] 数据配比调优
- [ ] 超参数搜索

### 8.2 目录结构

```
nanomind/
├── config/
│   ├── model/
│   │   └── nanomind_1b.yaml         # 模型架构配置
│   ├── training/
│   │   ├── base.yaml                # 基础训练配置
│   │   ├── phase1_short.yaml        # 阶段1: ≤512
│   │   ├── phase2_medium.yaml       # 阶段2: ≤1024
│   │   └── phase3_long.yaml         # 阶段3: ≤4096
│   ├── deepspeed/
│   │   ├── zero2.json               # ZeRO-2 配置
│   │   └── zero3.json               # ZeRO-3 配置
│   └── accelerate/
│       └── deepspeed_zero2.yaml     # Accelerate 配置
├── scripts/
│   ├── train_nanomind.py            # 主训练脚本
│   ├── calculate_token_lengths.py   # Token 长度计算
│   ├── bucket_documents.py          # 文档分桶
│   ├── validate_pretrain_data.py    # 数据验证
│   └── monitor_resources.py         # 资源监控
├── src/
│   └── training/
│       ├── __init__.py
│       ├── model_utils.py           # 模型创建/加载工具
│       ├── data_utils.py            # 数据处理工具
│       └── callbacks.py             # 自定义 callbacks
└── docs/
    └── pretrain_design.md           # 本文档
```

### 8.3 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| 显存 OOM | 高 | 启用 ZeRO-3 + CPU offload; 减小 batch size |
| 训练发散 | 高 | 降低 learning rate; 增加 warmup; 梯度裁剪 |
| 数据质量差 | 中 | 严格质量分桶; 初期使用高质量数据 |
| 训练速度慢 | 中 | 优化 dataloader; 增加 workers; 使用 BF16 |
|  checkpoint 损坏 | 低 | 频繁保存; 多版本备份 |

---

## 附录

### A. 参考文献

1. Hoffmann et al. (2022). Training Compute-Optimal Large Language Models. (Chinchilla)
2. Qwen3 Technical Report
3. DeepSpeed Documentation: https://www.deepspeed.ai/
4. Hugging Face Transformers Documentation
5. FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

### B. 相关配置快速参考

**模型参数量快速调整：**

```python
# Dense 模型配置:
configs_dense = {
    "tiny-0.3B": {"hidden": 1024, "layers": 18, "heads": 8, "kv_heads": 2, "ffn": 2816},
    "nano-0.6B": {"hidden": 1024, "layers": 36, "heads": 16, "kv_heads": 4, "ffn": 2816},
    "small-1.0B": {"hidden": 1536, "layers": 24, "heads": 12, "kv_heads": 4, "ffn": 4096},
    "medium-1.5B": {"hidden": 2048, "layers": 24, "heads": 16, "kv_heads": 4, "ffn": 5632},
}

# MoE 模型配置 (nanomind 采用此架构):
# 总参数 = 固定参数 + (层数 × 每专家参数 × 专家数)
# 激活参数 = 固定参数 + (层数 × 每专家参数 × 激活专家数)
configs_moe = {
    "nanomind-1.2B": {                          # 推荐配置 (激活占比~28.8%)
        "hidden": 1152,
        "layers": 20,                            # 20层 (15 DeltaNet + 5 Attention)
        "heads": 8,
        "kv_heads": 2,
        "experts_per_layer": 32,                # 32 路由专家
        "shared_experts": 1,                     # 1 共享专家
        "experts_per_tok": 3,                    # 激活 3 路由 + 1 共享 = 4
        "moe_intermediate": 448,                 # 专家中间维度
        "total_params": "~1.26B",
        "active_params": "~363M",
        "active_ratio": "~28.8%",
    },
}
```

---

**文档结束**

*本文档为 nanomind 预训练项目的设计蓝图，具体实现时需根据实际运行情况调整参数。*
