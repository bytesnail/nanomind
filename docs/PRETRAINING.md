# Nanomind 预训练设计文档

> ⚠️ **文档状态**: 设计草稿 v0.1 (2026-03-04)  
> 🚧 本文档描述计划中的预训练系统架构，**预训练代码尚未实现**。  
> 具体实现进度见 [实施路线图](#7-实施路线图)。

---

## 实现状态索引

| 模块 | 状态 | 参考文档 |
|------|------|----------|
| Tokenizer 训练 | ✅ 已实现 | [Tokenizer 训练设计](tokenizer_training_design.md) |
| FineWeb-Edu 数据重组 | ✅ 已实现 | [FineWeb-Edu 数据重组设计](fineweb_edu_data_reorganization_design.md) |
| 预训练代码 | 🚧 设计中 | 本文档 |

**本文档范围**: 预训练系统设计（模型架构、训练配置、实施计划）

---

## 目录

1. [项目概述](#1-项目概述)
   - 1.1 [项目目标](#11-项目目标)
   - 1.2 [设计原则](#12-设计原则)
   - 1.3 [技术栈](#13-技术栈)
   - 1.4 [Modular Transformers 简介](#14-modular-transformers-简介)
2. [模型架构设计](#2-模型架构设计)
   - 2.1 [参考架构：Qwen3-Next](#21-参考架构qwen3-next)
   - 2.2 [缩放设计参考](#22-缩放设计参考)
   - 2.3 [nanomind 架构配置](#23-nanomind-架构配置)
   - 2.4 [参数量明细](#24-参数量明细)
3. [Tokenizer 配置](#3-tokenizer-配置)
4. [数据预处理](#4-数据预处理)
5. [训练配置](#5-训练配置)
   - 5.1 [Chinchilla Scaling Law](#51-chinchilla-scaling-law)
   - 5.2 [超参数配置](#52-超参数配置)
   - 5.3 [DeepSpeed 配置](#53-deepspeed-配置)
   - 5.4 [显存估算](#54-显存估算)
   - 5.5 [硬件特定配置](#55-硬件特定配置)
   - 5.6 [Accelerate 配置](#56-accelerate-配置)
6. [训练基础设施](#6-训练基础设施)
   - 6.1 [启动命令](#61-启动命令)
   - 6.2 [WandB 集成](#62-wandb-集成)
   - 6.3 [断点续训](#63-断点续训)
   - 6.4 [训练脚本示例](#64-训练脚本示例)
7. [实施路线图](#7-实施路线图)
8. [附录：快速参考](#8-附录快速参考)

---

## 1. 项目概述

### 1.1 项目目标

基于 **Qwen3-Next 架构** 训练 **nanomind** —— 一个约 **1.26B 总参数、~363M 激活参数**（激活占比~28.8%）的轻量级 MoE 大语言模型。

| 属性 | 配置 |
|------|------|
| **模型名称** | nanomind |
| **总参数量** | ~1.26B |
| **激活参数** | ~363M |
| **激活占比** | ~28.8% |
| **基础架构** | Qwen3-Next-MoE |
| **Tokenizer** | 36K BPE |
| **硬件平台** | 2× RTX 2080 Ti (22GB) |

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **架构兼容** | 完全兼容 Qwen3-Next，复用其生态工具 |
| **资源友好** | 适配消费级 2×22GB GPU 配置 |
| **渐进迭代** | 先跑通流程，再优化数据配比和超参数 |
| **可观测性** | 全程使用 WandB 跟踪训练过程 |

### 1.3 技术栈

| 组件 | 技术选择 |
|------|----------|
| 模型框架 | Hugging Face Transformers + [Modular Transformers](#14-modular-transformers-简介) |
| 训练加速 | DeepSpeed ZeRO-2/3 + Accelerate |
| 数据流水线 | Datatrove |
| 实验跟踪 | Weights & Biases |
| 配置管理 | YAML |

### 1.4 Modular Transformers 简介

**Modular Transformers** 是 Hugging Face Transformers 的新特性，允许通过继承现有模型代码来创建新模型，无需从头重写。

**核心优势：**
- 从其他模型导入和继承代码
- 通过 `modular_xxx.py` 文件定义新模型
- 使用 linter 工具自动展开为传统 `modeling.py` 文件

**示例：**
```python
# modular_nanomind.py
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextMoeForCausalLM,
    Qwen3NextMoeConfig,
)

class NanomindConfig(Qwen3NextMoeConfig):
    model_type = "nanomind"
    
class NanomindForCausalLM(Qwen3NextMoeForCausalLM):
    config_class = NanomindConfig
```

**生成模型文件：**
```bash
python utils/modular_model_converter.py nanomind
```

---

## 2. 模型架构设计

### 2.1 参考架构：Qwen3-Next

**核心特性：**

- **混合注意力**: Linear Attention (Gated DeltaNet) + Full Attention (Gated Attention) 3:1 交替
- **MoE 结构**: 高稀疏度混合专家
- **位置编码**: RoPE (partial_rotary_factor=0.25)
- **归一化**: RMSNorm
- **注意力**: Grouped Query Attention (GQA), head_dim 固定 256
- **长上下文**: 支持 256K，可扩展至 1M tokens
- **多 Token 预测**: MTP 机制

### 2.2 缩放设计参考

基于 **Qwen3-Next-80B-A3B** 的缩放规律：

| 参数 | Qwen3-Next-80B-A3B | nanomind (目标) | 缩放比例 |
|------|-------------------|-----------------|----------|
| 总参数量 | 80B | **~1.26B** | 1/63 |
| 激活参数 | 3B | **~360M** | 1/8 |
| 隐藏维度 | 2048 | **1152** | ~1/1.8 |
| 层数 | 48 | **20** | 5/12 |
| 专家总数 | 512 | **32** | 1/16 |
| 激活专家 | 10+1 | **3+1** | ~1/3 |

### 2.3 nanomind 架构配置

```yaml
# config/model/nanomind_1b_moe.yaml (规划中)
model:
  model_type: "qwen3_next_moe"
  architectures: ["Qwen3NextMoeForCausalLM"]
  
  # 词表
  vocab_size: 36005
  
  # 基础架构
  hidden_size: 1152
  num_hidden_layers: 20
  
  # Gated Attention
  num_attention_heads: 8
  num_key_value_heads: 2
  head_dim: 256
  attn_output_gate: true
  
  # Gated DeltaNet
  linear_num_key_heads: 16
  linear_num_value_heads: 16
  linear_key_head_dim: 128
  linear_value_head_dim: 128
  
  # 混合布局: 15层 DeltaNet + 5层 Attention
  layer_types: ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
  full_attention_interval: 4
  
  # MoE 配置 (~28.5% 激活占比)
  num_experts: 32
  num_experts_per_tok: 3
  num_shared_experts: 1
  moe_intermediate_size: 448
  router_aux_loss_coef: 0.001
  
  # 其他
  hidden_act: "silu"
  max_position_embeddings: 4096
  rope_theta: 10000000.0
  partial_rotary_factor: 0.25
  rms_norm_eps: 1.0e-6
  tie_word_embeddings: true
  torch_dtype: "bfloat16"
  
  # 特殊 Token ID 配置
  pad_token_id: 36000  # <|endoftext|>
  eos_token_id: 36002  # <|im_end|>
  bos_token_id: null
```

### 2.4 参数量明细

| 组件 | 计算公式 | 参数量 | 占比 |
|------|----------|--------|------|
| Embedding (tied) | `vocab_size × hidden_size` | 41.5M | 3.3% |
| Gated Attention (5层) | `5 × hidden_size² × 4` | 36.1M | 2.9% |
| Gated DeltaNet (15层) | `15 × (k_heads × k_dim + v_heads × v_dim)` | 161.5M | 12.8% |
| MoE 专家 (20层 × 33专家) | `20 × 33 × (3 × hidden_size × moe_intermediate)` | ~1,023M | 81.2% |
| LayerNorm + 其他 | - | 0.1M | <0.1% |
| **总计** | - | **~1,262M** | 100% |
| **激活参数** | - | **~363M** | **28.8%** |

---

## 3. Tokenizer 配置

> 📚 **详细实现** 见 [Tokenizer 训练设计文档](tokenizer_training_design.md)

本节仅列出预训练相关的关键规格：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **词表大小** | 36005 | 36000 BPE + 5 特殊 token |
| **pad_token_id** | 36000 | `<\|endoftext\|>` |
| **eos_token_id** | 36002 | `<\|im_end\|>` |
| **bos_token_id** | null | 不使用 BOS |

**特殊 Token 列表：**
```yaml
- "<|endoftext|>"    # ID: 36000, padding
- "<|im_start|>"     # ID: 36001, 对话开始
- "<|im_end|>"       # ID: 36002, 对话结束 / eos
- "<think>"          # ID: 36003, 思考开始 (CoT)
- "</think>"         # ID: 36004, 思考结束 (CoT)
```

---

## 4. 数据预处理

> 📚 **详细实现** 见 [FineWeb-Edu 数据重组设计文档](fineweb_edu_data_reorganization_design.md)

### 4.1 数据源概要

| 数据集 | 类型 | 质量分桶 |
|--------|------|----------|
| FineWeb-EN | 英文教育文本 | 2.5/3.0/3.5/4.0 |
| FineWeb-ZH | 中文教育文本 | 2.5/3.0/3.5/4.0 |
| GitHub Code | 代码 | stars 过滤 |
| Nemotron Math | 数学文本 | quality 分数 |

### 4.2 两层分桶策略

**第一层：Token 长度分桶**

| 长度桶 | 范围 | 用途 |
|--------|------|------|
| short | (0, 512] | **预训练初期** |
| medium | (512, 1024] | 预训练中期 |
| long | (1024, 2048] | 预训练后期 |
| xl | (2048, 4096] | 长上下文训练 |

**第二层：质量分桶** - 详见数据重组设计文档

### 4.3 数据配比 (Phase 1)

| 数据源 | 文档数 | 占比 | 质量配比 |
|--------|--------|------|----------|
| FineWeb-EN | 720K | 24% | 4.0(40%)/3.5(25%)/3.0(20%)/2.5(15%) |
| FineWeb-ZH | 1.2M | 40% | 4.0(40%)/3.5(25%)/3.0(20%)/2.5(15%) |
| GitHub Code | 660K | 22% | above-2-stars(80%)/below-2-stars(20%) |
| Nemotron Math | 420K | 14% | 4plus(50%)/4plus_MIND(25%)/3(25%) |
| **总计** | **3M** | **100%** | ~900M tokens |

---

## 5. 训练配置

### 5.1 Chinchilla Scaling Law

根据 DeepMind 的 Chinchilla 论文，计算最优的模型应该使用约 **20 tokens/parameter**。

1B 参数模型的 Chinchilla optimal 训练需要约 **20B tokens**。

**学习率估算公式**（经验公式，来源：Chinchilla 论文扩展研究）：
```python
def estimate_lr(model_size_in_billions: float) -> float:
    """基于模型大小估算最优学习率"""
    base_lr = 3e-4
    scale_factor = (1.0 / model_size_in_billions) ** 0.5
    return base_lr * scale_factor

# nanomind ~1.26B
lr = estimate_lr(1.26)  # ~2.67e-4

# Batch size 调整 (平方根缩放规则)
def scale_lr_for_batch_size(base_lr: float, base_bs: int, target_bs: int) -> float:
    """根据 batch size 调整学习率"""
    return base_lr * (target_bs / base_bs) ** 0.5
```

| 阶段 | Tokens | 说明 |
|------|--------|------|
| 保守起步 | 1-3B | 验证流程 |
| 标准训练 | 10-20B | 接近 optimal |
| 过训练 | 30B+ | 小模型可受益 |

### 5.2 超参数配置

```yaml
# config/training/base.yaml (规划中)
training:
  num_train_epochs: 1
  
  # Batch size (2 GPUs × 2 × 16 = 64 global)
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  
  # 序列长度
  max_seq_length: 512  # Phase 1
  
  # 优化器
  learning_rate: 3.0e-4
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  max_grad_norm: 1.0
  
  # 学习率调度
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.01
  
  # 精度 (RTX 2080 Ti 使用 FP16)
  bf16: false
  fp16: true
  
  # 显存优化
  gradient_checkpointing: true
  deepspeed: "config/deepspeed/zero3.json"
  
  # 日志与保存
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
```

### 5.3 DeepSpeed 配置

#### ZeRO-2 (推荐，速度优先)

```json
// config/deepspeed/zero2.json (规划中)
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
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

#### ZeRO-3 (显存受限)

```json
// config/deepspeed/zero3.json (规划中)
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e9,
    "gather_16bit_weights_on_model_save": true
  }
}
```

### 5.4 显存估算

| 配置 | 单卡显存需求 | 适合场景 |
|------|-------------|----------|
| 无 DeepSpeed | ~16GB | - |
| ZeRO-2 | ~4-5GB | **推荐** |
| ZeRO-3 (2 GPUs) | ~2-3GB | 极致优化 |

**nanomind-1B-MoE 详细估算：**

```
模型参数: ~1.1GB (ZeRO-3 分片后 ~0.55GB/GPU)
梯度: ~1.1GB (ZeRO-3 分片后 ~0.55GB/GPU)
激活值: ~0.9GB
优化器状态: CPU Offload
开销: ~1GB
单卡总计 (ZeRO-3): ~3-4GB
```

### 5.5 硬件特定配置

**RTX 2080 Ti 限制：**

| 数据类型 | 支持 | 加速 | 建议 |
|----------|------|------|------|
| FP16 | ✅ | ✅ Tensor Cores | **使用** |
| BF16 | ⚠️ | ❌ 回退到 FP32 | **避免** |
| FP32 | ✅ | ✅ CUDA Cores | 基准 |

```yaml
# 2080 Ti 专用配置 (规划中)
training:
  bf16: false          # 禁用 BF16
  fp16: true           # 启用 FP16
  fp16_opt_level: "O1"
  
  # FP16 稳定性优化
  gradient_accumulation_steps: 16
  max_grad_norm: 1.0
  learning_rate: 2.0e-4  # 略低于 BF16
```

### 5.6 Accelerate 配置

```yaml
# config/accelerate/deepspeed_zero2.yaml (规划中)
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_processes: 2
gpu_ids: 0,1
mixed_precision: fp16

deepspeed_config:
  zero_stage: 2
  gradient_accumulation_steps: 16
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  zero3_init_flag: false
```

---

## 6. 训练基础设施

### 6.1 启动命令

> 🚧 以下命令基于设计配置，相关脚本尚未实现

```bash
# 使用 Accelerate (规划中)
accelerate launch \
  --config_file config/accelerate/deepspeed_zero2.yaml \
  scripts/train_nanomind.py \
  --config config/training/phase1_short.yaml

# 或使用 torchrun (规划中)
torchrun \
  --nnodes=1 --nproc_per_node=2 \
  scripts/train_nanomind.py \
  --deepspeed config/deepspeed/zero2.json
```

### 6.2 WandB 集成

```python
# 环境变量
export WANDB_PROJECT="nanomind-pretrain"
export WANDB_RUN_NAME="nanomind-1b-phase1"

# TrainingArguments
training_args = TrainingArguments(
    report_to="wandb",
    run_name="nanomind-1b-v1",
    logging_steps=100,
)
```

**监控指标：**
- 训练: loss, learning_rate, grad_norm, throughput
- 验证: eval_loss, perplexity
- 硬件: GPU 显存、利用率、温度

### 6.3 断点续训

```python
# TrainingArguments 中启用断点续训
training_args = TrainingArguments(
    # ... 其他参数
    resume_from_checkpoint=True,  # 自动检测最新 checkpoint
)

# 或在 trainer.train() 中指定
trainer.train(resume_from_checkpoint="output/checkpoint-1000")
```

**checkpoint 保存策略：**
- `save_steps`: 每 N 步保存一次
- `save_total_limit`: 保留最近的 N 个 checkpoint
- 定期将重要 checkpoint 备份到外部存储

### 6.4 训练脚本示例

> 🚧 **伪代码示例**：展示预期架构，实际脚本 `scripts/train_nanomind.py` 尚未实现

```python
#!/usr/bin/env python3
"""nanomind 预训练脚本 - 设计草案"""

import argparse
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

def main():
    # 初始化 Accelerator
    accelerator = Accelerator()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("output/tokenizer_36k")
    
    # 创建模型配置
    model_config = AutoConfig.for_model(
        "qwen3_next_moe",
        vocab_size=36005,
        hidden_size=1152,
        num_hidden_layers=20,
        num_attention_heads=8,
        num_key_value_heads=2,
        # ... 其他参数
    )
    
    # 初始化模型
    model = AutoModelForCausalLM.from_config(model_config)
    
    # 加载数据集
    dataset = load_from_disk(config["data_path"])
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="output/nanomind_pretrain",
        overwrite_output_dir=True,
        # ... 从 config 加载
    )
    
    # 初始化 WandB
    if accelerator.is_main_process:
        wandb.init(
            project="nanomind-pretrain",
            name="nanomind-1b-phase1",
            config=config,
        )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )
    
    # 训练
    trainer.train()
    
    # 保存
    if accelerator.is_main_process:
        trainer.save_model("output/nanomind_pretrain/final")

if __name__ == "__main__":
    main()
```

---

## 7. 实施路线图

### 7.1 阶段规划

#### Phase 0: 基础设施 (1-2 天)
- [ ] 创建模型配置文件
- [ ] 实现数据分桶脚本
- [ ] 配置 DeepSpeed + Accelerate
- [ ] 搭建 WandB 项目

#### Phase 1: 数据预处理 (2-3 天)
- [x] ~~运行 Token 长度计算~~ ✅ 已实现
- [x] ~~执行两层分桶聚合~~ ✅ 已实现
- [x] ~~验证输出数据质量~~ ✅ 已实现
- [ ] 生成 Phase 1 数据集（按长度分桶）

#### Phase 2: 流程验证 (1-2 天)
- [ ] 小规模试验（1% 数据）
- [ ] 验证显存占用
- [ ] 验证 checkpoint 保存/加载

#### Phase 3: 正式训练 (1-2 周)
- [ ] Phase 1: ≤512 tokens, 1-3B tokens
- [ ] Phase 2: ≤1024 tokens, 3-5B tokens
- [ ] Phase 3: ≤4096 tokens, 10B+ tokens

#### Phase 4: 评估迭代 (持续)
- [ ] 下游任务评估
- [ ] 数据配比调优
- [ ] 超参数搜索

### 7.2 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| 显存 OOM | 高 | 启用 ZeRO-3 + CPU offload; 减小 batch size |
| 训练发散 | 高 | 降低学习率; 增加 warmup; 梯度裁剪 |
| 数据质量差 | 中 | 严格质量分桶; 初期使用高质量数据 |
| 训练速度慢 | 中 | 优化 dataloader; 使用 FP16 |

---

## 8. 附录：快速参考

### 8.1 项目目录结构

```
nanomind/
├── config/
│   ├── model/
│   │   └── nanomind_1b_moe.yaml      # 🚧 规划中
│   ├── training/
│   │   ├── base.yaml                 # 🚧 规划中
│   │   ├── phase1_short.yaml         # 🚧 规划中
│   │   ├── phase2_medium.yaml        # 🚧 规划中
│   │   └── phase3_long.yaml          # 🚧 规划中
│   ├── deepspeed/
│   │   ├── zero2.json                # 🚧 规划中
│   │   └── zero3.json                # 🚧 规划中
│   └── accelerate/
│       └── deepspeed_zero2.yaml      # 🚧 规划中
├── scripts/
│   ├── train_nanomind.py             # 🚧 规划中
│   ├── calculate_token_lengths.py    # 🚧 规划中
│   ├── bucket_documents.py           # 🚧 规划中
│   └── validate_pretrain_data.py     # 🚧 规划中
├── src/
│   ├── data_processing/              # ✅ 已实现
│   └── training/                     # 🚧 规划中
├── docs/
│   ├── PRETRAINING.md                # 📍 本文档
│   ├── tokenizer_training_design.md  # ✅ 已完成
│   └── fineweb_edu_data_reorganization_design.md  # ✅ 已完成
└── output/
    ├── tokenizer_36k/                # ✅ 已生成
    └── nanomind_pretrain/            # 🚧 规划中
```

### 8.2 配置文件清单

| 文件 | 用途 | 路径 | 状态 |
|------|------|------|------|
| `nanomind_1b_moe.yaml` | 模型架构 | `config/model/` | 🚧 规划中 |
| `phase1_short.yaml` | Phase 1 训练 | `config/training/` | 🚧 规划中 |
| `zero2.json` | ZeRO-2 配置 | `config/deepspeed/` | 🚧 规划中 |
| `zero3.json` | ZeRO-3 配置 | `config/deepspeed/` | 🚧 规划中 |
| `deepspeed_zero2.yaml` | Accelerate | `config/accelerate/` | 🚧 规划中 |

### 8.3 参考文献

1. Hoffmann et al. (2022). Training Compute-Optimal Large Language Models. (Chinchilla)
2. Qwen3 Technical Report
3. DeepSpeed Documentation: https://www.deepspeed.ai/
4. Hugging Face Transformers Documentation

---

## 文档更新记录

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v0.1 | 2026-03-04 | 初始设计稿，添加实现状态标记，修复 Token ID 冲突 |

---

*本文档为 nanomind 预训练项目的统一设计蓝图。具体实现时请根据实际运行情况调整参数。*
