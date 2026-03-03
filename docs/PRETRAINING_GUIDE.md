# Nanomind 预训练配置指南

基于 Hugging Face Transformers + DeepSpeed + Accelerate 的 1B 参数模型预训练配置。

---

## 1. Modular Transformers 概述

### 1.1 什么是 Modular Transformers

**Modular Transformers** 是 Hugging Face Transformers 库的一个新特性，旨在降低贡献新模型的门槛，允许通过继承和导入其他模型的代码来创建新模型，而不需要从头重写所有代码。

**核心特性：**
- 允许从其他模型导入和继承代码
- 通过在模型目录中添加 `modular_xxx.py` 文件来定义新模型
- 使用 linter 工具将 modular 文件"解开"成传统的 `modeling.py` 文件
- 保持"单模型单文件"的最终输出结构

### 1.2 如何使用 Modular Transformers

**文件结构：**
```
src/transformers/models/your_model/
├── modular_your_model.py    # 模块化定义文件（你写的）
├── modeling_your_model.py   # 生成的单文件模型（自动）
├── configuration_your_model.py
└── ...
```

**示例：基于 LLaMA 创建新模型**

```python
# modular_nanomind.py
from ..llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaMLP,
)
from ..llama.configuration_llama import LlamaConfig


class NanomindConfig(LlamaConfig):
    """Nanomind 模型配置，继承自 LLaMA 配置"""
    model_type = "nanomind"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,  # GQA
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            **kwargs,
        )


class NanomindModel(LlamaModel):
    """Nanomind 基础模型，继承自 LLaMA"""
    config_class = NanomindConfig
    
    def __init__(self, config: NanomindConfig):
        super().__init__(config)
        # 可以在这里添加自定义修改


class NanomindForCausalLM(LlamaForCausalLM):
    """Nanomind Causal LM 模型"""
    config_class = NanomindConfig
    
    def __init__(self, config: NanomindConfig):
        super().__init__(config)
        # 继承 LLaMA 的 causal LM 实现
```

**生成 modeling.py 文件：**
```bash
python utils/modular_model_converter.py nanomind
```

### 1.3 继承规则与技巧

| 操作 | 方法 | 说明 |
|------|------|------|
| 继承并修改配置 | 重写 `__init__` | 使用 `super().__init__()` 调用父类 |
| 删除属性 | `del self.attribute` | 在 `super().__init__()` 之后使用 |
| 添加新层 | 重写 `__init__` 并定义 | 新层会在展开时自动添加 |
| 修改 forward | 完全重写 forward 方法 | 可以使用 `super().forward(**super_kwargs)` |
| 删除方法 | 重写并 `raise AttributeError` | 模拟删除父类方法 |

---

## 2. DeepSpeed ZeRO 配置

### 2.1 ZeRO 阶段对比

| 阶段 | 分区内容 | 内存节省 | 通信开销 | 适用场景 |
|------|----------|----------|----------|----------|
| ZeRO-1 | 优化器状态 | ~4x | 低 | 大 batch size |
| ZeRO-2 | 优化器状态 + 梯度 | ~8x | 中 | 中等模型 |
| ZeRO-3 | 参数 + 梯度 + 优化器 | 线性扩展 | 高 | 超大模型 |

### 2.2 2x 22GB GPU 推荐配置

对于 1B 参数模型，**ZeRO-2 是最佳选择**，平衡了内存使用和通信效率。

**配置 1: ZeRO-2 (推荐)**

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
    "stage": 2,
    "offload_optimizer": {
      "device": "none",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  }
}
```

**配置 2: ZeRO-3 (如果需要更大 batch size)**

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
      "device": "none",
      "pin_memory": true
    },
    "offload_param": {
      "device": "none",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 1e6,
    "stage3_prefetch_bucket_size": 1e6,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

**配置 3: ZeRO-Offload (CPU 内存充足时使用)**

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
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

### 2.3 内存估算参考

| 配置 | 模型参数 | 每 GPU 内存 |
|------|----------|-------------|
| 无 DeepSpeed | 1B | ~16 GB |
| ZeRO-1 | 1B | ~8 GB |
| ZeRO-2 | 1B | ~4 GB |
| ZeRO-3 (2 GPUs) | 1B | ~2 GB |

---

## 3. Hugging Face Accelerate + DeepSpeed 集成

### 3.1 安装与配置

**安装依赖：**
```bash
pip install accelerate deepspeed transformers torch
```

**使用 `accelerate config` 交互式配置：**

```bash
$ accelerate config

In which compute environment are you running? [0] This machine
Which type of machine are you using? [0] multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Do you wish to optimize your script with torch dynamo? [yes/NO]: NO
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: yes
Please enter the path to the json DeepSpeed config file: ds_config_zero2.json
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: no
How many GPU(s) should be used for distributed training? [1]: 2
accelerate configuration saved at /home/user/.cache/huggingface/accelerate/default_config.yaml
```

### 3.2 Accelerate 配置文件示例

**`accelerate_config.yaml` (使用 DeepSpeed):**

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  zero3_init_flag: false
  zero3_save_16bit_model: false
  zero_stage: 2
  deepspeed_config_file: ds_config_zero2.json
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
machine_rank: 0
main_training_function: main
megatron_lm_config: {}
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
use_cpu: false
```

### 3.3 代码集成方式

**方式 1: 使用 Accelerate 的 Trainer 集成**

```python
# train_with_accelerate.py
import os
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# DeepSpeed 插件配置
deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=2,
    gradient_accumulation_steps=4,
    gradient_clipping=1.0,
    offload_optimizer_device="none",
    offload_param_device="none",
    zero3_init_flag=False,
)

# 初始化 Accelerator
accelerator = Accelerator(
    mixed_precision="fp16",
    deepspeed_plugin=deepspeed_plugin,
    gradient_accumulation_steps=4,
)

# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained("nanomind-base")
tokenizer = AutoTokenizer.from_pretrained("nanomind-base")

# 准备数据集
dataset = load_dataset("json", data_files="data/train.jsonl")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果语言建模
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=1000,
    fp16=True,
    deepspeed="ds_config_zero2.json",  # DeepSpeed 配置文件
    report_to="wandb",  # WandB 集成
    run_name="nanomind-1b-pretrain",
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# 训练
trainer.train()

# 保存模型
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    "./output/final",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
    state_dict=accelerator.get_state_dict(model),
)
```

**方式 2: 原生 DeepSpeed 集成（更灵活）**

```python
# train_native_deepspeed.py
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as f:
            self.lines = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.lines[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }

# 初始化
model = AutoModelForCausalLM.from_pretrained("nanomind-base")
tokenizer = AutoTokenizer.from_pretrained("nanomind-base")

# 数据集
dataset = TextDataset("data/train.txt", tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# DeepSpeed 配置
ds_config = {
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 1000
        }
    },
    "fp16": {
        "enabled": True,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
    },
    "gradient_clipping": 1.0,
}

# 初始化 DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
)

# 训练循环
for epoch in range(3):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(model_engine.local_rank)
        attention_mask = batch["attention_mask"].to(model_engine.local_rank)
        labels = batch["labels"].to(model_engine.local_rank)
        
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        
        model_engine.backward(loss)
        model_engine.step()

# 保存模型
model_engine.save_checkpoint("./output")
```

---

## 4. 1B 参数模型训练超参数

### 4.1 Chinchilla Scaling Law 应用

根据 DeepMind 的 Chinchilla 论文，**计算最优**的模型应该使用约 **20 tokens/parameter**。

**1B 模型的推荐配置：**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 模型参数 | 1B (1,000M) | 可接受范围: 0.8B - 1.2B |
| 训练 tokens | 20B | Chinchilla 最优 = 20 × 1B |
| Batch size | 0.5M - 2M tokens | 每步处理的 tokens 数 |
| 学习率 | 1e-4 ~ 4e-4 | 通常 3e-4 是良好的起点 |
| 学习率调度 | Cosine decay | 带 warmup 的余弦退火 |
| Warmup steps | 1% ~ 2% total | 通常 1000-2000 steps |
| Weight decay | 0.01 ~ 0.1 | AdamW 常用 0.01 |
| Gradient clipping | 1.0 | 防止梯度爆炸 |

### 4.2 学习率计算公式

根据最新的 scaling law 研究：

```python
# 计算最优学习率 (近似公式)
def estimate_lr(model_size_in_billions: float) -> float:
    """
    基于模型大小估算最优学习率
    参考: https://arxiv.org/abs/2403.08518
    """
    # 经验公式: lr ∝ 1/sqrt(model_size)
    base_lr = 3e-4
    scale_factor = (1.0 / model_size_in_billions) ** 0.5
    return base_lr * scale_factor

# 1B 模型
lr_1b = estimate_lr(1.0)  # ~3e-4

# Batch size 调整 (linear scaling rule)
def scale_lr_for_batch_size(base_lr: float, base_bs: int, target_bs: int) -> float:
    """根据 batch size 调整学习率"""
    return base_lr * (target_bs / base_bs) ** 0.5  # 平方根缩放
```

### 4.3 训练脚本配置示例

```python
# training_config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class NanomindTrainingConfig:
    """Nanomind 1B 预训练配置"""
    
    # 模型配置
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4  # GQA
    max_position_embeddings: int = 4096
    
    # 训练超参数
    num_train_epochs: int = 1  # 通常一个 epoch 足够
    total_tokens: int = 20_000_000_000  # 20B tokens
    
    # Batch size 配置
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_gpus: int = 2
    seq_length: int = 2048
    
    @property
    def global_batch_size(self) -> int:
        """计算全局 batch size (in sequences)"""
        return self.per_device_batch_size * self.gradient_accumulation_steps * self.num_gpus
    
    @property
    def tokens_per_step(self) -> int:
        """每步处理的 tokens 数"""
        return self.global_batch_size * self.seq_length
    
    @property
    def total_steps(self) -> int:
        """总训练步数"""
        return self.total_tokens // self.tokens_per_step
    
    # 优化器参数
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    warmup_ratio: float = 0.01  # 1% warmup
    lr_scheduler_type: str = "cosine"
    
    @property
    def warmup_steps(self) -> int:
        """Warmup 步数"""
        return int(self.total_steps * self.warmup_ratio)
    
    # 保存与日志
    save_steps: int = 1000
    logging_steps: int = 100
    eval_steps: int = 500
    
    # 混合精度
    fp16: bool = True
    bf16: bool = False  # A100/H100 可启用
    
    # 梯度检查点 (节省内存)
    gradient_checkpointing: bool = True
    
    # DeepSpeed 配置
    deepspeed_config: str = "configs/ds_config_zero2.json"


# 使用示例
config = NanomindTrainingConfig()
print(f"总训练步数: {config.total_steps:,}")
print(f"每步 tokens: {config.tokens_per_step:,}")
print(f"Warmup 步数: {config.warmup_steps:,}")
print(f"全局 batch size: {config.global_batch_size}")
```

---

## 5. WandB 集成

### 5.1 基础配置

**环境变量设置：**
```bash
export WANDB_PROJECT="nanomind-pretraining"
export WANDB_ENTITY="your-username"  # 可选
export WANDB_API_KEY="your-api-key"  # 或运行 wandb login
```

**TrainingArguments 配置：**

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    
    # WandB 集成
    report_to="wandb",
    run_name="nanomind-1b-v1",  # 实验名称
    
    # 日志频率
    logging_steps=100,
    logging_first_step=True,
    
    # 保存模型到 WandB
    # 注意：这会保存所有 checkpoint，谨慎使用
    # save_safetensors=True,
)
```

### 5.2 高级 WandB 配置

```python
import wandb
from transformers import TrainingArguments, Trainer
from transformers.integrations import WandbCallback

# 初始化 WandB 运行
wandb.init(
    project="nanomind-pretraining",
    name="nanomind-1b-chinchilla",
    config={
        "model_size": "1B",
        "tokens": "20B",
        "batch_size": 512,
        "learning_rate": 3e-4,
        "architecture": "transformer",
        "attention": "gqa",
    },
    tags=["pretraining", "1b", "chinchilla"],
)

# 自定义 WandB 回调
class CustomWandbCallback(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """自定义日志记录"""
        if logs is not None:
            # 添加自定义指标
            logs["train/tokens_processed"] = state.global_step * args.train_batch_size * args.max_seq_length
            logs["train/learning_rate_current"] = logs.get("learning_rate", 0)
            
            # 计算吞吐量
            if hasattr(state, "throughput"):
                logs["perf/tokens_per_second"] = state.throughput
            
        super().on_log(args, state, control, model, logs, **kwargs)

# 在 Trainer 中使用
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[CustomWandbCallback],
)
```

### 5.3 监控关键指标

```python
# 在训练脚本中添加监控
import torch
import wandb

def log_model_info(model, step):
    """记录模型统计信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.log({
        "model/total_params_M": total_params / 1e6,
        "model/trainable_params_M": trainable_params / 1e6,
        "step": step,
    })

def log_memory_usage(step):
    """记录内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            wandb.log({
                f"system/gpu_{i}_memory_allocated_gb": allocated,
                f"system/gpu_{i}_memory_reserved_gb": reserved,
                "step": step,
            })
```

---

## 6. 混合精度训练配置

### 6.1 FP16 vs BF16

| 特性 | FP16 | BF16 |
|------|------|------|
| 指数位 | 5 bits | 8 bits |
| 尾数位 | 10 bits | 7 bits |
| 动态范围 | 较小 | 与 FP32 相同 |
| 精度 | 较高 | 较低 |
| 硬件支持 | V100+ | A100+, RTX 30+ |
| 稳定性 | 可能需要 loss scaling | 更稳定 |

### 6.2 配置选择建议

```python
# 检测硬件并选择最佳精度
def get_optimal_mixed_precision():
    """根据 GPU 选择最佳混合精度设置"""
    if not torch.cuda.is_available():
        return "no"
    
    device_name = torch.cuda.get_device_name(0).lower()
    
    # A100, H100, RTX 30/40 系列支持 BF16
    if any(x in device_name for x in ["a100", "h100", "rtx 30", "rtx 40", "a6000"]):
        return "bf16"
    else:
        return "fp16"

# DeepSpeed 配置中的混合精度
{
    "fp16": {
        "enabled": true,           # 或 false
        "loss_scale": 0,            # 0 = 动态 loss scaling
        "loss_scale_window": 1000,
        "initial_scale_power": 16,  # 初始 scale = 2^16
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": true,           # A100/H100 推荐
    }
}
```

### 6.3 完整混合精度训练示例

```python
# mixed_precision_training.py
from transformers import TrainingArguments

# BF16 配置 (A100/H100)
training_args_bf16 = TrainingArguments(
    output_dir="./output",
    bf16=True,                    # 启用 BF16
    bf16_full_eval=True,          # 评估时也使用 BF16
    fp16=False,
    # ... 其他参数
)

# FP16 配置 (V100, RTX 20 系列)
training_args_fp16 = TrainingArguments(
    output_dir="./output",
    fp16=True,                    # 启用 FP16
    fp16_full_eval=True,
    fp16_backend="auto",          # 或 "amp", "apex"
    fp16_opt_level="O1",          # Apex 优化级别
    # ... 其他参数
)

# DeepSpeed 中的混合精度 (自动处理)
{
    "fp16": {
        "enabled": "auto",       # 让 Trainer 决定
    },
    "bf16": {
        "enabled": "auto",
    }
}
```

---

## 7. 完整训练启动脚本

### 7.1 启动脚本

```bash
#!/bin/bash
# launch_training.sh

# 环境设置
export WANDB_PROJECT="nanomind-pretraining"
export WANDB_RUN_NAME="nanomind-1b-v1"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 多 GPU 训练 (使用 Accelerate)
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 2 \
    scripts/train.py \
    --config configs/nanomind_1b.yaml

# 或使用 torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train.py \
    --config configs/nanomind_1b.yaml \
    --deepspeed configs/ds_config_zero2.json
```

### 7.2 YAML 配置文件

```yaml
# configs/nanomind_1b.yaml

# 模型配置
model:
  vocab_size: 32000
  hidden_size: 2048
  intermediate_size: 5504
  num_hidden_layers: 22
  num_attention_heads: 32
  num_key_value_heads: 4
  max_position_embeddings: 4096
  rms_norm_eps: 1.0e-6
  rope_theta: 10000.0
  attention_dropout: 0.0
  tie_word_embeddings: false

# 训练配置
training:
  output_dir: "./output/nanomind-1b"
  num_train_epochs: 1
  total_tokens: 20000000000  # 20B
  
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  
  learning_rate: 3.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.01
  lr_scheduler_type: "cosine"
  
  max_grad_norm: 1.0
  gradient_checkpointing: true
  
  # 混合精度
  fp16: true
  bf16: false
  
  # 日志与保存
  logging_steps: 100
  save_steps: 1000
  eval_steps: 500
  save_total_limit: 3
  
  # DeepSpeed
  deepspeed: "configs/ds_config_zero2.json"
  
  # WandB
  report_to: "wandb"
  run_name: "nanomind-1b-v1"

# 数据配置
data:
  train_file: "data/train.jsonl"
  validation_file: "data/val.jsonl"
  max_seq_length: 2048
  preprocessing_num_workers: 8

# Tokenizer 配置
tokenizer:
  name_or_path: "nanomind-tokenizer"
  padding_side: "right"
```

---

## 8. 快速参考

### 8.1 配置文件速查表

| 配置文件 | 用途 | 位置 |
|----------|------|------|
| `ds_config_zero2.json` | ZeRO-2 训练 | `configs/` |
| `ds_config_zero3.json` | ZeRO-3 训练 | `configs/` |
| `accelerate_config.yaml` | Accelerate 配置 | `configs/` |
| `nanomind_1b.yaml` | 主训练配置 | `configs/` |

### 8.2 关键命令

```bash
# 估计内存使用
deepspeed --num_gpus=2 scripts/train.py --estimate_memory

# 测试配置
python scripts/train.py --config configs/nanomind_1b.yaml --dry_run

# 单 GPU 测试
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/test.yaml

# 多 GPU 训练
accelerate launch --config_file configs/accelerate_config.yaml scripts/train.py

# 使用 DeepSpeed 启动
deepspeed --num_gpus=2 scripts/train.py --deepspeed configs/ds_config_zero2.json
```

### 8.3 常见问题

**Q: OOM 错误怎么办？**
- 启用 ZeRO-3
- 启用 gradient checkpointing
- 减小 batch size
- 使用 CPU offload

**Q: 训练不稳定？**
- 降低学习率
- 增加 warmup steps
- 启用 gradient clipping
- 切换到 BF16 (如果硬件支持)

**Q: 如何加速训练？**
- 增加 batch size (如果内存允许)
- 使用 flash attention
- 使用更快的数据加载器
- 考虑使用 deepspeed-inference

---

*文档生成时间: 2026-03-04*
*适用于: nanomind 项目 1B 参数模型预训练*
