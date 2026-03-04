# Accelerate 配置文件目录

此目录包含 HuggingFace Accelerate 的配置文件。

## 文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| deepspeed_zero2.yaml | DeepSpeed ZeRO-2 配置 | 🚧 规划中 |
| deepspeed_zero3.yaml | DeepSpeed ZeRO-3 配置 | 🚧 规划中 |

## 生成配置

```bash
# 交互式生成
accelerate config

# 或复制示例
accelerate config default > config/accelerate/default.yaml
```

## 使用方式

```bash
accelerate launch \
  --config_file config/accelerate/deepspeed_zero2.yaml \
  scripts/train_nanomind.py \
  --config config/training/phase1_short.yaml
```

## 配置示例 (deepspeed_zero2.yaml)

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
gpu_ids: 0,1
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
deepspeed_config:
  gradient_accumulation_steps: 16
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  zero3_init_flag: false
  zero_stage: 2
```
