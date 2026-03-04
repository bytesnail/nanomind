# DeepSpeed 配置文件目录

此目录包含 DeepSpeed ZeRO 优化配置文件。

## 文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| zero2.json | ZeRO-2 配置（推荐，速度优先） | 🚧 规划中 |
| zero3.json | ZeRO-3 配置（显存受限） | 🚧 规划中 |

## 配置选择指南

| 场景 | 推荐配置 | 单卡显存 |
|------|----------|----------|
| 正常训练 | zero2.json | ~4-5GB |
| 显存受限 | zero3.json | ~2-3GB |
| 调试 | 无 DeepSpeed | ~16GB |

## 使用方式

```bash
# 方式1: 命令行指定
deepspeed --num_gpus=2 scripts/train_nanomind.py --deepspeed config/deepspeed/zero2.json

# 方式2: Accelerate 配置
accelerate launch --config_file config/accelerate/deepspeed_zero2.yaml scripts/train_nanomind.py
```

## ZeRO-2 配置说明

- `offload_optimizer`: 将优化器状态卸载到 CPU
- `allgather_bucket_size`: 通信桶大小 (2e8 = 200MB)
- `overlap_comm`: 通信与计算重叠

## ZeRO-3 配置说明

- `offload_param`: 将模型参数卸载到 CPU
- `stage3_max_live_parameters`: 最大驻留参数数
- `gather_16bit_weights_on_model_save`: 保存时收集权重
