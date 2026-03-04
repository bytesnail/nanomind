# 训练配置文件目录

此目录包含预训练各阶段的配置文件。

## 文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| base.yaml | 基础训练配置 | 🚧 规划中 |
| phase1_short.yaml | Phase 1: ≤512 tokens | 🚧 规划中 |
| phase2_medium.yaml | Phase 2: ≤1024 tokens | 🚧 规划中 |
| phase3_long.yaml | Phase 3: ≤4096 tokens | 🚧 规划中 |

## 配置说明

### base.yaml

包含所有阶段共享的基础训练参数：
- 优化器设置 (AdamW, lr, weight_decay)
- 学习率调度 (cosine, warmup)
- 精度设置 (fp16/bf16)
- 日志和保存频率

### phaseX_xxx.yaml

各阶段特定配置：
- 数据路径
- 序列长度
- 训练步数/epochs
- batch size

## 配置继承

```yaml
# phase1_short.yaml
extends: "base.yaml"

training:
  max_seq_length: 512
  data_path: "data/datasets/pretrain/short"
  num_train_epochs: 1
```
