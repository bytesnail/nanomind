# 实验管理

## 概述

本指南介绍如何管理和记录实验，包括实验追踪、实验清理和最佳实践。

---

## 实验记录

### 超参数记录

在实验脚本顶部记录所有超参数：

```python
"""实验 001: 基线模型

目的: 建立性能基线
超参数:
  - learning_rate: 0.001
  - batch_size: 32
  - epochs: 10
  - optimizer: Adam
  - model:
    - input_size: 784
    - hidden_size: 128
    - output_size: 10
"""

import torch
import torch.nn as nn
from typing import Dict, Any

# 超参数配置
CONFIG: Dict[str, Any] = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
    'optimizer': 'Adam',
    'model': {
        'input_size': 784,
        'hidden_size': 128,
        'output_size': 10,
    }
}
```

### 结果保存

```python
# 保存训练指标
metrics = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'config': CONFIG,
}

torch.save(metrics, f'outputs/results/exp_001/metrics.pth')

# 保存模型
torch.save(model.state_dict(), f'outputs/checkpoints/exp_001/model.pth')
```

### 日志管理

```bash
# 运行实验并输出到日志文件
python -m experiments.001 2>&1 | tee outputs/logs/exp_001.log

# 查看日志
tail -f outputs/logs/exp_001.log

# 搜索日志中的错误
grep -i "error" outputs/logs/exp_001.log
```

---

## 实验追踪

### 使用日志文件（推荐）

```bash
# 运行实验并输出到日志文件
python -m experiments.001 2>&1 | tee outputs/logs/exp_001.log

# 查看日志
tail -f outputs/logs/exp_001.log

# 搜索日志中的错误
grep -i "error" outputs/logs/exp_001.log
```

### 使用 TensorBoard（可选，未实现）

> **注意**：当前项目尚未实现 TensorBoard 集成。如需添加此功能，请参考以下示例：

```bash
# 安装 TensorBoard
pip install tensorboard

# 启动 TensorBoard
tensorboard --logdir=outputs/logs

# 在浏览器中访问 http://localhost:6006
```

```python
from torch.utils.tensorboard import SummaryWriter

# 创建日志记录器
writer = SummaryWriter('outputs/logs/exp_001')

# 记录超参数
writer.add_hparams(CONFIG, {})

# 记录标量
for epoch in range(CONFIG['epochs']):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

writer.close()
```

### 使用 Weights & Biases（可选，未实现）

> **注意**：当前项目尚未实现 W&B 集成。如需添加此功能，请参考以下示例：

```bash
# 安装 wandb
pip install wandb

# 初始化 wandb
wandb init --project nanomind
```

```python
import wandb

# 初始化 wandb
wandb.init(
    project="nanomind",
    config=CONFIG,
    name="exp_001"
)

# 记录指标
wandb.log({'loss': loss, 'accuracy': acc})

# 完成实验
wandb.finish()
```

---

## 实验对比

### 实验对比表格

创建 `outputs/results/experiments_comparison.md`：

```markdown
# 实验对比

| 实验 | 目的 | 学习率 | 批量大小 | 训练轮数 | 最终损失 | 最终准确率 | 备注 |
|-----|------|--------|----------|----------|----------|------------|------|
| exp_001 | 基线 | 0.001 | 32 | 10 | 0.1234 | 0.9567 | 基线性能 |
| exp_002 | 调参 | 0.0005 | 32 | 10 | 0.1156 | 0.9623 | 降低学习率 |
| exp_003 | 优化 | 0.001 | 64 | 10 | 0.0987 | 0.9745 | 增大 batch_size |
| exp_004 | 模型 | 0.001 | 32 | 20 | 0.0876 | 0.9812 | 增加训练轮数 |
```

### 对比脚本

```python
import torch

def compare_experiments(exp_ids: list) -> None:
    """对比多个实验的结果。"""
    print("=" * 80)
    print("实验对比")
    print("=" * 80)

    for exp_id in exp_ids:
        # 加载实验结果
        metrics = torch.load(f'outputs/results/exp_{exp_id:03d}/metrics.pth')

        config = metrics['config']
        final_loss = metrics['train_losses'][-1]
        final_acc = metrics['train_accuracies'][-1]

        print(f"exp_{exp_id:03d}:")
        print(f"  学习率: {config['learning_rate']}")
        print(f"  批量大小: {config['batch_size']}")
        print(f"  训练轮数: {config['epochs']}")
        print(f"  最终损失: {final_loss:.4f}")
        print(f"  最终准确率: {final_acc:.4f}")
        print()

# 对比实验 001-004
compare_experiments([1, 2, 3, 4])
```

---

## 实验清理

### 清理旧实验结果

```bash
# 删除超过 30 天的旧实验结果
find outputs/ -type d -mtime +30 -name "exp_*" -exec rm -rf {} \;

# 删除超过 30 天的日志文件
find outputs/logs/ -type f -mtime +30 -name "*.log" -delete

# 删除超过 30 天的检查点
find outputs/checkpoints/ -type f -mtime +30 -name "*.pth" -delete
```

### 清理策略

建议定期清理实验结果：

```bash
# 每周清理一次
# 删除除最新 5 个实验外的所有实验

# 保留实验清单
# exp_001: 基线（保留）
# exp_002: 调参（保留）
# exp_003: 优化（保留）
# exp_004: 模型（保留）
# exp_005: 最新（保留）
# exp_006-010: 删除（实验失败或效果不佳）
```

---

## 可选的实验追踪工具

### Weights & Biases

**优点**:
- 强大的可视化
- 易于共享实验
- 自动记录超参数
- 团队协作功能

**安装**:
```bash
pip install wandb
```

**使用**:
```python
import wandb

# 初始化
wandb.init(project="nanomind", config=CONFIG, name="exp_001")

# 记录指标
wandb.log({'loss': loss, 'accuracy': acc})

# 完成实验
wandb.finish()
```

### MLflow

**优点**:
- 开源
- 支持多种机器学习框架
- 模型版本管理
- 集成性强

**安装**:
```bash
pip install mlflow
```

**使用**:
```python
import mlflow

# 开始实验
mlflow.start_run()

# 记录参数
mlflow.log_params(CONFIG)

# 记录指标
mlflow.log_metric("loss", loss)
mlflow.log_metric("accuracy", acc)

# 保存模型
mlflow.pytorch.log_model(model, "model")

# 结束实验
mlflow.end_run()
```

---

## 最佳实践

### 1. 实验命名规范

```
exp_001_baseline.py          # 基线实验
exp_002_tune_lr.py           # 调整学习率
exp_003_increase_hidden.py   # 增加隐藏层
exp_004_dropout.py           # 添加 Dropout
exp_005_early_stopping.py    # 早停
exp_006_data_augmentation.py # 数据增强
```

### 2. 配置管理

**当前项目使用 argparse + dataclass 配置管理**：

```python
# experiments/001/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetConfig:
    """数据集配置类。"""
    name: str
    path: str
    text_key: str = "text"
    id_key: Optional[str] = "id"
    score_field: Optional[str] = "score"
    glob_pattern: str = "**/*.parquet"

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "dataset_name": DatasetConfig(
        name="dataset_name",
        path="data/datasets/dataset_name/",
        # ...
    ),
}
```

**如需使用 YAML 配置（可选）**：

```python
# config/exp_002.yaml
learning_rate: 0.001
batch_size: 32
epochs: 10
optimizer: Adam
```

```python
# 加载 YAML 配置
import yaml

with open('config/exp_002.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

### 3. 版本控制

```bash
# 提交实验代码
git add experiments/001/
git commit -m "exp: 添加实验 001 - 数据集统计"

# 提交配置文件（如使用 YAML）
# git add config/exp_002.yaml
# git commit -m "config: 添加实验 002 配置"

# 不要提交结果文件和检查点
# 在 .gitignore 中添加:
# outputs/
# *.pth
# *.log
```

### 4. 实验文档

当前项目的实验记录：
- `exp_000`: 环境验证（参见 [docs/environment/verification.md](../environment/verification.md)）
- `exp_001`: 数据集统计（参见 [docs/experiments/exp-001-overview.md](exp-001-overview.md)）
  - FineWeb-Edu 详细分析（参见 [docs/experiments/fineweb_stats.md](fineweb_stats.md)）

**建议创建** `experiments/README.md` 记录实验历史：

```markdown
# 实验历史

## 2026-01-31

### exp_000: 环境验证
- 目的: 验证项目环境配置
- 验证项: Python, PyTorch, CUDA, 数据加载
- 状态: ✅ 通过

### exp_001: 数据集统计
- 目的: 分析数据集的统计信息
- 数据集: FineWeb-Edu, FineMath, GitHub Code 2025
- 结果: 详见 outputs/results/exp_001/
```

**可选**：使用 Weights & Biases 或其他工具进行在线实验追踪（需要实现）

---

## 实验检查清单

开始实验前检查：

- [ ] 已记录所有超参数
- [ ] 已设置随机种子
- [ ] 已创建输出目录
- [ ] 已实现日志记录
- [ ] 已设置模型保存点
- [ ] 已实现评估指标
- [ ] 已运行环境检查

实验完成后检查：

- [ ] 已保存所有指标
- [ ] 已保存模型检查点
- [ ] 已更新实验对比表
- [ ] 已更新实验历史文档
- [ ] 已清理临时文件
- [ ] 已提交代码到 Git

---

## 下一步

- [开始实验](getting-started.md) - 创建您的第一个实验
- [项目结构](project-structure.md) - 目录规范和代码组织

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
