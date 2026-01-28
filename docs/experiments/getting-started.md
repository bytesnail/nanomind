# 开始第一个实验

## 概述

本指南介绍如何创建和运行您的第一个实验，包括实验模板、常用超参数和示例代码。

---

## 创建项目结构

```bash
# 创建必要的目录
mkdir -p models data training utils configs experiments outputs/{checkpoints,logs,results}
```

---

## 完整实验脚本模板

```python
"""实验 001: 基线模型

目的: 建立性能基线
超参数:
  - learning_rate: 0.001
  - batch_size: 32
  - epochs: 10
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

from models import SimpleModel
from training import Trainer
from utils import set_seed

# 设置随机种子
set_seed(42)

# 超参数配置
CONFIG: Dict[str, Any] = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
    'model': {
        'input_size': 784,
        'hidden_size': 128,
        'output_size': 10,
    }
}

def main() -> None:
    """运行实验。"""
    # 打印配置
    print("=" * 50)
    print("实验 001: 基线模型")
    print("=" * 50)
    print(f"学习率: {CONFIG['learning_rate']}")
    print(f"批量大小: {CONFIG['batch_size']}")
    print(f"训练轮数: {CONFIG['epochs']}")
    print()

    # 创建模型
    print("创建模型...")
    model = SimpleModel(**CONFIG['model'])

    # 创建训练器
    print("创建训练器...")
    trainer = Trainer(model, learning_rate=CONFIG['learning_rate'])

    # 训练
    print("开始训练...")
    metrics = trainer.train(
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
    )

    # 保存结果
    print("保存结果...")
    torch.save(metrics, 'outputs/results/exp_001_metrics.pth')

    # 打印最终结果
    print("=" * 50)
    print("实验完成!")
    print(f"最终损失: {metrics['loss'][-1]:.4f}")
    print(f"最终准确率: {metrics['accuracy'][-1]:.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
```

---

## Jupyter Notebook 模板

### Cell 1: 导入和配置

```python
# 导入必要的库
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
from utils import set_seed
set_seed(42)

# 超参数配置
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10
```

### Cell 2: 加载数据

```python
from torch.utils.data import DataLoader, TensorDataset

# 创建模拟数据
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
```

### Cell 3: 定义模型

```python
from models import SimpleModel

# 创建模型
model = SimpleModel(input_size=784, hidden_size=128, output_size=10)

# 打印模型结构
print(model)
```

### Cell 4: 训练

```python
import torch.optim as optim

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
train_losses = []

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    model.train()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # 记录平均损失
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
```

### Cell 5: 评估和可视化

```python
# 绘制训练曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 常用超参数范围

### 学习率（Learning Rate）

```python
# 常用范围: 1e-4 到 1e-2
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# 推荐
learning_rate = 0.001  # 1e-3

# 调试策略
# - 损失不下降: 增大学习率
# - 损失震荡: 减小学习率
```

### 批量大小（Batch Size）

```python
# 常用范围: 16 到 256
batch_sizes = [16, 32, 64, 128, 256]

# 推荐（取决于 GPU 内存）
batch_size = 32

# 调试策略
# - CUDA out of memory: 减小 batch_size
# - 训练太慢: 增大 batch_size
```

### 训练轮数（Epochs）

```python
# 常用范围: 10 到 100
epochs_list = [10, 20, 50, 100]

# 推荐
epochs = 10

# 调试策略
# - 过拟合（训练损失下降，验证损失上升）: 减小 epochs
# - 欠拟合（训练和验证损失都很高）: 增大 epochs
```

### 优化器

```python
# 常用优化器
import torch.optim as optim

# Adam（推荐用于大多数情况）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# SGD（推荐用于需要精细调整的情况）
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```

---

## 实验命名规范

### 文件命名

```
exp_001_baseline.py          # 基线实验
exp_002_tune_lr.py           # 调整学习率
exp_003_increase_hidden.py   # 增加隐藏层
exp_004_dropout.py           # 添加 Dropout
exp_005_early_stopping.py    # 早停
```

### 目录命名

```
outputs/
├── checkpoints/
│   ├── exp_001_baseline.pth
│   ├── exp_002_tune_lr.pth
│   └── exp_003_increase_hidden.pth
├── logs/
│   ├── exp_001_baseline.log
│   ├── exp_002_tune_lr.log
│   └── exp_003_increase_hidden.log
└── results/
    ├── exp_001_baseline/
    ├── exp_002_tune_lr/
    └── exp_003_increase_hidden/
```

---

## 运行实验

### 命令行运行

```bash
# 运行单个实验
python experiments/exp_001_baseline.py

# 运行实验并输出到日志文件
python experiments/exp_001_baseline.py 2>&1 | tee outputs/logs/exp_001_baseline.log

# 后台运行
nohup python experiments/exp_001_baseline.py > outputs/logs/exp_001_baseline.log 2>&1 &
```

### Jupyter Notebook 运行

```bash
# 启动 Jupyter Notebook
jupyter notebook

# 在浏览器中打开 notebook
# 运行所有 cells: Cell -> Run All
```

---

## 实验清单

开始实验前检查：

- [ ] 已设置随机种子
- [ ] 已记录所有超参数
- [ ] 已创建输出目录
- [ ] 已实现日志记录
- [ ] 已设置模型保存点
- [ ] 已实现评估指标
- [ ] 已运行环境检查

---

## 下一步

- [实验管理](management.md) - 实验记录、追踪、对比
- [项目结构](project-structure.md) - 目录规范和代码组织

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
