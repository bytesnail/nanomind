# 项目结构与目录规范

## 概述

本指南介绍 nanomind 项目的目录结构、模块导入规范、配置管理和代码组织原则。

---

## 标准项目结构

```
nanomind/
├── README.md                 # 项目简介和快速开始
├── AGENTS.md                 # 开发指南导航
├── main.py                   # 入口点
├── pyproject.toml            # 项目配置文件
├── requirements.txt          # 依赖列表
├── .python-version           # Python 版本（可选）
├── .gitignore               # Git 忽略文件
│
├── models/                  # 模型架构定义
│   ├── __init__.py
│   ├── base_model.py        # 基础模型类
│   ├── simple_model.py      # 简单模型
│   └── transformer_model.py # Transformer 模型
│
├── data/                    # 数据加载和预处理
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载器
│   ├── dataset.py           # 数据集类
│   └── preprocessing.py    # 数据预处理
│
├── training/                # 训练循环和优化器
│   ├── __init__.py
│   ├── trainer.py           # 训练器
│   ├── optimizer.py         # 优化器配置
│   └── scheduler.py         # 学习率调度器
│
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── config.py            # 配置加载
│   ├── metrics.py           # 评估指标
│   ├── seed.py              # 随机种子设置
│   └── logger.py            # 日志工具
│
├── configs/                 # 实验配置（YAML/JSON）
│   ├── exp_001.yaml         # 实验 001 配置
│   ├── exp_002.yaml         # 实验 002 配置
│   └── default.yaml        # 默认配置
│
├── experiments/             # 实验脚本和 Jupyter notebooks
│   ├── exp_000_environment_check.py  # 环境检查
│   ├── exp_001_baseline.py            # 实验 001
│   ├── exp_002_tune_lr.py             # 实验 002
│   └── notebooks/                      # Jupyter notebooks
│       ├── exp_001_baseline.ipynb
│       └── exp_002_tune_lr.ipynb
│
├── outputs/                 # 模型检查点、日志、结果
│   ├── checkpoints/        # 模型检查点
│   │   ├── exp_001/
│   │   │   └── model.pth
│   │   └── exp_002/
│   │       └── model.pth
│   ├── logs/               # 训练日志
│   │   ├── exp_001.log
│   │   └── exp_002.log
│   └── results/            # 实验结果
│       ├── exp_001/
│       │   ├── metrics.pth
│       │   ├── loss_curve.png
│       │   └── predictions.npy
│       └── exp_002/
│           ├── metrics.pth
│           ├── loss_curve.png
│           └── predictions.npy
│
├── docs/                    # 详细文档
│   ├── README.md           # 文档索引
│   ├── environment/
│   │   ├── setup.md       # 环境初始化
│   │   ├── dependencies.md # 依赖管理
│   │   └── verification.md # 环境验证
│   ├── development/
│   │   ├── code-style.md   # 代码风格
│   │   ├── best-practices.md # 最佳实践
│   │   └── debugging.md   # 调试技巧
│   └── experiments/
│       ├── getting-started.md # 开始实验
│       ├── management.md   # 实验管理
│       └── project-structure.md # 项目结构（本文档）
│
└── tests/                   # 测试文件（可选）
    ├── __init__.py
    ├── test_models.py
    ├── test_data.py
    └── test_training.py
```

---

## 模块导入规范

### 绝对导入 vs 相对导入

**推荐使用绝对导入**：

```python
# 推荐：绝对导入
from models import SimpleModel
from data import DataLoader
from training import Trainer
from utils import load_config, set_seed

# 不推荐：相对导入
from ..models import SimpleModel
from .utils import load_config
```

### 导入顺序

遵循 PEP 8：

```python
# 1. 标准库
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# 2. 第三方库
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel

# 3. 本地模块
from models import SimpleModel
from data import DataLoader
from training import Trainer
from utils import load_config
```

---

## 配置管理

### 使用 YAML 配置文件

创建 `configs/exp_001.yaml`：

```yaml
# 实验 001 配置

# 训练参数
learning_rate: 0.001
batch_size: 32
epochs: 10
optimizer: Adam

# 模型参数
model:
  input_size: 784
  hidden_size: 128
  output_size: 10

# 数据参数
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

# 输出参数
output:
  checkpoint_dir: outputs/checkpoints/exp_001
  log_dir: outputs/logs/exp_001.log
  result_dir: outputs/results/exp_001
```

### 加载配置

```python
# utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 使用配置
from utils import load_config

config = load_config('configs/exp_001.yaml')
learning_rate = config['learning_rate']
batch_size = config['batch_size']
```

---

## 代码组织原则

### 单一职责原则

每个模块应该有单一、明确的职责：

```python
# models/simple_model.py
"""定义简单的神经网络模型。"""
import torch.nn as nn

class SimpleModel(nn.Module):
    """简单的全连接神经网络。"""
    pass

# training/trainer.py
"""训练器类，负责训练循环。"""
class Trainer:
    """训练器。"""
    pass

# utils/metrics.py
"""评估指标计算函数。"""
def accuracy(predictions, labels):
    """计算准确率。"""
    pass
```

### 避免循环导入

```python
# 错误：循环导入
# models/__init__.py
from training import Trainer  # ❌

# training/__init__.py
from models import SimpleModel  # ❌

# 正确：使用类型提示（forward reference）
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training import Trainer

class SimpleModel(nn.Module):
    def __init__(self, trainer: 'Trainer'):
        pass
```

### 模块初始化

每个目录都应该包含 `__init__.py`：

```python
# models/__init__.py
"""模型模块。"""

from .base_model import BaseModel
from .simple_model import SimpleModel
from .transformer_model import TransformerModel

__all__ = ['BaseModel', 'SimpleModel', 'TransformerModel']
```

---

## .gitignore 规则

创建 `.gitignore` 文件：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# 虚拟环境
venv/
env/
ENV/

# 输出文件
outputs/
*.pth
*.pt
*.log

# 模型缓存
.cache/
*.ckpt

# 数据文件
data/
*.csv
*.json
*.parquet

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db
```

---

## 常见目录操作

### 创建目录结构

```bash
# 创建标准项目结构
mkdir -p models data training utils configs experiments outputs/{checkpoints,logs,results}

# 创建 __init__.py 文件
touch models/__init__.py
touch data/__init__.py
touch training/__init__.py
touch utils/__init__.py
touch tests/__init__.py
```

### 初始化项目

```bash
# 初始化 Git 仓库
git init

# 创建 .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
outputs/
*.pth
*.log
EOF

# 添加文件
git add .
git commit -m "Initial commit"
```

---

## 代码复用策略

### 工具函数

将常用工具函数放在 `utils/` 目录：

```python
# utils/seed.py
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """设置所有随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# utils/logger.py
import logging
from pathlib import Path

def setup_logger(name: str, log_file: str):
    """设置日志记录器。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
```

### 基础类

在 `models/base_model.py` 中定义基础模型类：

```python
import torch.nn as nn

class BaseModel(nn.Module):
    """所有模型的基类。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def get_num_params(self):
        """获取模型参数数量。"""
        return sum(p.numel() for p in self.parameters())
```

---

## 下一步

- [开始实验](getting-started.md) - 创建您的第一个实验
- [实验管理](management.md) - 实验记录、追踪、对比

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
