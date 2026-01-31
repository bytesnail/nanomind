# 项目结构与目录规范

## 概述

本指南介绍 nanomind 项目的实际目录结构、模块导入规范、配置管理和代码组织原则。

---

## 当前项目结构

```
nanomind/
├── README.md                          # 项目简介和快速开始
├── AGENTS.md                          # 开发指南导航
├── main.py                           # 项目入口（当前仅打印信息）
├── pyproject.toml                    # 项目配置文件
├── requirements.txt                   # 依赖列表
├── .python-version                    # Python 版本
├── .gitignore                        # Git 忽略文件
│
├── experiments/                      # 实验目录（核心）
│   ├── 000/                         # 环境验证实验
│   │   ├── exp_000_environment_check.py
│   │   ├── system_info.py
│   │   ├── torch_info.py
│   │   ├── __init__.py
│   │   └── __main__.py
│   ├── 001/                         # 数据集统计实验
│   │   ├── exp_001_datasets_stats.py
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── pipeline.py
│   │   ├── collector.py
│   │   ├── stats_utils.py
│   │   ├── io_utils.py
│   │   ├── __init__.py
│   │   └── __main__.py
│   └── utils/                       # 实验工具模块
│       ├── __init__.py
│       ├── common.py
│       ├── paths.py
│       └── constants.py
│
├── data/                             # 数据目录
│   └── datasets/                    # 数据集存储
│
├── outputs/                          # 实验输出
│   ├── checkpoints/                  # 模型检查点（如适用）
│   ├── logs/                        # 训练日志
│   └── results/                     # 实验结果
│       ├── exp_001_datasets_stats/
│       └── ...
│
├── docs/                             # 详细文档
│   ├── environment/                  # 环境管理
│   │   ├── setup.md
│   │   ├── dependencies.md
│   │   ├── verification.md
│   │   └── specs.md
│   ├── development/                  # 开发规范
│   │   ├── code-style.md
│   │   ├── best-practices.md
│   │   ├── debugging.md
│   │   └── git-workflow.md
│   └── experiments/                  # 实验管理
│       ├── getting-started.md
│       ├── management.md
│       ├── project-structure.md
│       └── fineweb_stats.md
│
├── .sisyphus/                       # 内部工作目录
│   ├── drafts/                       # 草稿文件
│   ├── plans/                        # 计划文档
│   └── developer-notes/              # 开发者笔记
│
├── .venv/                           # Python 虚拟环境（已忽略）
├── .pytest_cache/                    # Pytest 缓存（已忽略）
└── .ruff_cache/                     # Ruff 缓存（已忽略）
```

---

## 实验目录结构详解

### experiments/000/ - 环境验证实验

```
000/
├── exp_000_environment_check.py    # 主脚本：环境检查
├── system_info.py                   # 系统信息收集
├── torch_info.py                   # PyTorch 信息和测试
├── __init__.py
└── __main__.py                    # 模块入口
```

**运行方式**：
```bash
python -m experiments.000
```

### experiments/001/ - 数据集统计实验

```
001/
├── exp_001_datasets_stats.py       # 主脚本：数据集统计
├── cli.py                         # 命令行接口
├── config.py                      # 配置管理（dataclass）
├── pipeline.py                    # Datatrove 流水线
├── collector.py                   # 统计收集器
├── stats_utils.py                 # 统计工具
├── io_utils.py                   # I/O 工具
├── __init__.py
└── __main__.py                    # 模块入口
```

**运行方式**：
```bash
python -m experiments.001 explore --dataset <name> --data-dir <path> --workers 8
```

### experiments/utils/ - 共享工具模块

```
utils/
├── __init__.py
├── common.py                      # 通用工具（日志、datatrove工具等）
├── paths.py                       # 路径处理
└── constants.py                  # 常量定义
```

---

## 模块导入规范

### 绝对导入 vs 相对导入

**推荐使用绝对导入**：

```python
# 推荐：绝对导入
from experiments.utils import setup_logger, setup_experiment_paths
from experiments.utils.paths import project_root

# 推荐：相对导入（在实验模块内）
from .config import DATASET_CONFIGS
from .pipeline import create_pipeline
```

### 导入顺序

遵循 PEP 8：

```python
# 1. 标准库
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# 2. 第三方库
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datatrove.pipeline.base import PipelineStep

# 3. 本地模块
from experiments.utils import setup_logging, setup_experiment_paths
from experiments.utils.paths import project_root
from .config import DATASET_CONFIGS
```

---

## 配置管理

### 使用 argparse + dataclass

本项目使用 `argparse` 和 `dataclass` 组合进行配置管理，不使用 YAML 配置文件。

**配置示例**（experiments/001/config.py）：

```python
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
```

**命令行接口**（experiments/001/cli.py）：

```python
import argparse

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="多数据集统计与探索",
    )
    parser.add_argument("--dataset", type=str, nargs="+", required=True,
                       help="数据集名称")
    parser.add_argument("--output-dir", type=str, default="outputs/exp_001",
                       help="输出目录")
    parser.add_argument("--workers", type=int, default=8,
                       help="Worker 数量")
    return parser
```

---

## 代码组织原则

### 单一职责原则

每个实验目录应该有单一、明确的职责：

```
experiments/
├── 000/               # 环境验证职责
│   ├── exp_000...     # 主实验脚本
│   ├── system_info.py # 系统信息模块
│   └── torch_info.py  # PyTorch 信息模块
│
└── 001/               # 数据统计职责
    ├── exp_001...     # 主实验脚本
    ├── cli.py         # CLI 模块
    ├── config.py      # 配置模块
    ├── pipeline.py    # 流水线模块
    ├── collector.py   # 收集器模块
    └── stats_utils.py # 统计工具模块
```

### 工具模块复用

将常用工具函数放在 `experiments/utils/` 目录：

 ```python
 # experiments/utils/common.py
 import logging
 from pathlib import Path
 from datatrove.pipeline.base import PipelineStep, DocumentsPipeline
 from typing import List, Optional, Dict, Any

 def setup_logging(exp_name: str, log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
     """设置日志记录器。"""
     logger = logging.getLogger(exp_name)
     logger.setLevel(getattr(logging, log_level.upper()))
     # ... handler 设置
     return logger

 def setup_experiment_paths(script_path: str) -> Dict[str, Path]:
     """设置实验路径。"""
     project_root = Path(script_path).parent.parent.parent
     # ... 路径计算
     return paths
 ```

---

## .gitignore 规则

### 当前 .gitignore 内容

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
.venv/
venv/
env/
ENV/

# 输出文件
outputs/
*.pth
*.pt
*.log

# 数据文件
data/datasets/

# 模型缓存
.cache/
*.ckpt
*.safetensors

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db

# 内部目录
.sisyphus/
```

---

## 常见目录操作

### 创建新实验

```bash
# 1. 创建实验目录
mkdir -p experiments/002

# 2. 创建基本文件
touch experiments/002/__init__.py
touch experiments/002/__main__.py
touch experiments/002/exp_002_experiment.py

# 3. 复制工具模块（如需要）
cp experiments/001/cli.py experiments/002/
```

### 初始化实验

```bash
# 在 experiments/002/__main__.py 中添加：
from exp_002_experiment import main

if __name__ == "__main__":
    main()
```

---

## 下一步

- [开始实验](getting-started.md) - 创建您的第一个实验
- [实验管理](management.md) - 实验记录、追踪、对比

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
