# nanomind - 开发指南

## 项目概述
使用 Python 3.12 和 PyTorch 生态系统进行深度学习和大型语言模型（LLM）学习与试验。

**项目性质**: 学习与实验项目，优先保持灵活性，不强制使用传统的单元测试和集成测试。

## 快速开始

### 环境初始化
```bash
# 创建并激活 Conda 环境
conda create -n nanomind python=3.12 -y
conda activate nanomind

# 安装 uv（如果尚未安装）
conda install -c conda-forge uv -y

# 安装 CUDA（可选，用于 GPU 加速）
conda install -c nvidia cuda=12.8 -y

# 安装项目依赖
uv pip install -r requirements.txt
```

### 验证安装
```bash
# 基础验证
python main.py  # 应该输出 "Hello from nanomind!"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"  # 验证 PyTorch

# 完整环境检查
python experiments/exp_000_environment_check.py

# 同时输出到终端和日志文件
python experiments/exp_000_environment_check.py 2>&1 | tee outputs/logs/exp_000_environment_check.log
```

### 常用命令
```bash
# 添加新依赖
uv add <package_name> --no-sync
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt

# 运行代码
python main.py
python -m <module_name>

# 代码格式化和检查（可选）
uv add ruff black mypy --no-sync && uv pip compile pyproject.toml -o requirements.txt && uv pip install -r requirements.txt
black .        # 格式化
ruff check .    # 代码检查
mypy .         # 类型检查
```

---

## 代码风格

### 语言与框架
- **Python**: 3.12+
- **核心栈**: PyTorch, torchvision, transformers, datasets
- **包管理器**: uv（使用 `uv add <package> --no-sync`）

### 命名约定
- 函数/变量: `snake_case`（如 `train_model`, `batch_size`）
- 类: `PascalCase`（如 `TransformerModel`, `DataLoader`）
- 常量: `UPPER_SNAKE_CASE`（如 `MAX_EPOCHS`, `LEARNING_RATE`）
- 模块/文件: `snake_case.py`（如 `model.py`）

### 导入顺序
遵循 PEP 8 分组：
```python
# 1. 标准库
import os
from pathlib import Path

# 2. 第三方库
import torch
import torch.nn as nn
from transformers import AutoModel

# 3. 本地模块
from utils import load_config
from models import TransformerModel
```

### 类型提示
**所有函数签名都必须包含类型提示**：
```python
from typing import List, Optional, Dict, Any

def train_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float = 0.001,
) -> Dict[str, float]:
    """训练模型并返回指标。"""
```

### 错误处理
**永远不要静默地抑制异常**：
```python
# 正确：处理并重新抛出
try:
    model = AutoModel.from_pretrained(model_name)
except OSError as e:
    raise RuntimeError(f"无法加载模型 '{model_name}'") from e

# 错误：静默捕获
try:
    do_something()
except:
    pass
```

### 文档字符串
使用 Google 风格：
```python
def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """训练模型一个 epoch。

    Args:
        model: 要训练的神经网络模型。
        dataloader: 训练数据加载器。
        optimizer: 用于参数更新的优化器。

    Returns:
        该 epoch 的平均损失。
    """
```

---

## 项目结构

### 规划的结构（待创建）
```bash
mkdir -p models data training utils configs experiments outputs/{checkpoints,logs,results}
```

```
nanomind/
├── models/          # 模型架构定义
│   └── __init__.py
├── data/            # 数据加载和预处理
│   └── __init__.py
├── training/        # 训练循环和优化器
│   └── __init__.py
├── utils/           # 工具函数
│   └── __init__.py
├── configs/         # 实验配置（YAML/JSON）
├── experiments/     # 实验脚本和Jupyter notebooks
├── outputs/         # 模型检查点、日志、结果
│   ├── checkpoints/
│   ├── logs/
│   └── results/
├── main.py          # 入口点
└── tests/           # （可选）辅助代码测试
    └── __init__.py
```

---

## PyTorch/ML 最佳实践

### 设备处理
```python
import torch

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型和数据移动到设备
model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)
```

### 模型模式
```python
# 训练模式：启用 dropout、batchnorm 等
model.train()

# 评估模式：禁用 dropout、batchnorm 等
model.eval()
```

### 梯度控制
```python
# 训练时：计算梯度
with torch.set_grad_enabled(True):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()

# 推理时：禁用梯度计算（节省内存）
with torch.no_grad():
    outputs = model(inputs)
    predictions = outputs.argmax(dim=1)
```

### 随机种子（可复现性）
```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """设置所有随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在实验开始时调用
set_seed(42)
```

---

## 实验记录建议

### 环境检查
每个项目开始前运行环境检查（Exp 000）：
```bash
python experiments/exp_000_environment_check.py
```

记录信息包括：
- 系统配置（操作系统、CPU、内存、GPU）
- 软件版本（Python、PyTorch、CUDA、cuDNN）
- 功能验证（张量操作、矩阵运算、GPU 加速、自动求导）

### 实验日志
- **超参数**: 在实验脚本顶部或配置文件中记录所有超参数
- **结果**: 保存训练日志、模型性能指标、可视化图表
- **日志管理**:
  - 使用 `tee` 命令同时输出到终端和日志文件
  - 覆盖式保存：`command | tee log.log`
  - 追加式保存：`command | tee -a log.log`
- **可复现性**:
  - 固定所有随机种子（torch.manual_seed、numpy.random.seed、random.seed）
  - 锁定依赖版本（requirements.txt）
  - 记录环境信息（torch版本、CUDA版本、Python版本）
- **实验目录**: 为每个实验创建独立目录，包含配置、代码、输出

---

## Git 工作流程
- **提交信息**: Conventional Commits（feat:, fix:, docs:, exp:）
- **不要提交**: `.pyc`, `__pycache__/`, `.env`, `outputs/`, 大型数据文件
- **.gitignore**: 包含 `outputs/`, `*.pth`, `data/`, `*.log`

---

## 开发优先级
1. **代码清晰度** 优于优化
2. **可复现性**（种子设置、版本锁定）
3. **实验文档化**（记录目的、方法、结果）
4. **模块化** 以便于快速实验

---

## 快速参考
- **入口点**: `python main.py`
- **依赖**: `requirements.txt`
- **Python**: `3.12`（`.python-version`）
- **包管理**: `uv add <package> --no-sync`

---

## 开发建议

### 开始第一个实验
1. **创建项目结构**:
   ```bash
   mkdir -p models data training utils configs experiments outputs/{checkpoints,logs,results}
   ```

2. **创建简单模型** (`models/simple_model.py`):
   ```python
   import torch.nn as nn

   class SimpleModel(nn.Module):
       def __init__(self, input_size: int, hidden_size: int, output_size: int):
           super().__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.fc2 = nn.Linear(hidden_size, output_size)

       def forward(self, x):
           x = torch.relu(self.fc1(x))
           return self.fc2(x)
   ```

3. **创建实验脚本** (`experiments/exp_001_baseline.py`):
   ```python
   from models.simple_model import SimpleModel
   import torch
   from training.trainer import Trainer

   # 设置随机种子
   torch.manual_seed(42)

   # 超参数
   LEARNING_RATE = 0.001
   BATCH_SIZE = 32
   EPOCHS = 10

   # 创建模型
   model = SimpleModel(input_size=784, hidden_size=128, output_size=10)

   # 训练
   trainer = Trainer(model, lr=LEARNING_RATE, batch_size=BATCH_SIZE)
   trainer.train(epochs=EPOCHS)
   ```

### 实验管理
- 为每个实验创建独立的脚本文件（`exp_001_*.py`, `exp_002_*.py`）
- 在实验脚本顶部记录所有超参数和配置
- 将结果保存到 `outputs/` 对应的子目录
- 使用 Jupyter notebooks 进行数据探索和可视化
- 定期清理 `outputs/` 中的旧实验结果

### 调试技巧
- 使用 `torch.autograd.set_detect_anomaly(True)` 检测梯度问题
- 在 `torch.no_grad()` 块中验证模型推理
- 使用 `print` 或 `logging` 记录训练进度
- 使用 `tensorboard` 可视化训练过程（可选）

---

## 常见问题

### GPU 相关
```python
# 检查 CUDA 是否可用
print(torch.cuda.is_available())

# 查看 GPU 信息
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())
```

### 内存优化
- 使用 `.to(device)` 一次性移动数据到 GPU
- 训练循环中定期调用 `torch.cuda.empty_cache()`
- 使用混合精度训练 (`torch.cuda.amp`) 减少内存占用
- 适当减小 `batch_size`

### 性能优化
- 使用 `torch.utils.data.DataLoader` 的 `num_workers` 参数加速数据加载
- 使用 `pin_memory=True` 加速 CPU 到 GPU 的数据传输
- 避免在训练循环中进行不必要的 CPU 操作
- 使用 `torch.jit.script` 或 `torch.compile` 优化模型
