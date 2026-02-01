# 代码风格

## 概述

本指南介绍 nanomind 项目的代码风格规范，确保代码清晰、一致且易于维护。

---

## 语言与框架

- **Python**: 3.12+
- **核心栈**: PyTorch, torchvision, transformers, datasets
- **包管理器**: uv（使用 `uv add <package> --no-sync`）

---

## 命名约定

### 变量和函数
使用 `snake_case`（小写字母和下划线）：

```python
# 函数
def train_model(model, data_loader, epochs):
    pass

def load_config(config_path):
    pass

# 变量
learning_rate = 0.001
batch_size = 32
num_epochs = 10
```

### 类
使用 `PascalCase`（大驼峰命名）：

```python
class TransformerModel(nn.Module):
    pass

class DataLoader:
    pass

class SimpleLinearLayer:
    pass
```

### 常量
使用 `UPPER_SNAKE_CASE`（全大写字母和下划线）：

```python
MAX_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001
BATCH_SIZE = 32
CUDA_DEVICE = "cuda:0"
```

### 模块和文件
使用 `snake_case.py`：

```
models/
    transformer_model.py
    data_loader.py
utils/
    config_loader.py
    training_utils.py
```

---

## 导入顺序

遵循 PEP 8，按以下顺序导入：

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
from transformers import AutoModel, AutoTokenizer

# 3. 本地模块
from utils import load_config, set_seed
from models import TransformerModel, SimpleLinearLayer
from data import DataLoader
```

### 导入规范

- 每个导入语句占一行
- 使用绝对导入而非相对导入
- 避免使用 `import *`

```python
# 正确
import torch
import numpy as np

# 错误
import torch, numpy as np
```

---

## 类型提示

**所有函数签名都必须包含类型提示**：

```python
from typing import List, Optional, Dict, Any, Tuple

def train_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    learning_rate: float = 0.001,
) -> Dict[str, float]:
    """训练模型并返回指标。"""
    pass

def load_config(
    config_path: str,
    default_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """加载配置文件。"""
    pass

def predict(
    model: nn.Module,
    inputs: torch.Tensor,
    return_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """进行预测。"""
    pass
```

### 类型提示的好处

- 提高代码可读性
- 支持 IDE 自动补全
- 便于代码审查
- 提前发现类型错误

---

## 错误处理

**永远不要静默地抑制异常**：

```python
# 正确：处理并重新抛出
try:
    model = AutoModel.from_pretrained(model_name)
except OSError as e:
    raise RuntimeError(f"无法加载模型 '{model_name}'") from e

# 正确：记录并继续
try:
    data = load_data(path)
except FileNotFoundError as e:
    logger.warning(f"数据文件未找到: {e}")
    data = None

# 错误：静默捕获
try:
    do_something()
except:
    pass
```

### 异常处理最佳实践

- 使用具体的异常类型（如 `ValueError`, `KeyError`）
- 提供有用的错误信息
- 使用 `from e` 保留原始异常信息
- 考虑使用 `logging` 模块记录错误

```python
import logging

logger = logging.getLogger(__name__)

def load_model(model_name: str) -> nn.Module:
    """加载模型。"""
    try:
        model = AutoModel.from_pretrained(model_name)
        return model
    except OSError as e:
        logger.error(f"模型加载失败: {model_name}")
        raise RuntimeError(f"无法加载模型 '{model_name}'") from e
```

---

## 文档字符串

使用 Google 风格的文档字符串：

```python
def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
) -> float:
    """训练模型一个 epoch。

    Args:
        model: 要训练的神经网络模型。
        dataloader: 训练数据加载器。
        optimizer: 用于参数更新的优化器。
        device: 设备类型（"cuda" 或 "cpu"）。

    Returns:
        该 epoch 的平均损失。

    Raises:
        RuntimeError: 如果 GPU 不可用且 device="cuda"。

    Examples:
        >>> model = SimpleModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> loss = train_epoch(model, dataloader, optimizer)
        >>> print(f"平均损失: {loss:.4f}")
    """
    pass
```

### 文档字符串要点

- 简洁地描述函数功能
- 说明参数类型和用途
- 说明返回值类型
- 列出可能抛出的异常
- 提供使用示例（可选）

---

## PyTorch 特定规范

### 设备处理

⚠️ **注意**: `get_device()` 是推荐的最佳实践模式，当前代码库中尚未实现。请使用 `torch.device('cuda')` 直接管理设备。

```python
# 推荐：使用函数自动选择设备
def get_device() -> torch.device:
    """自动选择最佳设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
model = model.to(device)
inputs = inputs.to(device)
```

### 模型模式

```python
# 训练时
model.train()

# 评估时
model.eval()

# 推理时
with torch.no_grad():
    outputs = model(inputs)
```

### 梯度控制

```python
# 训练时：计算梯度
with torch.set_grad_enabled(True):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 推理时：禁用梯度计算（节省内存）
with torch.no_grad():
    outputs = model(inputs)
    predictions = outputs.argmax(dim=1)
```

---

## Python 3.12 特性使用

本项目使用 Python 3.12，可以使用以下新特性：

### 类型提示改进

```python
from typing import override

class BaseModel:
    def train(self) -> None:
        pass

class MyModel(BaseModel):
    @override
    def train(self) -> None:
        # 实现
        pass
```

### 性能优化

```python
# 使用 list.append() 的快速路径（PEP 688）
# 字符串转译速度提升
# 优化的 asyncio
```

**注意**: 为了保持代码可移植性，尽量避免使用过于新颖的特性。

---

## 代码格式化工具

### Black（代码格式化）

```bash
# 安装
uv add black --no-sync

# 格式化当前目录
black .

# 格式化特定文件
black file.py

# 检查（不修改）
black --check .
```

### Ruff（代码检查）

```bash
# 安装
uv add ruff --no-sync

# 检查当前目录
ruff check .

# 自动修复
ruff check --fix .

# 检查特定文件
ruff check file.py
```

### MyPy（类型检查）

```bash
# 安装
uv add mypy --no-sync

# 类型检查
mypy .

# 检查特定文件
mypy file.py
```

### 集成使用

```bash
# 安装所有工具
uv add ruff black mypy --no-sync
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt

# 格式化
black .
ruff check --fix .

# 类型检查
mypy .
```

---

## 常见反模式

### 避免硬编码

```python
# 错误
def train_model(model, data):
    for i in range(100):  # 硬编码
        pass

# 正确
def train_model(model, data, num_epochs: int = 100):
    for i in range(num_epochs):
        pass
```

### 避免魔法数字

```python
# 错误
def layer_sizes(input_dim):
    hidden_dim = input_dim * 2  # 魔法数字
    output_dim = input_dim // 2  # 魔法数字
    return hidden_dim, output_dim

# 正确
HIDDEN_SCALE = 2
OUTPUT_SCALE = 0.5

def layer_sizes(input_dim):
    hidden_dim = int(input_dim * HIDDEN_SCALE)
    output_dim = int(input_dim * OUTPUT_SCALE)
    return hidden_dim, output_dim
```

### 避免过长函数

```python
# 错误：函数过长
def train_model(model, data, config):
    # 100+ 行代码
    pass

# 正确：拆分为多个函数
def train_model(model, data, config):
    train_loader = create_train_loader(data, config)
    optimizer = create_optimizer(model, config)
    for epoch in range(config['num_epochs']):
        train_one_epoch(model, train_loader, optimizer)
    validate(model, config['val_loader'])
```

---

## 代码审查检查表

提交代码前检查：

- [ ] 所有函数都有类型提示
- [ ] 所有函数都有文档字符串
- [ ] 没有硬编码的魔法数字
- [ ] 没有静默捕获的异常
- [ ] 导入顺序符合 PEP 8
- [ ] 变量命名规范一致
- [ ] 使用了 Black 格式化
- [ ] Ruff 检查通过
- [ ] MyPy 类型检查通过（如果启用）

---

## 下一步

- [最佳实践](best-practices.md) - PyTorch 开发规范
- [调试技巧](debugging.md) - 常见问题与解决方案

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
