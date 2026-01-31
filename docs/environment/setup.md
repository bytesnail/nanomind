# 环境初始化

## 概述

本指南介绍如何为 nanomind 项目配置开发环境，包括 Conda 环境、uv 包管理器和 CUDA 支持。

---

## 快速开始

```bash
# 1. 创建并激活 Conda 环境
conda create -n nanomind python=3.12 -y
conda activate nanomind

# 2. 安装 uv（如果尚未安装）
conda install -c conda-forge uv -y

# 3. 初始化 uv 项目
uv init

# 4. 安装 CUDA（可选，用于 GPU 加速）
conda install -c nvidia cuda=12.8 -y

# 5. 添加项目依赖
uv add torch torchvision transformers datasets --no-sync

# 6. 生成 requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# 7. 安装到 Conda 环境
uv pip install -r requirements.txt
```

---

## 详细步骤

### 1. 创建 Conda 环境

```bash
# 创建 Python 3.12 环境
conda create -n nanomind python=3.12 -y

# 激活环境
conda activate nanomind

# 验证 Python 版本
python --version  # 应该输出 Python 3.12.12
```

### 2. 安装 uv

uv 是一个快速的 Python 包管理器，用于管理项目依赖。

```bash
# 安装 uv
conda install -c conda-forge uv -y

# 验证安装
uv --version
```

### 3. 初始化 uv 项目

```bash
# 在项目根目录运行
uv init

# 这会创建：
# - pyproject.toml  # 项目配置文件
# - uv.lock         # 依赖锁定文件
```

### 4. 安装 CUDA（可选）

如果需要 GPU 加速，请安装 CUDA 12.8。

```bash
# 安装 CUDA
conda install -c nvidia cuda=12.8 -y

# 验证 CUDA 安装
nvcc --version
```

### 5. 添加项目依赖

```bash
# 添加核心依赖（不自动安装）
uv add torch torchvision transformers datasets --no-sync

# 常用开发工具（可选）
uv add ruff black mypy --no-sync
```

### 6. 生成 requirements.txt

```bash
# 从 pyproject.toml 生成 requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# 查看生成的依赖
cat requirements.txt
```

### 7. 安装依赖

```bash
# 安装所有依赖到 Conda 环境
uv pip install -r requirements.txt

# 验证安装
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

---

## 验证安装

### 基础验证

```bash
# 运行环境检查脚本
python -m experiments.000

# 同时输出到终端和日志文件
python -m experiments.000 2>&1 | tee outputs/logs/exp_000_environment_check.log
```

### 手动验证

```bash
# 验证 PyTorch 和 CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# 验证 GPU
python -c "import torch; print('GPU 数量:', torch.cuda.device_count())"
```

---

## 常用命令

```bash
# 激活环境
conda activate nanomind

# 添加新依赖
uv add <package_name> --no-sync
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt

# 升级依赖
uv add <package_name> --upgrade --no-sync

# 运行代码
python main.py
python -m <module_name>
```

---

## 故障排查

### 问题 1: conda activate 不工作

**症状**: 运行 `conda activate nanomind` 后环境未激活。

**解决方案**:
```bash
# 初始化 Conda shell
conda init bash

# 重新加载 shell
source ~/.bashrc

# 再次尝试激活
conda activate nanomind
```

### 问题 2: uv pip install 失败

**症状**: 安装依赖时出现错误。

**解决方案**:
```bash
# 清理 uv 缓存
uv cache clean

# 重新安装
uv pip install -r requirements.txt
```

### 问题 3: CUDA 不可用

**症状**: `torch.cuda.is_available()` 返回 `False`。

**解决方案**:
```bash
# 检查 CUDA 版本匹配
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 确保 Conda 中的 CUDA 版本匹配
conda install -c nvidia cuda=12.8 -y

# 重新安装 PyTorch
uv add torch torchvision --upgrade --no-sync
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

### 问题 4: Python 版本不匹配

**症状**: 项目需要 Python 3.12，但当前环境是其他版本。

**解决方案**:
```bash
# 创建新的 Python 3.12 环境
conda create -n nanomind python=3.12 -y

# 检查 Python 版本
python --version
```

---

## 下一步

- [依赖管理](dependencies.md) - 管理项目依赖、升级和版本兼容性
- [环境验证](verification.md) - 验证环境配置是否正确

---

## 相关文档

- [README.md](../../README.md) - 项目简介和快速开始
- [AGENTS.md](../../AGENTS.md) - 开发指南导航
