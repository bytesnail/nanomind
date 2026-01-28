# 环境验证

## 概述

本指南介绍如何验证 nanomind 项目的环境配置是否正确，包括硬件、软件和功能验证。

---

## 快速验证

```bash
# 基础验证
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 完整环境检查

### 使用环境检查脚本

环境检查脚本位于 `experiments/exp_000_environment_check.py`。

**运行方式：**
```bash
# 仅输出到终端
python experiments/exp_000_environment_check.py

# 同时输出到终端和日志文件
python experiments/exp_000_environment_check.py 2>&1 | tee outputs/logs/exp_000_environment_check.log
```

**输出解读：**
- ✅ 表示功能正常
- ⚠️ 表示功能降级（如无 GPU）
- ❌ 表示功能缺失

**检查项目：**
- 系统配置（操作系统、CPU、内存、GPU）
- 软件版本（Python、PyTorch、CUDA、cuDNN）
- 功能验证（张量操作、矩阵运算、GPU 加速、自动求导）

---

## 手动验证清单

### 1. Python 环境

```bash
# 检查 Python 版本
python --version  # 应该输出 Python 3.12.12

# 检查 Python 路径
which python
```

### 2. 核心依赖

```bash
# 验证所有包导入正常
python -c "import torch, transformers, datasets, torchvision"

# 查看版本信息
python -c "import torch; print('torch:', torch.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import datasets; print('datasets:', datasets.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
```

### 3. GPU 和 CUDA

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 查看 GPU 信息
python -c "import torch; print(torch.cuda.get_device_name(0))"  # GPU 名称
python -c "import torch; print(torch.cuda.device_count())"       # GPU 数量

# 查看 GPU 内存
python -c "import torch; print('已分配:', torch.cuda.memory_allocated())"
python -c "import torch; print('已保留:', torch.cuda.memory_reserved())"
```

### 4. 基础功能测试

```bash
# 测试张量操作
python -c "
import torch
x = torch.randn(10, 10)
y = torch.randn(10, 10)
z = x @ y
print('✓ 张量操作正常')
"

# 测试 GPU 加速
python -c "
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
z = x @ y
print(f'✓ GPU 加速正常 (设备: {device})')
"

# 测试自动求导
python -c "
import torch
x = torch.randn(10, 10, requires_grad=True)
y = x @ x
y.backward()
print('✓ 自动求导正常')
"
```

### 5. Transformers 功能

```bash
# 测试模型加载
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print('✓ 模型加载正常')
"

# 测试 tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
inputs = tokenizer('Hello, world!', return_tensors='pt')
print('✓ tokenizer 正常')
"
```

---

## exp_000 使用说明

### 实验目的

验证 nanomind 项目的基础环境配置，确保所有依赖正确安装并正常工作。

### 运行方法

```bash
# 方法 1: 直接运行
python experiments/exp_000_environment_check.py

# 方法 2: 输出到日志文件
python experiments/exp_000_environment_check.py 2>&1 | tee outputs/logs/exp_000_environment_check.log
```

### 输出内容

实验脚本会输出以下信息：

1. **系统信息**
   - 操作系统
   - CPU 型号
   - 内存大小
   - GPU 信息（如果有）

2. **软件版本**
   - Python 版本
   - PyTorch 版本
   - CUDA 版本
   - cuDNN 版本

3. **功能测试**
   - 张量创建和运算
   - 矩阵乘法
   - GPU 加速（如果有 GPU）
   - 自动求导

### 预期结果

```
系统信息:
  操作系统: Linux 6.12.63
  CPU: Intel Xeon E5-2667 v4 @ 3.20GHz
  内存: 251.59 GB
  GPU: 2x NVIDIA GeForce RTX 2080 Ti

软件版本:
  Python: 3.12.12
  PyTorch: 2.10.0+cu128
  CUDA: 12.8
  cuDNN: 9.10.2

功能测试:
  ✓ 张量创建和运算
  ✓ 矩阵乘法
  ✓ GPU 加速
  ✓ 自动求导

环境验证完成！
```

### 常见问题

**Q: 脚本运行失败，提示 `ImportError`**

**A:** 说明某些依赖未安装。运行：
```bash
uv pip install -r requirements.txt
```

**Q: CUDA 不可用**

**A:** 确认 CUDA 12.8 已安装，并且 PyTorch 版本兼容。运行：
```bash
conda install -c nvidia cuda=12.8 -y
uv pip install -r requirements.txt
```

---

## 故障排查

### 问题 1: PyTorch 导入失败

**症状:** `ImportError: No module named 'torch'`

**解决方案:**
```bash
# 安装 PyTorch
uv pip install -r requirements.txt
```

### 问题 2: CUDA 不可用

**症状:** `torch.cuda.is_available()` 返回 `False`

**解决方案:**
```bash
# 检查 CUDA 安装
nvidia-smi

# 重新安装 CUDA
conda install -c nvidia cuda=12.8 -y

# 重新安装 PyTorch
uv pip install -r requirements.txt

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题 3: Transformers 导入失败

**症状:** `ImportError: No module named 'transformers'`

**解决方案:**
```bash
# 安装 Transformers
uv pip install transformers

# 或安装所有依赖
uv pip install -r requirements.txt
```

### 问题 4: GPU 内存不足

**症状:** `RuntimeError: CUDA out of memory`

**解决方案:**
```bash
# 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 减小 batch_size
# 或使用 CPU 进行测试
```

---

## 验证检查表

完成以下检查项以确保环境配置正确：

- [ ] Python 版本为 3.12.12
- [ ] PyTorch 版本为 2.10.0+cu128
- [ ] Transformers 版本为 5.0.0
- [ ] Datasets 版本为 4.5.0
- [ ] Torchvision 版本为 0.25.0+cu128
- [ ] CUDA 版本为 12.8
- [ ] cuDNN 版本为 9.10.2
- [ ] GPU 可用（如果有 GPU）
- [ ] 张量操作正常
- [ ] 矩阵乘法正常
- [ ] GPU 加速正常（如果有 GPU）
- [ ] 自动求导正常
- [ ] Transformers 模型加载正常
- [ ] exp_000 环境检查通过

---

## 下一步

- [环境初始化](setup.md) - 配置开发环境
- [依赖管理](dependencies.md) - 管理项目依赖、升级和版本兼容性

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
