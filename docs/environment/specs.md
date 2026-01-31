# 环境规格

## 概述

本文档详细说明了 nanomind 项目的硬件和软件环境规格，包括系统配置、硬件规格和软件版本信息。

---

## 硬件配置

| 组件 | 规格 |
|------|------|
| **CPU** | 2× Intel Xeon E5-2667 v4 @ 3.20GHz (16 核 / 32 线程) |
| **内存** | 251.59 GB |
| **GPU** | 2× NVIDIA GeForce RTX 2080 Ti (21.5 GB × 2) |

---

## 软件环境

| 组件 | 版本 |
|------|------|
| **Python** | 3.12.x |
| **PyTorch** | 2.10.0+cu128 |
| **CUDA** | 12.8 |
| **cuDNN** | 9.10.2 |
| **Transformers** | 5.0.0 |
| **Datasets** | 4.5.0 |
| **Torchvision** | 0.25.0+cu128 |

---

## 验证命令

以下命令可用于验证环境配置是否正确：

```bash
# 验证 Python 版本
python --version

# 验证 PyTorch 和 CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# 验证 GPU 数量
python -c "import torch; print('GPU 数量:', torch.cuda.device_count())"

# 验证 Transformers 版本
python -c "import transformers; print('Transformers:', transformers.__version__)"

# 验证 Datasets 版本
python -c "import datasets; print('Datasets:', datasets.__version__)"

# 验证 CUDA 版本
nvcc --version
```

---

## 环境信息

```bash
# 查看系统信息
uname -a

# 查看内存信息
free -h

# 查看 CPU 信息
lscpu

# 查看 GPU 信息
nvidia-smi
```

---

## 相关文档

- [环境初始化](setup.md) - 配置开发环境
- [依赖管理](dependencies.md) - 管理项目依赖和版本兼容性
- [环境验证](verification.md) - 验证环境配置和功能测试

---

## 更新日志

本文档反映了当前环境的实际配置。如需更新环境，请参考：
- [依赖管理](dependencies.md) - 了解如何升级依赖
- [环境初始化](setup.md) - 重新配置环境

---

- [README.md](../../README.md) - 项目简介和快速开始
- [AGENTS.md](../../AGENTS.md) - 开发指南导航
