# nanomind

深度学习、大语言模型学习与试验

## 项目目标

学习深度学习和LLM技术，快速原型设计，确保实验可复现

---

## 项目概览

nanomind 是一个以实验和数据探索为中心的深度学习项目，专注于：

- **环境验证**: exp_000 - 验证 Python、PyTorch、CUDA 等环境配置
- **数据探索**: exp_001 - 使用 Datatrove 进行大规模数据集统计分析
- **实验框架**: 模块化实验设计，支持 argparse + dataclass 配置管理

**技术栈**:
- Python 3.12
- PyTorch 2.10.0 + CUDA 12.8
- Transformers 5.0.0
- Datasets 4.5.0
- Datatrove 0.8.0+（数据流水线）

---

## 快速开始

```bash
# 1. 创建并激活 Conda 环境
conda create -n nanomind python=3.12 -y
conda activate nanomind

# 2. 安装 CUDA 12.8（用于 GPU 加速）
conda install -c nvidia cuda=12.8 -y

# 3. 安装项目依赖
uv pip install -r requirements.txt

# 4. 验证环境配置
python -m experiments.000
```

> 📚 **详细的环境初始化和依赖管理说明**请参考：[docs/environment/setup.md](docs/environment/setup.md)

---

## 硬件配置

| 组件 | 规格 |
|------|------|
| **CPU** | 2× Intel Xeon E5-2667 v4 @ 3.20GHz (16 核 / 32 线程) |
| **内存** | 251.59 GB |
| **GPU** | 2× NVIDIA GeForce RTX 2080 Ti (21.5 GB ×2) |

详细规格：[docs/environment/specs.md](docs/environment/specs.md)

---

## 文档导航

- [AGENTS.md](AGENTS.md) - 开发指南核心
- [docs/environment/](docs/environment/) - 环境管理（setup, dependencies, verification, specs）
- [docs/development/](docs/development/) - 开发规范（code-style, best-practices, debugging, git-workflow）
- [docs/experiments/](docs/experiments/) - 实验管理（getting-started, management, project-structure, exp-001-overview, fineweb_stats）

---

## 许可证

MIT License
