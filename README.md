# nanomind

深度学习、大语言模型学习与试验

## 快速开始

```bash
# 激活环境并初始化
conda activate nanomind && uv add torch torchvision transformers datasets --no-sync

# 安装依赖
uv pip compile pyproject.toml -o requirements.txt && uv pip install -r requirements.txt

# 验证环境
python experiments/exp_000_environment_check.py
```

## 文档

完整文档请查看 [AGENTS.md](AGENTS.md) - 开发指南与最佳实践。

## 硬件配置

- **CPU**: 2× Intel Xeon E5-2667 v4 @ 3.20GHz (16 核 / 32 线程)
- **内存**: 251.59 GB
- **GPU**: 2× NVIDIA GeForce RTX 2080 Ti (21.5 GB × 2)

## 软件环境

- **Python**: 3.12.12
- **PyTorch**: 2.10.0+cu128
- **CUDA**: 12.8
- **cuDNN**: 9.10.2
