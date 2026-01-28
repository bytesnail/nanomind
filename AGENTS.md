# nanomind - 开发指南

## 项目概述

使用 Python 3.12 和 PyTorch 生态系统进行深度学习和大型语言模型（LLM）学习与试验。

**项目性质**: 学习与实验项目，优先保持灵活性，不强制使用传统的单元测试和集成测试。

---

## 核心开发原则

1. **代码清晰度优于优化** - 先让代码清晰易懂，再考虑性能优化
2. **可复现性优先** - 固定随机种子、锁定依赖版本、记录环境信息
3. **实验文档化** - 记录实验目的、方法、结果，便于对比和回顾
4. **模块化设计** - 便于快速实验和迭代

---

## 快速导航

### 环境管理

| 文档 | 说明 | 关键内容 |
|------|------|---------|
| [环境初始化](docs/environment/setup.md) | 配置开发环境 | Conda、uv、CUDA 安装 |
| [依赖管理](docs/environment/dependencies.md) | 升级、版本兼容性 | 依赖升级、版本矩阵、回滚方案 |
| [环境验证](docs/environment/verification.md) | 检查清单和故障排查 | 硬件验证、软件验证、功能验证 |

### 开发指南

| 文档 | 说明 | 关键内容 |
|------|------|---------|
| [代码风格](docs/development/code-style.md) | 命名约定、类型提示 | PEP 8、类型提示、错误处理 |
| [最佳实践](docs/development/best-practices.md) | PyTorch 开发规范 | 设备处理、模型模式、梯度控制、随机种子 |
| [调试技巧](docs/development/debugging.md) | 常见问题与解决方案 | GPU 问题、内存优化、性能优化 |

### 实验管理

| 文档 | 说明 | 关键内容 |
|------|------|---------|
| [开始实验](docs/experiments/getting-started.md) | 创建您的第一个实验 | 实验模板、常用超参数、示例代码 |
| [实验管理](docs/experiments/management.md) | 记录、追踪、对比 | 实验记录、日志管理、实验追踪 |
| [项目结构](docs/experiments/project-structure.md) | 目录规范和代码组织 | 项目结构、模块导入、配置管理 |

---

## 常用命令速查

### 环境管理
```bash
conda activate nanomind                          # 激活环境
uv pip install -r requirements.txt               # 安装依赖
python experiments/exp_000_environment_check.py  # 环境检查
```

### 开发
```bash
python experiments/exp_001_baseline.py          # 运行实验
uv add <package> --no-sync                       # 添加依赖
black .                                          # 格式化代码
ruff check .                                     # 代码检查
```

### Git
```bash
git add .                                        # 暂存所有更改
git commit -m "exp: 添加实验 001"               # 提交更改
git push origin main                             # 推送到远程
```

---

## 当前环境版本

| 组件 | 版本 |
|------|------|
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.0.0 |
| Datasets | 4.5.0 |
| Torchvision | 0.25.0+cu128 |

---

## 快速参考

- **入口点**: `python main.py`
- **依赖**: `requirements.txt`
- **Python 版本**: 3.12
- **包管理**: `uv add <package> --no-sync`

---

## 相关文档

- [README.md](README.md) - 项目简介和快速开始
- [AGENTS.md](AGENTS.md) - 开发指南（本文档）
- [docs/](docs/) - 详细文档目录
