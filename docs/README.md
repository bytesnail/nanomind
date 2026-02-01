# nanomind 文档

本文档目录提供 nanomind 项目的完整文档导航。

---

## 快速开始

如果您是第一次使用本项目，建议按照以下顺序阅读：

1. [项目 README](../README.md) - 项目概述和快速开始
2. [开发指南 (AGENTS.md)](../AGENTS.md) - 项目原则和开发规范
3. [环境初始化](environment/setup.md) - 配置开发环境
4. [环境验证](environment/verification.md) - 验证环境配置
5. [开始实验](experiments/getting-started.md) - 创建您的第一个实验

---

## 文档目录

### 环境管理

快速搭建和验证 nanomind 开发环境。

| 文档 | 说明 |
|------|------|
| [setup.md](environment/setup.md) | 环境初始化：Conda、uv、CUDA 安装配置 |
| [dependencies.md](environment/dependencies.md) | 依赖管理：版本、升级、兼容性矩阵 |
| | [data-setup.md](environment/data-setup.md) | 数据获取和初始化：下载、验证数据集 |
| [verification.md](environment/verification.md) | 环境验证：功能测试、exp_000 使用说明 |
| | [troubleshooting.md](environment/troubleshooting.md) | 环境故障排查：常见问题、解决方案 |
| [specs.md](environment/specs.md) | 环境规格：硬件配置、软件版本 |

---

### 开发规范

确保代码质量、一致性和可维护性。

| 文档 | 说明 |
|------|------|
| [code-style.md](development/code-style.md) | 代码风格：命名约定、导入顺序、类型提示 |
| [best-practices.md](development/best-practices.md) | PyTorch 最佳实践：设备处理、梯度控制、随机种子 |
| [debugging.md](development/debugging.md) | 调试技巧：常见问题、GPU 内存优化、性能优化 |
| [git-workflow.md](development/git-workflow.md) | Git 工作流：提交规范、分支管理 |

---

### 实验管理

创建、运行和管理实验。

| 文档 | 说明 |
|------|------|
| [getting-started.md](experiments/getting-started.md) | 开始实验：实验模板、超参数、命名规范 |
| [management.md](experiments/management.md) | 实验管理：记录、追踪、对比、清理策略 |
| [project-structure.md](experiments/project-structure.md) | 项目结构：目录规范、模块导入、配置管理 |
| [exp-001-overview.md](experiments/exp-001-overview.md) | 实验详解：数据集统计与探索完整说明 |
| [exp-001-fineweb-example.md](experiments/exp-001-fineweb-example.md) | 实验案例：FineWeb-Edu 统计实验案例分析 |

---

## 核心概念

### 配置管理

本项目使用 **argparse + dataclass** 配置管理方式：

```python
@dataclass
class Config:
    dataset: str
    batch_size: int = 32
```

```bash
python -m experiments.001 --dataset <name> --batch-size 64
```

详见：[项目结构](experiments/project-structure.md)

### 实验运行

使用 `-m` 参数运行实验模块：

```bash
# 环境验证 ✅ 已验证
python -m experiments.000

# 数据集统计（单数据集） ✅ 已验证
python -m experiments.001 --dataset HuggingFaceFW/fineweb-edu --workers 8

# 数据集统计（所有有 score 的数据集） ✅ 已验证
python -m experiments.001 --dataset all --workers 8
```

详见：[开始实验](experiments/getting-started.md)

### DataTrove

DataTrove 是本项目的核心数据处理工具：

- 用于大规模数据集处理
- 支持分布式处理
- 流水线式数据处理

详见：[AGENTS.md](../AGENTS.md) - DataTrove 说明

---

## 当前实验

### exp_000：环境验证

验证 Python、PyTorch、CUDA 等环境配置。

**运行**：
```bash
python -m experiments.000
```

### exp_001：数据集统计

使用 Datatrove 对多个数据集进行统计分析。

**运行**：
```bash
# 所有有 score 的数据集 ✅ 已验证
python -m experiments.001 --dataset all --workers 8

# 指定数据集 ✅ 已验证
python -m experiments.001 --dataset HuggingFaceFW/fineweb-edu --workers 8
```

**支持的数据集**：
- HuggingFaceFW/fineweb-edu
- opencsg/Fineweb-Edu-Chinese-V2.1
- HuggingFaceTB/finemath
- nvidia/Nemotron-CC-Math-v1
- nick007x/github-code-2025

详见：[exp-001 概览](experiments/exp-001-overview.md)

---

## 常用命令

### 环境管理

```bash
# 激活环境
conda activate nanomind

# 添加新依赖
uv add <package> --no-sync
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt

# 运行环境检查
python -m experiments.000
```

### 开发工具

```bash
# 代码格式化
black .

# 代码检查
ruff check .
ruff check --fix .
```

### 实验运行

```bash
# 运行实验（数据集统计） ✅ 已验证
python -m experiments.001 --dataset <name> --workers 8

# 运行所有有 score 的数据集 ✅ 已验证
python -m experiments.001 --dataset all --workers 8

# 查看实验帮助 ✅ 已验证
python -m experiments.001 --help

# 运行并保存日志
python -m experiments.001 2>&1 | tee outputs/logs/exp_001.log
```

---

## 贡献指南

### 提交规范

使用以下前缀标识提交类型：

- `exp:` - 实验相关代码
- `fix:` - Bug 修复
- `docs:` - 文档更新
- `chore:` - 杂项（依赖、配置等）
- `refactor:` - 代码重构

### 代码规范

- 所有函数必须包含类型提示和 docstring
- 遵循 PEP 8 代码风格
- 使用 Black 格式化和 Ruff 检查
- 提交前运行：`black . && ruff check .`

详见：[代码风格](development/code-style.md)

---

## 故障排查

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| ImportError: No module named 'torch' | 运行 `uv pip install -r requirements.txt` |
| CUDA 不可用 | 确认 CUDA 12.8 已安装，重新运行 `uv pip install -r requirements.txt` |
| 实验路径错误 | 使用 `python -m experiments.xxx` 而非直接运行脚本 |
| DataTrove 处理失败 | 检查数据路径、减少 batch_size 或 workers 数量 |

详见：
- [环境验证](environment/verification.md)
- [调试技巧](development/debugging.md)
- [exp-001 故障排查](experiments/exp-001-overview.md#故障排查)

---


## 快速修复路径

遇到环境或实验问题？快速导航到解决方案：

- [环境故障排查指南](environment/troubleshooting.md) - 完整的故障排查流程和解决方案
- [exp-001 故障排查](experiments/exp-001-overview.md#故障排查) - 数据集统计实验的常见问题

## 相关资源

- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [DataTrove Documentation](https://github.com/huggingface/datatrove)

---

