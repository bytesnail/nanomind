# nanomind - 开发指南

## 项目概述

使用 Python 3.12 和 PyTorch 生态系统进行深度学习和LLM学习与试验。

**核心依赖**：
- **PyTorch >=2.10.0**: 深度学习框架
- **Transformers >=5.0.0**: HuggingFace 模型库
- **Datasets >=4.5.0**: 数据集处理库
- **Datatrove >=0.8.0**: 大规模数据处理流水线（用于数据统计和探索）

**核心原则**:
1. **代码清晰度优于优化**
2. **可复现性优先**
3. **实验文档化**
4. **模块化设计**
5. **实用主义优于过度工程化**
6. **配置管理清晰化** - 使用 argparse + dataclass 而非硬编码

---

## Do/Don'ts

### 环境
- **DO**: `uv add <package> --no-sync`、环境检查、提交前格式化
- **DON'T**: 不使用 `pip install`、不跳过验证
- **NEVER**: 不提交密钥

### 代码
- **DO**: 类型提示、docstrings、绝对导入、固定随机种子
- **DON'T**: 不硬编码配置、不使用 `import *`

### AI 代理工作模式
- **DO**: 自主决策和执行本项目的操作（无需等待交互授权）
- **DO**: 直接执行符合项目原则的修改
- **DON'T**: 为简单操作等待用户确认（如修复 lint 错误、运行测试等）
- **DON'T**: 因等待交互而中断工作流程
- **NOTE**: 仅针对本项目目录下的操作无需等待交互授权

### PyTorch
- **DO**: `get_device()`（推荐模式）、`model.train()`/`model.eval()`、`torch.no_grad()`
- **DON'T**: 不手动管理设备、推理时不禁用梯度

**注意事项**：
- ⚠️ `get_device()` 是推荐的最佳实践模式，当前代码库中尚未实现。请使用 `torch.device('cuda')` 直接管理设备。
- ⚠️ `model.train()`/`model.eval()` 和 `torch.no_grad()` 是最佳实践，当前数据探索实验中尚未应用这些模式。

### 实验
- **DO**:
  - 使用 `experiments/` 目录存放实验脚本
  - 使用 `argparse` + dataclass 管理实验配置
  - 使用 `--help` 查看参数说明
  - 使用 `outputs/` 记录实验结果
  - 使用 `exp:` 前缀提交实验相关代码
  - 使用 datatrove 进行大规模数据处理
- **DON'T**:
  - 不硬编码超参数到代码中
  - 不提交模型检查点（outputs/ 下的 .pth 文件）
  - 不提交大型数据集文件

**DataTrove 说明**：
- 用于大规模数据处理和流水线操作
- 在 exp_001 中用于多数据集统计
- 支持分布式处理和并行统计收集

---

## 安全边界

| 操作 | 允许 | 说明 |
|------|------|------|
| 写入到 `experiments/` | ✅ | 实验代码 |
| 写入到 `outputs/` | ✅ | 日志、结果（checkpoints/ 在实验运行时创建） |
| 修改 `configs/` | ⚠️ | 需要询问 |
| 提交密钥 | 🚫 | 绝对禁止 |

---

## 常用命令

### 环境
```bash
conda activate nanomind
uv pip install -r requirements.txt
python -m experiments.000
```

### 开发
```bash
# 查看实验参数
python -m experiments.001 --help

# 运行实验（数据集统计）
python -m experiments.001 --dataset <name> --output-dir <path> --workers 8

# 运行所有有 score 的数据集
python -m experiments.001 --dataset all --workers 8

# 代码工具
uv add <package> --no-sync
black .
ruff check .
```

### Git 提交规范
```bash
# 使用前缀标识提交类型
git commit -m "exp: add new experiment"
git commit -m "fix: correct device handling"
```

---

## 代码模式快速参考

**核心要点**：
- 类型提示 + docstring → [代码风格](docs/development/code-style.md)
- 使用 `get_device()` 自动选择设备（推荐模式，当前未实现）→ [最佳实践](docs/development/best-practices.md)
- `model.train()` / `model.eval()` 切换模式（最佳实践，待应用）
- 推理时使用 `torch.no_grad()` 禁用梯度（最佳实践，待应用）
- 使用 `argparse` + `dataclass` 管理配置 → [实验管理](docs/experiments/management.md)

---

## 快速参考

- [环境初始化](docs/environment/setup.md)
- [依赖管理](docs/environment/dependencies.md)
- [环境验证](docs/environment/verification.md)
- [代码风格](docs/development/code-style.md)
- [最佳实践](docs/development/best-practices.md)
- [调试技巧](docs/development/debugging.md)
- [Git 工作流](docs/development/git-workflow.md)
- [开始实验](docs/experiments/getting-started.md)
- [实验管理](docs/experiments/management.md)
- [项目 README](README.md)
