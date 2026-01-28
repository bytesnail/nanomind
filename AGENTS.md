# nanomind - 开发指南

## 项目概述

使用 Python 3.12 和 PyTorch 生态系统进行深度学习和LLM学习与试验。

**核心原则**:
1. **代码清晰度优于优化**
2. **可复现性优先**
3. **实验文档化**
4. **模块化设计**

---

## Do/Don'ts

### 环境
- **DO**: `uv add <package> --no-sync`、环境检查、提交前格式化
- **DON'T**: 不使用 `pip install`、不跳过验证
- **NEVER**: 不提交密钥

### 代码
- **DO**: 类型提示、docstrings、绝对导入、固定随机种子
- **DON'T**: 不硬编码配置、不使用 `import *`

### PyTorch
- **DO**: `get_device()`、`model.train()`/`model.eval()`、`torch.no_grad()`
- **DON'T**: 不手动管理设备、推理时不禁用梯度

### 实验
- **DO**: `experiments/` 目录、配置文件、`outputs/` 记录、`exp:` 提交
- **DON'T**: 不硬编码超参数、不提交检查点

---

## 安全边界

| 操作 | 允许 | 说明 |
|------|------|------|
| 写入到 `experiments/` | ✅ | 实验代码 |
| 写入到 `outputs/` | ✅ | 模型、日志、结果 |
| 修改 `configs/` | ⚠️ | 需要询问 |
| 提交密钥 | 🚫 | 绝对禁止 |

---

## 常用命令

### 环境
```bash
conda activate nanomind
uv pip install -r requirements.txt
python experiments/exp_000_environment_check.py
```

### 开发
```bash
python experiments/exp_001_baseline.py
uv add <package> --no-sync
black .
ruff check .
```

---

## 代码模式

```python
# ✅ GOOD: 类型提示 + docstring
def train_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 10
) -> Dict[str, float]:
    """训练模型并返回指标。"""
    set_seed(42)
    return {"loss": 0.123, "accuracy": 0.956}

# 🚫 BAD
def train_model(model, dataloader):
    torch.manual_seed(42)
    return 0.123

# ✅ GOOD: 自动选择设备
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🚫 BAD: 手动管理设备
if torch.cuda.is_available():
    model = model.cuda()

# ✅ GOOD: 正确的模型模式
model.train()
for epoch in range(epochs):
    pass
model.eval()
with torch.no_grad():
    pass

# 🚫 BAD: 忘记切换模式

# ✅ GOOD: 使用配置文件
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
config = load_config('configs/exp_001.yaml')

# 🚫 BAD: 硬编码超参数
learning_rate = 0.001
```

---

## 技术栈

| 组件 | 版本 |
|------|------|
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.0.0 |
| Datasets | 4.5.0 |
| Torchvision | 0.25.0+cu128 |

详细规格：[docs/environment/specs.md](docs/environment/specs.md)

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
- [项目结构](docs/experiments/project-structure.md)

---

## 相关文档

- [README.md](README.md) - 项目简介和快速开始
- [docs/](docs/) - 详细文档目录
