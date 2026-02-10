# AGENTS.md - nanomind

深度学习、大语言模型学习与试验项目。

## 项目概览

| 属性 | 值 |
|------|-----|
| **语言** | Python 3.13+ |
| **包管理** | uv (requirements.txt 工作流) |
| **核心栈** | torch, transformers, datatrove, datasets |
| **主入口** | `python -m src.data_processing.fineweb_edu` |

## 目录导航

| 目录 | 内容 | AGENTS.md |
|------|------|-----------|
| `src/data_processing/` | 核心数据处理模块 | [✅ 查看](./src/data_processing/AGENTS.md) |
| `config/` | YAML 配置文件 (dataset/processing/paths) | - |
| `scripts/` | 试运行/验证工具 | - |
| `tests/` | pytest 单元测试 | - |
| `docs/` | 设计文档 | - |

## 环境配置

```bash
# 创建 conda 环境
conda create -n nanomind python=3.13 -y
conda activate nanomind
conda install -c conda-forge uv -y
conda install -c nvidia cuda=12.9 -y  # GPU 可选
```

## 依赖管理

```bash
# 安装依赖
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt

# 添加新依赖（必须使用 --no-sync）
uv add <package> --no-sync
uv add --dev <dev-package> --no-sync
```

### 核心依赖

- **运行时**: torch, torchvision, transformers, datasets, datatrove, matplotlib, tqdm
- **开发**: black, ruff, pytest

## 常用命令

```bash

# 代码检查
ruff check .                 # 检查
ruff check --fix .           # 自动修复
ruff format .                # 格式化

# 测试
pytest                       # 全部测试
pytest -xvs                  # 详细输出，遇错停止
pytest -k "test_name"        # 指定测试

# 运行
python -m src.data_processing.fineweb_edu      # 主流程
python scripts/trial_run.py                     # 试运行
python scripts/validate_output.py --input ...   # 验证结果
```

## 代码规范

| 规则 | 说明 |
|------|------|
| **行长度** | 88 字符 (Black 默认) |
| **引号** | 双引号优先 |
| **命名** | snake_case(函数/变量), PascalCase(类), UPPER_CASE(常量), _leading_underscore(私有) |
| **导入** | 标准库 → 第三方 → 本地模块，绝对导入 |
| **类型** | 函数签名必须类型注解 |
| **路径** | 优先相对路径，使用 `pathlib.Path` |

### 导入规范

- 分组顺序：**标准库 → 第三方库 → 本地模块**
- 使用绝对导入，避免相对导入
- 使用 `isort` 风格排序（ruff 自动处理）

### 错误处理

- 使用具体的异常类型，避免裸 `except:`
- 资源管理使用上下文管理器 (`with` 语句)
- 记录错误时提供上下文信息

## 关键约定

### 不使用 uv.lock
项目采用 `requirements.txt` 工作流。uv.lock 已加入 .gitignore。

### 数据集下载
使用 `hfd` 脚本下载数据集到 `data/datasets/`：
```bash
hfd HuggingFaceFW/fineweb-edu --local-dir data/datasets/HuggingFaceFW/fineweb-edu
```

### 深度学习规范
- 推理使用 `torch.no_grad()` 上下文
- 显式张量设备移动
- 长操作使用 `tqdm` 进度条
- 数据集缓存到 `data/datasets/`

### 配置层级
1. YAML 配置 (`config/`)
2. 代码默认值

## 输出目录结构

```
data/datasets/fineweb/
├── en/                 # 英文数据集
│   ├── 2.5/           # 质量分桶
│   ├── 3.0/
│   ├── 3.5/
│   └── 4.0/
└── zh/                 # 中文数据集
    └── ...
```
