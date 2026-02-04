# AGENTS.md - nanomind 编码规范

> 深度学习、大语言模型学习与试验项目

## 构建 / 格式化 / 测试命令

```bash
# 安装依赖（uv 工作流）
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt

# 添加新依赖（必须使用 --no-sync）
uv add <package> --no-sync
uv add --dev <dev-package> --no-sync

# 代码检查与格式化
ruff check .                    # 检查所有文件
ruff check --fix .              # 自动修复问题
ruff format .                   # 格式化所有文件

# 测试
pytest                          # 运行所有测试
pytest -xvs                     # 详细输出，遇到第一个失败停止
pytest -k "test_name"           # 运行匹配名称的测试
```

## 代码风格指南

### Python 版本
- **Python 3.13+** 必需

### 导入规范
- 分组顺序：标准库 → 第三方库 → 本地模块
- 使用绝对导入，避免相对导入
- 使用 `isort` 风格排序（ruff 自动处理）

### 格式化
- **行长度**：88 字符（Black 默认）
- **引号**：字符串使用双引号
- 使用 Ruff 进行代码检查和格式化

### 命名规范
- `snake_case`：函数、变量、模块
- `PascalCase`：类名
- `UPPER_CASE`：常量
- `_leading_underscore`：私有方法或暂不使用的变量，下划线前缀

### 类型注解
- 函数签名必须使用类型注解

### 错误处理
- 使用具体的异常类型
- 资源管理使用上下文管理器
- 记录错误时提供上下文信息

### 文件路径规范
- **优先使用相对路径**，避免使用绝对路径
- 使用 `pathlib.Path` 处理路径，避免字符串拼接
- 通过环境变量或配置文件管理可配置路径

### 深度学习特定规范
- 推理时使用 `torch.no_grad()` 上下文
- 显式将张量移动到目标设备
- 长时间操作使用 `tqdm` 显示进度条
- 数据集缓存到 `data/datasets/`

## 项目结构

```
nanomind/
├── pyproject.toml          # 项目配置
├── requirements.txt        # 编译后的依赖（自动生成）
├── data/
│   └── datasets/           # 预下载数据集
├── src/                    # 源代码（按需创建）
└── tests/                  # 测试文件（按需创建）
```

## 依赖说明

**核心依赖**：torch, torchvision, transformers, datasets, datatrove, matplotlib, tqdm
**开发依赖**：black, ruff, pytest

## 环境配置

```bash
conda create -n nanomind python=3.13 -y
conda activate nanomind
conda install -c conda-forge uv -y
conda install -c nvidia cuda=12.9 -y  # GPU 可选
```

## 注意事项

- **不使用 uv.lock**：项目采用 `requirements.txt` 工作流（uv.lock 已加入 .gitignore）
- **数据集**：使用 `hfd` 脚本下载数据集到 `data/datasets/`
- **GPU**：推荐使用 CUDA 12.9 进行 GPU 计算
