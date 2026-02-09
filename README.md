# nanomind

深度学习、大语言模型学习与试验项目

## 简介

**nanomind** 专注于深度学习和大语言模型相关的学习与实验，当前核心功能是 **FineWeb-Edu 数据集的质量评分分桶重组**——支持多语言、分层采样和高性能并行处理的数据流水线。

### 核心特性

- **单次读取多桶处理**: I/O 效率提升约 75%
- **确定性分层采样**: 基于 MD5 哈希，结果可复现
- **多语言支持**: 自动处理英文原版和中文版本
- **灵活配置**: YAML 配置 + 环境变量覆盖
- **模块化架构**: 基于 Datatrove，易于扩展

## 快速开始

### 环境要求

- **Python**: 3.13+
- **CUDA**: 12.9+（推荐，用于 GPU 加速）
- **内存**: 建议 32GB+

### 安装

```bash
# 创建 conda 环境
conda create -n nanomind python=3.13 -y
conda activate nanomind

# 安装 uv 包管理器
conda install -c conda-forge uv -y

# 安装 CUDA（如使用 GPU）
conda install -c nvidia cuda=12.9 -y

# 安装依赖
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

### 使用方法

```bash
# 处理所有配置的数据集（英文 + 中文）
python -m src.data_processing.fineweb_edu

# 验证输出结果
python scripts/validate_output.py --input data/datasets/fineweb/en

# 试运行（小规模测试）
python scripts/trial_run.py
```

## 项目结构

```
nanomind/
├── src/data_processing/          # 数据重组核心代码
│   ├── fineweb_edu/             # FineWeb-Edu 专用子模块
│   ├── bucket_config.py         # 评分桶配置管理
│   ├── score_filter.py          # 评分过滤 + 采样
│   ├── bucket_path_writer.py    # 多桶并行写入器
│   └── ...
├── scripts/                     # 工具脚本
├── config/                      # 配置文件
├── tests/                       # 测试文件
├── docs/                        # 设计文档
└── pyproject.toml               # 项目配置
```

## 文档

- **[设计文档](./docs/fineweb_edu_data_reorganization_design.md)** - 架构设计、核心组件详解、性能优化、扩展指南
- **[模块文档](./src/data_processing/README.md)** - 数据处理模块 API 说明

## 开发

### 代码检查与格式化

```bash
ruff check .           # 检查代码
ruff check --fix .     # 自动修复
ruff format .          # 格式化
```

### 运行测试

```bash
pytest                 # 运行所有测试
pytest -xvs            # 详细输出，遇错停止
pytest -k "test_name"  # 运行特定测试
```

### 依赖管理

⚠️ 使用 `requirements.txt` 工作流，添加依赖必须使用 `--no-sync`：

```bash
uv add <package> --no-sync
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

## 许可证

- **项目代码**: MIT License
- **FineWeb-Edu 数据集**: ODC-BY 1.0 License
