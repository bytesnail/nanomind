# nanomind

深度学习、大语言模型学习与试验项目

## 项目简介

本项目专注于深度学习和大语言模型相关的学习与实验，核心功能是实现 **FineWeb-Edu 数据集的质量评分分桶重组**，支持多语言、分层采样和高性能并行处理。

## 核心功能

### FineWeb-Edu 数据重组

基于 [Datatrove](https://github.com/huggingface/datatrove) 构建的高性能数据处理流水线，实现：

- **多语言支持**: 同时处理英文原版（HuggingFaceFW/fineweb-edu）和中文版本（opencsg/Fineweb-Edu-Chinese-V2.1）
- **一次读取多桶处理**: 单遍读取输入数据，同时处理所有评分桶，I/O 效率提升约 75%
- **确定性分层采样**: 基于 MD5 哈希的伪随机采样，确保结果可复现
- **灵活配置**: YAML 配置文件 + 环境变量覆盖机制

## 环境设置

### 系统要求

- **Python**: 3.13+
- **CUDA**: 12.9+（推荐，用于 GPU 加速）
- **内存**: 建议 32GB+

### 快速开始

```bash
# 1. 创建 conda 环境
conda create -n nanomind python=3.13 -y
conda activate nanomind

# 2. 安装 uv 包管理器
conda install -c conda-forge uv -y

# 3. 安装 CUDA（如使用 GPU）
conda install -c nvidia cuda=12.9 -y

# 4. 安装依赖
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

### 依赖管理

⚠️ **重要**: 本项目使用 `requirements.txt` 工作流，添加依赖必须使用 `--no-sync` 参数。

```bash
# 添加生产依赖
uv add <package> --no-sync

# 添加开发依赖
uv add --dev <package> --no-sync

# 重新编译并安装
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

## FineWeb-Edu 数据重组使用指南

### 评分桶配置

#### 英文数据集（HuggingFaceFW/fineweb-edu）

| 质量评分区间 | 桶名称 | 采样率 | 说明 |
|-------------|--------|--------|------|
| 2.5 ≤ score < 3.0 | 2.5 | 25% | 中低质量数据 |
| 3.0 ≤ score < 3.5 | 3.0 | 50% | 中等质量数据 |
| 3.5 ≤ score < 4.0 | 3.5 | 80% | 高质量数据 |
| score ≥ 4.0 | 4.0 | 100% | 顶级质量数据 |

#### 中文数据集（Fineweb-Edu-Chinese-V2.1）

中文数据集使用归一化评分（0.0-1.0），自动转换为原始评分（×5）：

| 质量评分区间 | 桶名称 | 采样率 | 归一化范围 |
|-------------|--------|--------|-----------|
| 2.5 ≤ score < 3.0 | 2.5 | 40% | 0.50-0.60 |
| 3.0 ≤ score < 3.5 | 3.0 | 60% | 0.60-0.70 |
| 3.5 ≤ score < 4.0 | 3.5 | 90% | 0.70-0.80 |
| score ≥ 4.0 | 4.0 | 100% | ≥ 0.80 |

### 快速开始

```bash
# 处理所有配置的数据集（英文 + 中文）
python -m src.data_processing.fineweb_reorganizer

# 验证输出结果
python scripts/validate_output.py --input data/datasets/fineweb/en
```

### 生产环境运行

```bash
# 使用 time 统计运行时间
time python -m src.data_processing.fineweb_reorganizer

# 处理完成后自动验证
python -m src.data_processing.fineweb_reorganizer && \
  python scripts/validate_output.py --input data/datasets/fineweb/en
```

### 试运行

```bash
# 创建小规模测试数据并运行完整流程
python scripts/trial_run.py

# 分析采样准确性
python scripts/trial_run.py --analyze-sampling
```

### 配置调优

编辑 `config/processing.yaml` 调整性能参数：

```yaml
workers: 32                       # 本地并行工作进程数
tasks: 2500                       # Datatrove 任务数
random_seed: 42                   # 采样随机种子
compression: "zstd"               # Parquet 压缩格式
max_file_size_bytes: 2147483648   # 输出文件大小限制（2GB）
```

### 环境变量

使用环境变量覆盖配置：

```bash
# 覆盖日志目录
export FINEWEB_LOG_DIR="custom/logs"

# 覆盖试运行目录
export FINEWEB_TRIAL_INPUT_DIR="data/test_input"
export FINEWEB_TRIAL_OUTPUT_DIR="data/test_output"

python -m src.data_processing.fineweb_reorganizer
```

## 项目结构

```
nanomind/
├── src/data_processing/          # 数据重组核心代码
│   ├── fineweb_reorganizer.py   # CLI 主入口
│   ├── adapters.py              # 数据适配器
│   ├── score_filter.py          # 评分过滤 + 采样
│   ├── bucket_path_writer.py    # 多桶并行写入器
│   ├── bucket_config.py         # 评分桶配置管理
│   └── config_loader.py         # YAML 配置加载器
├── scripts/                     # 工具脚本
│   ├── trial_run.py             # 试运行脚本
│   └── validate_output.py       # 输出验证工具
├── config/                      # 配置文件
│   ├── dataset.yaml             # 数据集定义
│   ├── processing.yaml          # 处理参数
│   └── paths.yaml               # 路径配置
├── tests/                       # 测试文件
├── docs/                        # 设计文档
│   └── fineweb_edu_data_reorganization_design.md  # 完整设计文档（含中文评分分析）
└── pyproject.toml               # 项目配置
```

## 开发

### 代码检查与格式化

```bash
# 检查代码
ruff check .

# 自动修复问题
ruff check --fix .

# 格式化代码
ruff format .
```

### 运行测试

```bash
# 运行所有测试
pytest

# 详细输出，遇到第一个失败停止
pytest -xvs

# 运行匹配名称的测试
pytest -k "test_name"
```

## 文档

- [完整设计文档](./docs/fineweb_edu_data_reorganization_design.md) - 架构设计、组件详解、配置说明（含中文数据集评分分析）
- [模块文档](./src/data_processing/README.md) - 数据处理模块详细说明

## 许可证

- **项目代码**: MIT License
- **FineWeb-Edu 数据集**: ODC-BY 1.0 License
