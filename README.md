# nanomind

深度学习、大语言模型学习与试验项目。

## 核心功能

### 1. FineWeb-Edu 数据处理

基于 Datatrove 的高性能数据处理流水线，支持多语言数据集的自动化质量分层与分层采样。

```bash
# 处理所有配置的数据集（英文 + 中文）
python -m src.data_processing.fineweb_edu

# 试运行（小规模测试）
python scripts/trial_run.py

# 验证输出结果
python scripts/validate_output.py --all
```

### 2. Tokenizer 训练

从采样数据训练与 Qwen3-Next 兼容的 36K 词表 BPE Tokenizer。

```bash
# 1. 准备模板（从 HuggingFace 下载）
python scripts/prepare_tokenizer_template.py

# 2. 准备训练数据（从配置的数据集采样）
python scripts/prepare_tokenizer_data.py

# 3. 训练 Tokenizer
python scripts/train_tokenizer.py --validate
```

## 快速开始

### 环境要求

- Python 3.13+
- CUDA 12.9+（可选，用于 GPU 加速）

### 安装

```bash
conda create -n nanomind python=3.13 -y
conda activate nanomind
conda install -c conda-forge uv -y

uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

## 项目结构

```
nanomind/
├── src/data_processing/     # 数据处理模块
│   ├── fineweb_edu/         # FineWeb-Edu 处理流水线
│   ├── bucket_config.py     # 评分桶配置
│   ├── score_filter.py      # 评分过滤 + 采样
│   └── ...
├── scripts/                 # 可执行脚本
│   ├── trial_run.py         # 试运行工具
│   ├── validate_output.py   # 输出验证
│   ├── prepare_tokenizer_template.py  # Tokenizer 模板准备
│   ├── prepare_tokenizer_data.py      # Tokenizer 数据准备
│   └── train_tokenizer.py             # Tokenizer 训练
├── config/                  # YAML 配置
│   ├── dataset.yaml         # 数据集定义
│   ├── processing.yaml      # 处理参数
│   └── tokenizer_data.yaml  # Tokenizer 数据配置
└── docs/                    # 设计文档
```

## 配置文件

| 文件 | 用途 |
|------|------|
| `config/dataset.yaml` | 数据集评分桶、归一化配置、输入输出路径 |
| `config/processing.yaml` | 处理参数（workers、compression、文件大小） |
| `config/tokenizer_data.yaml` | Tokenizer 训练数据采样配置 |

## 文档

| 文档 | 说明 |
|------|------|
| [数据处理模块](src/data_processing/README.md) | API 参考、使用示例 |
| [设计文档](docs/fineweb_edu_data_reorganization_design.md) | 系统架构与扩展指南 |
| [AGENTS.md](AGENTS.md) | 开发规范与代码风格 |

## 许可证

- 项目代码: MIT License
- FineWeb-Edu 数据集: ODC-BY 1.0 License
