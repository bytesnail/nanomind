# nanomind

深度学习与大语言模型实验项目，专注于高质量数据处理和 Tokenizer 训练。

## 功能特性

- **FineWeb-Edu 数据处理**: 基于 Datatrove 的多语言数据质量分层与采样流水线
- **Tokenizer 训练**: 与 Qwen3-Next 兼容的 36K 词表 BPE Tokenizer

## 快速开始

### 环境要求

- Python 3.13+
- CUDA 12.9+ (可选，用于 GPU 加速)

### 安装

```bash
conda create -n nanomind python=3.13 -y
conda activate nanomind
conda install -c conda-forge uv -y

uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

## 使用指南

### FineWeb-Edu 数据处理

```bash
# 处理所有配置的数据集
python -m src.data_processing.fineweb_edu

# 试运行（小规模测试）
python scripts/trial_run.py --dataset zh

# 验证输出
python scripts/validate_output.py --all
```

### Tokenizer 训练

```bash
# 1. 下载模板
python scripts/prepare_tokenizer_template.py

# 2. 准备训练数据
python scripts/prepare_tokenizer_data.py

# 3. 训练并验证
python scripts/train_tokenizer.py --validate
```

## 项目结构

```
nanomind/
├── src/data_processing/     # 数据处理模块
│   ├── fineweb_edu/         # FineWeb-Edu 流水线
│   └── ...
├── scripts/                 # 可执行脚本
├── config/                  # YAML 配置文件
├── tests/                   # 测试文件
└── docs/                    # 设计文档
```

## 配置文件

| 文件 | 用途 |
|------|------|
| `config/dataset.yaml` | 数据集评分桶与路径配置 |
| `config/processing.yaml` | 处理参数（workers、compression） |
| `config/tokenizer_data.yaml` | Tokenizer 训练数据采样配置 |
| `config/tokenizer.yaml` | Tokenizer 训练参数配置 |
| `config/paths.yaml` | 路径配置 |

## 文档

- [数据处理模块文档](src/data_processing/README.md) - API 参考与使用示例
- [AGENTS.md](AGENTS.md) - 开发规范与代码风格指南

## 测试

```bash
pytest                    # 运行所有测试
pytest -xvs tests/test_bucket_config.py  # 单文件测试
```

## 许可证

- 项目代码: MIT License
- FineWeb-Edu 数据集: ODC-BY 1.0 License
