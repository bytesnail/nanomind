# nanomind

深度学习、大语言模型学习与试验项目

## 简介

nanomind 是一个专注于深度学习和大语言模型学习实验的项目。

当前核心功能：**FineWeb-Edu 数据集质量评分分桶重组**——基于 Datatrove 的高性能数据处理流水线，支持多语言数据集的自动化质量分层与分层采样。

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

### 运行

```bash
# 处理所有配置的数据集（英文 + 中文）
python -m src.data_processing.fineweb_edu

# 试运行（小规模测试）
python scripts/trial_run.py

# 验证输出结果
python scripts/validate_output.py --input data/datasets/fineweb/en
```

## 文档

| 文档 | 说明 |
|------|------|
| [设计文档](docs/fineweb_edu_data_reorganization_design.md) | 系统架构、核心组件、配置系统、性能优化、扩展指南 |
| [模块文档](src/data_processing/README.md) | API 参考、使用示例、CLI 说明 |
| [AGENTS.md](AGENTS.md) | 开发规范与代码风格指南 |

## 许可证

- 项目代码: MIT License
- FineWeb-Edu 数据集: ODC-BY 1.0 License
