# nanomind

深度学习、大语言模型学习与试验项目

## 环境设置

**⚠️ 重要**：本项目使用 `requirements.txt` 工作流，**添加依赖必须使用 `--no-sync` 参数**。

### 环境准备

```bash
conda create -n nanomind python=3.13 -y
conda activate nanomind
conda install -c conda-forge uv -y
conda install -c nvidia cuda=12.9 -y
```

### 依赖管理

```bash
# 添加依赖
uv add --dev black ruff pytest --no-sync
uv add torch torchvision transformers datasets datatrove[all] matplotlib tqdm --no-sync

# 编译并安装
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

## FineWeb-Edu 数据重组

项目实现 FineWeb-Edu 数据集按质量评分分桶重组，支持分层采样。详见设计文档：

- [`docs/fineweb_edu_data_reorganization_design.md`](./docs/fineweb_edu_data_reorganization_design.md) - 完整设计文档
- [`docs/fineweb-edu-chinese-score-analysis.md`](./docs/fineweb-edu-chinese-score-analysis.md) - 中文数据集评分分析

### 快速开始

```bash
# 处理所有评分桶
python -m src.data_processing.fineweb_reorganizer

# 验证输出
python scripts/validate_output.py --input data/datasets/fineweb/en
```

### 评分桶配置

| 桶名称 | 评分区间 | 采样率 |
|--------|----------|--------|
| 2.8 | 2.8 ≤ score < 3.0 | 30% |
| 3.0 | 3.0 ≤ score < 3.5 | 60% |
| 3.5 | 3.5 ≤ score < 4.0 | 80% |
| 4.0 | score ≥ 4.0 | 100% |

### 常用命令

```bash
# 处理指定评分桶
python -m src.data_processing.fineweb_reorganizer --bucket 3.0

# 指定 workers 和随机种子
python -m src.data_processing.fineweb_reorganizer --workers 16 --seed 42

# 并行处理多个桶
python -m src.data_processing.fineweb_reorganizer --parallel-buckets 4

# 试运行（创建小规模测试数据）
python scripts/trial_run.py

# 批量运行（生产环境推荐）
bash scripts/run_processing.sh --workers 16 --parallel-buckets 2
```

## 项目结构

```
nanomind/
├── src/data_processing/    # 数据重组核心代码
├── scripts/                # 工具脚本
├── config/                 # 配置文件
├── tests/                  # 测试文件
└── docs/                   # 设计文档
```
