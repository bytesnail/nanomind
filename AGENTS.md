# AGENTS.md - nanomind

**Generated:** 2026-03-04
**Commit:** f5aee16
**Branch:** main

## OVERVIEW

深度学习/LLM 实验项目。核心功能：FineWeb-Edu 数据处理流水线 + Tokenizer 训练。

## STRUCTURE

```
nanomind/
├── src/data_processing/         # 数据处理核心 [AGENTS.md]
│   └── fineweb_edu/             # FineWeb-Edu 流水线 [AGENTS.md]
├── scripts/                     # 工具脚本
│   ├── trial_run.py             # 试运行
│   ├── validate_output.py       # 输出验证
│   ├── prepare_tokenizer_*.py   # Tokenizer 数据/模板准备
│   └── train_tokenizer.py       # 36K BPE 训练
├── tests/                       # pytest
├── config/                      # YAML 配置
│   ├── dataset.yaml             # 数据集 + 评分桶
│   ├── processing.yaml          # workers/tasks/compression
│   ├── tokenizer_data.yaml      # Tokenizer 采样配置
│   ├── tokenizer.yaml           # Tokenizer 训练配置
│   └── paths.yaml               # 路径
└── docs/                        # 设计文档 + KNOWLEDGE_BASE.md
```

## WHERE TO LOOK

| 任务 | 位置 | 说明 |
|------|------|------|
| 添加数据集 | `config/dataset.yaml` | 评分桶、归一化、路径 |
| Tokenizer 训练配置 | `config/tokenizer.yaml` | vocab_size、batch_size 等训练参数 |
| Tokenizer 数据配比 | `config/tokenizer_data.yaml` | EN/ZH/Code/Math 采样数 |
| 修改流水线参数 | `config/processing.yaml` | workers=32, tasks=2500 |
| 数据适配器 | `src/data_processing/fineweb_edu/adapters.py` | normalize_score |
| 试运行 | `scripts/trial_run.py --dataset zh` | 小规模测试 |
| 验证输出 | `scripts/validate_output.py --all` | 分桶结果校验 |
| 经验教训 | `docs/KNOWLEDGE_BASE.md` | 踩坑记录 + 最佳实践 |

## PUBLIC API

```python
from src.data_processing import (
    # 配置
    BucketConfig, Compression,
    find_bucket_for_score, get_all_bucket_configs,
    # PipelineSteps
    BucketPathWriter, ScoreFilter,
    # 处理
    fineweb_adapter, normalize_score,
    process_all_datasets, process_single_dataset,
    # 工具
    merge_all_buckets, merge_bucket_files,
    validate_all_buckets, validate_bucket, validate_file,
    print_report,
)
```

## COMMANDS

```bash
# 环境
conda create -n nanomind python=3.13 -y && conda activate nanomind
uv pip compile pyproject.toml -o requirements.txt && uv pip install -r requirements.txt

# 数据处理
python -m src.data_processing.fineweb_edu    # 处理所有数据集
python scripts/trial_run.py --dataset zh     # 试运行 (≤5 文件)

# Tokenizer 训练
python scripts/prepare_tokenizer_template.py
python scripts/prepare_tokenizer_data.py
python scripts/train_tokenizer.py --validate

# 测试
pytest                                        # 全部
pytest -xvs tests/test_bucket_config.py       # 单文件

# 代码质量
ruff check . && ruff format .
```

## CONVENTIONS

| 规则 | 标准 |
|------|------|
| Python | 3.13+ |
| 行长度 | 88 字符 |
| 引号 | 双引号 |
| 命名 | snake_case(函数), PascalCase(类), UPPER_CASE(常量) |
| 导入 | 标准库 → 第三方 → 本地模块 |
| 类型注解 | 函数签名必须 |
| 路径 | pathlib.Path |
| 缓存 | `@lru_cache` 延迟加载配置 |
| 日志 | `logger = logging.getLogger(__name__)` |

## ANTI-PATTERNS (NEVER)

| 禁止 | 原因 |
|------|------|
| `uv add` 不带 `--no-sync` | 使用 requirements.txt 工作流 |
| 提交 `uv.lock` | 已加入 .gitignore |
| 裸 `except:` | 使用具体异常类型 |
| 循环内 `import` | 影响性能 |
| 一次性加载大数据集 | 使用 `iter_batches()` 流式处理 |
| 共享 Datatrove `logging_dir` | 导致任务跳过 |
| 模块顶层加载配置 | 使用 `@lru_cache` 延迟加载 |
| 假设配置字段存在 | 使用 `.get(key, default)` |

## DEEP LEARNING

| 规则 | 说明 |
|------|------|
| `torch.no_grad()` | 推理必须 |
| 显式设备移动 | `.to(device)` |
| `tqdm` 进度条 | 长操作必须 |
| `gc.collect()` | 大数据集定期调用 |
| jemalloc | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2` |

## DEPENDENCY MANAGEMENT

```bash
# 添加依赖（必须 --no-sync）
uv add <package> --no-sync
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

## TESTING

```bash
pytest tests/test_bucket_config.py       # BucketConfig 测试
pytest tests/test_score_filter.py        # 采样测试
pytest tests/test_bucket_path_writer.py  # Writer 测试
pytest tests/test_parquet_merger.py      # 合并测试
```

Fixtures: `tests/conftest.py` (sample_document, sample_buckets, create_parquet)

## NOTES

- **不使用 uv.lock**: requirements.txt 工作流
- **数据集下载**: `hfd` 脚本下载到 `data/datasets/`
- **内存泄漏**: 启用 jemalloc (见 KNOWLEDGE_BASE.md 7.4)
- **详细文档**: `docs/KNOWLEDGE_BASE.md` 包含完整经验教训
- **无 CI/CD**: 手动运行测试和质量检查

## KNOWLEDGE BASE

> 📚 **按需查阅**: `docs/KNOWLEDGE_BASE.md` 包含项目全周期积累的技术知识、踩坑记录和最佳实践。在实现复杂功能或遇到问题时，可在此文档中搜索相关主题获取经验参考，无需通读全文。
