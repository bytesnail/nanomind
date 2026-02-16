# AGENTS.md - nanomind

**Generated:** 2026-02-17
**Commit:** 2afba8e
**Branch:** main

## OVERVIEW

深度学习、大语言模型学习与试验项目。

## STRUCTURE

```
nanomind/
├── src/data_processing/     # 核心数据处理模块 [AGENTS.md]
│   └── fineweb_edu/         # FineWeb-Edu 专用流水线 [AGENTS.md]
├── scripts/                 # 工具脚本 (trial_run, validate_output)
├── tests/                   # pytest 测试套件
├── config/                  # YAML 配置文件
└── docs/                    # 设计文档 + KNOWLEDGE_BASE.md
```

## WHERE TO LOOK

| 任务 | 位置 | 说明 |
|------|------|------|
| 添加新数据集 | `config/dataset.yaml` | 定义评分桶和路径 |
| 修改处理参数 | `config/processing.yaml` | workers, compression, 文件大小 |
| 扩展 PipelineStep | `src/data_processing/` | ScoreFilter, BucketPathWriter |
| 数据适配器 | `src/data_processing/fineweb_edu/adapters.py` | normalize_score |
| 试运行 | `scripts/trial_run.py` | 小规模测试 |
| 验证输出 | `scripts/validate_output.py` | 分桶结果校验 |

## COMMANDS

```bash
# 环境
conda create -n nanomind python=3.13 -y && conda activate nanomind
uv pip compile pyproject.toml -o requirements.txt && uv pip install -r requirements.txt

# 运行
python -m src.data_processing.fineweb_edu    # 主流程
python scripts/trial_run.py                   # 试运行

# 测试
pytest                                        # 全部
pytest -xvs tests/test_bucket_config.py       # 单文件

# 代码质量
ruff check . && ruff format .                 # 检查+格式化
```

## CONVENTIONS

| 规则 | 标准 |
|------|------|
| Python | 3.13+ |
| 行长度 | 88 字符 |
| 引号 | 双引号优先 |
| 命名 | snake_case(函数), PascalCase(类), UPPER_CASE(常量) |
| 导入 | 标准库 → 第三方 → 本地模块 |
| 类型注解 | 函数签名必须 |
| 路径 | pathlib.Path，优先相对路径 |

## ANTI-PATTERNS (NEVER)

| 禁止 | 原因 |
|------|------|
| `uv add` 不带 `--no-sync` | 项目使用 requirements.txt 工作流 |
| 提交 `uv.lock` | 已加入 .gitignore |
| 裸 `except:` | 使用具体异常类型 |
| 循环内 `import` | 影响性能 |
| 一次性加载大数据集 | 使用流式处理 |
| 共享 Datatrove `logging_dir` | 导致任务跳过 |

## DEEP LEARNING

| 规则 | 说明 |
|------|------|
| `torch.no_grad()` | 推理必须使用 |
| 显式设备移动 | `.to(device)` |
| `tqdm` 进度条 | 长操作必须 |
| `gc.collect()` | 大数据集定期调用 |

## DEPENDENCY MANAGEMENT

```bash
# 添加依赖（必须 --no-sync）
uv add <package> --no-sync
uv add --dev <package> --no-sync

# 更新锁定
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

## NOTES

- **不使用 uv.lock**: 项目采用 requirements.txt 工作流
- **数据集下载**: 使用 `hfd` 脚本下载到 `data/datasets/`
- **内存泄漏**: 启用 jemalloc `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`
- **详细文档**: `docs/KNOWLEDGE_BASE.md` 包含经验教训
