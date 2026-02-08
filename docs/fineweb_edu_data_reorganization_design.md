# FineWeb-Edu 数据集质量评分分桶重组

> **目标数据集**: HuggingFaceFW/fineweb-edu (score≥2 版本)  
> **输出目录**: `data/datasets/fineweb/en/`

## 设计目标

按质量评分（`score` 字段）对 FineWeb-Edu 数据集进行分桶重组，低质量数据通过采样率控制保留比例：

| 质量评分区间 | 桶名称 | 采样率 | 说明 |
|-------------|--------|--------|------|
| 2.8 ≤ score < 3.0 | 2.8 | 30% | 中低质量数据 |
| 3.0 ≤ score < 3.5 | 3.0 | 60% | 中等质量数据 |
| 3.5 ≤ score < 4.0 | 3.5 | 80% | 高质量数据 |
| score ≥ 4.0 | 4.0 | 100% | 顶级质量数据 |

**输出字段**: `id`, `text`, `score`

**输出目录结构**:
```
data/datasets/fineweb/en/
├── 2.8/
│   └── {rank}_{counter}.parquet
├── 3.0/
│   └── {rank}_{counter}.parquet
├── 3.5/
│   └── {rank}_{counter}.parquet
└── 4.0/
    └── {rank}_{counter}.parquet
```

## 核心组件

### 1. 评分桶配置 (`bucket_config.py`)

```python
@dataclass(frozen=True)
class BucketConfig:
    name: str          # 桶名称
    min_score: float   # 最小评分（含）
    max_score: float   # 最大评分（不含），None表示无上界
    sampling_rate: float  # 采样率 0.0-1.0
```

- 区间采用**左闭右开**（最后桶无上界）
- 配置从 `config/buckets.yaml` 加载

### 2. 评分过滤器 (`score_filter.py`)

基于 Datatrove PipelineStep 实现：

- **区间过滤**: 根据评分桶配置过滤数据
- **确定性采样**: 使用 MD5 哈希确保可复现性
  ```python
  data = f"{random_seed}_{doc_id}".encode()
  h = int.from_bytes(hashlib.md5(data).digest()[:8], "big")
  return h / (2**64) < rate
  ```
- **统计追踪**: 记录 `missing_score`, `filtered_out`, `kept_{bucket}`, `sampled_out_{bucket}`

### 3. 数据适配器 (`adapters.py`)

- **ID 生成**: `{相对路径}#{索引}`，相对路径从 `fineweb-edu` 后开始
- **字段提取**: `text`, `score`, `dump` → `text`, `id`, `score`, `cc_main`
- **数据验证**: 空文本返回 `None`

### 4. 桶路径写入器 (`bucket_path_writer.py`)

继承 Datatrove PipelineStep：

- 输出文件名: `{rank:05d}_{counter:05d}.parquet`
- 支持压缩格式: `zstd`, `gzip`, `snappy`, `brotli`, `lz4`
- 默认文件大小限制: 512MB
- 多桶并行写入支持

### 5. 配置加载器 (`config_loader.py`)

支持从 `config/` 目录加载 YAML 配置：

| 文件 | 内容 |
|------|------|
| `buckets.yaml` | 评分桶定义 |
| `processing.yaml` | workers、tasks、random_seed、compression、max_file_size_bytes |
| `paths.yaml` | input_dir、output_dir，支持 `FINEWEB_{KEY}` 环境变量覆盖 |
| `dataset.yaml` | 数据集字段配置、root_marker |

### 6. 主处理器 (`fineweb_reorganizer.py`)

CLI 入口，实现**一次读取，多桶并行处理**的高效模式：

```python
# 处理流程：Reader -> ScoreFilter -> BucketPathWriter
pipeline = [
    ParquetReader(str(input_dir), adapter=fineweb_adapter, glob_pattern="**/*.parquet"),
    ScoreFilter(buckets=buckets, random_seed=seed),
    BucketPathWriter(
        output_dir=str(output_dir),
        buckets=buckets,
        compression=compression,
        max_file_size=max_size,
    ),
]
```

**性能优化**:
- **单次读取**: 输入数据集只被读取一次，避免重复 I/O
- **并行分发**: 根据评分将文档路由到对应桶，同时应用各桶的采样率
- **内存缓冲**: 每个桶独立缓冲，达到文件大小限制时批量写入
- **统计追踪**: 按桶统计 `kept_{bucket}`, `sampled_out_{bucket}`, `written_{bucket}`

## 使用方法

### 主程序

```bash
# 处理所有评分桶
python -m src.data_processing.fineweb_reorganizer

# 处理指定评分桶
python -m src.data_processing.fineweb_reorganizer --bucket 3.0

# 指定 workers 和随机种子
python -m src.data_processing.fineweb_reorganizer --workers 16 --seed 42

# 单独指定 tasks 数量（控制 Datatrove pipeline 并行度）
python -m src.data_processing.fineweb_reorganizer --workers 8 --tasks 16

# 指定压缩格式和文件大小
python -m src.data_processing.fineweb_reorganizer --compression zstd --max-file-size 536870912
```

### 生产环境运行

```bash
# 使用 time 命令统计运行时间
time python -m src.data_processing.fineweb_reorganizer --workers 16

# 处理完成后自动验证
python -m src.data_processing.fineweb_reorganizer --workers 16 && \
  python scripts/validate_output.py --input data/datasets/fineweb/en
```

### 试运行

```bash
# 创建测试数据并运行完整流程
python scripts/trial_run.py

# 分析采样准确性
python scripts/trial_run.py --analyze-sampling
```

### 验证输出

```bash
# 验证所有桶
python scripts/validate_output.py --input data/datasets/fineweb/en

# 验证指定桶
python scripts/validate_output.py --input data/datasets/fineweb/en --bucket 3.0

# 输出 JSON 报告
python scripts/validate_output.py --input data/datasets/fineweb/en --json report.json
```

## 项目结构

```
nanomind/
├── src/data_processing/
│   ├── __init__.py              # 模块导出
│   ├── adapters.py              # 数据适配器（fineweb_adapter）
│   ├── bucket_config.py         # 评分桶配置（BucketConfig）
│   ├── bucket_path_writer.py    # Parquet 写入器（多桶支持）
│   ├── config_loader.py         # YAML 配置加载器
│   ├── fineweb_reorganizer.py   # CLI 主入口（多桶处理器）
│   └── score_filter.py          # 评分过滤器和采样
├── scripts/
│   ├── trial_run.py             # 试运行脚本
│   └── validate_output.py       # 输出验证脚本
├── config/
│   ├── buckets.yaml             # 评分桶配置
│   ├── dataset.yaml             # 数据集字段配置
│   ├── paths.yaml               # 路径配置
│   └── processing.yaml          # 处理参数配置
├── tests/
│   ├── test_adapters.py         # 适配器测试
│   ├── test_bucket_config.py    # 评分桶配置测试
│   ├── test_bucket_path_writer.py  # 桶写入器测试
│   └── test_score_filter.py     # 评分过滤器测试
└── docs/
    ├── fineweb_edu_data_reorganization_design.md  # 本文档
    └── fineweb-edu-chinese-score-analysis.md      # 中文数据集评分分析
```

## 相关参考

### FineWeb-Edu 中文数据集评分分析

> 详见: [`fineweb-edu-chinese-score-analysis.md`](./fineweb-edu-chinese-score-analysis.md)

本地数据集 `opencsg/Fineweb-Edu-Chinese-V2.1` 的 `score` 字段使用**归一化浮点数**（0.0-1.0），与英文原版（1.0-5.0）不同：

- **映射公式**: `original_score = normalized_score × 5`
- **官方验证**: [OpenCSG 回复](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu-v2/discussions/2)

| 文件夹 | 归一化范围 | 原始评分 |
|--------|-----------|---------|
| `2_3/` | 0.40-0.60 | 2.0-3.0 |
| `3_4/` | 0.60-0.80 | 3.0-4.0 |
| `4_5/` | 0.80-0.94 | 4.0-4.70 |

**注意**: 本项目当前实现针对英文版 FineWeb-Edu（1.0-5.0 评分），如需处理中文数据集需先将归一化评分转换为原始评分。

## 依赖

- **Python**: 3.13+
- **核心库**: datatrove, pyarrow, pyyaml
- **开发库**: pytest, ruff

**数据许可**: FineWeb-Edu 基于 ODC-BY 1.0 许可证发布
