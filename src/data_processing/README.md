# FineWeb-Edu 数据集重组项目

## 项目结构

```
nanomind/
├── src/data_processing/
│   ├── __init__.py
│   ├── adapters.py              # 数据适配器（字段筛选）
│   ├── bucket_config.py         # 评分桶配置
│   ├── score_filter.py          # 评分过滤器和采样
│   ├── metadata_cleaner.py      # 元数据清理器
│   ├── cc_main_path_writer.py   # CC-MAIN 路径写入器
│   ├── fineweb_reorganizer.py   # 主入口和 CLI
│   └── README.md                # 模块说明
├── tests/
│   ├── __init__.py
│   ├── test_bucket_config.py    # 评分桶配置测试
│   ├── test_adapters.py         # 适配器测试
│   └── test_score_filter.py     # 评分过滤器测试
├── scripts/
│   ├── validate_output.py       # 验证脚本
│   ├── monitor_processing.py    # 监控脚本
│   └── run_processing.sh        # 批量运行脚本
└── docs/
    └── fineweb_edu_data_reorganization_design.md  # 设计文档
```

## 使用方法

### 处理所有评分桶

```bash
python -m src.data_processing.fineweb_reorganizer
```

### 处理指定评分桶

```bash
python -m src.data_processing.fineweb_reorganizer --bucket 3.0
```

### 指定 workers 和随机种子

```bash
python -m src.data_processing.fineweb_reorganizer --workers 16 --seed 42
```

### 并行处理多个桶

```bash
python -m src.data_processing.fineweb_reorganizer --parallel-buckets 4
```

### 验证输出

```bash
python scripts/validate_output.py --input data/datasets/fineweb/en
```

### 监控处理过程

```bash
python scripts/monitor_processing.py --output data/datasets/fineweb
```

### 使用批量运行脚本

```bash
./scripts/run_processing.sh --workers 16 --parallel-buckets 2
```

## 评分桶配置

| 桶名称 | 评分区间 | 采样率 |
|--------|----------|--------|
| 2.8 | 2.8 ≤ score < 3.0 | 30% |
| 3.0 | 3.0 ≤ score < 3.5 | 60% |
| 3.5 | 3.5 ≤ score < 4.0 | 80% |
| 4.0 | score ≥ 4.0 | 100% |

## 输出目录结构

```
data/datasets/fineweb/
└── en/
    ├── 2.8/
    │   └── CC-MAIN-XXXX-XX/
    │       └── {rank}.parquet
    ├── 3.0/
    │   └── CC-MAIN-XXXX-XX/
    │       └── {rank}.parquet
    ├── 3.5/
    │   └── CC-MAIN-XXXX-XX/
    │       └── {rank}.parquet
    └── 4.0/
        └── CC-MAIN-XXXX-XX/
            └── {rank}.parquet
```
