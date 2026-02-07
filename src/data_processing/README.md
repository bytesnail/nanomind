# FineWeb-Edu 数据处理模块

将 FineWeb-Edu 数据集按质量评分分桶重组，支持分层采样。

## 核心组件

| 文件 | 功能 |
|------|------|
| `adapters.py` | 数据适配器（字段筛选、ID 生成） |
| `bucket_config.py` | 评分桶配置（区间定义、采样率） |
| `score_filter.py` | 评分过滤器（区间过滤、确定性采样） |
| `bucket_path_writer.py` | Parquet 写入器（输出到桶目录） |
| `config_loader.py` | YAML 配置加载器（支持环境变量覆盖） |
| `fineweb_reorganizer.py` | CLI 主入口 |

## 评分桶配置

配置位于 `config/buckets.yaml`：

| 桶名称 | 评分区间 | 采样率 |
|--------|----------|--------|
| 2.8 | 2.8 ≤ score < 3.0 | 30% |
| 3.0 | 3.0 ≤ score < 3.5 | 60% |
| 3.5 | 3.5 ≤ score < 4.0 | 80% |
| 4.0 | score ≥ 4.0 | 100% |

区间采用**左闭右开**，最后桶无上界。

## 使用方法

```bash
# 处理所有评分桶
python -m src.data_processing.fineweb_reorganizer

# 处理指定评分桶
python -m src.data_processing.fineweb_reorganizer --bucket 3.0

# 指定 workers 和随机种子
python -m src.data_processing.fineweb_reorganizer --workers 16 --seed 42

# 单独指定 tasks 数量（控制 Datatrove pipeline 并行度）
python -m src.data_processing.fineweb_reorganizer --workers 8 --tasks 16
```

## 配置文件

位于 `config/` 目录：

- `buckets.yaml`: 评分桶定义
- `processing.yaml`: 处理参数
- `paths.yaml`: 路径配置（支持环境变量 `FINEWEB_{KEY}` 覆盖）
- `dataset.yaml`: 数据集字段配置

详见设计文档 `docs/fineweb_edu_data_reorganization_design.md`
