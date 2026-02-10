# AGENTS.md - src/data_processing

FineWeb-Edu 数据集质量评分分桶重组的数据处理流水线。

## 模块结构

```
src/data_processing/
├── __init__.py              # 模块导出
├── bucket_config.py         # 评分桶配置管理
├── bucket_path_writer.py    # 多桶并行 Parquet 写入器
├── config_loader.py         # YAML 配置加载器
├── parquet_merger.py        # Parquet 文件合并工具
├── score_filter.py          # 评分过滤 + 确定性采样
└── fineweb_edu/             # FineWeb-Edu 专用子模块
    ├── __init__.py          # 子模块导出
    ├── __main__.py          # CLI 入口
    ├── adapters.py          # 数据适配器
    └── reorganizer.py       # 数据处理流水线
```

## 核心组件速查

| 组件 | 文件 | 说明 |
|------|------|------|
| `BucketConfig` | `bucket_config.py` | 评分桶配置数据类 |
| `ScoreFilter` | `score_filter.py` | 评分过滤 + 确定性采样 |
| `BucketPathWriter` | `bucket_path_writer.py` | 多桶并行写入器 |
| `merge_all_buckets` | `parquet_merger.py` | 合并所有桶的文件 |
| `fineweb_adapter` | `adapters.py` | 数据适配器函数 |
| `process_all_datasets` | `reorganizer.py` | 处理所有配置的数据集 |

## 关键模式

### 评分桶区间
- 采用**左闭右开**区间：`[min_score, max_score)`
- 最后一个桶 `max_score=None` 表示无上界
- 桶列表按 `min_score` 排序，支持二分查找

### 确定性采样
```python
# 基于 MD5 哈希的确定性采样
h = int.from_bytes(hashlib.md5(f"{seed}_{doc_id}".encode()).digest()[:8], "big")
return h / (2**64) < sampling_rate
```

### 配置加载
```python
from src.data_processing import config_loader

# 自动处理环境变量覆盖
config = config_loader.get_dataset_config("en")
```

## 评分桶配置

| 数据集 | 评分范围 | 归一化 | 采样策略 |
|--------|----------|--------|----------|
| en | 1.0-5.0 | 无 | 2.5(25%), 3.0(50%), 3.5(80%), 4.0(100%) |
| zh | 0.0-1.0 | ×5 | 2.5(40%), 3.0(60%), 3.5(90%), 4.0(100%) |

## 测试覆盖

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_adapters.py` | ID 生成、评分归一化 |
| `test_bucket_config.py` | 区间匹配、配置加载 |
| `test_score_filter.py` | 过滤逻辑、采样算法 |
| `test_bucket_path_writer.py` | 多桶写入、文件分片 |
| `test_fineweb_reorganizer.py` | 流水线集成测试 |

## 相关文档

- [完整设计文档](../docs/fineweb_edu_data_reorganization_design.md) - 架构设计、扩展指南
- [模块 API 文档](./README.md) - 使用示例、CLI 说明
- [项目根 AGENTS.md](../../AGENTS.md) - 全局代码规范、环境配置
