# 数据获取和初始化指南

## 概述

本指南介绍如何为 nanomind 项目配置数据目录、下载数据集以及验证数据完整性。

---

## 数据目录初始化

### 数据目录结构

nanomind 项目使用标准的数据目录结构：

```
data/
└── datasets/               # 数据集存储目录
    ├── HuggingFaceFW/      # HuggingFace 数据集
    ├── opencsg/           # 其他数据源
    ├── HuggingFaceTB/
    ├── nick007x/
    └── nvidia/
```

### 设置数据目录

**方式一：使用符号链接（推荐）**

```bash
# 创建数据目录符号链接（指向外部数据存储）
ln -s /mnt/usr/data/datasets/ data/datasets

# 验证符号链接
ls -la data/
```

**方式二：本地存储**

```bash
# 创建数据目录
mkdir -p data/datasets

# 设置环境变量（可选）
export DATADIR=data/datasets
```

### 数据目录权限

```bash
# 确保有写入权限
chmod -R 755 data/datasets

# 或设置更宽松的权限（仅限开发环境）
chmod -R 777 data/datasets
```

---

## 下载数据集

### 使用 HuggingFace Datasets API

**方式一：从 HuggingFace Hub 下载**

```python
from datasets import load_dataset

# 下载 FineWeb-Edu 数据集
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train")

# 指定下载目录
dataset.save_to_disk("data/datasets/HuggingFaceFW/fineweb-edu")
```

**方式二：使用 CLI**

```bash
# 安装 HuggingFace CLI（如果尚未安装）
uv add huggingface_hub --no-sync

# 下载数据集
huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset --local-dir data/datasets/HuggingFaceFW/fineweb-edu
```

### 常用数据集下载

#### 1. FineWeb-Edu

```python
from datasets import load_dataset

# 下载 FineWeb-Edu 数据集
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train")

# 保存到本地
dataset.save_to_disk("data/datasets/HuggingFaceFW/fineweb-edu")

print(f"数据集大小: {len(dataset)} 样本")
```

#### 2. FineWeb-Edu Chinese

```python
from datasets import load_dataset

# 下载中文 FineWeb-Edu 数据集
dataset = load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", split="train")

# 保存到本地
dataset.save_to_disk("data/datasets/opencsg/Fineweb-Edu-Chinese-V2.1")

print(f"数据集大小: {len(dataset)} 样本")
```

#### 3. FineMath

```python
from datasets import load_dataset

# 下载 FineMath 数据集
dataset = load_dataset("HuggingFaceTB/finemath", split="train")

# 保存到本地
dataset.save_to_disk("data/datasets/HuggingFaceTB/finemath")

print(f"数据集大小: {len(dataset)} 样本")
```

#### 4. GitHub Code 2025

```python
from datasets import load_dataset

# 下载 GitHub Code 数据集
dataset = load_dataset("nick007x/github-code-2025", split="train")

# 保存到本地
dataset.save_to_disk("data/datasets/nick007x/github-code-2025")

print(f"数据集大小: {len(dataset)} 样本")
```

#### 5. Nemotron-CC-Math

```python
from datasets import load_dataset

# 下载 Nemotron-CC-Math 数据集
dataset = load_dataset("nvidia/Nemotron-CC-Math-v1", split="train")

# 保存到本地
dataset.save_to_disk("data/datasets/nvidia/Nemotron-CC-Math-v1")

print(f"数据集大小: {len(dataset)} 样本")
```

---

## 验证数据完整性

### 基本验证

```python
from datasets import load_from_disk

# 加载本地数据集
dataset = load_from_disk("data/datasets/HuggingFaceFW/fineweb-edu")

# 验证数据集信息
print(f"数据集大小: {len(dataset)}")
print(f"列名: {dataset.column_names}")
print(f"特征: {dataset.features}")

# 查看第一个样本
print(dataset[0])
```

### 数据集统计

```python
# 计算基本统计信息
from collections import Counter

# 统计 dump 分布（如果有）
if "dump" in dataset.column_names:
    dump_counts = Counter(dataset["dump"])
    print(f"Dump 分布: {dump_counts}")

# 统计 token 数量（如果有）
if "token_count" in dataset.column_names:
    token_counts = dataset["token_count"]
    print(f"平均 token 数: {sum(token_counts) / len(token_counts):.2f}")
    print(f"最大 token 数: {max(token_counts)}")
    print(f"最小 token 数: {min(token_counts)}")
```

### 数据完整性检查

```bash
# 检查数据目录结构
ls -lh data/datasets/

# 检查特定数据集
ls -lh data/datasets/HuggingFaceFW/fineweb-edu/

# 检查数据文件
du -sh data/datasets/*/

# 统计数据集数量
find data/datasets/ -name "dataset_info.json" | wc -l
```

---

## 数据集配置

### 更新 exp_001 配置

如果下载了新的数据集，需要在 `experiments/001/config.py` 中添加配置：

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetConfig:
    """数据集配置类。"""
    name: str
    path: str
    text_key: str = "text"
    id_key: Optional[str] = "id"
    group_field: Optional[str] = None
    group_by: Optional[str] = None
    score_field: Optional[str] = "score"
    int_score_field: Optional[str] = "int_score"
    glob_pattern: str = "**/*.parquet"

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    # 添加新的数据集配置
    "your-dataset-name": DatasetConfig(
        name="your-dataset-name",
        path="data/datasets/your-dataset-name/",
        text_key="text",  # 根据实际情况调整
        id_key="id",      # 根据实际情况调整
        glob_pattern="**/*.parquet",
    ),
    # ... 保留现有配置
}
```

### 测试新数据集

```bash
# 运行 exp_001 测试新数据集
python -m experiments.001 explore --dataset your-dataset-name --workers 4
```

---

## 常见问题

### 问题 1: 磁盘空间不足

**症状**: 下载过程中报错"磁盘空间不足"。

**解决方案**:
```bash
# 检查磁盘空间
df -h

# 清理缓存
rm -rf ~/.cache/huggingface/datasets

# 使用更小的数据集或部分数据
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train[:1000]")
```

### 问题 2: 下载速度慢

**症状**: 数据集下载非常缓慢。

**解决方案**:
```bash
# 配置镜像源（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### 问题 3: 数据集格式不兼容

**症状**: 加载数据集时报错"格式不支持"。

**解决方案**:
```python
# 检查数据集格式
from datasets import load_dataset

try:
    # 尝试加载
    dataset = load_from_disk("data/datasets/your-dataset")
except Exception as e:
    print(f"错误: {e}")
    print("可能需要转换数据集格式")
```

### 问题 4: 权限问题

**症状**: 保存数据集时报错"权限拒绝"。

**解决方案**:
```bash
# 检查权限
ls -ld data/datasets/

# 修改权限
chmod -R 755 data/datasets

# 或更改所有者
sudo chown -R $USER:$USER data/datasets
```

---

## 下一步

- [环境初始化](setup.md) - 配置开发环境
- [环境验证](verification.md) - 验证环境配置
- [exp-001 概览](../experiments/exp-001-overview.md) - 了解如何使用数据集进行实验

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介
