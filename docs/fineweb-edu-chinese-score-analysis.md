# Fineweb-Edu-Chinese-V2.1 Score 字段分析

## 问题背景

本地数据集 `opencsg/Fineweb-Edu-Chinese-V2.1` 中的 `score` 字段值为 0.4-0.93，而用户期望的是 1.0-5.0 质量评分。

## 核心发现

- **存储格式**: 归一化浮点数 0.0-1.0（原始评分 0.0-5.0 经 BERT 分类器归一化）
- **映射公式**: `original_score = normalized_score × 5`
- **官方验证**: [OpenCSG 回复](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu-v2/discussions/2) *"multiply the score by 5"*

数据集经过质量过滤，仅保留原始评分 ≥ 3.0 的样本：

| 文件夹 | 归一化范围 | 原始评分 |
|--------|-----------|---------|
| `2_3/` | 0.40-0.60 | 2.0-3.0 |
| `3_4/` | 0.60-0.80 | 3.0-4.0 |
| `4_5/` | 0.80-0.94 | 4.0-4.70 |

## 常见误区

| ❌ 错误 | ✅ 正确 |
|--------|--------|
| score 是 1-5 分 | 实际为归一化值 0-1 |
| 最高是 5.0 分 | 实际最高约 4.70 分 |
| 使用 `(score-1)/4` | 正确公式是 `score×5` |
