# 环境故障排查指南

## 概述

本指南提供 nanomind 项目开发过程中常见环境问题的解决方案，涵盖安装、配置和运行时的各类问题。

---

## 快速诊断

### 一键环境检查

```bash
# 运行环境验证脚本
python -m experiments.000

# 查看完整输出
python -m experiments.000 2>&1 | tee outputs/logs/troubleshooting_check.log
```

### 手动诊断步骤

```bash
# 1. 检查 Python 版本
python --version  # 应该是 Python 3.12.x

# 2. 检查核心依赖
python -c "import torch, transformers, datasets, torchvision; print('✓ 核心依赖已安装')"

# 3. 检查 GPU 和 CUDA
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"

# 4. 检查数据目录
ls -la data/datasets/

# 5. 检查输出目录
ls -la outputs/
```

---

## 安装问题

### 问题 1: conda 命令找不到

**症状**: 运行 `conda` 命令时报错"command not found"。

**解决方案**:
```bash
# 初始化 Conda shell
conda init bash

# 重新加载 shell
source ~/.bashrc

# 再次尝试
conda --version
```

### 问题 2: uv 安装失败

**症状**: 运行 `conda install -c conda-forge uv` 时报错。

**解决方案**:
```bash
# 方式一：使用 pip 安装
pip install uv

# 方式二：使用官方脚本安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证安装
uv --version
```

### 问题 3: 依赖安装失败

**症状**: 运行 `uv pip install -r requirements.txt` 时报错。

**解决方案**:
```bash
# 清理 uv 缓存
uv cache clean

# 更新 uv
uv self update

# 重新安装依赖
uv pip install -r requirements.txt

# 如果仍然失败，尝试逐个安装
uv add torch torchvision transformers datasets --no-sync
```

### 问题 4: PyTorch 版本不兼容

**症状**: `torch.cuda.is_available()` 返回 `False`。

**解决方案**:
```bash
# 检查 CUDA 版本
nvidia-smi

# 确认 CUDA 12.8 已安装
conda list | grep cuda

# 重新安装 CUDA
conda install -c nvidia cuda=12.8 -y

# 重新安装 PyTorch
uv pip install -r requirements.txt

# 验证
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"
```

---

## 配置问题

### 问题 1: Python 版本不匹配

**症状**: 项目需要 Python 3.12，但当前环境是其他版本。

**解决方案**:
```bash
# 创建新的 Python 3.12 环境
conda create -n nanomind python=3.12 -y

# 激活环境
conda activate nanomind

# 验证
python --version  # 应该输出 Python 3.12.x
```

### 问题 2: GPU 内存不足

**症状**: 运行实验时报错"RuntimeError: CUDA out of memory"。

**解决方案**:
```bash
# 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 减小 batch_size
python -m experiments.001 explore --dataset your-dataset --batch-size 16

# 使用更少的 workers
python -m experiments.001 explore --dataset your-dataset --workers 2

# 或使用 CPU 进行测试
CUDA_VISIBLE_DEVICES="" python -m experiments.001 explore --dataset your-dataset
```

### 问题 3: 数据目录权限问题

**症状**: 保存数据或结果时报错"Permission denied"。

**解决方案**:
```bash
# 检查目录权限
ls -la data/

# 修改权限
chmod -R 755 data/

# 或更改所有者
sudo chown -R $USER:$USER data/

# 验证
touch data/test.txt && rm data/test.txt
```

### 问题 4: 环境变量未设置

**症状**: 某些程序找不到数据目录或配置文件。

**解决方案**:
```bash
# 设置数据目录环境变量（可选）
export DATADIR=data/datasets

# 设置 CUDA 设备（可选）
export CUDA_VISIBLE_DEVICES=0,1

# 添加到 .bashrc 永久生效
echo 'export DATADIR=data/datasets' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0,1' >> ~/.bashrc
source ~/.bashrc
```

---

## 运行时问题

### 问题 1: 模块导入失败

**症状**: `ImportError: No module named 'xxx'`。

**解决方案**:
```bash
# 安装缺失的模块
uv add <module_name> --no-sync
uv pip install -r requirements.txt

# 如果是本地模块，检查 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 或使用 Python 的 -m 参数运行
python -m experiments.001 explore --dataset your-dataset
```

### 问题 2: DataTrove 处理失败

**症状**: 运行 exp_001 时 DataTrove 报错。

**解决方案**:
```bash
# 检查数据路径
ls -la data/datasets/your-dataset/

# 减少并发数量
python -m experiments.001 explore --dataset your-dataset --workers 2

# 检查数据集格式
python -c "from datasets import load_from_disk; ds = load_from_disk('data/datasets/your-dataset'); print(ds.column_names)"

# 查看 DataTrove 日志
tail -f outputs/logs/exp_001.log
```

### 问题 3: 实验脚本执行失败

**症状**: 运行实验脚本时出现意外错误。

**解决方案**:
```bash
# 查看详细错误信息
python -m experiments.001 --help

# 使用调试模式运行
python -m experiments.001 explore --dataset your-dataset --debug

# 查看完整日志
python -m experiments.001 2>&1 | tee outputs/logs/debug.log

# 检查输出目录
ls -la outputs/
```

### 问题 4: Jupyter Notebook 无法连接内核

**症状**: 启动 Jupyter Notebook 后无法连接 Python 内核。

**解决方案**:
```bash
# 安装 ipykernel
uv add ipykernel --no-sync
uv pip install -r requirements.txt

# 注册内核
python -m ipykernel install --user --name=nanomind

# 启动 Jupyter
jupyter notebook

# 在 Notebook 中选择 "nanomind" 内核
```

---

## 性能优化问题

### 问题 1: 训练速度慢

**症状**: 实验运行速度比预期慢很多。

**解决方案**:
```bash
# 检查 GPU 使用率
nvidia-smi -l 1

# 启用 torch.compile（PyTorch 2.10.0+）
# 在代码中添加：
# model = torch.compile(model)

# 增加 batch_size（如果内存允许）
python -m experiments.001 explore --dataset your-dataset --batch-size 64

# 使用混合精度训练（如适用）
# 在代码中添加：
# scaler = torch.cuda.amp.GradScaler()
```

### 问题 2: 数据加载慢

**症状**: 数据加载阶段耗时很长。

**解决方案**:
```bash
# 使用更多 workers
python -m experiments.001 explore --dataset your-dataset --workers 8

# 启用数据缓存
export DATATROVE_CACHE_DIR=~/.cache/datatrove

# 使用 SSD 存储数据（如果使用 HDD）
# 将 data/ 目录移动到 SSD
```

### 问题 3: 内存使用过高

**症状**: 程序占用大量系统内存。

**解决方案**:
```bash
# 监控内存使用
free -h

# 减小 batch_size
python -m experiments.001 explore --dataset your-dataset --batch-size 16

# 使用数据流式处理（如果支持）
# 在 DataTrove 配置中启用流式处理

# 清理未使用的变量
import gc
gc.collect()
```

---

## 日志和调试

### 查看日志

```bash
# 查看最新的实验日志
tail -f outputs/logs/exp_001.log

# 查看环境检查日志
cat outputs/logs/exp_000_environment_check.log

# 搜索错误信息
grep -i "error" outputs/logs/*.log

# 搜索警告信息
grep -i "warning" outputs/logs/*.log
```

### 启用详细日志

```bash
# 设置环境变量启用详细日志
export PYTHONPATH=$(pwd)
export LOG_LEVEL=DEBUG

# 运行实验
python -m experiments.001 explore --dataset your-dataset --verbose
```

### 调试技巧

```python
# 使用 Python 调试器
import pdb; pdb.set_trace()

# 或使用 ipdb（如果已安装）
import ipdb; ipdb.set_trace()

# 打印变量值
print(f"变量值: {variable}")

# 检查变量类型
print(f"变量类型: {type(variable)}")
```

---

## 网络问题

### 问题 1: HuggingFace 下载慢

**症状**: 从 HuggingFace 下载数据集非常慢。

**解决方案**:
```bash
# 配置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 测试连接
ping huggingface.co
```

### 问题 2: 依赖下载失败

**症状**: `uv pip install` 下载包时失败。

**解决方案**:
```bash
# 使用国内镜像源
uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云镜像
uv pip install -r requirements.txt --index-url https://mirrors.aliyun.com/pypi/simple/
```

---

## 重置环境

### 完全重置

```bash
# 1. 停止所有运行中的进程
pkill -f python
pkill -f jupyter

# 2. 退出当前环境
conda deactivate

# 3. 删除环境（谨慎使用）
conda env remove -n nanomind -y

# 4. 重新创建环境
conda create -n nanomind python=3.12 -y
conda activate nanomind

# 5. 重新安装依赖
uv pip install -r requirements.txt

# 6. 验证环境
python -m experiments.000
```

### 清理缓存

```bash
# 清理 Python 缓存
find . -type d -name __pycache__ -exec rm -rf {} +

# 清理 HuggingFace 缓存
rm -rf ~/.cache/huggingface/

# 清理 uv 缓存
uv cache clean

# 清理 DataTrove 缓存
rm -rf ~/.cache/datatrove/

# 清理 outputs 目录（可选）
rm -rf outputs/logs/*
rm -rf outputs/results/*
```

---

## 获取帮助

### 查看文档

```bash
# 查看项目文档
cat README.md
cat AGENTS.md

# 查看环境设置文档
cat docs/environment/setup.md
cat docs/environment/verification.md
```

### 搜索解决方案

```bash
# 在日志中搜索关键词
grep -r "error" outputs/logs/

# 搜索代码中的特定模式
grep -r "cuda" experiments/

# 查找文档中的相关信息
grep -r "GPU" docs/
```

### 联系支持

如果以上解决方案都无法解决您的问题，请：

1. 收集完整的错误信息和日志
2. 记录您的环境配置（Python 版本、CUDA 版本等）
3. 查看项目 Issue 页面或提交新的 Issue

---

## 下一步

- [环境初始化](setup.md) - 配置开发环境
- [依赖管理](dependencies.md) - 管理项目依赖
- [数据获取](data-setup.md) - 下载数据集

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介
