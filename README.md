# nanomind

深度学习、大语言模型学习与试验

## 环境配置

### 硬件配置
- **CPU**: 2× Intel Xeon E5-2667 v4 @ 3.20GHz
  - 物理核心数: 16 核（每颗 8 核）
  - 逻辑处理器: 32 核（超线程启用）
- **内存**: 251.59 GB
- **GPU**: 2× NVIDIA GeForce RTX 2080 Ti
  - 显存: 约 21.5 GB × 2
  - 计算能力: 7.5

### 软件环境
- **操作系统**: Linux (Kernel 6.12.63)
- **Python**: 3.12.12
- **PyTorch**: 2.9.1+cu128
- **CUDA**: 12.8
- **cuDNN**: 9.10.2

### 环境验证
运行环境检查脚本验证配置：
```bash
python experiments/exp_000_environment_check.py
```

同时输出到终端和日志文件：
```bash
python experiments/exp_000_environment_check.py 2>&1 | tee outputs/logs/exp_000_environment_check.log
```

---

## 环境初始化

### 1. 创建 Conda 环境

```bash
conda create -n nanomind python=3.12 -y
```

### 2. 激活环境并安装 uv

```bash
conda activate nanomind
conda install -c conda-forge uv -y
```

### 3. 初始化 uv 项目

```bash
uv init
```

### 4. 安装依赖

```bash
# 安装cuda
conda install -c nvidia cuda=12.8 -y

# 添加项目依赖
uv add torch torchvision transformers datasets --no-sync

# 生成requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# 安装到conda环境
uv pip install -r requirements.txt
```

