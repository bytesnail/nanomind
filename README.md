# nanomind

深度学习、大语言模型学习与试验

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

