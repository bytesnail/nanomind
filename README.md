# nanomind
深度学习、大语言模型学习与试验

## 环境设置

**⚠️ 重要**：本项目使用 `requirements.txt` 工作流，**添加依赖必须使用 `--no-sync` 参数**，否则生成 `uv.lock`（已在 `.gitignore` 忽略）。

### 环境准备

```bash
conda create -n nanomind python=3.13 -y
conda activate nanomind
conda install -c conda-forge uv -y
conda install -c nvidia cuda=12.9 -y
```

### 项目初始化

```bash
uv init
```

### 依赖管理

**添加依赖**：
```bash
uv add --dev black ruff pytest --no-sync
uv add torch torchvision transformers datasets datatrove[all] matplotlib tqdm --no-sync
```

**编译并安装**：
```bash
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

**工作流**：`uv add --no-sync` → `uv pip compile` → `uv pip install`

**注意**：修改 `pyproject.toml` 后需重新编译 requirements.txt
