# 依赖管理

## 概述

本指南介绍如何管理 nanomind 项目的依赖包，包括查看、升级、版本兼容性和回滚。

---

## 查看依赖状态

### 查看过期的依赖

```bash
# 查看可以升级的包
uv pip list --outdated
```

### 查看当前已安装版本

```bash
# 核心依赖
python -c "import torch; print('torch:', torch.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import datasets; print('datasets:', datasets.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"

# 查看所有已安装的包
uv pip list
```

---

## 当前版本

### 当前版本（2026-01-28）

| 组件 | 版本 |
|------|------|
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.0.0 |
| Datasets | 4.5.0 |
| Torchvision | 0.25.0+cu128 |

---

## 版本兼容性矩阵

### 核心依赖兼容性

| PyTorch | CUDA | Transformers | 兼容性 | 说明 |
|---------|------|--------------|--------|------|
| 2.10.0 | 12.8 | 5.0.0 | ✅ | 当前版本 |
| 2.9.1 | 12.8 | 4.57.6 | ✅ | 上一版本 |
| 2.10.0 | 11.8 | 5.0.0 | ❌ | CUDA 版本不兼容 |
| 2.9.1 | 12.8 | 5.0.0 | ⚠️ | 部分功能可能降级 |

### 升级路径

**从 2.9.1 升级到 2.10.0:**
```bash
# 1. 升级 PyTorch
uv add torch>=2.10.0 --no-sync

# 2. 升级 Transformers
uv add transformers>=5.0.0 --no-sync

# 3. 重新生成 requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# 4. 安装更新
uv pip install -r requirements.txt

# 5. 验证
python experiments/exp_000_environment_check.py
```

### 常见版本冲突

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `ImportError: torch` | PyTorch 未安装或版本不兼容 | `uv pip install -r requirements.txt` |
| CUDA 不可用 | CUDA 版本与 PyTorch 不匹配 | 确认 CUDA 12.8 与 PyTorch 2.10.0 兼容 |
| Transformers API 错误 | 使用了已废弃的 API | 查看 [Transformers 5.0.0 变更说明](#transformers-500-api-变更) |

---

## 升级依赖

### 升级所有主要依赖到最新版本

```bash
# 1. 更新 pyproject.toml 中的版本约束
uv add torch torchvision transformers datasets --upgrade --no-sync

# 2. 重新生成 requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# 3. 安装更新后的依赖
uv pip install -r requirements.txt

# 4. 验证安装
python -c "
import torch
import transformers
import datasets
import torchvision
print('✓ torch:', torch.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ datasets:', datasets.__version__)
print('✓ torchvision:', torchvision.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
"

# 5. 运行环境检查
python experiments/exp_000_environment_check.py
```

### 升级单个包

```bash
# 例如只升级 torch
uv add torch --upgrade --no-sync
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

### 升级到特定版本

```bash
# 编辑 pyproject.toml，指定精确版本
# 例如: "torch==2.10.0" 或 "torch>=2.10.0"

uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

---

## Transformers 5.0.0 API 变更

### 主要变更

Transformers 5.0.0 移除了一些破坏性变更，需要注意：

| 旧 API | 新 API | 兼容性 |
|--------|--------|--------|
| `tokenizer.encode_plus()` | `tokenizer()` | 部分兼容（取决于 tokenizer） |
| `tokenizer.batch_decode()` | `tokenizer.decode(batch=True)` | 仍可用 |
| `load_in_4bit` / `load_in_8bit` | `quantization_config` | 已移除 |
| `torchscript` / `torch.fx` 支持 | - | 已移除 |

### 代码示例

**旧代码（可能不兼容）:**
```python
# GPT2Tokenizer 不再支持 encode_plus
inputs = tokenizer.encode_plus(text, return_tensors="pt")
```

**新代码（推荐）:**
```python
# 使用新的 API
inputs = tokenizer(text, return_tensors="pt")

# decode 仍可用
result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# batch_decode 仍可用于批处理
results = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
```

### 详细变更

详细信息见：[Transformers v5 发布说明](https://github.com/huggingface/transformers/releases/tag/v5.0.0)

---

## 升级后验证

### 基础验证

```bash
# 验证所有包导入正常
python -c "import torch, transformers, datasets, torchvision"

# 验证 GPU 可用
python -c "import torch; print('GPU 数量:', torch.cuda.device_count())"
```

### 模型加载测试

创建测试脚本 `test_upgrade.py`：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 测试基础模型加载
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("✓ 模型加载成功")

# 测试生成功能
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=20)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("✓ 生成测试通过:", result[:50])
```

运行测试：
```bash
python test_upgrade.py
```

### 完整环境检查

```bash
python experiments/exp_000_environment_check.py 2>&1 | tee outputs/logs/exp_000_environment_check.log
```

---

## 回滚方案

### Git 回滚（推荐）

```bash
# 1. 查看之前的提交
git log --oneline -5

# 2. 回滚到升级前的提交
git revert <commit-hash>

# 3. 重新安装旧版本
uv pip install -r requirements.txt

# 4. 验证
python experiments/exp_000_environment_check.py
```

### 手动回滚

```bash
# 1. 编辑 pyproject.toml 恢复旧版本约束
# 例如: 将 "torch>=2.10.0" 改回 "torch==2.9.1"

# 2. 重新生成 requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# 3. 安装旧版本
uv pip install -r requirements.txt

# 4. 验证
python experiments/exp_000_environment_check.py
```

---

## 常见问题

### Q: 升级后 CUDA 不可用？

**解决方案:**
```bash
# 检查 CUDA 版本匹配
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 确保 conda 中的 CUDA 版本匹配
conda install -c nvidia cuda=12.8 -y

# 重新安装 PyTorch
uv pip install -r requirements.txt
```

### Q: transformers 加载模型报错？

**解决方案:**
```bash
# 清理缓存
rm -rf ~/.cache/huggingface/transformers

# 重新下载模型
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
```

### Q: uv pip install 失败？

**解决方案:**
```bash
# 清理 uv 缓存
uv cache clean

# 重新安装
uv pip install -r requirements.txt
```

---

## 提交依赖变更

```bash
# 添加文件
git add pyproject.toml requirements.txt

# 提交
git commit -m "chore: 升级依赖包到最新版本

- torch 2.9.1 → 2.10.0
- torchvision 0.24.1 → 0.25.0
- transformers 4.57.6 → 5.0.0
- datasets 保持 4.5.0"
```

---

## 下一步

- [环境初始化](setup.md) - 配置开发环境
- [环境验证](verification.md) - 验证环境配置是否正确

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
