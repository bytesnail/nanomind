# 依赖管理

## 概述

本指南介绍如何管理 nanomind 项目的依赖包，包括查看、升级、版本兼容性和回滚。

---

## uv 包管理器

### 什么是 uv

**uv** 是一个用 Rust 编写的超快 Python 包管理器，专为现代 Python 开发设计。

**主要特性**：
- **超快速度**: 比 pip 快 10-100 倍
- **兼容性**: 完全兼容 PyPI 和 pip 命令
- **依赖解析**: 快速且精确的依赖解析
- **多环境支持**: 支持虚拟环境管理和依赖锁定

### uv 与 pip 对比

| 特性 | uv | pip |
|------|-----|-----|
| **速度** | 超快（Rust 实现）| 较慢（Python 实现）|
| **依赖解析** | 快速且精确 | 可能较慢 |
| **兼容性** | 完全兼容 pip | 标准 |
| **虚拟环境** | 内置支持 | 需要额外工具 |
| **依赖锁定** | 支持 | 不支持 |
| **推荐场景** | 日常开发、CI/CD | 系统级安装 |

### 基本使用

#### 添加依赖

```bash
# 添加依赖到 pyproject.toml（不自动安装）
uv add <package> --no-sync

# 示例：添加 torch
uv add torch --no-sync

# 添加多个依赖
uv add torch torchvision transformers --no-sync

# 指定版本
uv add torch==2.10.0 --no-sync
```

#### 移除依赖

```bash
# 移除依赖
uv remove <package>

# 示例：移除 tensorboard
uv remove tensorboard
```

#### 同步依赖

```bash
# 同步依赖到虚拟环境
uv sync

# 这会：
# 1. 安装 pyproject.toml 中的所有依赖
# 2. 移除不再需要的依赖
# 3. 使用 uv.lock 锁定版本
```

#### 生成 requirements.txt

```bash
# 从 pyproject.toml 生成 requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# 安装 requirements.txt 中的所有依赖
uv pip install -r requirements.txt
```

#### 搜索包

```bash
# 搜索包
uv search <package>

# 示例：搜索 torch
uv search torch
```

### 项目工作流

#### 标准流程

```bash
# 1. 添加新依赖
uv add <package> --no-sync

# 2. 更新 requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# 3. 同步到虚拟环境
uv pip install -r requirements.txt

# 4. 验证
python -c "import <package>; print(<package>.__version__)"
```

#### 快速流程

```bash
# 直接添加并安装（不推荐，绕过 requirements.txt）
uv add <package>
```

### 高级用法

#### 使用锁文件

```bash
# uv.lock 文件会自动创建
# 它确保所有环境使用相同版本的依赖

# 查看锁文件
cat uv.lock

# 使用锁文件同步
uv sync --lockfile uv.lock
```

#### 离线模式

```bash
# 在离线环境中使用预下载的包
uv pip install -r requirements.txt --offline

# 或使用本地包目录
uv pip install -r requirements.txt --no-index --find-links ./packages/
```

#### 缓存管理

```bash
# 查看缓存
uv cache dir

# 清理缓存
uv cache clean

# 查看缓存大小
du -sh $(uv cache dir)
```

### 常见问题

**Q: uv 和 pip 可以混用吗？**

A: 可以，但不推荐。uv 完全兼容 pip，但混用可能导致依赖冲突。

**Q: uv 支持所有 Python 包吗？**

A: uv 支持绝大多数 PyPI 包，但某些特殊包可能仍需要使用 pip。

**Q: 如何在 CI/CD 中使用 uv？**

A: 使用 `uv sync` 命令，它会确保环境的一致性。

```bash
# CI/CD 示例
python -m pip install uv
uv sync
python -m pytest
```

**Q: uv 的性能优势在哪里？**

A: 主要在以下方面：
- 依赖解析：使用 Rust 实现，速度更快
- 并行下载：支持多线程下载
- 缓存机制：更智能的缓存策略

### 最佳实践

1. **使用 `--no-sync` 标志**: 添加依赖时使用 `--no-sync`，然后使用 `uv pip compile` 和 `uv pip install`。
2. **提交 requirements.txt**: 始终将 requirements.txt 提交到版本控制。
3. **锁定版本**: 使用 `uv.lock` 文件确保环境一致性。
4. **定期清理缓存**: 使用 `uv cache clean` 清理过期的缓存。
5. **团队协作**: 团队成员使用相同的 uv 版本和 requirements.txt。

### 相关文档

- [uv 官方文档](https://docs.astral.sh/uv/)
- [uv GitHub 仓库](https://github.com/astral-sh/uv)

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

### 当前版本（2026-01-31）

| 组件 | 版本 |
|------|------|
| Python | 3.12.x |
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
python -m experiments.000
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
python -m experiments.000
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

Transformers 5.0.0 是一个重要版本更新，引入了多项变更：

#### 1. 分词器变更

- **移除"Fast"和"Slow"概念**：不再区分 FastTokenizer 和 SlowTokenizer
- **统一使用 tokenizers 后端**：所有分词器现在都使用 Rust 实现的 tokenizers 库
- **API 简化**：`tokenizer.encode_plus()` 现在可以直接使用 `tokenizer()` 替代

#### 2. 框架支持变更

- **专注 PyTorch**：放弃 Flax 和 TensorFlow 支持，专注优化 PyTorch 实现
- **性能提升**：专注于单一框架后端，带来了更好的性能和代码简洁性
- **迁移建议**：如果之前使用 Flax/TensorFlow，需要迁移到 PyTorch

#### 3. 注意力接口（AttentionInterface）

- **引入 AttentionInterface**：抽象注意力方法，支持自定义注意力实现
- **灵活性提升**：允许开发者更容易地实现和集成自定义注意力机制
- **向前兼容**：大多数现有代码无需修改，但新 API 提供了更强的扩展性

#### 4. 量化支持

- **量化成为一等公民**：`load_in_4bit` 和 `load_in_8bit` 参数仍然支持
- **量化配置**：使用 `quantization_config` 进行更灵活的量化配置
- **性能优化**：针对量化的性能进行了优化

### API 变更对照表

| 旧 API | 新 API | 兼容性 |
|--------|--------|--------|
| `tokenizer.encode_plus()` | `tokenizer()` | 推荐使用 |
| `tokenizer.batch_decode()` | `tokenizer.decode(batch=True)` | 仍可用 |
| `load_in_4bit` | 仍可用 | ✅ 兼容 |
| `load_in_8bit` | 仍可用 | ✅ 兼容 |
| `torchscript` / `torch.fx` 支持 | `torch.export` | 推荐迁移 |

### 代码示例

**分词器使用（推荐方式）:**
```python
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# 使用新的 API（推荐）
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# decode 仍可用
result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# batch_decode 仍可用于批处理
results = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
```

**量化配置（推荐方式）:**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 使用量化配置加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

### 详细变更

详细信息见：[Transformers v5.0.0 发布说明](https://huggingface.co/docs/transformers/en/whats_new#version-5-0-0)

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
python -m experiments.000 2>&1 | tee outputs/logs/exp_000_environment_check.log
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
python -m experiments.000
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
python -m experiments.000
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
