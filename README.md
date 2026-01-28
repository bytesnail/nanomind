# nanomind

深度学习、大语言模型学习与试验

## 项目目标

学习深度学习和LLM技术，快速原型设计，确保实验可复现

---

## 快速开始

```bash
conda activate nanomind && uv add torch torchvision transformers datasets --no-sync
uv pip compile pyproject.toml -o requirements.txt && uv pip install -r requirements.txt
python experiments/exp_000_environment_check.py
```

---

## 硬件配置

- **CPU**: 2× Intel Xeon E5-2667 v4 @ 3.20GHz (16 核 / 32 线程)
- **内存**: 251.59 GB
- **GPU**: 2× NVIDIA GeForce RTX 2080 Ti (21.5 GB ×2)

详细规格：[docs/environment/specs.md](docs/environment/specs.md)

---

## 文档导航

- [AGENTS.md](AGENTS.md) - 开发指南
- [docs/environment/](docs/environment/) - 环境管理
- [docs/development/](docs/development/) - 开发规范
- [docs/experiments/](docs/experiments/) - 实验管理

---

## 贡献

欢迎提交PR！格式化代码，使用 `exp:`、`fix:`、`docs:`、`chore:` 前缀。

---

## 许可证

MIT License
