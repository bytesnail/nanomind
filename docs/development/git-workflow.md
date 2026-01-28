# Git 工作流

## 概述

本指南介绍 nanomind 项目的 Git 工作流和提交规范。

---

## 常用命令

```bash
git add .                    # 暂存所有
git commit -m "exp: 添加实验"
git push origin main         # 推送
git pull origin main         # 拉取
git status                   # 查看状态
```

---

## 提交规范

### 提交格式

使用 `<type>: <description>` 格式：

| 类型 | 说明 | 示例 |
|------|------|------|
| `exp` | 新增实验 | `exp: 添加 Transformer 实验` |
| `fix` | 修复问题 | `fix: 修复环境检查脚本错误` |
| `docs` | 文档更新 | `docs: 更新实验管理指南` |
| `chore` | 构建/配置 | `chore: 更新依赖版本` |

### 提交示例

```bash
git commit -m "exp: 添加 GPT-2 文本生成实验"
git commit -m "fix: 修复数据加载器批处理错误"
```

---

## 注意事项

- 提交前运行 `black .` 和 `ruff check --fix .`
- 提交消息使用中文，描述清晰
- 保持提交小而专注
- 实验文件使用 `exp:` 前缀

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
