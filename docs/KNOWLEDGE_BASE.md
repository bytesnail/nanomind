# nanomind 分类知识库

| 属性 | 值 |
|------|-----|
| **覆盖范围** | `94eeacd` ~ `de2c58d` (178 commits) |
| **最后更新** | `de2c58d` @ 2026-03-03 |

> 💡 **使用方式**：`根据docs/KNOWLEDGE_BASE.md的增量更新指南，分析 所有未分析commit 的知识库更新计划` → 输出计划书 → `执行计划` → `应用更新`

---

## 增量更新指南

> ⚠️ 本章节提供更新规则，如需修改本章节请先与用户确认。

### 术语定义

| 术语 | 定义 |
|------|------|
| **元信息** | 文档头部表格（覆盖范围 + 最后更新），格式固定，仅更新内容 |
| **新知识** | 代码库中首次出现的模式/技术/解决方案 |
| **知识变更** | 已记录知识的版本升级、替代方案、废弃或修正 |
| **重大变更** | 满足以下任一：新增/删除模块、修改公开API、涉及安全漏洞、变更核心依赖 |

### 触发条件

| 条件 | 动作 |
|------|------|
| 含任意重大变更（commit 数不限） | 走简化流程：直接输出更新内容 → 用户确认 |
| 待分析 commit 数 ≥ 3 | 启动更新分析（走完整流程） |
| 其他情况 | 忽略，不做任何更新 |
### 更新原则

| 场景 | 动作 |
|------|------|
| **新知识** | 对应章节追加 |
| **知识变更** | 直接修改原内容 |
| **过时/错误** | 直接删除 |
| **重复内容** | 合并至一处 |

### 代码示例规则

- ✅ 保留：≤10行（核心逻辑超限时，拆分为摘要 + 多个关键片段），优先级：核心逻辑 > 简洁 > 完整上下文
- ❌ 禁止：整段复制完整函数/类实现
- 📎 替代：摘要（功能描述 + 关键 API/参数）+ 文件路径:行号

### 章节归属

| 知识类型 | 目标章节 |
|----------|----------|
| 纯 Python 语法 | 一、Python 语法知识 |
| 框架/库 API | 二、框架知识 |
| 项目结构/配置 | 三、项目架构知识 |
| 参数/命令 | 四、相关参数与命令 |
| 最佳实践 | 五、最佳实践 |
| 踩坑/修复 | 六、经验教训 |
| FAQ | 七、常见问题 |
| **跨章节** | 主章节记录核心，其他章节用 → 引用 |

### 计划书格式

每次更新前输出分析框架，而非分析结果：

```markdown
## 更新计划 [波次 N/M]

**待分析 Commits**: `abc123` `def456` ...（共 X 个）
**分组策略**: 按章节/相关性拆分为 M 个波次

### 本波次分析项

| 分析项 | 输出格式 | 说明 |
|--------|----------|------|
| commit 归属章节 | 章节名称列表 | 每个 commit 影响哪些章节 |
| 变更类型判定 | 新增/修改/删除 | 对现有知识的影响类型 |
| 代码示例提取 | 文件路径:行号 | 需提取的代码片段位置 |
| 冗余检测 | 重复条目列表 | 方法：关键词搜索 + 相同文件路径 + 技术点重合 |
```

> 💡 分批次原因：待分析内容可能较多，一次性分析难以保证准确性和完整性。
### 执行流程

```
1. 列出待分析 commits → 按数量/相关性分组
2. 输出计划书（分析框架）→ 等待用户确认
3. 用户批准 → 按波次执行分析
4. 每波次分析完成 → 输出该批次更新内容 → 用户确认
5. 全部波次完成 → 更新元信息
6. 用户拒绝/超时（>24h）→ 结束本次更新，不做任何修改
```

> 📝 **指南演进**: 本指南经过多次迭代（`c9fbe67`, `e8ac07e`, `31e7790`, `0fe4cfd`, `8197409`），逐步增强结构化与可操作性。

### 优先级

```
用户明确指令 > 简洁性 > 完整性
```

**冲突场景处理**：
- 简洁性 vs 完整性冲突 → 优先摘要 + 文件路径:行号引用
- 不确定时 → 询问用户

### 底部版本号

文档底部 `*文档版本: vX.Y*` 随每次内容更新递增：
- 小修（错别字、格式）：X.Y → X.Y+1
- 内容变更（新增/修改/删除）：X.Y → X+1.0
---


## 一、Python 语法知识

### 1.1 类型注解

```python
# Union (Python 3.10+)
def find_bucket(score: float) -> BucketConfig | None: ...  # 推荐
# 等价: Optional[BucketConfig]

# Final 不可变 (PEP 591)
EPSILON: Final = 1e-6

# Literal 字面量 (PEP 586)
CompressionType = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]

# TypeAlias (Python 3.10+)
DocHash: TypeAlias = tuple[int, str, Path, int]

# Forward Reference 避免循环导入
self._bloom_filter: "ScalableBloomFilter | None" = None

# TYPE_CHECKING 条件导入
if TYPE_CHECKING:
    from pybloom_live import ScalableBloomFilter
```

### 1.2 数据类

```python
@dataclass(frozen=True)
class BucketConfig:  # frozen=True 使实例不可变，可作字典键
    name: str
    min_score: float
    max_score: float | None

@dataclass
class SamplingConfig:
    buckets: dict[str, int] = field(default_factory=dict)  # 避免可变默认参数陷阱

class DocHash(NamedTuple):  # 轻量数据结构
    hash_value: int
    doc_id: str
    file_path: Path
    row_index: int
```

### 1.3 海象运算符与生成器

```python
# 海象运算符
if not (text := raw.get("text", "")): raise ValueError("text is missing")
if bucket := _BUCKET_MAP.get(name): return bucket
(path := log_dir / bucket).mkdir(parents=True, exist_ok=True)

# 生成器
def stream_file_rows(file_path: Path) -> Generator[tuple[int, str], None, None]:
    with pq.ParquetFile(file_path) as pf:
        for batch in pf.iter_batches(batch_size=10000):
            for row_idx, text in enumerate(batch.column("text").to_pylist()):
                yield row_idx, text
```

### 1.4 装饰器与常用技巧

```python
# 延迟加载
@property
def buckets(self) -> dict[str, Any]:
    return self._load("buckets")

# 缓存
@lru_cache(maxsize=1024)
def get_file_row_count(file_path: Path) -> int:
    return pq.read_metadata(file_path).num_rows

# next + 生成器表达式：返回第一个匹配项
return next((b for b in DEFAULT_BUCKETS if b.contains(score)), None)
```

### 1.5 其他语法

```python
# 链式比较
return self.min_score - EPSILON <= score < self.max_score

# 三元表达式
interval = f"[{self.min_score}, +∞)" if self.max_score is None else f"[{self.min_score}, {self.max_score})"

# 数字分隔符 (PEP 515)
bloom_capacity: int = 2_000_000_000
max_file_size: int = 512 * 1024 * 1024

# hashlib 确定性哈希
def compute_doc_hash(doc_id: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.md5(f"{seed}_{doc_id}".encode(), usedforsecurity=False).digest()[:8], "big"
    )
```

---

## 二、框架知识

### 2.1 datatrove Pipeline

```python
# 自定义 PipelineStep
class ScoreFilter(PipelineStep):
    name = "Score Filter"
    type = "🎯 - FILTER"
    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            if self._should_keep(doc): yield doc

# LocalPipelineExecutor
executor = LocalPipelineExecutor(
    pipeline=[ParquetReader(...), ScoreFilter(...), BucketPathWriter(...)],
    tasks=2500, workers=32, logging_dir=str(log_path),
)

# ParquetReader adapter
def fineweb_adapter(_reader, raw: dict, source: str, idx: int) -> dict:
    return {"text": raw.get("text", ""), "id": f"{source}#{idx}",
            "metadata": {"score": raw.get("score")}}

# Document 结构: text(必需), id(必需), metadata(dict), media(可选)
```

### 2.2 PyArrow/Parquet

```python
# 流式读取（推荐）
with pq.ParquetFile(file_path) as pf:
    for batch in pf.iter_batches(batch_size=10000, columns=["text"]):
        process(batch)
# 避免: pq.read_table(file_path)  # 可能 OOM

# 快速获取行数
row_count = pq.read_metadata(file_path).num_rows

# 流式写入
schema = pa.schema([("text", pa.string()), ("id", pa.string())])
with pq.ParquetWriter(output_path, schema, compression="zstd") as writer:
    for batch in batches: writer.write_table(pa.table(batch))
```

### 2.3 concurrent.futures

```python
# ThreadPoolExecutor (IO 密集型)
with ThreadPoolExecutor(max_workers=32) as executor:
    future_to_file = {executor.submit(read_func, f): f for f in files}
    for future in as_completed(future_to_file):  # as_completed 按完成顺序，更快
        file_path, result = future.result()
```

### 2.4 argparse / pytest / YAML

```python
# argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", "-i", type=Path, default=Path("data/..."))
parser.add_argument("--bucket", choices=["2.8", "3.0", "3.5", "4.0"])

# pytest 参数化
@pytest.mark.parametrize("score,expected", [(3.5, "3.5"), (4.2, "4.0")])
def test_find_bucket(score, expected):
    assert find_bucket_for_score(score).name == expected

# YAML 安全加载
config = yaml.safe_load(f) or {}
```

### 2.5 HuggingFace Tokenizers

```python
# train_new_from_iterator — 自动继承模板配置
new_tokenizer = template.train_new_from_iterator(
    text_iterator,
    vocab_size=36005,
)
# 自动继承: normalizer, pre_tokenizer, decoder, post_processor

# 特殊 Token 定义
SPECIAL_TOKENS = ["", "<|im_start|>", "<|im_end|>", "oliniht", ""]

# tokenizer.json 后处理：vocab 重新映射
special_set = set(SPECIAL_TOKENS)
filtered = [(t, id) for t, id in vocab.items() if t not in special_set]
new_vocab = {t: new_id for new_id, (t, _) in enumerate(sorted(filtered))}

# extra_special_tokens vs added_tokens
# added_tokens: 5个（含 ）
# extra_special_tokens: 4个（不含 ，仅对话+推理token）
```

**训练流程**:
1. 加载模板 tokenizer: `AutoTokenizer.from_pretrained(template_dir)`
2. 创建文本迭代器: `create_text_iterator()` 流式读取 Parquet
3. 训练: `template.train_new_from_iterator(text_iterator, vocab_size)`
4. 后处理: vocab 重新映射 + added_tokens ID 调整
5. 保存: `tokenizer.save_pretrained(output_dir)`

---

## 三、项目架构知识

### 3.1 包结构

```
src/data_processing/
├── __init__.py              # 公共 API 导出
├── config_loader.py         # 配置加载
├── bucket_config.py         # 评分桶配置（通用）
├── score_filter.py          # 评分过滤器（通用）
├── bucket_path_writer.py    # 桶路径写入器（通用）
└── fineweb_edu/             # FineWeb-Edu 专用
    ├── __main__.py          # CLI: python -m src.data_processing.fineweb_edu
    └── adapters.py          # 数据适配器
```

### 3.2 配置分层与设计模式

```
config/
├── buckets.yaml      # 业务：评分桶定义
├── processing.yaml   # 运行：workers, tasks, compression
└── paths.yaml        # 路径：输入/输出目录
```

```python
# 延迟加载模式
_DEFAULT_BUCKETS: list[BucketConfig] | None = None
def get_all_bucket_configs() -> list[BucketConfig]:
    global _DEFAULT_BUCKETS
    if _DEFAULT_BUCKETS is None: _DEFAULT_BUCKETS = _load_buckets()
    return _DEFAULT_BUCKETS
```

### 3.3 Pipeline 架构

```
ParquetReader → ScoreFilter → BucketPathWriter
     ↓              ↓              ↓
  读取数据      过滤+采样      写入文件
```

| 阶段 | 操作 | 内存 |
|------|------|------|
| 预计算 | 采样索引 | O(target × 16 bytes) |
| 处理 | 流式 Pipeline | 不累积 |

### 3.4 Tokenizer 训练系统

```
scripts/
├── train_tokenizer.py              # BPE 训练主脚本
├── prepare_tokenizer_data.py       # 多数据源采样（两遍处理）
├── prepare_tokenizer_template.py   # 模板准备
config/
└── tokenizer_data.yaml             # 数据采样配置
output/
├── qwen3_next_tokenizer/           # 模板 tokenizer
└── tokenizer_36k/                  # 训练输出
```

**Pipeline 架构**:
```
ParquetReader → IndexFilter → LanguageTagger → SourceTagger → TokenizerDataWriter
```

**两遍处理架构**:
| 阶段 | 操作 | 内存占用 |
|------|------|----------|
| 第一遍 | 预计算采样索引（只读元数据 + 计算哈希） | O(target × 16 bytes) |
| 第二遍 | 流式读取选中文档并写入 Parquet | 流式，不累积 |

**采样算法**:
- 使用确定性哈希（MD5 前 8 字节）+ 最大堆采样
- 目标数 ≥ 总数 90% 时自动切换为全量处理模式

**GitHub Code 语言过滤**:
| 语言 | 扩展名 |
|------|--------|
| Python | `.py`, `.pyw`, `.pyi` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` |
| C/C++ | `.c`, `.h`, `.cpp`, `.hpp`, `.cc` |
| Rust | `.rs` |
| 其他 | HTML/CSS/Markdown/JSON/XML/TOML 等 |

---

## 四、相关参数与命令

### 4.1 uv 包管理

```bash
uv add <package> --no-sync              # 添加（必须带 --no-sync）
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```

### 4.2 性能参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `workers` | `min(16, cpu_count)` | 并行进程数 |
| `tasks` | `workers` | 数据分片数 |
| `max_file_size` | 10GB | 输出文件上限 |
| `batch_size` | 50,000 | 批次大小 |

```python
# 并发公式
max_workers = min(32, cpu_count * 2)  # IO 密集型
io_workers = max_workers * 2
```

### 4.3 数据集转换

| 数据集 | 原始范围 | 转换 |
|--------|----------|------|
| FineWeb-EN | 0-5 | 无 |
| FineWeb-ZH | 0-1 | `× 5` |

### 4.4 Tokenizer 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vocab-size` | 36005 | 词表大小（36000 BPE + 5 特殊 token） |
| `--batch-size` | 10000 | Parquet 读取批次大小 |
| `--data-dir` | `data/datasets/nanomind_tokenizer` | 训练数据目录 |
| `--template-dir` | `output/qwen3_next_tokenizer` | 模板目录 |
| `--output-dir` | `output/tokenizer_36k` | 输出目录 |
| `--validate` | True | 是否执行验证 |

**数据采样配置** (`config/tokenizer_data.yaml`):
| 数据集 | 样本数 | 占比 |
|--------|--------|------|
| FineWeb-EN | 720K | 24% |
| FineWeb-ZH | 1.2M | 40% |
| GitHub Code | 660K | 22% |
| Nemotron Math | 420K | 14% |

**采样配置字段**:
| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 数据集名称，用于标记输出数据来源 |
| `source` | 是 | 源数据目录路径 |
| `samples` | 是 | 目标采样数量 |
| `buckets` | 二选一 | 质量分桶配置（FineWeb、Nemotron） |
| `stars_filter` | 二选一 | Stars 过滤配置（GitHub Code） |

**配置演进历史**:
| 日期 | 词表 | 样本 | EN | ZH | Code | Math | Commit |
|------|------|------|----|----|------|------|--------|
| 2026-02-20 | 24K | - | - | - | - | - | `f3fdebf` |
| 2026-02-20 | 32K | - | - | - | - | - | `88ed2ec` |
| 2026-02-22 | 32K | 0.8M | 24% | 28% | 32% | 16% | `b496fb0` |
| 2026-02-22 | 32K | 1.2M | - | - | 30% | 18% | `97dc0c7` |
| 2026-02-27 | 36K | 2M | 26% | 30% | 26% | 18% | `c6bff2c` |
| 2026-02-28 | 36K | 2.4M | - | - | - | - | `e286bd8` |
| 2026-03-01 | 36K | 3M | - | - | - | - | `ca27976` |
| 2026-03-01 | 36K | 3M | 24% | 40% | 22% | 14% | `fbbf238` |

> 💡 最终配置：36K 词表 + 3M 样本（EN:720K/24%, ZH:1.2M/40%, Code:660K/22%, Math:420K/14%）

---

## 五、最佳实践

- **类型注解**：函数签名必须类型注解
- **路径处理**：使用 `pathlib.Path`
- **资源管理**：使用 `with` 语句
- **错误处理**：精确异常捕获，显式检查替代 assert
- **日志**：使用 `logging` 而非 `print`
- **CLI**：提供短/长选项，返回标准退出码

---

## 六、经验教训 ⚠️

### 6.1 DO（推荐做法）

**依赖管理**
- `uv add <package> --no-sync` → `uv pip compile` → `uv pip install -r`
- 必须带 `--no-sync` 避免 uv.lock

**配置管理**
- `yaml.safe_load()` 而非 `yaml.load()`
- 延迟加载避免循环依赖，保留合理默认值

**代码质量**
- 命名常量替代魔法数字
- 可复用逻辑提取到模块
- 定期 `gc.collect()` 防止内存碎片

**数据处理**
- `encode("utf-8")` 获取字符串实际字节大小
- `iter_batches()` 流式读取大文件
- 存储整数索引而非完整对象

**并发处理**
- `as_completed()` 获取已完成任务
- IO 密集型用 `ThreadPoolExecutor`
- 在 `with` 语句中使用 Executor

**框架使用**
- PipelineStep 设置 `name` 和 `type` 属性
- adapter 函数扩展 Reader 行为

**Tokenizer 训练**
- `train_new_from_iterator` 自动继承模板配置，避免手动复制
- 两遍处理分离预计算和数据读取，降低内存峰值
- 使用 `gc.collect()` 每批次后清理内存
- `ParquetFile.iter_batches()` 实现真正的流式读取

### 6.2 DO NOT（不推荐做法）

| 类别 | 禁止 |
|------|------|
| 配置 | 模块顶层直接加载、假设字段一定存在 |
| 代码 | 裸 `except:`、生产代码用 `assert`、`print` 输出日志 |
| 数据 | 假设固定宽度编码、全量加载后处理、循环内 import |
| 并发 | 串行逻辑处理并行场景、堆中存储 Path 等大对象 |
| 架构 | 可复用逻辑留在脚本、数据集逻辑放通用模块 |
| Tokenizer | 忽略 `added_tokens` 与 `extra_special_tokens` 的差异、手动复制模板配置项 |

### 6.3 NEVER（禁止做法）

| 类别 | 禁止 |
|------|------|
| 依赖 | `pip install`、`uv add` 不带 `--no-sync`、提交 uv.lock |
| 类型 | `as any`、`@ts-ignore`、`@ts-expect-error` |
| 数据 | 负数索引/空路径生成 ID、采样循环创建大量临时对象 |
| 框架 | 忽略 Datatrove 任务检测、Writer 返回非 None |
| 内存 | 一次性加载大规模数据集、共享 Datatrove 日志目录 |
| Tokenizer | 训练时遗漏 `--validate` 验证、跳过 tokenizer.json 后处理步骤 |

### 6.4 修复记录

| Commit | 问题 | 解决 |
|--------|------|------|
| `8d51fea` | TokenizerDataWriter 多批次写入覆盖 | 修复 IndexFilter 索引格式兼容性 |
| `2d6b14f` | IndexFilter 路径匹配失败 | 修复路径匹配逻辑 |
| `c9cae5e` | metadata 合并顺序 + 日志目录冲突 | 调整合并顺序，独立日志目录 |

---

## 七、常见问题

### 7.1 文件大小估算错误
- **问题**：输出 20-200MB 而非 1-2GB
- **原因**：`len(text) * 2` 估算 UTF-8
- **解决**：`len(text.encode("utf-8")) + 32`

### 7.2 Datatrove 跳过任务
- **问题**：后续数据集被跳过
- **原因**：共享 `logging_dir`
- **解决**：`log_name = f"multi_bucket_{output_dir.name}"`

### 7.3 IndexFilter 不生效
- **问题**：无法过滤文档
- **原因**：ParquetReader 默认不设置 `row_idx`
- **解决**：adapter 添加 `metadata["row_idx"] = id_in_file`

### 7.4 内存泄漏
- **问题**：长时间运行内存增长
- **原因**：glibc ptmalloc2 碎片
- **解决**：启用 jemalloc
```python
if os.path.exists("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"):
    os.environ.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")
```

---

*文档版本: v3.0 | 生成日期: 2026-03-03*
