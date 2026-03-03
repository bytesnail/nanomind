# nanomind 分类知识库

| 属性 | 值 |
|------|-----|
| **覆盖范围** | `94eeacd` ~ `b551a08` (191 commits) |
| **最后更新** | `b551a08` @ 2026-03-04 |
| **文档版本** | v4.0 |

---

## 增量更新指南

> ⚠️ **执行前置条件**：本指南定义知识库增量更新的标准操作流程。更新本文档其他内容前，必须先执行本指南。

### 核心目标

根据代码库新 commits 的变化，将相关知识同步更新到本文档，保持知识库与实际代码的一致性。

---

### 1. 前置检查（必须执行）

#### 步骤 1.1：读取当前元信息
```python
# 从本文档第 1-9 行读取以下信息
current_start = "94eeacd"  # 当前覆盖范围起始 commit
current_end = "b551a08"    # 当前覆盖范围结束 commit
current_version = "v4.0"   # 当前文档版本
```

#### 步骤 1.2：获取待分析 commits
```bash
# 获取从 current_end 到 HEAD 的所有新 commits
git log --oneline {current_end}..HEAD

# 示例输出：
# abc1234 新增 Tokenizer 数据采样配置
# def5678 修复 IndexFilter 路径匹配 bug
# hij9012 更新依赖版本
```

#### 步骤 1.3：判定更新策略

| 条件 | 判定结果 | 执行动作 |
|------|----------|----------|
| **重大变更** | 满足任一：新增/删除模块、修改公开 API、安全漏洞、核心依赖变更 | **完整流程**：执行第 2-5 章 |
| **常规变更** | 新 commits ≥ 3 且无重大变更 | **简化流程**：输出变更摘要 → 等待用户确认 → 直接更新文档 |
| **无需更新** | 新 commits < 3 且无重大变更 | **终止**：不做任何更新，向用户说明原因 |

**重大变更判定示例**：
- ✅ 是：`新增 src/data_processing/new_module/`
- ✅ 是：`修改 BucketConfig.__init__` 签名
- ✅ 是：`升级 pyarrow 从 12.x 到 14.x`
- ❌ 否：`修复注释错别字`
- ❌ 否：`调整日志输出格式`

---

### 2. 核心术语定义

执行更新前，必须明确以下术语的精确含义：

| 术语 | 定义                  | 判定标准 | 示例 |
|------|---------------------|----------|------|
| **元信息** | 文档第 1-9 行的头部表格     | 每次更新必须同步修改 | 覆盖范围、最后更新 commit |
| **新知识** | 知识库中首次出现的模式/技术/解决方案 | 全文搜索关键词无结果 | 新依赖、新脚本、新配置项 |
| **知识变更** | 已记录知识的版本升级/替代/废弃/修正 | 已有内容需要更新 | 配置字段更名、API 参数变更 |
| **重大变更** | 影响系统架构或公共接口的变更      | 可能导致用户代码失效 | 新增模块、删除脚本、修改函数签名 |

---

### 3. 知识处理决策矩阵

对于每个新发现的知识点，必须按以下矩阵执行：

| 变更类型 | 判定条件 | 执行动作 | 禁止动作 |
|----------|----------|----------|----------|
| **新增知识** | 知识库中无相同/相似内容 | 在目标章节末尾追加 | 不要插入到章节中间 |
| **知识变更** | 已有内容需要更新/替换 | **直接修改**原内容 | 不要保留旧版本描述 |
| **过时/错误** | 技术已废弃或记录错误 | **直接删除**相关条目 | 不要标记为"已废弃" |
| **重复内容** | 多章节出现相同技术点 | **合并**至一处，其他用 `→` 引用 | 不要重复记录完整内容 |

**判定流程**：
1. 在本文档全文搜索关键词
2. 若未找到 → **新增知识**
3. 若找到相同主题 → 对比内容是否一致
4. 若不一致 → **知识变更**（更新）或 **过时/错误**（删除）
5. 若相同内容出现在多处 → **重复内容**（合并）

---

### 4. 章节归属判定

#### 4.1 知识类型映射表

| 知识类型 | 判定特征 | 目标章节 | 章节锚点 |
|----------|----------|----------|----------|
| Python 语法技巧 | `typing`、`dataclass`、装饰器等语言特性 | 一、Python 语法知识 | #python-语法知识 |
| 第三方库使用 | `datatrove`、`pyarrow`、`tokenizers` 等 | 二、框架知识 | #框架知识 |
| 项目结构/配置 | 目录结构、YAML 配置、模块关系 | 三、项目架构知识 | #项目架构知识 |
| 命令/参数 | CLI 命令、脚本参数、配置项 | 四、相关参数与命令 | #相关参数与命令 |
| 代码规范 | 命名约定、最佳实践 | 五、最佳实践 | #最佳实践 |
| 问题与修复 | 踩坑记录、bug 修复 | 六、经验教训 | #经验教训 |
| 常见疑问 | FAQ 类问题 | 七、常见问题 | #常见问题 |

#### 4.2 跨章节处理规则

- **单一知识**：在**最主要**的章节记录完整内容
- **引用格式**：其他章节使用 `→` 指向主位置
  ```markdown
  详见 [项目架构知识](#项目架构知识) →
  ```
- **禁止**：在不同章节重复记录相同技术的完整细节

---

### 5. 代码示例规范

#### 5.1 保留条件（必须同时满足）
1. 代码行数 ≤ 10 行
2. 能独立、完整表达核心逻辑

#### 5.2 超长代码处理

**正确做法**（摘要 + 精确引用）：
```markdown
使用 StreamWriter 实现分批写入（scripts/parquet_merger.py:45-78）
```

**错误做法**：
```markdown
- ❌ 复制完整函数/类实现
- ❌ 省略关键 API/参数
- ❌ 使用模糊路径引用（如"见 scripts 目录"）
```

#### 5.3 路径引用格式

| 场景 | 格式 | 示例 |
|------|------|------|
| 文件引用 | `路径:行号` | `src/config.py:25-30` |
| 函数引用 | `路径:函数名` | `src/utils.py:load_config()` |
| 范围引用 | `路径:起始-结束` | `scripts/train.py:45-78` |

---

### 6. 完整更新流程

#### 6.1 流程概览

适用于：常规变更（commits ≥ 3，无重大变更）

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: 分组 → 按章节/相关性将 commits 分为 M 个波次        │
├─────────────────────────────────────────────────────────────┤
│ 步骤 2: 计划 → 输出【更新计划书】→ 等待用户确认              │
├─────────────────────────────────────────────────────────────┤
│ 步骤 3: 执行 → 启动子 agent 并行分析各波次                   │
│         （详见 6.3 子 agent 使用规范）                       │
├─────────────────────────────────────────────────────────────┤
│ 步骤 4: 整合 → 汇总各波次结果，去重并检测冲突                │
├─────────────────────────────────────────────────────────────┤
│ 步骤 5: 更新 → 修改文档内容 + 同步元信息                     │
│         （覆盖范围、最后更新 commit、版本号）                │
└─────────────────────────────────────────────────────────────┘
```

#### 6.2 Commit 分组策略

**分组原则**：
- 同一模块/章节的 commits 归为一组
- 存在依赖关系的 commits 归为一组
- 每组 3-5 个 commits 为宜

**分组示例**：
```
波次 1/3: Tokenizer 相关 (commits: abc123, def456, hij789)
- 新增 tokenizer_data.yaml 配置
- 修改采样算法
- 更新文档

波次 2/3: 配置系统 (commits: klm012, nop345)
- 统一配置加载方式
- 新增延迟加载模式

波次 3/3: 工具脚本 (commits: qrs678, tuv901, wxy234)
- 新增 parquet_merger.py
- 修复路径处理 bug
- 优化日志输出
```

#### 6.3 子 agent 使用规范（关键步骤）

**为什么必须使用子 agent？**
- 每批次分析需阅读多个 commit 的代码变更、关联文件、上下文
- 主 agent 上下文窗口有限（通常 128K-200K tokens）
- 全部分析放入主 agent 将导致早期信息被挤出，降低质量

**执行策略**：

| 场景 | 方案 | 说明 |
|------|------|------|
| M 个波次 | 启动 M 个子 agent（`run_in_background=true`） | 每个子 agent 独立分析一个波次 |
| 单波次 > 5 commits | 波次内再拆分 | 每 3-5 个 commits 一个子任务 |
| 依赖复杂 | 分层分析 | 先分析依赖，再分析具体变更 |

**子 agent 任务模板**（必须按此格式分配）：

```markdown
## 子 agent 任务 [波次 X/M]

**待分析 commits**: `abc123`, `def456`, `hij789`（共 3 个）

**分析目标**:
1. 逐 commit 分析变更内容
2. 判定归属章节（使用第 4 章映射表）
3. 判定变更类型（使用第 3 章决策矩阵）
4. 提取需记录的代码示例位置

**约束条件**:
- 只分析本波次的 commits
- 不要跨波次关联
- 输出格式见【波次分析结果格式】

**输出要求**:
- 按【更新决策矩阵】判定每项变更类型
- 明确建议的文档修改操作
```

**波次分析结果格式**（子 agent 必须输出）：

```markdown
## 波次 X 分析结果

### Commit `abc123` - 新增 Tokenizer 数据采样配置
- **归属章节**: 三、项目架构知识
- **变更类型**: 新增知识
- **内容摘要**: 新增 config/tokenizer_data.yaml，定义 EN/ZH/Code/Math 数据采样配比
- **代码提取**: config/tokenizer_data.yaml:1-25
- **建议操作**: 在 3.4 Tokenizer 训练系统 追加采样配置说明

### Commit `def456` - 修复 IndexFilter 路径匹配 bug
- **归属章节**: 六、经验教训
- **变更类型**: 新知识（踩坑记录）
- **内容摘要**: ParquetReader 默认不设置 row_idx，导致 IndexFilter 无法匹配
- **代码提取**: src/data_processing/fineweb_edu/adapters.py:15
- **建议操作**: 在 6.3 修复记录 追加条目
```

**主 agent 整合职责**：
1. 并行启动所有波次的子 agent
2. 收集各波次分析结果
3. 跨波次去重（相同技术点合并）
4. 冲突检测（相同章节的多项修改）
5. 执行文档实际修改

#### 6.4 终止条件

遇到以下情况必须停止，向用户说明原因：
- 用户明确拒绝更新
- 等待用户确认超过 24 小时
- 无法判定知识归属或类型（不确定时应询问，而非猜测）

---

### 7. 更新计划书模板

**使用时机**：完整流程的步骤 2

**输出要求**：仅输出分析框架，不输出分析结果

```markdown
## 知识库更新计划书

### 基本信息
- **当前版本**: v4.0
- **当前覆盖**: 94eeacd ~ b551a08
- **待分析 commits**: abc123, def456, hij789, klm012, nop345（共 5 个）

### 分组策略

| 波次 | Commits | 分组逻辑 |
|------|---------|----------|
| 1/2 | abc123, def456 | Tokenizer 配置相关 |
| 2/2 | hij789, klm012, nop345 | 数据处理优化相关 |

### 波次 1/2 分析项

| 分析项 | 输出格式 | 说明 |
|--------|----------|------|
| 归属章节 | 章节名称 | 使用第 4 章映射表判定 |
| 变更类型 | 新增/修改/删除 | 使用第 3 章决策矩阵判定 |
| 代码引用 | 路径:行号 | 需提取的代码片段位置 |
| 冗余检查 | 是/否 | 是否已存在相似内容 |

### 波次 2/2 分析项
（同上格式）
```

---

### 8. 优先级与冲突解决

#### 8.1 优先级排序

```
用户明确指令 > 简洁性 > 完整性
```

**解释**：
- 用户指令优先于任何规则
- 在保证准确的前提下，简洁优于冗长
- 用摘要 + 代码引用替代大段复制

#### 8.2 冲突解决规则

| 冲突场景 | 解决方案 | 决策人 |
|----------|----------|--------|
| 简洁性 vs 完整性 | 使用摘要 + 文件路径:行号引用 | AI Agent |
| 不确定归属章节 | 列出可能选项，询问用户 | 用户 |
| 不确定是否记录 | 默认不记录，询问用户 | 用户 |
| 多个波次建议修改同一章节 | 合并修改，避免重复 | AI Agent |

---

### 9. 元信息与版本管理

#### 9.1 元信息更新

**位置**：文档第 1-9 行

**每次更新必须修改**：
```markdown
| 属性 | 值 |
|------|-----|
| **覆盖范围** | `94eeacd` ~ `new_commit_hash` |
| **最后更新** | `new_commit_hash` @ YYYY-MM-DD |
| **文档版本** | vX.Y |
```

#### 9.2 版本号管理

**递增规则**：

| 变更类型 | 版本变化 | 示例 |
|----------|----------|------|
| 小修（错别字、格式） | X.Y → X.Y+1 | v4.0 → v4.1 |
| 内容变更（新增/修改/删除） | X.Y → X+1.0 | v4.0 → v5.0 |

**变更类型判定**：
- 仅修改文字描述、修正错别字 → **小修**
- 新增知识点、修改技术内容、删除条目 → **内容变更**

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

# Iterator 类型注解 (Python 3.9+)
from collections.abc import Iterator
from datatrove.data import Document

def run(self, data, rank: int = 0, world_size: int = 1) -> Iterator[Document]:
    for doc in data:
        if self._should_keep(doc):
            yield doc
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
@pytest.mark.parametrize("score,expected", [
    (2.5, "2.8"), (3.0, "3.0"), (3.7, "3.5"), (4.5, "4.0")
])
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
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

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
├── parquet_merger.py        # Parquet 合并工具
└── fineweb_edu/             # FineWeb-Edu 专用
    ├── __main__.py          # CLI: python -m src.data_processing.fineweb_edu
    └── adapters.py          # 数据适配器

src/
└── constants.py             # 项目级常量定义

scripts/
└── utils.py                 # 脚本工具模块（日志、JSON工具）
```

### 3.2 配置分层与设计模式

```
config/
├── buckets.yaml      # 业务：评分桶定义
├── processing.yaml   # 运行：workers, tasks, compression
├── paths.yaml        # 路径：输入/输出目录
├── tokenizer.yaml    # Tokenizer：训练参数统一配置
└── tokenizer_data.yaml  # Tokenizer：数据采样配置
```

```python
# 延迟加载模式
_DEFAULT_BUCKETS: list[BucketConfig] | None = None
def get_all_bucket_configs() -> list[BucketConfig]:
    global _DEFAULT_BUCKETS
    if _DEFAULT_BUCKETS is None: _DEFAULT_BUCKETS = _load_buckets()
    return _DEFAULT_BUCKETS

# 统一配置加载
def get_tokenizer_config() -> dict[str, Any]:
    return load_config("tokenizer")
```

**项目级常量** (`src/constants.py`):

```python
# 特殊 Token 定义
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

# 代码语言扩展名映射
LANGUAGE_EXTENSIONS = {
    "python": [".py", ".pyw", ".pyi"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx", ".mts", ".cts"],
    "cpp": [".c", ".h", ".cpp", ".hpp", ".cc"],
    "rust": [".rs"],
}
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

### 4.2 脚本工具模块

**scripts/utils.py** 提供跨脚本复用工具：

```python
from scripts.utils import setup_logging, read_json, write_json

# 统一日志配置
logger = setup_logging("prepare_tokenizer_data")

# JSON 读写工具
data = read_json(path)           # 支持 Path 对象
write_json(path, data, indent=2) # 自动创建父目录
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

**统一配置** (`config/tokenizer.yaml`):

```yaml
training:
  vocab_size: 36005          # 36000 BPE + 5 特殊 token
  batch_size: 10000          # Parquet 读取批次
  min_frequency: 2           # BPE 最小频率
  
paths:
  data_dir: "data/datasets/nanomind_tokenizer"
  template_dir: "output/qwen3_next_tokenizer"
  output_dir: "output/tokenizer_36k"
  
special_tokens:
  - "<|endoftext|>"
  - "<|im_start|>"
  - "<|im_end|>"
  - "<think>"
  - "</think>"
```

**配置加载**:

```python
from src.data_processing import load_config
from src.constants import SPECIAL_TOKENS, LANGUAGE_EXTENSIONS

config = load_config("tokenizer")
vocab_size = config["training"]["vocab_size"]
```

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
| `c74171d` | Tokenizer 配置分散在多个脚本 | 新增 `config/tokenizer.yaml` 统一配置 |
| `6e45111` | 脚本间日志/JSON 工具重复 | 新增 `scripts/utils.py` 公共工具模块 |
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
