# nanomind 分类知识库

| 属性 | 值 |
|------|-----|
| **覆盖范围** | `94eeacd` ~ `7731e66` (89 commits) |
| **最后更新** | `7731e66` @ 2026-02-16 |

> 更新时同步修改 commit hash 和日期。触发条件：新增 commit ≥ 3 或有重大变更。


---

## 增量更新指南(严禁修改或删除本章节)

**更新原则**：
- **新知识** → 在对应章节追加
- **已有知识变更** → 直接修改原内容
- **过时/错误/不需要** → 直接删除
- **保持简洁** → 合并重复，避免冗余；代码示例只保留最相关最重要的最短代码片段，严禁整段摘抄复制完整代码实现
- **元信息** → 元信息(本文件开头的覆盖范围和最后更新时间等信息)格式严格保持不变，只更新元信息内容
- **分波次分析与用户主导** → 先将待分析 commit 按相关性和数量拆分为连贯的分析波次；不主动更新本文件，只输出详细计划书，由用户主导执行

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

### 6.2 DO NOT（不推荐做法）

| 类别 | 禁止 |
|------|------|
| 配置 | 模块顶层直接加载、假设字段一定存在 |
| 代码 | 裸 `except:`、生产代码用 `assert`、`print` 输出日志 |
| 数据 | 假设固定宽度编码、全量加载后处理、循环内 import |
| 并发 | 串行逻辑处理并行场景、堆中存储 Path 等大对象 |
| 架构 | 可复用逻辑留在脚本、数据集逻辑放通用模块 |

### 6.3 NEVER（禁止做法）

| 类别 | 禁止 |
|------|------|
| 依赖 | `pip install`、`uv add` 不带 `--no-sync`、提交 uv.lock |
| 类型 | `as any`、`@ts-ignore`、`@ts-expect-error` |
| 数据 | 负数索引/空路径生成 ID、采样循环创建大量临时对象 |
| 框架 | 忽略 Datatrove 任务检测、Writer 返回非 None |
| 内存 | 一次性加载大规模数据集、共享 Datatrove 日志目录 |

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

*文档版本: v1.1 | 生成日期: 2026-02-16*
