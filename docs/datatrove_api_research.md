# DataTrove API 和模式研究文档

## 1. PipelineStep基类

### 定义和继承关系

**源码位置**: `datatrove/pipeline/base.py`

```python
from abc import ABC, abstractmethod
from datatrove.data import Document, DocumentsPipeline, Media
from datatrove.utils.stats import Stats

class PipelineStep(ABC):
    """Base pipeline block, all blocks should inherit from this one.
        Takes care of some general things such as handling dependencies, and stats
    
    Args:
        name: Name of the step
        type: Type of the step
            Types are high-level categories of steps, e.g. "Reader", "Tokenizer", "Filters", etc.
    """
    
    name: str = None
    type: str = None
    
    def __init__(self):
        super().__init__()
        self.stats = Stats(str(self))
```

### 必需方法签名

#### `run()` 方法
```python
@abstractmethod
def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    """
    Main entrypoint for any pipeline step. `data` is a generator of `Document`, and this method should
    yield `Document` (either add new documents if it is reading them, modify their content or metadata,
    or drop a few if it is a filter)

    Args:
      data: DocumentsPipeline:
      rank: int:  (Default value = 0) used when each worker needs to choose a shard of data to work on
      world_size: int:  (Default value = 1) used when each worker needs to choose a shard of data to work on

    Returns:

    """
    if data:
        yield from data
```

#### `__init__()` 方法
```python
def __init__(self):
    super().__init__()
    self.stats = Stats(str(self))
```

### 数据处理模式

**Yield vs Return**: PipelineStep使用`yield`模式传递数据，每个步骤接收DocumentsPipeline（生成器）并返回DocumentsPipeline。

```python
# DocumentsPipeline类型定义
DocumentsPipeline = NewType("DocumentsPipeline", Generator[Document, None, None] | None)
```

### Document数据结构

```python
@dataclass(slots=True)
class Document:
    """Main Document dataclass going through the processing pipeline

    Args:
        text: str
             the actual text content for each sample
        id: str
            a unique id (string) for this sample
        media: list[Media]
            The media associated with the document
        metadata: dict[str, Any]
            a dictionary where any additional info may be stored
    """

    text: str
    id: str
    media: list[Media] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 自定义PipelineStep示例

#### 简单过滤器示例
```python
import re
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter

class RegexFilter(BaseFilter):
    name = "🕵 Regex"

    def __init__(self, regex_exp: str, exclusion_writer: DiskWriter = None):
        """
        filters if regex finds at least one match

        Args:
            regex_exp: regex expression
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.regex = re.compile(regex_exp)

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        return not self.regex.search(doc.text)
```

#### 自定义处理器示例
```python
from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document, DocumentsPipeline

class CustomProcessor(PipelineStep):
    name = "🔧 Custom Processor"
    type = "PROCESSOR"
    
    def __init__(self, custom_param: str = "default"):
        super().__init__()
        self.custom_param = custom_param
    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            # 自定义处理逻辑
            with self.track_time():
                processed_text = self.process_text(doc.text)
                doc.text = processed_text
                doc.metadata["processed_by"] = self.name
                doc.metadata["custom_param"] = self.custom_param
                self.stat_update("processed_docs")
                self.update_doc_stats(doc)
            yield doc
    
    def process_text(self, text: str) -> str:
        """自定义文本处理逻辑"""
        return text.upper()  # 示例：转换为大写
```

## 2. Stats组件和输出格式

### Stats系统架构

**核心类**:
- `Stats`: 单个pipeline步骤的统计信息
- `MetricStats`: 单个指标的统计信息
- `MetricStatsDict`: 多个统计指标的字典
- `PipelineStats`: 整个pipeline的统计信息聚合

### Stats组件初始化和输出

#### DocStats
```python
class DocStats(BaseStats):
    """
    Summary stats of document level metrics:

    Available stats:
    length: Length of the document
    white_space_ratio: Ratio of whitespace characters
    non_alpha_digit_ratio: Ratio of non-alphabetic and non-digit characters
    digit_ratio: Ratio of digits
    uppercase_ratio: Ratio of uppercase letters
    elipsis_ratio: Ratio of elipsis characters
    punctuation_ratio: Punctuation ratio
    """

    name = "📜 Doc stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
```

#### TokenStats
```python
class TokenStats(BaseStats, PipelineStepWithTokenizer):
    """
    Token stats of a document.

    Available metrics:
    token_count: Number of tokens in the document
    """

    name = "🔗 Token counter"

    def __init__(
        self,
        output_folder: DataFolderLike,
        tokenizer_name_or_path: str = "gpt2",
        groups_to_compute: list[GROUP] = ["fqdn", "suffix", "summary", "histogram"],
        histogram_rounding: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        BaseStats.__init__(self, output_folder, groups_to_compute, histogram_rounding, top_k_config)
        PipelineStepWithTokenizer.__init__(self, tokenizer_name_or_path)
```

### Stats输出存储位置和格式

**存储路径结构**:
```
{output_folder}/
├── {group}/
│   ├── {stat_name}/
│   │   ├── 00000.json  # rank 0的统计
│   │   ├── 00001.json  # rank 1的统计
│   │   └── ...
└── stats.json  # 聚合后的统计信息
```

**文件格式**: JSON格式，包含详细的统计信息

```python
# 示例stats文件内容
{
    "doc_len": {
        "total": 1234567,
        "mean": 123.4,
        "variance": 456.7,
        "std_dev": 21.3,
        "min": 1,
        "max": 1000
    },
    "doc_len_tokens": {
        "total": 234567,
        "mean": 23.4,
        "variance": 12.3,
        "std_dev": 3.5,
        "min": 1,
        "max": 200
    }
}
```

### Stats访问和聚合API

#### 访问单个PipelineStep的统计
```python
# 在PipelineStep内部使用
self.stat_update("my_metric", value=10, unit="doc")
self.stat_update("another_metric")  # 默认value=1

# 访问统计信息
my_stats = self.stats["my_metric"]
print(f"Total: {my_stats.total}, Mean: {my_stats.mean}")
```

#### 聚合多个worker的统计
```python
# PipelineStats自动聚合
from datatrove.utils.stats import PipelineStats

# 自动从多个rank的统计文件聚合
pipeline_stats = PipelineStats()
print(pipeline_stats.get_repr("All tasks"))
```

### Stats聚合示例

```python
# 自定义统计聚合
def aggregate_stats(stats_folder: str):
    """聚合多个rank的统计信息"""
    aggregated = {}
    
    for rank_file in os.listdir(stats_folder):
        if rank_file.endswith(".json"):
            rank = int(rank_file.split("_")[1].split(".")[0])
            with open(os.path.join(stats_folder, rank_file)) as f:
                rank_stats = json.load(f)
                
            for metric, values in rank_stats.items():
                if metric not in aggregated:
                    aggregated[metric] = {
                        "total": 0,
                        "count": 0,
                        "values": []
                    }
                
                aggregated[metric]["total"] += values.get("total", 0)
                aggregated[metric]["count"] += 1
                aggregated[metric]["values"].append(values.get("mean", 0))
    
    # 计算全局统计
    for metric, data in aggregated.items():
        data["global_mean"] = sum(data["values"]) / len(data["values"])
        
    return aggregated
```

## 3. Checkpoint机制

### skip_completed参数工作原理

**源码位置**: `datatrove/executor/base.py`

```python
def is_rank_completed(self, rank: int) -> bool:
    """
        Checks if a given task has already been completed.
    Args:
        rank: the rank of the task to check

    Returns: whether task is already completed. If `skip_completed=False`, will always return `False`.

    """
    return self.skip_completed and self.logging_dir.isfile(f"completions/{rank:05d}")

def mark_rank_as_completed(self, rank: int):
    """
        Marks a given task as completed.
        In practice this involves creating an empty file with the rank in the filename.
    Args:
        rank: the rank of the task to mark as completed

    Returns:

    """
    self.logging_dir.open(f"completions/{rank:05d}", "w").close()
```

**工作原理**:
1. 当`skip_completed=True`时，executor在运行每个任务前会检查是否存在对应的completion文件
2. 如果completion文件存在，则跳过该任务
3. 任务成功完成后，会创建一个空的completion文件标记完成

### Checkpoint文件位置

**存储路径**: `{logging_dir}/completions/`

**文件命名**: `{rank:05d}` (例如: `00000`, `00001`, ...)

**目录结构示例**:
```
{logging_dir}/
├── completions/
│   ├── 00000    # rank 0完成标记
│   ├── 00001    # rank 1完成标记
│   └── ...
├── logs/
│   ├── task_00000.log
│   ├── task_00001.log
│   └── ...
└── stats/
    ├── 00000.json
    ├── 00001.json
    └── ...
```

### Checkpoint文件格式

**格式**: 空文件
- checkpoint文件只是简单的标记文件，不包含具体数据
- 文件存在即表示该任务已完成

### Checkpoint相关API

```python
# LocalPipelineExecutor中的checkpoint使用
from datatrove.executor.local import LocalPipelineExecutor

executor = LocalPipelineExecutor(
    pipeline=my_pipeline,
    tasks=100,
    logging_dir="outputs/my_experiment",
    skip_completed=True,  # 启用checkpoint
)

# 获取未完成的任务列表
incomplete_ranks = executor.get_incomplete_ranks()
print(f"需要运行的任务: {incomplete_ranks}")

# 检查特定任务是否完成
if executor.is_rank_completed(5):
    print("任务5已完成")
else:
    print("任务5未完成")
```

### Checkpoint最佳实践

```python
# 推荐的checkpoint配置
executor = LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=1000,
    logging_dir="outputs/experiment_2024_01_15",
    skip_completed=True,  # 启用checkpoint以支持断点续传
    workers=10,  # 并行worker数量
    randomize_start_duration=30,  # 随机延迟开始，避免资源竞争
)

# 运行pipeline
stats = executor.run()

# 检查完成状态
completed_tasks = 1000 - len(executor.get_incomplete_ranks())
print(f"已完成任务: {completed_tasks}/1000")
```

## 4. 日志配置选项

### LocalPipelineExecutor日志配置

**源码位置**: `datatrove/utils/logging.py`

```python
def add_task_logger(
    logging_dir,
    rank: int,
    local_rank: int = 0,
    node_rank: int = -1,
):
    """
    Sets up logging for a given task
    Args:
      logging_dir: DataFolder
      rank: int:
      local_rank: int:  (Default value = 0)
      node_rank: int: Node rank for logging prefix. Logs will be prefixed with [NODE X] if node_rank != -1. node_rank == -1 means single node mode (default).
    Returns:

    """
    logger.remove()
    logfile = logging_dir.open(f"logs/task_{rank:05d}.log", "w")

    # Create format string with node prefix at start (only if node_rank != -1)
    # Format: [NODE X] timestamp | level | module:function:line - message
    node_prefix = f"[NODE {node_rank}] " if node_rank != -1 else ""
    format_string = f"{node_prefix}{{time:YYYY-MM-DD HH:mm:ss.SSS}} | <level>{{level: <8}}</level> | {{name}}:{{function}}:{{line}} - <level>{{message}}</level>"

    logger.add(
        sys.stderr,
        colorize=DATATROVE_COLORIZE_LOGS,
        level="INFO" if local_rank == 0 else "ERROR",
        format=format_string,
    )
    logger.add(
        logfile,
        colorize=DATATROVE_COLORIZE_LOG_FILES,
        level="DEBUG",
        format=format_string,
    )
    logger.info(f"Launching pipeline for {rank=}")
    return logfile
```

### 日志级别设置

**环境变量控制**:
- `DATATROVE_COLORIZE_LOGS`: 控制控制台日志颜色化
- `DATATROVE_COLORIZE_LOG_FILES`: 控制日志文件颜色化（默认False）

**日志级别策略**:
- `local_rank == 0`: INFO级别（主worker显示详细日志）
- `local_rank != 0`: ERROR级别（其他worker只显示错误）

### 自定义Formatter支持

**格式字符串**:
```
{node_prefix}{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} - <level>{message}</level>
```

**格式组件**:
- `node_prefix`: 节点前缀（分布式模式）
- `time`: 时间戳
- `level`: 日志级别
- `name`: 模块名
- `function`: 函数名
- `line`: 行号
- `message`: 日志消息

### 默认日志详细程度

**控制台输出**:
- 主worker (local_rank=0): INFO级别及以上
- 其他workers: 仅ERROR级别

**文件输出**:
- 所有workers: DEBUG级别及以上
- 存储位置: `{logging_dir}/logs/task_{rank:05d}.log`

### 日志配置示例

```python
# 设置环境变量
import os
os.environ["DATATROVE_COLORIZE_LOGS"] = "true"  # 启用颜色
os.environ["DATATROVE_COLORIZE_LOG_FILES"] = "false"  # 禁用文件颜色

# 创建executor
executor = LocalPipelineExecutor(
    pipeline=my_pipeline,
    tasks=100,
    logging_dir="outputs/my_experiment",
    workers=5,
)

# 运行时会自动配置日志
stats = executor.run()
```

### 自定义日志处理

```python
from datatrove.utils.logging import logger, add_task_logger, close_task_logger

# 在PipelineStep中使用日志
class CustomStep(PipelineStep):
    def run(self, data, rank=0, world_size=1):
        logger.info(f"开始处理rank {rank}")
        
        for doc in data:
            logger.debug(f"处理文档: {doc.id}")
            
            try:
                # 处理文档
                result = self.process_document(doc)
                logger.info(f"成功处理文档 {doc.id}")
                yield result
            except Exception as e:
                logger.error(f"处理文档 {doc.id} 时出错: {e}")
                # 可以选择跳过或重新抛出异常
                
        logger.success(f"Rank {rank} 处理完成")
```

## 5. 其他关键点和示例代码

### 完整Pipeline示例

**FineWeb处理流程示例**（来自DataTrove官方示例）:

```python
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

DUMP_TO_PROCESS = "CC-MAIN-2023-50"
MAIN_OUTPUT_PATH = "s3://some_s3_bucket"
FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"

# 主要处理流程
main_processing_executor = SlurmPipelineExecutor(
    job_name=f"cc_{DUMP_TO_PROCESS}",
    pipeline=[
        WarcReader(
            f"s3://commoncrawl/crawl-data/{DUMP_TO_PROCESS}/segments/",
            glob_pattern="*/warc/*",
            default_metadata={"dump": DUMP_TO_PROCESS},
        ),
        URLFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")),
        Trafilatura(favour_precision=True),
        LanguageFilter(
            exclusion_writer=JsonlWriter(
                f"{FILTERING_OUTPUT_PATH}/2_non_english/",
                output_filename="${language}/" + DUMP_TO_PROCESS + "/${rank}.jsonl.gz",
            )
        ),
        GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/3_gopher_rep/{DUMP_TO_PROCESS}")),
        GopherQualityFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/4_gopher_qual/{DUMP_TO_PROCESS}")),
        C4QualityFilter(
            filter_no_terminal_punct=False,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/5_c4/{DUMP_TO_PROCESS}"),
        ),
        FineWebQualityFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/6_fineweb_qual/{DUMP_TO_PROCESS}")),
        JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"),
    ],
    tasks=8000,
    time="10:00:00",
    logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP_TO_PROCESS}",
    slurm_logs_folder=f"logs/base_processing/{DUMP_TO_PROCESS}/slurm_logs",
    randomize_start_duration=180,
    mem_per_cpu_gb=2,
    partition="hopper-cpu",
)

main_processing_executor.run()
```

### 多worker情况下的数据结构处理

**线程安全注意事项**:
1. **避免共享状态**: 每个worker应该有独立的状态
2. **文件输出**: 使用`${rank}`模板避免文件冲突
3. **统计聚合**: 使用线程安全的统计方法

```python
# 线程安全的PipelineStep示例
class ThreadSafeProcessor(PipelineStep):
    def __init__(self, output_folder: str):
        super().__init__()
        self.output_folder = output_folder
        # 每个worker独立的状态
        self.worker_stats = {}
    
    def run(self, data, rank=0, world_size=1):
        # 使用rank来避免冲突
        worker_output = f"{self.output_folder}/worker_{rank:05d}.jsonl"
        
        with open(worker_output, "w") as f:
            for doc in data:
                # 处理文档
                processed_doc = self.process(doc)
                
                # 线程安全的统计更新
                self.stat_update("processed_docs")
                self.update_doc_stats(processed_doc)
                
                # 写入文件
                f.write(json.dumps({
                    "id": processed_doc.id,
                    "text": processed_doc.text,
                    "metadata": processed_doc.metadata
                }) + "\n")
                
                yield processed_doc
```

### PipelineStep中的自定义数据存储

**存储模式**:
1. **文件系统存储**: 使用`DataFolder`进行跨平台存储
2. **内存存储**: 在`__init__`中初始化数据结构
3. **元数据存储**: 利用Document.metadata字段

```python
class DataStorageExample(PipelineStep):
    def __init__(self, storage_dir: str):
        super().__init__()
        self.storage_dir = storage_dir
        self.data_folder = get_datafolder(storage_dir)
        
        # 内存中的数据结构（每个worker独立）
        self.processing_stats = {
            "total_docs": 0,
            "error_count": 0,
            "processing_times": []
        }
    
    def run(self, data, rank=0, world_size=1):
        for doc in data:
            try:
                with self.track_time():
                    # 处理文档
                    result = self.complex_processing(doc)
                    
                    # 存储到文件系统
                    storage_key = f"processed/{rank:05d}/{doc.id}.json"
                    with self.data_folder.open(storage_key, "w") as f:
                        json.dump({
                            "original_id": doc.id,
                            "processed_text": result.text,
                            "processing_metadata": result.metadata,
                            "worker_rank": rank
                        }, f)
                    
                    # 更新内存统计
                    self.processing_stats["total_docs"] += 1
                    
                    yield result
                    
            except Exception as e:
                self.processing_stats["error_count"] += 1
                logger.error(f"处理文档 {doc.id} 失败: {e}")
                # 继续处理下一个文档
        
        # 保存处理统计
        stats_file = f"stats/worker_{rank:05d}_stats.json"
        with self.data_folder.open(stats_file, "w") as f:
            json.dump(self.processing_stats, f)
    
    def complex_processing(self, doc):
        """复杂处理逻辑示例"""
        # 模拟耗时处理
        import time
        time.sleep(0.01)
        
        # 创建处理后的文档
        result = Document(
            text=doc.text.upper(),
            id=f"processed_{doc.id}",
            metadata={
                **doc.metadata,
                "processed_at": time.time(),
                "processor": self.name
            }
        )
        
        return result
```

### 注意事项和最佳实践

1. **资源管理**:
   - 使用`with self.track_time()`自动跟踪处理时间
   - 及时释放不需要的资源
   - 合理设置worker数量避免内存溢出

2. **错误处理**:
   - 在PipelineStep中妥善处理异常
   - 使用`exclusion_writer`保存被过滤的文档
   - 记录详细的错误日志

3. **性能优化**:
   - 使用批处理提高吞吐量
   - 避免不必要的文档复制
   - 合理使用统计更新避免性能开销

4. **调试技巧**:
   - 使用小数据集进行测试
   - 启用详细日志（DEBUG级别）
   - 监控统计信息确保数据处理正确性

5. **扩展性考虑**:
   - 设计可配置的PipelineStep
   - 使用依赖管理（`_requires_dependencies`）
   - 支持分布式执行环境

---

**总结**: DataTrove提供了强大而灵活的数据处理框架，通过理解PipelineStep基类、Stats系统、Checkpoint机制、日志配置等核心组件，可以构建高效、可靠的大规模数据处理流水线。关键是要正确处理数据流、统计聚合、错误处理和资源管理。