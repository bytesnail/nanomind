#!/usr/bin/env python3
"""Tokenizer 训练脚本.

基于 Qwen3-Next 架构训练 64K 词表的 BPE Tokenizer。

训练步骤:
1. 从模板加载 pretokenizer/normalizer/decoder 配置
2. 空白初始化 BPE 模型
3. 在采样数据上学习 64000 个 BPE 合并规则
4. 添加 5 个特殊 token（ID 64000-64004）
5. 配置 eos/pad/bos/unk 映射

用法:
    python scripts/train_tokenizer.py \\
        --data-dir data/datasets/nanomind_tokenizer \\
        --template-dir output/qwen3_next_tokenizer \\
        --output-dir output/tokenizer_64k \\
        --vocab-size 64005 \\
        --validate

输出:
    output/tokenizer_64k/
    ├── tokenizer.json              # 词表与合并规则
    ├── tokenizer_config.json       # Tokenizer 配置
    ├── special_tokens_map.json     # 特殊 token 映射
    └── vocab.txt                   # 可读词汇表

依赖: tokenizers>=0.22.0, transformers>=4.40.0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
from collections import Counter
from collections.abc import Generator
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_DATA_DIR = Path("data/datasets/nanomind_tokenizer")
DEFAULT_TEMPLATE_DIR = Path("output/qwen3_next_tokenizer")
DEFAULT_OUTPUT_DIR = Path("output/tokenizer_64k")
DEFAULT_VOCAB_SIZE = 64005
DEFAULT_BATCH_SIZE = 300
DEFAULT_MIN_FREQUENCY = 4
DEFAULT_NUM_CHUNKS = 28
DEFAULT_CHUNK_VOCAB_RATIO = 4
SHUFFLE_SEED = 42

# 特殊 Token 定义（ID 64000-64004）
SPECIAL_TOKENS = [
    AddedToken("<|endoftext|>", normalized=False, special=True),  # ID 64000
    AddedToken("<|im_start|>", normalized=False, special=True),  # ID 64001
    AddedToken("<|im_end|>", normalized=False, special=True),  # ID 64002
    AddedToken("<|think|>", normalized=False, special=True),  # ID 64003
    AddedToken("<|/think|>", normalized=False, special=True),  # ID 64004
]

SPECIAL_TOKENS_MAP = {
    "pad_token": "<|endoftext|>",
    "eos_token": "<|im_end|>",
    "bos_token": None,
    "unk_token": None,
}


def load_template_components(template_dir: Path) -> dict:
    """从模板 tokenizer 文件加载组件配置。

    Args:
        template_dir: 模板 tokenizer 目录

    Returns:
        包含 normalizer, pre_tokenizer, decoder 配置的字典
    """
    tokenizer_path = template_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"模板 tokenizer 文件不存在: {tokenizer_path}")

    logger.info(f"从模板加载组件: {tokenizer_path}")
    template_tokenizer = Tokenizer.from_file(str(tokenizer_path))

    components = {}

    if template_tokenizer.normalizer is not None:
        components["normalizer"] = template_tokenizer.normalizer
        logger.info("  已复制 normalizer")

    if template_tokenizer.pre_tokenizer is not None:
        components["pre_tokenizer"] = template_tokenizer.pre_tokenizer
        logger.info("  已复制 pre_tokenizer")

    if template_tokenizer.decoder is not None:
        components["decoder"] = template_tokenizer.decoder
        logger.info("  已复制 decoder")

    return components


def get_data_stats(data_dir: Path) -> tuple[int, int]:
    """获取数据集统计信息。

    Args:
        data_dir: 数据目录

    Returns:
        (总行数, 文件数量)
    """
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {data_dir}")

    logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")

    total_rows = 0
    for fp in tqdm(parquet_files, desc="统计行数", leave=False):
        try:
            total_rows += pq.read_metadata(fp).num_rows
        except Exception as e:
            logger.warning(f"无法读取 {fp} 元数据: {e}")

    logger.info(f"总样本数: {total_rows:,}")

    return total_rows, len(parquet_files)


def batch_iterator(
    data_dir: Path, batch_size: int = DEFAULT_BATCH_SIZE
) -> Generator[list[str]]:
    """流式迭代文本数据（内存安全版）。

    使用 ParquetFile.iter_batches() 逐批次读取，避免 OOM。

    Args:
        data_dir: 数据目录
        batch_size: 批次大小

    Yields:
        文本批次的列表
    """
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    total_rows_yielded = 0

    for fp in tqdm(parquet_files, desc="读取数据", leave=False):
        try:
            parquet_file = pq.ParquetFile(fp)
            for record_batch in parquet_file.iter_batches(
                batch_size=batch_size, columns=["text"]
            ):
                texts = [t for t in record_batch["text"].to_pylist() if t]
                if texts:
                    yield texts
                    total_rows_yielded += len(texts)
                del record_batch
            parquet_file.close()
            gc.collect()
        except Exception as e:
            logger.warning(f"读取文件失败 {fp}: {e}")

    logger.info(f"总计处理 {total_rows_yielded:,} 条文本")


def chunked_batch_iterator(
    data_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    resume_chunk: int = 0,
) -> Generator[tuple[int, list[str]]]:
    """分块迭代器，支持断点续训和乱序分块。

    分块前会对文件进行乱序，确保每块包含混合内容类型（英文、中文、代码、数学），
    避免某一块只包含单一类型数据导致词表偏斜。

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_chunks: 分块数量
        resume_chunk: 从第几个块开始（断点续训）

    Yields:
        (chunk_idx, batch) 元组
    """
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {data_dir}")

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(parquet_files)
    logger.info(f"文件已乱序（种子: {SHUFFLE_SEED}），确保每块包含混合内容类型")

    chunk_size = max(1, len(parquet_files) // num_chunks)
    total_rows_yielded = 0

    for chunk_idx in range(resume_chunk, num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = (
            start_idx + chunk_size if chunk_idx < num_chunks - 1 else len(parquet_files)
        )
        chunk_files = parquet_files[start_idx:end_idx]

        chunk_rows = 0
        batch = []
        for fp in tqdm(chunk_files, desc=f"块 {chunk_idx + 1}", leave=False):
            try:
                parquet_file = pq.ParquetFile(fp)
                for record_batch in parquet_file.iter_batches(
                    batch_size=batch_size * 2, columns=["text"]
                ):
                    for text in record_batch["text"].to_pylist():
                        if text:
                            batch.append(text)
                            if len(batch) >= batch_size:
                                yield chunk_idx, batch
                                chunk_rows += len(batch)
                                batch = []
                parquet_file.close()
                gc.collect()
            except Exception as e:
                logger.warning(f"读取文件失败 {fp}: {e}")

        if batch:
            yield chunk_idx, batch
            chunk_rows += len(batch)

        logger.info(f"第 {chunk_idx + 1} 块完成，产出 {chunk_rows:,} 条样本")
        total_rows_yielded += chunk_rows

    logger.info(f"总计处理 {total_rows_yielded:,} 条样本")


def save_chunk_vocab(
    vocab: dict[str, int],
    chunk_idx: int,
    checkpoint_dir: Path,
    token_frequencies: dict[str, int] | None = None,
) -> Path:
    """保存词表及可选的频率信息。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = checkpoint_dir / f"vocab_chunk_{chunk_idx}.json"

    data = {"vocab": vocab}
    if token_frequencies:
        data["frequencies"] = token_frequencies

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"词表已保存: {vocab_path} ({len(vocab)} tokens)")
    return vocab_path


def load_chunk_vocab(
    checkpoint_dir: Path, chunk_idx: int
) -> tuple[dict[str, int], dict[str, int]] | None:
    """加载词表及频率信息。"""
    vocab_path = checkpoint_dir / f"vocab_chunk_{chunk_idx}.json"
    if vocab_path.exists():
        logger.info(f"加载词表: {vocab_path}")
        with open(vocab_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "vocab" in data:
            return data["vocab"], data.get("frequencies", {})
        # 兼容旧格式
        return data, {}
    return None


def find_latest_chunk(checkpoint_dir: Path) -> int:
    if not checkpoint_dir.exists():
        return -1

    vocab_files = list(checkpoint_dir.glob("vocab_chunk_*.json"))
    if not vocab_files:
        return -1

    latest = max(vocab_files, key=lambda p: int(p.stem.split("_")[-1]))
    return int(latest.stem.split("_")[-1])


def train_tokenizer(
    data_dir: Path,
    template_dir: Path,
    output_dir: Path,
    vocab_size: int,
    validate: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    resume: bool = False,
    chunk_vocab_ratio: int = DEFAULT_CHUNK_VOCAB_RATIO,
) -> None:
    """训练 BPE tokenizer，使用分块增量训练模式。

    Args:
        data_dir: 数据目录
        template_dir: 模板 tokenizer 目录
        output_dir: 输出目录
        vocab_size: 词表大小（包含特殊 token）
        validate: 是否执行验证
        batch_size: 数据批次大小
        min_frequency: BPE 最小词频
        num_chunks: 分块数量
        resume: 是否从检查点恢复训练
        chunk_vocab_ratio: 每块词表大小比例，每块训练 vocab_size // ratio
    """

    checkpoint_dir = output_dir / "checkpoints"
    components = load_template_components(template_dir)

    bpe_vocab_size = vocab_size - len(SPECIAL_TOKENS)
    logger.info(f"初始化 BPE tokenizer (词表大小: {bpe_vocab_size})")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.normalizer = components.get("normalizer")
    tokenizer.pre_tokenizer = components.get("pre_tokenizer")
    tokenizer.decoder = components.get("decoder")

    logger.info("使用分块增量训练模式")
    logger.info(f"分块数: {num_chunks}")
    logger.info(f"检查点目录: {checkpoint_dir}")

    resume_chunk = 0
    if resume:
        resume_chunk = find_latest_chunk(checkpoint_dir) + 1
        if resume_chunk > 0:
            logger.info(f"从检查点恢复，从第 {resume_chunk} 块开始")

    if resume_chunk == 0:
        total_rows, total_files = get_data_stats(data_dir)
        logger.info(f"数据集: {total_files} 个文件, {total_rows:,} 个样本")

    # 每块训练减小后的词表大小（加速第一阶段），最后从全局候选池择优
    chunk_vocab_size = bpe_vocab_size // chunk_vocab_ratio
    logger.info(
        f"每块训练词表大小: {chunk_vocab_size} (目标 {bpe_vocab_size} // {chunk_vocab_ratio}, "
        f"从 {num_chunks}×{chunk_vocab_size}={num_chunks * chunk_vocab_size:,} 候选池择优)"
    )

    all_vocabs: dict[str, int] = {}
    all_token_freq: dict[str, int] = {}
    processed_chunks: set[int] = set()
    current_chunk_idx = None
    current_chunk_texts = []

    # 断点续训：加载已处理块的频率数据
    if resume and resume_chunk > 0:
        logger.info(f"加载已处理块 0-{resume_chunk - 1} 的频率数据...")
        for prev_chunk_idx in range(resume_chunk):
            loaded = load_chunk_vocab(checkpoint_dir, prev_chunk_idx)
            if loaded:
                vocab, freq = loaded
                all_vocabs.update(vocab)
                for token, count in freq.items():
                    all_token_freq[token] = all_token_freq.get(token, 0) + count
                processed_chunks.add(prev_chunk_idx)
        logger.info(
            f"已加载 {len(processed_chunks)} 个块，累计 {len(all_vocabs)} 个候选 token"
        )

    def train_current_chunk() -> dict[str, int] | None:
        """训练当前 chunk 的词表，返回该块的 token 频率统计。"""
        nonlocal current_chunk_texts, current_chunk_idx
        if not current_chunk_texts or current_chunk_idx is None:
            return None

        logger.info(
            f"训练第 {current_chunk_idx + 1} 块词表 ({len(current_chunk_texts):,} 条文本)..."
        )
        chunk_tokenizer = Tokenizer(BPE(unk_token=None))
        chunk_tokenizer.normalizer = components.get("normalizer")
        chunk_tokenizer.pre_tokenizer = components.get("pre_tokenizer")
        chunk_tokenizer.decoder = components.get("decoder")

        chunk_trainer = BpeTrainer(
            vocab_size=chunk_vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=[],
        )

        chunk_tokenizer.train_from_iterator(current_chunk_texts, trainer=chunk_trainer)

        chunk_vocab = chunk_tokenizer.get_vocab()

        # 统计当前 chunk 中各 token 的频率（批处理优化）
        logger.info(
            f"统计第 {current_chunk_idx + 1} 块 token 频率 ({len(current_chunk_texts):,} 条文本)..."
        )
        token_id_counts = Counter()
        encode_batch_size = 20

        for batch_start_idx in tqdm(
            range(0, len(current_chunk_texts), encode_batch_size),
            desc=f"统计块 {current_chunk_idx + 1} 频率",
            leave=False,
        ):
            encode_batch = current_chunk_texts[
                batch_start_idx : batch_start_idx + encode_batch_size
            ]
            encodings = chunk_tokenizer.encode_batch(encode_batch)
            for enc in encodings:
                token_id_counts.update(enc.ids)

        id_to_token_map = {v: k for k, v in chunk_vocab.items()}
        token_frequencies = {
            id_to_token_map[tid]: cnt
            for tid, cnt in token_id_counts.items()
            if tid in id_to_token_map
        }

        save_chunk_vocab(
            chunk_vocab, current_chunk_idx, checkpoint_dir, dict(token_frequencies)
        )
        all_vocabs.update(chunk_vocab)
        processed_chunks.add(current_chunk_idx)

        del chunk_tokenizer, chunk_vocab
        gc.collect()
        current_chunk_texts = []

        return token_frequencies

    for chunk_idx, batch in chunked_batch_iterator(
        data_dir, batch_size, num_chunks, resume_chunk
    ):
        if chunk_idx in processed_chunks:
            continue

        loaded = load_chunk_vocab(checkpoint_dir, chunk_idx)
        if loaded:
            vocab, freq = loaded
            logger.info(f"第 {chunk_idx + 1} 块已处理，跳过")
            all_vocabs.update(vocab)
            # 累加频率
            for token, count in freq.items():
                all_token_freq[token] = all_token_freq.get(token, 0) + count
            processed_chunks.add(chunk_idx)
            continue

        # 切换到新 chunk 时，先训练前一个 chunk
        if chunk_idx != current_chunk_idx:
            chunk_freq = train_current_chunk()
            if chunk_freq:
                for token, count in chunk_freq.items():
                    all_token_freq[token] = all_token_freq.get(token, 0) + count
            current_chunk_idx = chunk_idx

        current_chunk_texts.extend(batch)

    # 训练最后一个 chunk
    last_chunk_freq = train_current_chunk()
    if last_chunk_freq:
        for token, count in last_chunk_freq.items():
            all_token_freq[token] = all_token_freq.get(token, 0) + count

    # 从全局候选池择优：每块训练64000，最后从640000候选中选最优64000
    candidate_count = len(all_vocabs)
    logger.info(
        f"全局候选池: {candidate_count:,} 个 unique token (来自 {num_chunks} 块 × {bpe_vocab_size:,})"
    )
    logger.info(f"从中择优选择: {bpe_vocab_size:,} 个 token")

    # 使用累积频率选择 top tokens
    top_tokens = [t for t, _ in Counter(all_token_freq).most_common(bpe_vocab_size)]

    final_vocab = {token: idx for idx, token in enumerate(top_tokens)}

    logger.info("=" * 60)
    logger.info("第二步：用完整数据训练 merges（vocab 固定）")
    logger.info("=" * 60)

    tokenizer = Tokenizer(BPE(vocab=final_vocab, merges=[], unk_token=None))
    tokenizer.normalizer = components.get("normalizer")
    tokenizer.pre_tokenizer = components.get("pre_tokenizer")
    tokenizer.decoder = components.get("decoder")

    merge_trainer = BpeTrainer(
        vocab_size=len(final_vocab),
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=[],
    )

    total_rows, _ = get_data_stats(data_dir)

    tokenizer.train_from_iterator(
        batch_iterator(data_dir, batch_size),
        trainer=merge_trainer,
        length=total_rows,
    )

    logger.info(f"Merges 训练完成，词表大小: {tokenizer.get_vocab_size()}")

    # 6. 添加特殊 token（强制指定 ID）
    # 由于 tokenizers 库会在现有词表末尾添加新 token，
    # 我们需要确保 BPE 词表正好是 64000 个，然后添加 5 个特殊 token
    current_vocab_size = tokenizer.get_vocab_size()
    logger.info(f"当前词表大小: {current_vocab_size}")

    num_added = tokenizer.add_tokens(SPECIAL_TOKENS)
    logger.info(f"添加 {num_added} 个特殊 token")

    # 验证特殊 token ID
    for i, token in enumerate(SPECIAL_TOKENS):
        token_str = str(token).lstrip('AddedToken("').rstrip('")')
        token_id = tokenizer.token_to_id(token_str)
        expected_id = bpe_vocab_size + i
        if token_id != expected_id:
            logger.warning(
                f"Token '{token_str}' ID 不匹配: 期望 {expected_id}, 实际 {token_id}"
            )
        else:
            logger.info(f"  {token_str}: ID {token_id}")

    # 7. 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 8. 保存 tokenizer.json
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)
    logger.info(f"已保存: {tokenizer_path}")

    # 9. 使用 transformers 包装并配置特殊 token
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=SPECIAL_TOKENS_MAP["bos_token"],
        eos_token=SPECIAL_TOKENS_MAP["eos_token"],
        unk_token=SPECIAL_TOKENS_MAP["unk_token"],
        pad_token=SPECIAL_TOKENS_MAP["pad_token"],
    )

    # 添加额外特殊 token 标记
    additional_special_tokens = ["<|im_start|>", "<|think|>", "<|/think|>"]
    hf_tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )

    # 10. 保存完整配置
    hf_tokenizer.save_pretrained(output_dir)
    logger.info(f"已保存 transformers tokenizer 到: {output_dir}")

    # 11. 生成可读词汇表
    vocab_path = output_dir / "vocab.txt"
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, idx in sorted_vocab:
            f.write(f"{idx}\t{token}\n")
    logger.info(f"已保存可读词汇表: {vocab_path}")

    # 12. 输出文件列表
    saved_files = list(output_dir.iterdir())
    logger.info(f"输出文件 ({len(saved_files)} 个):")
    for f in sorted(saved_files):
        if f.is_file():
            size = f.stat().st_size
            logger.info(f"  - {f.name} ({size:,} bytes)")

    # 13. 验证
    if validate:
        validate_tokenizer(output_dir, vocab_size)


def validate_tokenizer(output_dir: Path, expected_vocab_size: int) -> bool:
    """验证训练后的 tokenizer。

    Args:
        output_dir: tokenizer 输出目录
        expected_vocab_size: 期望的词表大小

    Returns:
        验证是否通过
    """
    logger.info("=" * 60)
    logger.info("开始验证")
    logger.info("=" * 60)

    errors = []

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(output_dir))

    # 1. 验证词表大小
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != expected_vocab_size:
        errors.append(
            f"词表大小不匹配: 期望 {expected_vocab_size}, 实际 {actual_vocab_size}"
        )
    else:
        logger.info(f"✓ 词表大小: {actual_vocab_size}")

    # 2. 验证特殊 token ID
    special_token_ids = {
        "<|endoftext|>": 64000,
        "<|im_start|>": 64001,
        "<|im_end|>": 64002,
        "<|think|>": 64003,
        "<|/think|>": 64004,
    }

    for token, expected_id in special_token_ids.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            errors.append(
                f"Token '{token}' ID 不匹配: 期望 {expected_id}, 实际 {actual_id}"
            )
        else:
            logger.info(f"✓ {token}: ID {actual_id}")

    # 3. 验证特殊 token 映射
    if tokenizer.pad_token != "<|endoftext|>":
        errors.append(
            f"pad_token 不匹配: 期望 '<|endoftext|>', 实际 '{tokenizer.pad_token}'"
        )
    else:
        logger.info(f"✓ pad_token: {tokenizer.pad_token}")

    if tokenizer.eos_token != "<|im_end|>":
        errors.append(
            f"eos_token 不匹配: 期望 '<|im_end|>', 实际 '{tokenizer.eos_token}'"
        )
    else:
        logger.info(f"✓ eos_token: {tokenizer.eos_token}")

    # 4. 编解码一致性测试
    test_texts = [
        "<|im_start|>user\n问题<|im_end|>",
        "<|im_start|>assistant\n<|think|>推理过程...<|/think|>\n答案<|im_end|>",
        "Hello, world! 你好，世界！",
        "def hello():\n    print('Hello')\n",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        if decoded != text:
            errors.append(f"编解码不一致:\n  原文: {text!r}\n  解码: {decoded!r}")
        else:
            logger.info(f"✓ 编解码测试通过: {text[:50]}...")

    if errors:
        logger.error("验证失败:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        return False

    logger.info("=" * 60)
    logger.info("验证通过！")
    logger.info("=" * 60)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="训练 BPE Tokenizer - 基于 Qwen3-Next 架构训练 64K 词表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置训练
  python scripts/train_tokenizer.py

  # 指定参数并验证
  python scripts/train_tokenizer.py \\
      --data-dir data/datasets/nanomind_tokenizer \\
      --template-dir output/qwen3_next_tokenizer \\
      --output-dir output/tokenizer_64k \\
      --vocab-size 64005 \\
      --validate

  # 调整批次大小（内存优化）
  python scripts/train_tokenizer.py --batch-size 5000

依赖版本:
  tokenizers>=0.22.0
  transformers>=4.40.0
  pyarrow>=15.0.0
        """,
    )

    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"训练数据目录 (默认: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--template-dir",
        "-t",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        help=f"模板 tokenizer 目录 (默认: {DEFAULT_TEMPLATE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"词表大小（包含特殊 token）(默认: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"数据批次大小 (默认: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--min-frequency",
        "-m",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
        help=f"BPE 最小词频 (默认: {DEFAULT_MIN_FREQUENCY})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="训练后执行验证",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help=f"分块数量 (默认: {DEFAULT_NUM_CHUNKS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从检查点恢复训练",
    )
    parser.add_argument(
        "--chunk-vocab-ratio",
        type=int,
        default=DEFAULT_CHUNK_VOCAB_RATIO,
        help=f"每块词表大小比例 (每块训练 vocab_size // ratio，默认: {DEFAULT_CHUNK_VOCAB_RATIO})",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"数据目录不存在: {args.data_dir}")
        return 1

    if not args.template_dir.exists():
        logger.error(f"模板目录不存在: {args.template_dir}")
        return 1

    logger.info("=" * 60)
    logger.info("Tokenizer 训练配置")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"模板目录: {args.template_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"词表大小: {args.vocab_size}")
    logger.info(f"BPE tokens: {args.vocab_size - len(SPECIAL_TOKENS)}")
    logger.info(f"特殊 tokens: {len(SPECIAL_TOKENS)}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"最小词频: {args.min_frequency}")
    logger.info(f"验证模式: {args.validate}")
    logger.info(f"分块数: {args.num_chunks}")
    logger.info(f"恢复训练: {args.resume}")
    logger.info(f"每块词表比例: 1/{args.chunk_vocab_ratio}")
    logger.info("=" * 60)

    try:
        train_tokenizer(
            data_dir=args.data_dir,
            template_dir=args.template_dir,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            validate=args.validate,
            batch_size=args.batch_size,
            min_frequency=args.min_frequency,
            num_chunks=args.num_chunks,
            resume=args.resume,
            chunk_vocab_ratio=args.chunk_vocab_ratio,
        )
        logger.info("训练完成！")
        return 0
    except Exception as e:
        logger.exception(f"训练失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
