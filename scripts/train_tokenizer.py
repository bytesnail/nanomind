#!/usr/bin/env python3
"""Tokenizer 训练脚本 - 基于 Hugging Face tokenizers 库.

用法:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --validate

输出:
    output/tokenizer_32k/
    ├── tokenizer.json
    └── tokenizer_config.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path


import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/datasets/nanomind_tokenizer")
DEFAULT_TEMPLATE_DIR = Path("output/qwen3_next_tokenizer")
DEFAULT_OUTPUT_DIR = Path("output/tokenizer_32k")
DEFAULT_VOCAB_SIZE = 32005
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_BATCH_SIZE = 2000

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|think|>",
    "<|/think|>",
]

SPECIAL_TOKENS_MAP = {
    "bos_token": None,
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
    "unk_token": None,
}


def find_parquet_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    files = sorted(data_dir.rglob("*.parquet"))
    logger.info(f"找到 {len(files)} 个parquet文件")
    return files


def stream_texts_from_parquet(
    files: list[Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[list[str]]:
    total_rows = 0
    for fp in tqdm(files, desc="读取数据文件"):
        try:
            parquet_file = pq.ParquetFile(fp)
            for record_batch in parquet_file.iter_batches(
                columns=["text"], batch_size=batch_size
            ):
                texts = [
                    text
                    for text in record_batch["text"].to_pylist()
                    if isinstance(text, str) and text.strip()
                ]
                if texts:
                    total_rows += len(texts)
                    yield texts
        except Exception as e:
            logger.warning(f"读取文件失败 {fp}: {e}")
            continue
    logger.info(f"总共读取 {total_rows:,} 条文本")


def load_template_tokenizer(template_dir: Path) -> Tokenizer:
    if not template_dir.exists():
        raise FileNotFoundError(f"模板目录不存在: {template_dir}")
    template_path = template_dir / "tokenizer.json"
    if not template_path.exists():
        raise FileNotFoundError(f"模板tokenizer文件不存在: {template_path}")
    logger.info(f"加载模板tokenizer: {template_path}")
    template = Tokenizer.from_file(str(template_path))
    tokenizer = Tokenizer(BPE())
    if template.pre_tokenizer is not None:
        tokenizer.pre_tokenizer = template.pre_tokenizer
    if template.normalizer is not None:
        tokenizer.normalizer = template.normalizer
    if template.decoder is not None:
        tokenizer.decoder = template.decoder
    logger.info("已复制模板架构配置 (pretokenizer/normalizer/decoder)")
    return tokenizer


def train_bpe_tokenizer(
    tokenizer: Tokenizer,
    data_dir: Path,
    vocab_size: int,
    min_frequency: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    files = find_parquet_files(data_dir)
    if not files:
        raise ValueError(f"在 {data_dir} 中未找到任何parquet文件")
    bpe_vocab_size = vocab_size - len(SPECIAL_TOKENS)
    logger.info("开始训练BPE tokenizer")
    logger.info(f"  目标词表大小: {vocab_size}")
    logger.info(f"  BPE词表大小: {bpe_vocab_size}")
    logger.info(f"  特殊token数: {len(SPECIAL_TOKENS)}")
    logger.info(f"  最小词频: {min_frequency}")
    trainer = BpeTrainer(
        vocab_size=bpe_vocab_size,
        min_frequency=min_frequency,
        special_tokens=[],
        show_progress=True,
    )
    text_iterator = stream_texts_from_parquet(files, batch_size=batch_size)
    tokenizer.train_from_iterator(
        (text for batch in text_iterator for text in batch),
        trainer=trainer,
        length=sum(pq.read_metadata(fp).num_rows for fp in files),
    )
    logger.info(f"BPE训练完成，当前词表大小: {tokenizer.get_vocab_size()}")


def add_special_tokens(tokenizer: Tokenizer) -> None:
    logger.info(f"添加 {len(SPECIAL_TOKENS)} 个特殊token")
    current_vocab_size = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    logger.info(f"添加后词表大小: {tokenizer.get_vocab_size()}")
    vocab = tokenizer.get_vocab()
    for i, token in enumerate(SPECIAL_TOKENS):
        expected_id = current_vocab_size + i
        actual_id = vocab.get(token)
        if actual_id != expected_id:
            logger.warning(
                f"特殊token ID不匹配: {token} 期望 {expected_id}, 实际 {actual_id}"
            )
        else:
            logger.info(f"  {token}: ID {actual_id}")


def save_vocab_text(tokenizer: Tokenizer, output_path: Path) -> None:
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(output_path, "w", encoding="utf-8") as f:
        for token, _token_id in sorted_vocab:
            f.write(f"{token}\n")
    logger.info(f"词表已保存: {output_path}")


def create_transformers_tokenizer(
    tokenizer: Tokenizer,
    output_dir: Path,
) -> PreTrainedTokenizerFast:
    transformers_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        **SPECIAL_TOKENS_MAP,
    )
    transformers_tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer已保存到: {output_dir}")
    return transformers_tokenizer


def validate_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    vocab_size: int,
) -> bool:
    logger.info("开始验证tokenizer...")
    is_valid = True
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != vocab_size:
        logger.error(f"词表大小不匹配: 期望 {vocab_size}, 实际 {actual_vocab_size}")
        is_valid = False
    else:
        logger.info(f"词表大小正确: {actual_vocab_size}")
    for i, token in enumerate(SPECIAL_TOKENS):
        expected_id = 32000 + i
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            logger.error(
                f"特殊token ID不匹配: {token} 期望 {expected_id}, 实际 {actual_id}"
            )
            is_valid = False
        else:
            logger.info(f"{token}: ID {actual_id}")
    logger.info(f"bos_token: {tokenizer.bos_token}")
    logger.info(f"eos_token: {tokenizer.eos_token}")
    logger.info(f"pad_token: {tokenizer.pad_token}")
    logger.info(f"unk_token: {tokenizer.unk_token}")
    test_text = "<|im_start|>assistant\n<|think|>推理<|/think|>答案<|im_end|>"
    encoded = tokenizer(test_text, return_tensors="pt")
    decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
    if test_text.replace(" ", "") == decoded.replace(" ", ""):
        logger.info("编解码一致性测试通过")
    else:
        logger.warning(f"编解码可能有差异:\n  原始: {test_text}\n  解码: {decoded}")
    normal_text = "Hello, 世界! 这是一个测试。"
    encoded_normal = tokenizer(normal_text)
    _ = tokenizer.decode(encoded_normal["input_ids"], skip_special_tokens=False)
    logger.info(
        f"常规文本测试: '{normal_text[:20]}...' -> {len(encoded_normal['input_ids'])} tokens"
    )
    if is_valid:
        logger.info("验证通过")
    else:
        logger.error("验证失败")
    return is_valid


def train_tokenizer(
    data_dir: Path,
    template_dir: Path,
    output_dir: Path,
    vocab_size: int,
    min_frequency: int,
    validate: bool = False,
) -> int:
    try:
        logger.info("=" * 60)
        logger.info("开始训练 Tokenizer")
        logger.info("=" * 60)
        logger.info(f"数据目录: {data_dir}")
        logger.info(f"模板目录: {template_dir}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"词表大小: {vocab_size}")
        logger.info(f"最小词频: {min_frequency}")
        tokenizer = load_template_tokenizer(template_dir)
        train_bpe_tokenizer(
            tokenizer=tokenizer,
            data_dir=data_dir,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        add_special_tokens(tokenizer)
        output_dir.mkdir(parents=True, exist_ok=True)
        transformers_tokenizer = create_transformers_tokenizer(tokenizer, output_dir)
        if validate:
            logger.info("")
            is_valid = validate_tokenizer(transformers_tokenizer, vocab_size)
            if not is_valid:
                logger.error("Tokenizer 验证失败")
                return 1
        logger.info("")
        logger.info("=" * 60)
        logger.info("训练完成")
        logger.info("=" * 60)
        logger.info(f"最终词表大小: {len(transformers_tokenizer)}")
        logger.info(f"输出目录: {output_dir}")
        saved_files = list(output_dir.iterdir())
        logger.info(f"输出文件 ({len(saved_files)} 个):")
        for f in sorted(saved_files):
            size = f.stat().st_size if f.is_file() else 0
            logger.info(f"  - {f.name} ({size:,} bytes)")
        return 0
    except Exception as e:
        logger.exception(f"训练失败: {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="训练 BPE Tokenizer - 流式处理避免OOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/train_tokenizer.py

  # 启用验证
  python scripts/train_tokenizer.py --validate

  # 自定义参数
  python scripts/train_tokenizer.py \
      --data-dir data/datasets/nanomind_tokenizer \
      --template-dir output/qwen3_next_tokenizer \
      --output-dir output/tokenizer_32k \
      --vocab-size 32005 \
      --min-frequency 2
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
        help=f"模板tokenizer目录 (默认: {DEFAULT_TEMPLATE_DIR})",
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
        help=f"目标总词表大小 (默认: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--min-frequency",
        "-f",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
        help=f"最小词频 (默认: {DEFAULT_MIN_FREQUENCY})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="训练后验证tokenizer",
    )
    args = parser.parse_args()
    return train_tokenizer(
        data_dir=args.data_dir,
        template_dir=args.template_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        validate=args.validate,
    )


if __name__ == "__main__":
    sys.exit(main())
