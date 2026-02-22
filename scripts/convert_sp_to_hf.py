#!/usr/bin/env python3
"""将 SentencePiece 模型转换为 HuggingFace Tokenizer 格式

完全对齐 qwen3_next_tokenizer 的配置：
1. 复制 qwen3 的 pre-tokenizer 配置（Split + ByteLevel）
2. 复制 qwen3 的 post-processor 配置
3. 添加相同的特殊 token
4. 复制 chat_template
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tokenizers import (
    Tokenizer,
    decoders,
    pre_tokenizers,
    processors,
)
from tokenizers.models import BPE

if TYPE_CHECKING:
    from sentencepiece import SentencePieceProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

__all__ = [
    "QWEN3_SPECIAL_TOKENS",
    "load_qwen3_config",
    "convert_sp_to_hf_bpe",
    "setup_pre_tokenizer",
    "setup_decoder",
    "setup_post_processor",
    "add_special_tokens",
    "create_tokenizer_config",
    "create_special_tokens_map",
    "convert",
    "test_tokenizer",
]

# qwen3 的特殊 token 列表
QWEN3_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def _load_sp_model(sp_model_path: Path) -> "SentencePieceProcessor":
    """加载 SentencePiece 模型。"""
    from sentencepiece import SentencePieceProcessor as SPP

    processor = SPP()
    processor.load(str(sp_model_path))
    return processor


def load_qwen3_config(qwen3_dir: Path) -> dict[str, Any] | None:
    """加载 qwen3 tokenizer 的配置"""
    tokenizer_json = qwen3_dir / "tokenizer.json"
    config_json = qwen3_dir / "tokenizer_config.json"

    config: dict[str, Any] = {}

    if tokenizer_json.exists():
        with open(tokenizer_json, encoding="utf-8") as f:
            config["tokenizer"] = json.load(f)
        logger.info("已加载 tokenizer.json")

    if config_json.exists():
        with open(config_json, encoding="utf-8") as f:
            config["config"] = json.load(f)
        logger.info("已加载 tokenizer_config.json")

    return config if config else None


def convert_sp_to_hf_bpe(
    sp_model_path: Path,
    vocab_size: int = 32000,
) -> Tokenizer:
    """
    将 SentencePiece 模型转换为 HuggingFace BPE 模型

    注意：SentencePiece 训练使用 BPE 算法（--model_type=bpe）
    所以我们使用 tokenizers 的 BPE 模型
    """
    logger.info(f"加载 SentencePiece 模型: {sp_model_path}")
    sp = _load_sp_model(sp_model_path)

    # 提取词表（只取前 vocab_size 个）
    vocab: dict[str, int] = {}
    sp_vocab_size = sp.vocab_size()
    for i in range(min(vocab_size, sp_vocab_size)):
        token = sp.id_to_piece(i)
        vocab[token] = i

    logger.info(f"基础词表大小: {len(vocab)}")

    # 创建 BPE 模型（没有 unk_token，因为 qwen3 没有）
    model = BPE(
        vocab=vocab,
        merges=[],  # BPE 不需要显式 merges
        unk_token=None,
        fuse_unk=False,
        byte_fallback=True,
    )
    return Tokenizer(model)


def setup_pre_tokenizer(tokenizer: Tokenizer) -> None:
    """
    设置 pre-tokenizer，完全复制 qwen3 的配置

    qwen3 配置：
    - Sequence: [Split (regex pattern), ByteLevel]
    """
    # qwen3 的 regex pattern (来自官方配置)  # noqa: E501
    qwen3_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    pre_tokenizer = pre_tokenizers.Sequence(
        [
            # 第一步：使用 regex split
            pre_tokenizers.Split(
                pattern=qwen3_pattern,
                behavior="isolated",  # type: ignore[arg-type]
                invert=False,
            ),
            # 第二步：ByteLevel（不添加 prefix space）
            pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                trim_offsets=True,
                use_regex=False,
            ),
        ]
    )

    tokenizer.pre_tokenizer = pre_tokenizer
    logger.info("已设置 pre-tokenizer（复制 qwen3 配置）")


def setup_decoder(tokenizer: Tokenizer) -> None:
    """设置 decoder，使用 ByteLevel"""
    tokenizer.decoder = decoders.ByteLevel()
    logger.info("已设置 decoder")


def setup_post_processor(tokenizer: Tokenizer) -> None:
    """
    设置 post-processor

    qwen3 使用 ByteLevel post-processor
    """
    tokenizer.post_processor = processors.ByteLevel(
        add_prefix_space=False,
        trim_offsets=False,
    )
    logger.info("已设置 post-processor")


def add_special_tokens(tokenizer: Tokenizer) -> None:
    """
    添加特殊 token

    Args:
        tokenizer: Tokenizer 实例
    """
    for token in QWEN3_SPECIAL_TOKENS:
        tokenizer.add_tokens([token])
        logger.debug(f"添加特殊 token: {token}")

    logger.info(f"添加了 {len(QWEN3_SPECIAL_TOKENS)} 个特殊 token")


def create_tokenizer_config(
    vocab_size: int,
    extra_special_tokens: tuple[str, ...],
) -> dict[str, Any]:
    """创建 tokenizer_config.json"""
    return {
        "add_prefix_space": False,
        "bos_token": None,
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "errors": "replace",
        "extra_special_tokens": list(extra_special_tokens),
        "is_local": False,
        "model_max_length": 1010000,
        "pad_token": "<|endoftext|>",
        "split_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": None,
        "vocab_size": vocab_size,
    }


def create_special_tokens_map() -> dict[str, str | None]:
    """创建 special_tokens_map.json"""
    return {
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    }


def convert(
    sp_model_path: Path,
    qwen3_dir: Path,
    output_dir: Path,
    vocab_size: int = 32000,
) -> Path:
    """
    主转换函数

    Args:
        sp_model_path: SentencePiece 模型文件路径
        qwen3_dir: qwen3 tokenizer 目录（用于复制配置）
        output_dir: 输出目录
        vocab_size: 基础词表大小

    Returns:
        Path: 输出的 tokenizer 目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SentencePiece -> HuggingFace 转换")
    logger.info("=" * 60)

    # 1. 加载 qwen3 配置（用于验证和日志记录）
    logger.info("\n[1/6] 加载 qwen3 配置...")
    qwen3_config = load_qwen3_config(qwen3_dir)
    if qwen3_config is None:
        logger.warning("未找到 qwen3 配置文件，将使用默认配置")

    # 2. 转换模型
    logger.info("\n[2/6] 转换 SentencePiece 模型...")
    tokenizer = convert_sp_to_hf_bpe(sp_model_path, vocab_size)

    # 3. 设置 pre-tokenizer
    logger.info("\n[3/6] 设置 pre-tokenizer...")
    setup_pre_tokenizer(tokenizer)

    # 4. 设置 decoder
    logger.info("\n[4/6] 设置 decoder...")
    setup_decoder(tokenizer)

    # 5. 添加特殊 token
    logger.info("\n[5/6] 添加特殊 token...")
    add_special_tokens(tokenizer)

    # 6. 设置 post-processor
    logger.info("\n[6/6] 设置 post-processor...")
    setup_post_processor(tokenizer)

    # 保存 tokenizer
    tokenizer_file = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    logger.info(f"已保存 tokenizer: {tokenizer_file}")

    # 创建 tokenizer_config.json
    config = create_tokenizer_config(
        vocab_size=vocab_size + len(QWEN3_SPECIAL_TOKENS),
        extra_special_tokens=QWEN3_SPECIAL_TOKENS,
    )
    config_file = output_dir / "tokenizer_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存 config: {config_file}")

    # 创建 special_tokens_map.json
    special_map = create_special_tokens_map()
    special_map_file = output_dir / "special_tokens_map.json"
    with open(special_map_file, "w", encoding="utf-8") as f:
        json.dump(special_map, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存 special_tokens_map: {special_map_file}")

    # 复制 chat_template
    chat_template_src = qwen3_dir / "chat_template.jinja"
    if chat_template_src.exists():
        chat_template_dst = output_dir / "chat_template.jinja"
        shutil.copy(chat_template_src, chat_template_dst)
        logger.info(f"已复制 chat_template: {chat_template_dst}")
    else:
        logger.warning(f"未找到 chat_template: {chat_template_src}")

    # 复制 SP 模型文件（用于参考）
    sp_model_dst = output_dir / "sp_model.model"
    shutil.copy(sp_model_path, sp_model_dst)
    logger.info(f"已复制 SP 模型: {sp_model_dst}")

    logger.info("\n" + "=" * 60)
    logger.info("转换完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)

    return output_dir


def test_tokenizer(tokenizer_dir: Path) -> None:
    """测试转换后的 tokenizer"""
    from transformers import PreTrainedTokenizerFast

    logger.info("\n测试 tokenizer...")

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_dir,
            trust_remote_code=True,
        )

        test_texts = [
            "Hello world!",
            "你好，世界！",
            "<|im_start|>user\n你好<|im_end|>",
            "def hello_world(): pass",
        ]

        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            match = "✓" if decoded.strip() == text.strip() else "✗"
            logger.info(f"  {match} '{text[:30]}...' -> {len(encoded)} tokens")

        logger.info(f"\n词表大小: {tokenizer.vocab_size}")
        logger.info(f"EOS token: {tokenizer.eos_token}")
        logger.info(f"PAD token: {tokenizer.pad_token}")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 SentencePiece 模型转换为 HuggingFace Tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基础用法
    python scripts/convert_sp_to_hf.py \\
        --sp-model output/tokenizer_32k_sp/tokenizer.model \\
        --qwen3-dir output/qwen3_next_tokenizer \\
        --output-dir output/tokenizer_32k

    # 指定词表大小
    python scripts/convert_sp_to_hf.py \\
        --sp-model output/tokenizer_32k_sp/tokenizer.model \\
        --qwen3-dir output/qwen3_next_tokenizer \\
        --output-dir output/tokenizer_32k \\
        --vocab-size 32000
        """,
    )

    parser.add_argument(
        "--sp-model",
        type=Path,
        required=True,
        help="SentencePiece 模型文件路径",
    )
    parser.add_argument(
        "--qwen3-dir",
        type=Path,
        required=True,
        help="qwen3 tokenizer 目录（用于复制配置）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tokenizer_32k"),
        help="输出目录（默认: output/tokenizer_32k）",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="基础词表大小（默认: 32000，不包括特殊 token）",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="转换后测试 tokenizer",
    )

    args = parser.parse_args()

    # 检查输入文件
    if not args.sp_model.exists():
        logger.error(f"未找到 SP 模型文件: {args.sp_model}")
        sys.exit(1)

    if not args.qwen3_dir.exists():
        logger.error(f"未找到 qwen3 目录: {args.qwen3_dir}")
        sys.exit(1)

    # 执行转换
    result_dir = convert(
        sp_model_path=args.sp_model,
        qwen3_dir=args.qwen3_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )

    # 测试
    if args.test:
        test_tokenizer(result_dir)
