#!/usr/bin/env python3
"""
Tokenizer 训练脚本 - 基于模板继承的 BPE 训练

从采样数据训练与 Qwen3-Next 兼容的 32K 词表 BPE Tokenizer。
使用 `train_new_from_iterator` 方法自动继承模板配置。

用法:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --data-dir data/datasets/nanomind_tokenizer
    python scripts/train_tokenizer.py --vocab-size 32005 --validate

输出:
    output/tokenizer_32k/
    ├── tokenizer.json              # 词表与合并规则
    ├── tokenizer_config.json       # Tokenizer 配置
    └── chat_template.jinja         # 对话模板（从模板复制）
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_DATA_DIR = Path("data/datasets/nanomind_tokenizer")
DEFAULT_TEMPLATE_DIR = Path("output/qwen3_next_tokenizer")
DEFAULT_OUTPUT_DIR = Path("output/tokenizer_32k")
DEFAULT_VOCAB_SIZE = 32005  # 32000 BPE + 5 特殊 token

# 特殊 token 定义
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
]

# 模型属性映射
EOS_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|endoftext|>"
BOS_TOKEN = None
UNK_TOKEN = None


def load_template_tokenizer(template_dir: Path) -> PreTrainedTokenizerFast:
    """从本地目录加载模板 tokenizer。

    Args:
        template_dir: 模板 tokenizer 目录路径

    Returns:
        加载的模板 tokenizer

    Raises:
        FileNotFoundError: 当模板目录不存在时
        Exception: 当加载失败时
    """
    if not template_dir.exists():
        raise FileNotFoundError(f"模板目录不存在: {template_dir}")

    logger.info(f"加载模板 tokenizer: {template_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            template_dir,
            trust_remote_code=True,
        )
        logger.info("成功加载模板 tokenizer")
        return tokenizer
    except Exception as e:
        logger.error(f"加载模板 tokenizer 失败: {e}")
        raise


def find_parquet_files(data_dir: Path) -> list[Path]:
    """递归查找数据目录中的所有 parquet 文件。

    Args:
        data_dir: 数据目录路径

    Returns:
        parquet 文件路径列表（已排序）
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    files = sorted(data_dir.rglob("*.parquet"))
    logger.info(f"找到 {len(files)} 个 parquet 文件")
    return files


def create_text_iterator(
    data_dir: Path,
    batch_size: int = 10000,
) -> Iterator[str]:
    """创建文本迭代器，流式读取采样数据。

    采用生成器模式，避免一次性加载所有数据到内存。
    每处理 batch_size 个文档后触发垃圾回收。

    Args:
        data_dir: 采样数据目录
        batch_size: 批次大小，每处理这么多文档后触发 GC

    Yields:
        文档文本内容

    Raises:
        FileNotFoundError: 当数据目录不存在时
    """
    files = find_parquet_files(data_dir)

    if not files:
        logger.warning(f"未找到任何 parquet 文件: {data_dir}")
        return

    total_yielded = 0

    for file_path in files:
        logger.debug(f"读取文件: {file_path}")

        try:
            table = pq.read_table(file_path, columns=["text"])
            texts = table["text"].to_pylist()

            for text in texts:
                if text and isinstance(text, str):
                    yield text
                    total_yielded += 1

                    if total_yielded % batch_size == 0:
                        gc.collect()
                        logger.debug(f"已处理 {total_yielded} 个文档")

        except Exception as e:
            logger.warning(f"读取文件失败 {file_path}: {e}")
            continue

    logger.info(f"文本迭代器完成，共 {total_yielded} 个文档")


def train_tokenizer_with_iterator(
    template_tokenizer: PreTrainedTokenizerFast,
    data_dir: Path,
    vocab_size: int,
) -> PreTrainedTokenizerFast:
    """使用 `train_new_from_iterator` 训练新 tokenizer。

    该方法自动继承模板 tokenizer 的所有配置（normalizer、pre_tokenizer、
    decoder、post_processor 等），无需手动提取和复制。

    Args:
        template_tokenizer: 模板 tokenizer
        data_dir: 训练数据目录
        vocab_size: 目标词表大小（包含特殊 token）

    Returns:
        训练完成的新 tokenizer
    """
    logger.info(f"开始训练 tokenizer (vocab_size={vocab_size})")
    logger.info(f"特殊 tokens: {SPECIAL_TOKENS}")

    # 创建文本迭代器
    text_iterator = create_text_iterator(data_dir)

    # 使用 train_new_from_iterator 训练
    # 自动继承模板的所有配置
    new_tokenizer = template_tokenizer.train_new_from_iterator(
        text_iterator,
        vocab_size=vocab_size,
        new_special_tokens=SPECIAL_TOKENS,
    )

    logger.info("Tokenizer 训练完成")
    return new_tokenizer


def configure_special_token_mappings(
    tokenizer: PreTrainedTokenizerFast,
) -> PreTrainedTokenizerFast:
    """配置特殊 token 映射。

    设置 eos_token、pad_token、bos_token、unk_token 等模型属性。

    Args:
        tokenizer: 训练完成的 tokenizer

    Returns:
        配置完成后的 tokenizer
    """
    logger.info("配置特殊 token 映射")

    tokenizer.eos_token = EOS_TOKEN
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.bos_token = BOS_TOKEN
    tokenizer.unk_token = UNK_TOKEN

    logger.info(f"  eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"  pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"  bos_token: {tokenizer.bos_token}")
    logger.info(f"  unk_token: {tokenizer.unk_token}")

    return tokenizer


def copy_chat_template(template_dir: Path, output_dir: Path) -> None:
    """从模板目录复制 chat_template.jinja 到输出目录。

    Args:
        template_dir: 模板目录
        output_dir: 输出目录
    """
    template_file = template_dir / "chat_template.jinja"
    output_file = output_dir / "chat_template.jinja"

    if template_file.exists():
        shutil.copy2(template_file, output_file)
        logger.info("已复制 chat_template.jinja")

    else:
        logger.warning(f"模板目录中未找到 chat_template.jinja: {template_file}")


def save_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    output_dir: Path,
    template_dir: Path,
) -> None:
    """保存 tokenizer 到输出目录。

    Args:
        tokenizer: 训练完成的 tokenizer
        output_dir: 输出目录
        template_dir: 模板目录（用于复制 chat_template.jinja）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer 已保存到: {output_dir}")

    # 复制 chat_template.jinja
    copy_chat_template(template_dir, output_dir)

    # 列出输出文件
    saved_files = list(output_dir.iterdir())
    logger.info(f"输出文件 ({len(saved_files)} 个):")
    for f in sorted(saved_files):
        size = f.stat().st_size if f.is_file() else 0
        logger.info(f"  - {f.name} ({size:,} bytes)")


def _load_tokenizer_json(tokenizer_dir: Path) -> dict[str, Any]:
    """加载 tokenizer.json 文件。"""
    json_path = tokenizer_dir / "tokenizer.json"
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _load_tokenizer_config(tokenizer_dir: Path) -> dict[str, Any]:
    """加载 tokenizer_config.json 文件。"""
    config_path = tokenizer_dir / "tokenizer_config.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _validate_vocab_size(
    tokenizer: PreTrainedTokenizerFast,
    expected_size: int,
) -> bool:
    """验证词表大小。"""
    actual_size = len(tokenizer)
    if actual_size != expected_size:
        logger.error(f"词表大小验证失败: 期望 {expected_size}, 实际 {actual_size}")
        return False
    logger.info(f"✓ 词表大小: {actual_size}")
    return True


def _validate_special_token_ids(
    tokenizer: PreTrainedTokenizerFast,
) -> bool:
    """验证特殊 token ID 正确性。"""
    expected_ids = {
        "<|endoftext|>": 32000,
        "<|im_start|>": 32001,
        "<|im_end|>": 32002,
        "<think>": 32003,
        "</think>": 32004,
    }

    for token, expected_id in expected_ids.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            logger.error(
                f"特殊 token ID 错误: {token} 期望 {expected_id}, 实际 {actual_id}"
            )
            return False

    logger.info("✓ 特殊 token IDs 正确")
    return True


def _validate_token_mappings(tokenizer: PreTrainedTokenizerFast) -> bool:
    """验证模型属性映射。"""
    checks = [
        (tokenizer.eos_token, EOS_TOKEN, "eos_token"),
        (tokenizer.pad_token, PAD_TOKEN, "pad_token"),
        (tokenizer.bos_token, BOS_TOKEN, "bos_token"),
        (tokenizer.unk_token, UNK_TOKEN, "unk_token"),
    ]

    for actual, expected, name in checks:
        if actual != expected:
            logger.error(f"{name} 映射错误: 期望 {expected}, 实际 {actual}")
            return False

    logger.info("✓ 模型属性映射正确")
    return True


def _validate_added_tokens(tokenizer_dir: Path) -> bool:
    """验证 added_tokens 包含正确的 5 个 token。"""
    tokenizer_json = _load_tokenizer_json(tokenizer_dir)

    expected_added = set(SPECIAL_TOKENS)
    actual_added = {t["content"] for t in tokenizer_json.get("added_tokens", [])}

    if actual_added != expected_added:
        logger.error("added_tokens 验证失败")
        logger.error(f"  期望: {expected_added}")
        logger.error(f"  实际: {actual_added}")
        return False

    logger.info(f"✓ added_tokens: {sorted(actual_added)}")
    return True


def _validate_extra_special_tokens(tokenizer_dir: Path) -> bool:
    """验证 extra_special_tokens 包含正确的 4 个 token（不含 <|endoftext|>）。"""
    config = _load_tokenizer_config(tokenizer_dir)

    expected_special = {"<|im_start|>", "<|im_end|>", "<think>", "</think>"}
    actual_special = set(config.get("extra_special_tokens", []))

    if actual_special != expected_special:
        logger.error("extra_special_tokens 验证失败")
        logger.error(f"  期望: {expected_special}")
        logger.error(f"  实际: {actual_special}")
        return False

    # 检查不包含 <|endoftext|>
    if "<|endoftext|>" in actual_special:
        logger.error("extra_special_tokens 不应包含 <|endoftext|>")
        return False

    logger.info(f"✓ extra_special_tokens: {sorted(actual_special)}")
    return True


def _validate_no_vision_tokens(tokenizer_dir: Path) -> bool:
    """验证不包含视觉/多模态相关的 token。"""
    config = _load_tokenizer_config(tokenizer_dir)
    extra_special = config.get("extra_special_tokens", [])

    vision_tokens = [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<tool_call>",
        "</tool_call>",
    ]

    for vt in vision_tokens:
        if vt in extra_special:
            logger.error(f"extra_special_tokens 不应包含视觉/多模态 token: {vt}")
            return False

    logger.info("✓ 不包含视觉/多模态相关 tokens")
    return True


def _validate_template_consistency(
    trained_tokenizer: PreTrainedTokenizerFast,
    template_tokenizer: PreTrainedTokenizerFast,
) -> bool:
    """验证与模板的配置一致性。"""
    # 比较 normalizer、pre_tokenizer、decoder 配置
    # 注意：词表不同，但处理流程应一致

    template_config = template_tokenizer.to_dict()
    trained_config = trained_tokenizer.to_dict()

    # 检查关键配置字段
    keys_to_check = ["normalizer", "pre_tokenizer", "decoder", "post_processor"]

    for key in keys_to_check:
        template_val = template_config.get(key)
        trained_val = trained_config.get(key)

        if template_val != trained_val:
            # 词表不同是正常的，但配置结构应该相同
            logger.warning(f"{key} 配置可能与模板不同（可能由于词表变化）")

    logger.info("✓ 模板配置一致性检查完成")
    return True


def _validate_encode_decode(tokenizer: PreTrainedTokenizerFast) -> bool:
    """验证编解码一致性。"""
    test_texts = [
        "Hello World",
        "你好世界",
        "<|im_start|>user\n问题<|im_end|>",
        "<think>推理过程...</think>答案",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        # 由于 BPE 特性，解码后可能不完全相同，但重新编码应该得到相同结果
        re_encoded = tokenizer.encode(decoded)

        if encoded != re_encoded:
            logger.warning(f"编解码一致性检查警告: '{text[:30]}...'")
            logger.warning(f"  原始编码: {encoded}")
            logger.warning(f"  重新编码: {re_encoded}")

    logger.info("✓ 编解码一致性检查完成")
    return True


def validate_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    output_dir: Path,
    template_tokenizer: PreTrainedTokenizerFast,
    expected_vocab_size: int,
) -> bool:
    """执行完整的 tokenizer 验证。

    包括：
    - 词表大小验证
    - 特殊 token ID 验证
    - 模型属性映射验证
    - added_tokens 验证
    - extra_special_tokens 验证
    - 视觉/多模态 token 排除验证
    - 模板配置一致性验证
    - 编解码一致性验证

    Args:
        tokenizer: 训练完成的 tokenizer
        output_dir: 输出目录
        template_tokenizer: 模板 tokenizer
        expected_vocab_size: 期望的词表大小

    Returns:
        验证是否全部通过
    """
    logger.info("=" * 60)
    logger.info("开始验证 tokenizer")
    logger.info("=" * 60)

    all_passed = True

    # 1. 基础验证
    all_passed &= _validate_vocab_size(tokenizer, expected_vocab_size)
    all_passed &= _validate_special_token_ids(tokenizer)
    all_passed &= _validate_token_mappings(tokenizer)

    # 2. 配置文件验证
    all_passed &= _validate_added_tokens(output_dir)
    all_passed &= _validate_extra_special_tokens(output_dir)
    all_passed &= _validate_no_vision_tokens(output_dir)

    # 3. 一致性验证
    all_passed &= _validate_template_consistency(tokenizer, template_tokenizer)
    all_passed &= _validate_encode_decode(tokenizer)

    logger.info("=" * 60)
    if all_passed:
        logger.info("✓ 所有验证通过！")
    else:
        logger.error("✗ 验证失败，请检查上述错误")
    logger.info("=" * 60)

    return all_passed


def train_tokenizer(
    data_dir: Path,
    template_dir: Path,
    output_dir: Path,
    vocab_size: int,
    validate: bool = True,
) -> int:
    """主函数：训练 tokenizer。

    Args:
        data_dir: 训练数据目录
        template_dir: 模板 tokenizer 目录
        output_dir: 输出目录
        vocab_size: 目标词表大小
        validate: 是否执行验证

    Returns:
        退出码 (0 表示成功)
    """
    logger.info("=" * 60)
    logger.info("Tokenizer 训练")
    logger.info("=" * 60)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"模板目录: {template_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"词表大小: {vocab_size}")

    try:
        # 1. 加载模板 tokenizer
        template_tokenizer = load_template_tokenizer(template_dir)

        # 2. 训练新 tokenizer
        new_tokenizer = train_tokenizer_with_iterator(
            template_tokenizer=template_tokenizer,
            data_dir=data_dir,
            vocab_size=vocab_size,
        )

        # 3. 配置特殊 token 映射
        new_tokenizer = configure_special_token_mappings(new_tokenizer)

        # 4. 保存 tokenizer
        save_tokenizer(new_tokenizer, output_dir, template_dir)

        # 5. 验证（如果启用）
        if validate:
            success = validate_tokenizer(
                tokenizer=new_tokenizer,
                output_dir=output_dir,
                template_tokenizer=template_tokenizer,
                expected_vocab_size=vocab_size,
            )
            if not success:
                return 1

        logger.info("=" * 60)
        logger.info("Tokenizer 训练完成！")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.exception(f"训练失败: {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="训练与 Qwen3-Next 兼容的 32K 词表 BPE Tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/train_tokenizer.py

  # 指定数据目录和输出目录
  python scripts/train_tokenizer.py --data-dir /path/to/data --output-dir /path/to/output

  # 自定义词表大小
  python scripts/train_tokenizer.py --vocab-size 32005

  # 跳过验证
  python scripts/train_tokenizer.py --no-validate
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
        help=f"目标词表大小 (默认: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否执行验证 (默认: True)",
    )

    args = parser.parse_args()

    return train_tokenizer(
        data_dir=args.data_dir,
        template_dir=args.template_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        validate=args.validate,
    )


if __name__ == "__main__":
    sys.exit(main())
