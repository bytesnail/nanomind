#!/usr/bin/env python3
"""Tokenizer 训练脚本 - 基于 Qwen3-Next 模板的 BPE 训练.

训练流程:
1. 加载 Qwen3-Next 模板 Tokenizer
2. 提取模板配置 (normalizer/pretokenizer/decoder)
3. 在 800K 采样数据上训练 32K BPE 词表
4. 组合新词表与模板配置，配置特殊 token
5. 保存并验证输出

用法:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --data-dir data/datasets/nanomind_tokenizer
    python scripts/train_tokenizer.py --template-dir output/qwen3_next_tokenizer --validate

输出:
    output/tokenizer_32k/
    ├── tokenizer.json              # 词表与处理配置
    ├── tokenizer_config.json       # Tokenizer 配置
    └── chat_template.jinja         # 对话模板 (从模板复制)
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
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/datasets/nanomind_tokenizer")
DEFAULT_TEMPLATE_DIR = Path("output/qwen3_next_tokenizer")
DEFAULT_OUTPUT_DIR = Path("output/tokenizer_32k")
DEFAULT_VOCAB_SIZE = 32005
DEFAULT_MIN_FREQUENCY = 2

EXTRA_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
]

SPECIAL_TOKENS = ["<|endoftext|>"] + EXTRA_SPECIAL_TOKENS  # ID 32000-32004

# 实际 BPE 词表大小 (不含特殊 token)
BPE_VOCAB_SIZE = DEFAULT_VOCAB_SIZE - len(SPECIAL_TOKENS)


VISION_TOKENS = [
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
]


def load_template_tokenizer(template_dir: Path) -> AutoTokenizer:
    """从模板目录加载完整的 Qwen2Tokenizer.

    Args:
        template_dir: 模板目录路径

    Returns:
        加载的 AutoTokenizer

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


def extract_template_config(template_dir: Path) -> dict[str, Any]:
    """从模板提取 normalizer/pretokenizer/decoder 配置.

    读取 tokenizer.json 中的关键组件配置，用于与新训练的 BPE 词表组合。

    Args:
        template_dir: 模板目录路径

    Returns:
        包含 normalizer, pre_tokenizer, decoder, post_processor 的字典

    Raises:
        FileNotFoundError: 当 tokenizer.json 不存在时
        json.JSONDecodeError: 当 JSON 解析失败时
    """
    tokenizer_json_path = template_dir / "tokenizer.json"
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(f"模板 tokenizer.json 不存在: {tokenizer_json_path}")

    logger.info(f"提取模板配置: {tokenizer_json_path}")

    with open(tokenizer_json_path, encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    config = {
        "normalizer": tokenizer_data.get("normalizer"),
        "pre_tokenizer": tokenizer_data.get("pre_tokenizer"),
        "decoder": tokenizer_data.get("decoder"),
        "post_processor": tokenizer_data.get("post_processor"),
        "model_type": tokenizer_data.get("model", {}).get("type"),
    }

    special_tokens_map_path = template_dir / "special_tokens_map.json"
    if special_tokens_map_path.exists():
        with open(special_tokens_map_path, encoding="utf-8") as f:
            config["special_tokens_map"] = json.load(f)

    logger.info(f"提取配置项: {list(config.keys())}")
    return config


def get_parquet_files(data_dir: Path) -> list[Path]:
    """递归获取数据目录中的所有 parquet 文件.

    Args:
        data_dir: 数据目录路径

    Returns:
        parquet 文件路径列表 (按名称排序)
    """
    if not data_dir.exists():
        logger.warning(f"数据目录不存在: {data_dir}")
        return []

    files = sorted(data_dir.rglob("*.parquet"))
    logger.info(f"找到 {len(files)} 个 parquet 文件")
    return files


def text_iterator(data_dir: Path, batch_size: int = 10000) -> Iterator[str]:
    """流式迭代 parquet 文件中的文本数据.

    使用生成器模式避免一次性加载所有数据到内存。

    Args:
        data_dir: 数据目录路径
        batch_size: 每批处理的文档数

    Yields:
        文本字符串
    """
    files = get_parquet_files(data_dir)
    if not files:
        logger.warning("未找到 parquet 文件")
        return

    total_rows = 0
    for fp in files:
        try:
            metadata = pq.read_metadata(fp)
            total_rows += metadata.num_rows
        except Exception as e:
            logger.warning(f"无法读取文件元数据 {fp}: {e}")

    logger.info(f"开始流式读取 {len(files)} 个文件，总计约 {total_rows:,} 行")

    processed = 0
    batch_count = 0

    with tqdm(total=total_rows, desc="读取训练数据") as pbar:
        for fp in files:
            try:
                table = pq.read_table(fp, columns=["text"])
                texts = table["text"].to_pylist()

                for text in texts:
                    if text and isinstance(text, str):
                        yield text
                        processed += 1
                        pbar.update(1)

                        if processed % batch_size == 0:
                            batch_count += 1
                            if batch_count % 10 == 0:
                                gc.collect()

            except Exception as e:
                logger.warning(f"读取文件失败 {fp}: {e}")

    logger.info(f"完成数据迭代，共处理 {processed:,} 个文档")


def train_bpe_vocab(
    data_dir: Path,
    vocab_size: int = BPE_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    batch_size: int = 10000,
) -> Tokenizer:
    """在采样数据上训练 BPE 词表.

    使用 Hugging Face tokenizers 库训练 BPE tokenizer。

    Args:
        data_dir: 训练数据目录
        vocab_size: 目标词表大小 (不含特殊 token)
        min_frequency: 最小词频
        batch_size: 数据迭代批次大小

    Returns:
        训练后的 Tokenizer
    """
    logger.info(
        f"开始 BPE 训练: vocab_size={vocab_size}, min_frequency={min_frequency}"
    )

    tokenizer = Tokenizer(BPE(unk_token=None))

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    logger.info("启动训练...")
    tokenizer.train_from_iterator(
        text_iterator(data_dir, batch_size=batch_size),
        trainer=trainer,
        length=800000,  # 预期总文档数
    )

    logger.info(f"训练完成，词表大小: {tokenizer.get_vocab_size()}")
    return tokenizer


def build_tokenizer_from_template(
    trained_tokenizer: Tokenizer,
    template_config: dict[str, Any],
) -> Tokenizer:
    """将新训练的 BPE 词表与模板配置组合.

    继承模板的 normalizer/pre_tokenizer/decoder/post_processor，
    仅替换 model 为训练后的 BPE 词表。

    Args:
        trained_tokenizer: 训练后的 BPE tokenizer
        template_config: 模板配置字典

    Returns:
        组合后的完整 Tokenizer
    """
    logger.info("开始组合 tokenizer 配置...")

    new_tokenizer = Tokenizer(trained_tokenizer.model)

    if template_config.get("normalizer"):
        new_tokenizer.normalizer = Tokenizer.from_str(
            json.dumps({"normalizer": template_config["normalizer"]})
        ).normalizer

    if template_config.get("pre_tokenizer"):
        new_tokenizer.pre_tokenizer = Tokenizer.from_str(
            json.dumps({"pre_tokenizer": template_config["pre_tokenizer"]})
        ).pre_tokenizer

    if template_config.get("decoder"):
        new_tokenizer.decoder = Tokenizer.from_str(
            json.dumps({"decoder": template_config["decoder"]})
        ).decoder

    if template_config.get("post_processor"):
        new_tokenizer.post_processor = Tokenizer.from_str(
            json.dumps({"post_processor": template_config["post_processor"]})
        ).post_processor

    logger.info("配置组合完成")
    return new_tokenizer


def configure_special_tokens(tokenizer: Tokenizer) -> dict[str, Any]:
    """配置特殊 token 并生成 tokenizer_config.json 内容.

    Args:
        tokenizer: 训练后的 tokenizer

    Returns:
        tokenizer_config.json 的字典内容
    """
    logger.info("配置特殊 token...")

    vocab = tokenizer.get_vocab()

    for i, token in enumerate(SPECIAL_TOKENS):
        token_id = BPE_VOCAB_SIZE + i
        if token not in vocab:
            vocab[token] = token_id

    tokenizer_config = {
        "added_tokens_decoder": {
            str(BPE_VOCAB_SIZE + i): {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            for i, token in enumerate(SPECIAL_TOKENS)
        },
        "additional_special_tokens": EXTRA_SPECIAL_TOKENS,
        "bos_token": None,
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "extra_special_tokens": EXTRA_SPECIAL_TOKENS,
        "model_max_length": 1010000,
        "pad_token": "<|endoftext|>",
        "unk_token": None,
    }

    logger.info(
        f"配置完成: {len(SPECIAL_TOKENS)} 个 special_tokens, "
        f"{len(EXTRA_SPECIAL_TOKENS)} 个 extra_special_tokens"
    )
    return tokenizer_config


def save_tokenizer(
    tokenizer: Tokenizer,
    tokenizer_config: dict[str, Any],
    output_dir: Path,
    template_dir: Path,
) -> None:
    """保存 tokenizer 到输出目录.

    Args:
        tokenizer: 训练后的 tokenizer
        tokenizer_config: tokenizer 配置字典
        output_dir: 输出目录
        template_dir: 模板目录 (用于复制 chat_template.jinja)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_dir / "tokenizer.json"))
    logger.info(f"tokenizer.json 已保存: {output_dir / 'tokenizer.json'}")

    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    logger.info(f"tokenizer_config.json 已保存: {output_dir / 'tokenizer_config.json'}")

    special_tokens_map = {
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    }
    with open(output_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
    logger.info("special_tokens_map.json 已保存")

    chat_template_path = template_dir / "chat_template.jinja"
    if chat_template_path.exists():
        shutil.copy(chat_template_path, output_dir / "chat_template.jinja")
        logger.info("chat_template.jinja 已复制")


def verify_tokenizer(
    output_dir: Path,
    template_dir: Path,
    expected_vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> bool:
    """验证训练后的 tokenizer.

    检查:
    - 词表大小
    - 特殊 token ID
    - 编解码一致性
    - 模板一致性

    Args:
        output_dir: 输出目录
        template_dir: 模板目录
        expected_vocab_size: 预期词表大小

    Returns:
        验证是否通过
    """
    logger.info("=" * 60)
    logger.info("开始验证 tokenizer")
    logger.info("=" * 60)

    try:
        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        template = AutoTokenizer.from_pretrained(template_dir, trust_remote_code=True)
    except Exception as e:
        logger.error(f"加载 tokenizer 失败: {e}")
        return False

    all_passed = True

    # 1. 验证词表大小
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != expected_vocab_size:
        logger.error(
            f"词表大小错误: 预期 {expected_vocab_size}, 实际 {actual_vocab_size}"
        )
        all_passed = False
    else:
        logger.info(f"✓ 词表大小: {actual_vocab_size}")

    # 2. 验证特殊 token
    vocab = tokenizer.get_vocab()
    for i, token in enumerate(SPECIAL_TOKENS):
        expected_id = BPE_VOCAB_SIZE + i
        actual_id = vocab.get(token)
        if actual_id != expected_id:
            logger.error(
                f"特殊 token ID 错误: {token} 预期 {expected_id}, 实际 {actual_id}"
            )
            all_passed = False
        else:
            logger.info(f"✓ {token}: ID {actual_id}")

    # 3. 验证模型属性
    if tokenizer.eos_token != "<|im_end|>":
        logger.error(f"eos_token 错误: 预期 '<|im_end|>', 实际 '{tokenizer.eos_token}'")
        all_passed = False
    else:
        logger.info("✓ eos_token = <|im_end|>")

    if tokenizer.pad_token != "<|endoftext|>":
        logger.error(
            f"pad_token 错误: 预期 '<|endoftext|>', 实际 '{tokenizer.pad_token}'"
        )
        all_passed = False
    else:
        logger.info("✓ pad_token = <|endoftext|>")

    # 4. 验证与模板的一致性
    if tokenizer.eos_token != template.eos_token:
        logger.error("eos_token 与模板不一致")
        all_passed = False
    else:
        logger.info("✓ eos_token 与模板一致")

    if tokenizer.pad_token != template.pad_token:
        logger.error("pad_token 与模板不一致")
        all_passed = False
    else:
        logger.info("✓ pad_token 与模板一致")

    # 5. 验证 extra_special_tokens
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    actual_extra = set(config.get("extra_special_tokens", []))
    expected_extra = set(EXTRA_SPECIAL_TOKENS)
    if actual_extra != expected_extra:
        logger.error(
            f"extra_special_tokens 错误: 预期 {expected_extra}, 实际 {actual_extra}"
        )
        all_passed = False
    else:
        logger.info(f"✓ extra_special_tokens: {list(actual_extra)}")

    # 6. 检查视觉 token 未包含
    for vt in VISION_TOKENS:
        if vt in actual_extra:
            logger.error(f"不应包含视觉 token: {vt}")
            all_passed = False
    logger.info("✓ 无视觉 token")

    # 7. 编解码测试
    test_text = "<|im_start|>assistant\n<think>推理过程...</think>答案<|im_end|>"
    try:
        encoded = tokenizer(test_text, return_tensors="pt")
        decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
        if test_text not in decoded and decoded not in test_text:
            logger.warning(f"编解码不完全一致:\n  原始: {test_text}\n  解码: {decoded}")
        else:
            logger.info("✓ 编解码测试通过")
    except Exception as e:
        logger.error(f"编解码测试失败: {e}")
        all_passed = False

    # 8. 验证 added_tokens
    tokenizer_json_path = output_dir / "tokenizer.json"
    with open(tokenizer_json_path, encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    actual_added = [t["content"] for t in tokenizer_json.get("added_tokens", [])]
    if set(actual_added) != set(SPECIAL_TOKENS):
        logger.error(f"added_tokens 错误: 预期 {SPECIAL_TOKENS}, 实际 {actual_added}")
        all_passed = False
    else:
        logger.info(f"✓ added_tokens: {len(actual_added)} 个")

    if all_passed:
        logger.info("=" * 60)
        logger.info("✓ 所有验证通过!")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("✗ 验证失败")
        logger.error("=" * 60)

    return all_passed


def train_tokenizer(
    data_dir: Path,
    template_dir: Path,
    output_dir: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    validate: bool = True,
) -> int:
    """主训练函数.

    Args:
        data_dir: 训练数据目录
        template_dir: 模板 tokenizer 目录
        output_dir: 输出目录
        vocab_size: 总词表大小 (含特殊 token)
        min_frequency: BPE 最小词频
        validate: 是否运行验证

    Returns:
        退出码 (0 表示成功)
    """
    logger.info("=" * 60)
    logger.info("Tokenizer 训练")
    logger.info("=" * 60)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"模板目录: {template_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(
        f"词表大小: {vocab_size} (BPE: {BPE_VOCAB_SIZE}, 特殊: {len(SPECIAL_TOKENS)})"
    )
    logger.info(f"最小词频: {min_frequency}")

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return 1

    try:
        _ = load_template_tokenizer(template_dir)
        template_config = extract_template_config(template_dir)

        trained_tokenizer = train_bpe_vocab(
            data_dir=data_dir,
            vocab_size=BPE_VOCAB_SIZE,
            min_frequency=min_frequency,
        )

        new_tokenizer = build_tokenizer_from_template(
            trained_tokenizer=trained_tokenizer,
            template_config=template_config,
        )

        tokenizer_config = configure_special_tokens(
            tokenizer=new_tokenizer,
        )

        save_tokenizer(
            tokenizer=new_tokenizer,
            tokenizer_config=tokenizer_config,
            output_dir=output_dir,
            template_dir=template_dir,
        )

        if validate:
            success = verify_tokenizer(
                output_dir=output_dir,
                template_dir=template_dir,
                expected_vocab_size=vocab_size,
            )
            if not success:
                return 1

        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info(f"输出目录: {output_dir}")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.exception(f"训练失败: {e}")
        return 1


def main() -> int:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        description="训练 32K BPE Tokenizer - 基于 Qwen3-Next 模板",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/train_tokenizer.py

  # 指定数据目录
  python scripts/train_tokenizer.py --data-dir data/datasets/nanomind_tokenizer

  # 跳过验证
  python scripts/train_tokenizer.py --no-validate

  # 使用自定义词表大小
  python scripts/train_tokenizer.py --vocab-size 32005
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"训练数据目录 (默认: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        help=f"模板 tokenizer 目录 (默认: {DEFAULT_TEMPLATE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"总词表大小 (默认: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
        help=f"BPE 最小词频 (默认: {DEFAULT_MIN_FREQUENCY})",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否运行验证 (默认: True)",
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
