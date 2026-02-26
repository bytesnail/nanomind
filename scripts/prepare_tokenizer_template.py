#!/usr/bin/env python3
"""
准备 Qwen3-Next Tokenizer 模板

从 Hugging Face 下载 Qwen3-Next-80B-A3B-Instruct 的 tokenizer，
保存到本地作为后续训练的模板（复制 pretokenizer/normalizer/decoder 配置）。

用法:
    python scripts/prepare_tokenizer_template.py
    python scripts/prepare_tokenizer_template.py --output-dir output/qwen3_next_tokenizer_origin
    python scripts/prepare_tokenizer_template.py --model Qwen/Qwen3-Next-80B-A3B-Instruct

输出:
    output/qwen3_next_tokenizer_origin/
    ├── chat_template.jinja     # 对话模板
    ├── tokenizer.json          # 词表与处理配置
    └── tokenizer_config.json   # Tokenizer 配置

    output/qwen3_next_tokenizer/
    ├── chat_template.jinja     # 修改后的对话模板（保持原样）
    ├── tokenizer.json          # 修改后的词表（仅保留基础 special tokens）
    └── tokenizer_config.json   # 修改后的配置（仅保留基础 special tokens）
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
DEFAULT_OUTPUT_DIR = Path("output/qwen3_next_tokenizer_origin")
MODIFIED_OUTPUT_DIR = Path("output/qwen3_next_tokenizer")

# 需要保留的基础 special tokens（按顺序）
BASE_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

# 需要添加的 think tokens
THINK_TOKENS = ["<think>", "</think>"]


def _create_modified_tokenizer_config(config: dict) -> dict:
    """修改 tokenizer_config.json，仅保留基础 special tokens 和 think tokens。"""
    config["extra_special_tokens"] = (
        BASE_SPECIAL_TOKENS[1:] + THINK_TOKENS
    )  # 去掉<|endoftext|>
    return config


def _create_modified_tokenizer_json(tokenizer_data: dict) -> dict:
    """修改 tokenizer.json，仅保留基础 special tokens 并将 think tokens 设为 special。"""
    added_tokens = tokenizer_data.get("added_tokens", [])

    # 找到基础 special tokens
    base_tokens = []
    for token in added_tokens:
        if token["content"] in BASE_SPECIAL_TOKENS:
            base_tokens.append(token)

    # 按 id 排序确保顺序正确
    base_tokens.sort(key=lambda x: x["id"])

    # 计算新的 id：基于最后一个 base token 的 id
    next_id = base_tokens[-1]["id"] + 1

    # 构建新的 added_tokens 列表
    new_added_tokens = []

    # 添加基础 tokens
    for token in base_tokens:
        new_added_tokens.append(token.copy())

    # 添加 think tokens（使用连续的 id，并设为 special）
    for think_token_content in THINK_TOKENS:
        new_added_tokens.append(
            {
                "id": next_id,
                "content": think_token_content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        next_id += 1

    tokenizer_data["added_tokens"] = new_added_tokens
    return tokenizer_data


def prepare_tokenizer_template(model_name: str, output_dir: Path, modified_dir: Path) -> None:
    """从指定模型下载并保存 tokenizer 模板。

    Args:
        model_name: Hugging Face 模型名称
        output_dir: 原始输出目录路径
        modified_dir: 修改后输出目录路径

    Raises:
        Exception: 当下载或保存失败时抛出
    """
    logger.info(f"开始下载 tokenizer: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"成功加载 tokenizer: {model_name}")
    except Exception as e:
        logger.error(f"加载 tokenizer 失败: {e}")
        raise

    # 保存原始版本
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer.save_pretrained(output_dir)
        logger.info(f"原始 tokenizer 已保存到: {output_dir}")
    except Exception as e:
        logger.error(f"保存原始 tokenizer 失败: {e}")
        raise

    saved_files = list(output_dir.iterdir())
    logger.info(f"原始输出文件 ({len(saved_files)} 个):")
    for f in sorted(saved_files):
        size = f.stat().st_size if f.is_file() else 0
        logger.info(f"  - {f.name} ({size:,} bytes)")

    # 创建修改版本
    logger.info("开始创建修改版本...")
    modified_dir.mkdir(parents=True, exist_ok=True)

    # 复制并修改 tokenizer_config.json
    config_path = output_dir / "tokenizer_config.json"
    modified_config_path = modified_dir / "tokenizer_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config = _create_modified_tokenizer_config(config)
    with open(modified_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("已修改并保存 tokenizer_config.json")

    # 复制并修改 tokenizer.json
    tokenizer_json_path = output_dir / "tokenizer.json"
    modified_tokenizer_path = modified_dir / "tokenizer.json"
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    tokenizer_data = _create_modified_tokenizer_json(tokenizer_data)
    with open(modified_tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
    logger.info("已修改并保存 tokenizer.json")

    # 复制 chat_template.jinja（保持不变）
    template_path = output_dir / "chat_template.jinja"
    modified_template_path = modified_dir / "chat_template.jinja"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
        with open(modified_template_path, "w", encoding="utf-8") as f:
            f.write(template_content)
        logger.info("已复制 chat_template.jinja")

    modified_files = list(modified_dir.iterdir())
    logger.info(f"修改版本输出文件 ({len(modified_files)} 个):")
    for f in sorted(modified_files):
        size = f.stat().st_size if f.is_file() else 0
        logger.info(f"  - {f.name} ({size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="准备 Qwen3-Next Tokenizer 模板 - 下载并保存基础 tokenizer 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/prepare_tokenizer_template.py

  # 指定输出目录
  python scripts/prepare_tokenizer_template.py --output-dir /path/to/output

  # 使用不同的基础模型
  python scripts/prepare_tokenizer_template.py --model Qwen/Qwen3-Next-0.6B
        """,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face 模型名称 (默认: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"原始输出目录路径 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--modified-dir",
        type=Path,
        default=MODIFIED_OUTPUT_DIR,
        help=f"修改后输出目录路径 (默认: {MODIFIED_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    try:
        prepare_tokenizer_template(args.model, args.output_dir, args.modified_dir)
        logger.info("模板准备完成！")
        return 0
    except Exception as e:
        logger.error(f"准备模板失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
