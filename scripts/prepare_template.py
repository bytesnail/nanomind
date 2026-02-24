#!/usr/bin/env python3
"""
准备 Qwen3-Next Tokenizer 模板

从 Hugging Face 下载 Qwen3-Next-80B-A3B-Instruct 的 tokenizer，
保存到本地作为后续训练的模板（复制 pretokenizer/normalizer/decoder 配置）。

用法:
    python scripts/prepare_template.py
    python scripts/prepare_template.py --output-dir output/qwen3_next_tokenizer
    python scripts/prepare_template.py --model Qwen/Qwen3-Next-80B-A3B-Instruct

输出:
    output/qwen3_next_tokenizer/
    ├── chat_template.jinja     # 对话模板
    ├── tokenizer.json          # 词表与处理配置
    └── tokenizer_config.json   # Tokenizer 配置
"""

import argparse
import logging
import sys
from pathlib import Path

from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
DEFAULT_OUTPUT_DIR = Path("output/qwen3_next_tokenizer")


def prepare_template(model_name: str, output_dir: Path) -> None:
    """从指定模型下载并保存 tokenizer 模板。

    Args:
        model_name: Hugging Face 模型名称
        output_dir: 输出目录路径

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

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Tokenizer 模板已保存到: {output_dir}")
    except Exception as e:
        logger.error(f"保存 tokenizer 失败: {e}")
        raise

    saved_files = list(output_dir.iterdir())
    logger.info(f"输出文件 ({len(saved_files)} 个):")
    for f in sorted(saved_files):
        size = f.stat().st_size if f.is_file() else 0
        logger.info(f"  - {f.name} ({size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="准备 Qwen3-Next Tokenizer 模板 - 下载并保存基础 tokenizer 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/prepare_template.py

  # 指定输出目录
  python scripts/prepare_template.py --output-dir /path/to/output

  # 使用不同的基础模型
  python scripts/prepare_template.py --model Qwen/Qwen3-Next-0.6B
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
        help=f"输出目录路径 (默认: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    try:
        prepare_template(args.model, args.output_dir)
        logger.info("模板准备完成！")
        return 0
    except Exception as e:
        logger.error(f"准备模板失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
