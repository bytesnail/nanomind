#!/usr/bin/env python3
"""使用 SentencePiece 流式训练 Tokenizer（修正版）

核心修正：
1. 正确处理换行符 - 不替换为空格，保留段落结构
2. 使用空行（\n\n）作为文档分隔符
3. 支持 40M 全量样本的流式处理
4. 内存友好 - 分批处理，不累积所有数据

使用说明：
    python scripts/train_tokenizer_sp.py \
        --data-dir data/datasets/nanomind_tokenizer \
        --output-dir output/tokenizer_32k_sp \
        --vocab-size 32000
"""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


__all__ = [
    "parquet_text_iterator",
    "export_to_text_streaming",
    "train_sentencepiece",
    "main",
]


def parquet_text_iterator(
    data_dir: Path,
    columns: list[str] | None = None,
    batch_size: int = 10000,
) -> Iterator[dict[str, object]]:
    """
    流式读取 parquet 文件的文本内容

    Args:
        data_dir: 数据目录
        columns: 要读取的列，默认 ["text"]
        batch_size: 每批读取的行数

    Yields:
        dict: 包含 text 等字段的字典
    """
    if columns is None:
        columns = ["text"]

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")

    total_docs = 0

    for pf_path in tqdm(parquet_files, desc="读取 parquet 文件"):
        try:
            pf = pq.ParquetFile(pf_path)

            for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                # 转换为 Python 对象
                column_data = {}
                for col in columns:
                    if col in batch.column_names:
                        column_data[col] = batch.column(col).to_pylist()
                    else:
                        column_data[col] = [None] * batch.num_rows

                # 逐行 yield
                for i in range(batch.num_rows):
                    yield {col: column_data[col][i] for col in columns}
                    total_docs += 1

                # 定期 GC
                if total_docs % 100000 == 0:
                    gc.collect()

        except Exception as e:
            logger.warning(f"处理 {pf_path} 时出错: {e}")
            continue

    logger.info(f"总计读取 {total_docs:,} 个文档")


def export_to_text_streaming(
    data_dir: Path,
    output_file: Path,
    doc_separator: str = "\n\n",
) -> int:
    """
    将 parquet 数据流式导出为文本文件

    关键设计：
    1. 保留原始文本中的所有换行符（不替换为空格）
    2. 使用空行（\n\n）作为文档分隔符
    3. 流式处理，内存占用低

    Args:
        data_dir: 输入数据目录
        output_file: 输出文本文件路径
        doc_separator: 文档分隔符，默认 "\n\n"

    Returns:
        int: 处理的文档总数
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始导出数据到: {output_file}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"文档分隔符: {repr(doc_separator)}")

    total_docs = 0
    empty_docs = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in parquet_text_iterator(data_dir, columns=["text"]):
            text = doc.get("text", "")

            # 只去除首尾空白，保留内部所有字符（包括换行符）
            text = str(text).strip()

            if not text:
                empty_docs += 1
                continue

            # 写入文档内容（保留所有换行符）
            f.write(text)
            # 使用文档分隔符（默认空行）
            f.write(doc_separator)

            total_docs += 1

            # 定期日志
            if total_docs % 1000000 == 0:
                logger.info(f"已处理 {total_docs:,} 个文档")
                gc.collect()

    logger.info("导出完成:")
    logger.info(f"  总文档数: {total_docs:,}")
    logger.info(f"  空文档数: {empty_docs:,}")
    logger.info(f"  输出文件: {output_file}")
    logger.info(f"  文件大小: {output_file.stat().st_size / (1024**3):.2f} GB")

    return total_docs


def train_sentencepiece(
    input_file: Path,
    output_dir: Path,
    vocab_size: int = 32000,
    model_prefix: str = "tokenizer",
    num_threads: int = 32,
) -> Path:
    """
    使用 spm_train 训练 SentencePiece 模型

    配置说明：
    - 使用 BPE 算法
    - 保留换行符作为句子边界
    - 内存友好的参数设置

    Args:
        input_file: 输入文本文件
        output_dir: 输出目录
        vocab_size: 词表大小
        model_prefix: 模型文件前缀
        num_threads: 训练线程数

    Returns:
        Path: 训练好的模型文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_prefix}.model"

    # 特殊 token 列表 - 确保它们在训练时就被当作不可分割的单元
    special_tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|think|>",
        "<|/think|>",
    ]
    num_special = len(special_tokens)

    # 基础 BPE 词表大小 = vocab_size（用户指定）
    # 最终总词表大小 = vocab_size + num_special（基础 + 特殊token）
    bpe_vocab_size = vocab_size
    total_vocab_size = vocab_size + num_special

    logger.info("开始训练 SentencePiece 模型...")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"基础 BPE 词表: {vocab_size}, 特殊 token: {num_special}, 总计: {total_vocab_size}")
    logger.info(f"特殊 tokens: {special_tokens}")
    logger.info(f"输出目录: {output_dir}")
    # 构建 spm_train 命令
    cmd = [
        "spm_train",
        f"--input={input_file}",
        f"--model_prefix={output_dir / model_prefix}",
        f"--vocab_size={bpe_vocab_size}",
        "--model_type=bpe",
        "--character_coverage=1.0",  # 全覆盖所有字符（与 qwen3 tokenizer 一致）
        f"--num_threads={num_threads}",
        # 句子长度限制
        "--max_sentence_length=0",  # 0 = 不限制
        "--max_sentencepiece_length=64",
        # 分割设置
        "--split_by_unicode_script=false",
        "--split_by_number=false",
        "--split_by_whitespace=true",
        # 大语料处理
        "--train_extremely_large_corpus=true",
        # 采样设置（内部使用 reservoir sampling）
        "--input_sentence_size=0",  # 0 = 不限制，使用全部语料
        "--shuffle_input_sentence=true",
        # 控制词表大小
        "--seed_sentencepiece_size=500000",
        "--mining_sentence_size=5000000",
        "--shrinking_factor=0.75",
        # 添加用户定义的特殊token（确保不可分割）
        f"--user_defined_symbols={','.join(special_tokens)}",
    ]

    logger.info(f"训练命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout:
            logger.info(result.stdout)

        logger.info("训练完成！")
        logger.info(f"模型文件: {model_path}")
        logger.info(f"词表文件: {output_dir / f'{model_prefix}.vocab'}")

        # 显示词表信息
        vocab_file = output_dir / f"{model_prefix}.vocab"
        if vocab_file.exists():
            with open(vocab_file) as f:
                line_count = sum(1 for _ in f)
            logger.info(f"词表条目数: {line_count:,}")

        return model_path

    except subprocess.CalledProcessError as e:
        logger.error("训练失败！")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("未找到 spm_train 命令。请先安装 sentencepiece:")
        logger.error("  pip install sentencepiece")
        logger.error("或从源码编译安装：https://github.com/google/sentencepiece")
        raise


def main(
    data_dir: Path,
    output_dir: Path,
    work_dir: Path | None = None,
    vocab_size: int = 32000,
    num_threads: int = 32,
    skip_export: bool = False,
) -> Path:
    """
    主函数

    流程：
    1. 将 parquet 数据导出为文本文件（保留换行符）
    2. 使用 SentencePiece 训练 BPE 模型
    3. 输出模型文件
    """
    logger.info("=" * 60)
    logger.info("SentencePiece Tokenizer 训练")
    logger.info("=" * 60)

    # 设置工作目录
    if work_dir is None:
        work_dir = output_dir / "work"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 步骤1：导出数据
    text_file = work_dir / "train.txt"

    if skip_export and text_file.exists():
        logger.info(f"跳过导出，使用已有文件: {text_file}")
    else:
        logger.info("\n[步骤 1/2] 导出数据...")
        export_to_text_streaming(data_dir, text_file)

    # 步骤2：训练
    logger.info("\n[步骤 2/2] 训练 SentencePiece 模型...")
    model_path = train_sentencepiece(
        input_file=text_file,
        output_dir=output_dir,
        vocab_size=vocab_size,
        num_threads=num_threads,
    )

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info(f"模型文件: {model_path}")
    logger.info("下一步: 运行 convert_sp_to_hf.py 转换为 HF 格式")
    logger.info("=" * 60)

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 SentencePiece 训练 BPE Tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基础用法
    python scripts/train_tokenizer_sp.py \
        --data-dir data/datasets/nanomind_tokenizer \
        --output-dir output/tokenizer_32k_sp
    
    # 指定词表大小和线程数
    python scripts/train_tokenizer_sp.py \
        --data-dir data/datasets/nanomind_tokenizer \
        --output-dir output/tokenizer_64k_sp \
        --vocab-size 64000 \
        --num-threads 32
    
    # 跳过导出（使用已导出的文本文件）
    python scripts/train_tokenizer_sp.py \
        --data-dir data/datasets/nanomind_tokenizer \
        --output-dir output/tokenizer_32k_sp \
        --skip-export
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/datasets/nanomind_tokenizer"),
        help="输入数据目录（包含 parquet 文件）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tokenizer_32k_sp"),
        help="输出目录",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="工作目录（用于存储临时文件）",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="基础 BPE 词表大小（默认: 32000，不包括特殊 token）",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="训练线程数（默认: 16）",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="跳过数据导出（如果已有导出的文本文件）",
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        vocab_size=args.vocab_size,
        num_threads=args.num_threads,
        skip_export=args.skip_export,
    )
