#!/usr/bin/env python3
"""验证训练好的 Tokenizer 与 qwen3 的行为一致性

提供详细的对比测试，包括：
- 特殊 token 配置对比
- 编码/解码一致性测试
- Token 数量对比
- 压缩率测试

使用示例:
    python scripts/verify_tokenizer.py --ours output/tokenizer_32k \
        --qwen3 output/qwen3_next_tokenizer
    python scripts/verify_tokenizer.py --ours output/tokenizer_32k \
        --qwen3 output/qwen3_next_tokenizer -v
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

__all__ = [
    "TestCase",
    "TestResult",
    "ComparisonSummary",
    "TEST_CASES",
    "compare_special_tokens",
    "run_single_test",
    "compare_tokenizers",
    "print_results",
    "test_compression_ratio",
    "main",
]


@dataclass(frozen=True)
class TestCase:
    """测试用例定义"""

    name: str
    text: str
    description: str = ""


# 预定义测试用例集
TEST_CASES: list[TestCase] = [
    # 基础英文
    TestCase("hello", "Hello world!", "基础英文"),
    TestCase("punctuation", "What's your name?", "标点符号"),
    TestCase("numbers", "The answer is 42.", "数字"),
    TestCase("code_simple", "def hello():", "简单代码"),
    TestCase("code_complex", "def hello_world(x, y=10):\n    return x + y", "复杂代码"),
    # 中文
    TestCase("chinese_simple", "你好", "简单中文"),
    TestCase("chinese_sentence", "你好，世界！", "中文句子"),
    TestCase("chinese_mixed", "Hello 你好 world", "中英混合"),
    # 特殊字符
    TestCase("symbols", "!@#$%^&*()", "特殊符号"),
    TestCase("math", "x = 1 + 2 * 3", "数学表达式"),
    TestCase("url", "https://example.com/path", "URL"),
    # 特殊 token
    TestCase("special_im", "<|im_start|>user\n你好<|im_end|>", "对话格式"),
    TestCase("special_eot", "<|endoftext|>", "结束符"),
    # 非训练语言回退测试（验证 byte-level 回退能力）
    TestCase("japanese", "こんにちは", "非训练语言-日语（回退测试）"),
    TestCase("korean", "안녕하세요", "非训练语言-韩语（回退测试）"),
    TestCase("arabic", "مرحبا", "非训练语言-阿拉伯语（回退测试）"),
    TestCase("russian", "Привет", "非训练语言-俄语（回退测试）"),
    # 代码
    TestCase("python_class", "class MyClass:\n    pass", "Python 类"),
    TestCase("python_import", "import numpy as np", "Python 导入"),
    TestCase("markdown", "# Title\n\nSome text", "Markdown"),
    # 边界情况
    TestCase("empty", "", "空字符串"),
    TestCase("whitespace", "   ", "纯空格"),
    TestCase("newline", "line1\nline2", "换行符"),
    TestCase("multiple_newlines", "para1\n\npara2", "多段换行"),
    # 长文本
    TestCase("long_text", "This is a sentence. " * 100, "长文本"),
]


@dataclass
class TestResult:
    """单个测试用例的结果"""

    name: str
    text: str
    status: str
    ours_token_count: int = 0
    qwen3_token_count: int = 0
    ours_decoded: str = ""
    qwen3_decoded: str = ""
    error: str = ""


@dataclass
class ComparisonSummary:
    """对比测试汇总结果"""

    total: int = 0
    exact_match: int = 0
    token_count_match: int = 0
    decode_match: int = 0
    failed: int = 0
    results: list[TestResult] = field(default_factory=list)


def _format_text_for_display(text: str, max_length: int = 100) -> str:
    """格式化文本用于显示，处理空字符串和超长文本"""
    if not text:
        return "(empty)"
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def compare_special_tokens(
    tok_ours: PreTrainedTokenizer,
    tok_qwen3: PreTrainedTokenizer,
) -> None:
    """对比特殊 token 配置"""
    logger.info("\n特殊 token 对比:")
    special_tokens = ["eos_token", "pad_token", "bos_token", "unk_token"]

    for st in special_tokens:
        ours_val = getattr(tok_ours, st)
        qwen3_val = getattr(tok_qwen3, st)
        match = "✓" if ours_val == qwen3_val else "✗"
        logger.info(f"  {match} {st}: ours='{ours_val}' vs qwen3='{qwen3_val}'")


def run_single_test(
    tok_ours: PreTrainedTokenizer,
    tok_qwen3: PreTrainedTokenizer,
    test_case: TestCase,
) -> TestResult:
    """运行单个测试用例"""
    result = TestResult(
        name=test_case.name,
        text=_format_text_for_display(test_case.text),
        status="unknown",
    )

    try:
        # 编码
        ours_encoded = tok_ours.encode(test_case.text)
        qwen3_encoded = tok_qwen3.encode(test_case.text)

        result.ours_token_count = len(ours_encoded)
        result.qwen3_token_count = len(qwen3_encoded)

        # 解码
        result.ours_decoded = tok_ours.decode(ours_encoded)
        result.qwen3_decoded = tok_qwen3.decode(qwen3_encoded)

        # 判断结果
        exact_match = ours_encoded == qwen3_encoded
        token_count_match = result.ours_token_count == result.qwen3_token_count
        decode_match = result.ours_decoded.strip() == test_case.text.strip()

        if exact_match:
            result.status = "exact"
        elif token_count_match and decode_match:
            result.status = "functional"
        elif decode_match:
            result.status = "decode_ok"
        else:
            result.status = "different"

    except Exception as e:
        result.status = "error"
        result.error = str(e)

    return result


def compare_tokenizers(
    ours_dir: Path,
    qwen3_dir: Path,
    test_cases: Sequence[TestCase] | None = None,
) -> ComparisonSummary:
    """
    对比两个 tokenizer 的行为

    Args:
        ours_dir: 我们的 tokenizer 目录
        qwen3_dir: qwen3 tokenizer 目录
        test_cases: 自定义测试用例列表（可选）

    Returns:
        ComparisonSummary: 测试结果汇总
    """
    if test_cases is None:
        test_cases = TEST_CASES

    logger.info("=" * 60)
    logger.info("Tokenizer 对比验证")
    logger.info("=" * 60)

    # 加载 tokenizer
    logger.info("\n加载 tokenizer...")
    tok_ours = AutoTokenizer.from_pretrained(ours_dir, trust_remote_code=True)
    logger.info(f"  ✓ 我们的 tokenizer ({ours_dir})")
    logger.info(f"    词表大小: {tok_ours.vocab_size}")

    tok_qwen3 = AutoTokenizer.from_pretrained(qwen3_dir, trust_remote_code=True)
    logger.info(f"  ✓ qwen3 tokenizer ({qwen3_dir})")
    logger.info(f"    词表大小: {tok_qwen3.vocab_size}")

    # 对比特殊 token
    compare_special_tokens(tok_ours, tok_qwen3)

    # 运行测试用例
    logger.info(f"\n运行 {len(test_cases)} 个测试用例...")
    summary = ComparisonSummary(total=len(test_cases))

    for test_case in test_cases:
        result = run_single_test(tok_ours, tok_qwen3, test_case)
        summary.results.append(result)

        # 统计
        if result.status == "exact":
            summary.exact_match += 1
            summary.token_count_match += 1
            summary.decode_match += 1
        elif result.status in ("functional", "decode_ok"):
            summary.token_count_match += 1
            summary.decode_match += 1
        elif result.status == "error":
            summary.failed += 1

    return summary


def _calculate_percentage(value: int, total: int) -> float:
    """计算百分比，避免除零"""
    if total == 0:
        return 0.0
    return value / total * 100


def print_results(summary: ComparisonSummary, verbose: bool = False) -> None:
    """打印测试结果"""
    logger.info("\n" + "=" * 60)
    logger.info("验证结果")
    logger.info("=" * 60)
    logger.info(f"总测试数: {summary.total}")
    exact_pct = _calculate_percentage(summary.exact_match, summary.total)
    logger.info(f"完全匹配: {summary.exact_match} ({exact_pct:.1f}%)")
    token_pct = _calculate_percentage(summary.token_count_match, summary.total)
    logger.info(f"token数匹配: {summary.token_count_match} ({token_pct:.1f}%)")
    decode_pct = _calculate_percentage(summary.decode_match, summary.total)
    logger.info(f"解码正确: {summary.decode_match} ({decode_pct:.1f}%)")
    logger.info(f"失败: {summary.failed}")

    # 显示失败的用例
    failed_cases = [
        r for r in summary.results if r.status not in ("exact", "functional")
    ]
    if failed_cases:
        logger.info("\n需要关注的测试用例:")
        for result in failed_cases[:10]:
            logger.info(f"  [{result.status}] {result.name}: '{result.text}'")
            if result.ours_token_count > 0 or result.qwen3_token_count > 0:
                logger.info(
                    f"    token数: ours={result.ours_token_count}, qwen3={result.qwen3_token_count}"
                )
            if result.error:
                logger.info(f"    错误: {result.error}")

    # 详细输出
    if verbose:
        logger.info("\n详细测试结果:")
        for result in summary.results:
            logger.info(f"\n[{result.status}] {result.name}")
            logger.info(f"  文本: {result.text}")
            if result.ours_token_count > 0 or result.qwen3_token_count > 0:
                logger.info(
                    f"  token数: ours={result.ours_token_count}, qwen3={result.qwen3_token_count}"
                )
            if result.error:
                logger.info(f"  错误: {result.error}")


def test_compression_ratio(
    ours_dir: Path,
    test_file: Path | None = None,
) -> dict[str, float | int]:
    """测试压缩率

    Returns:
        dict: 包含压缩率信息的字典
    """
    logger.info("\n" + "=" * 60)
    logger.info("压缩率测试")
    logger.info("=" * 60)

    tok = AutoTokenizer.from_pretrained(ours_dir, trust_remote_code=True)

    if test_file and test_file.exists():
        text = test_file.read_text(encoding="utf-8")
    else:
        text = "This is a test sentence. " * 1000

    encoded = tok.encode(text)

    original_bytes = len(text.encode("utf-8"))
    token_count = len(encoded)
    compressed_bytes = token_count * 2  # uint16

    compression_ratio = original_bytes / compressed_bytes
    tokens_per_char = token_count / len(text) if text else 0.0

    logger.info(f"文本长度: {len(text):,} 字符")
    logger.info(f"原始字节: {original_bytes:,} bytes")
    logger.info(f"Token数: {token_count:,}")
    logger.info(f"压缩率: {compression_ratio:.2f}x")
    logger.info(f"平均每字符token数: {tokens_per_char:.2f}")

    return {
        "original_bytes": original_bytes,
        "token_count": token_count,
        "compression_ratio": compression_ratio,
        "tokens_per_char": tokens_per_char,
    }


def main(
    ours_dir: Path,
    qwen3_dir: Path,
    test_file: Path | None = None,
    verbose: bool = False,
) -> int:
    """
    主函数

    Returns:
        int: 退出码 (0=成功, 1=失败)
    """
    try:
        # 对比测试
        summary = compare_tokenizers(ours_dir, qwen3_dir)
        print_results(summary, verbose)

        # 压缩率测试
        test_compression_ratio(ours_dir, test_file)

        # 判断整体结果
        if summary.exact_match == summary.total:
            logger.info("\n✓ 所有测试完全匹配！")
            return 0
        if summary.decode_match >= summary.total * 0.9:
            logger.info("\n✓ 大部分测试通过（解码正确）")
            return 0
        logger.warning("\n✗ 测试未完全通过")
        return 1

    except Exception as e:
        logger.error(f"验证失败: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="验证 Tokenizer 与 qwen3 的一致性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基础用法
    python scripts/verify_tokenizer.py --ours output/tokenizer_32k \
        --qwen3 output/qwen3_next_tokenizer  # noqa: E501

    # 详细输出
    python scripts/verify_tokenizer.py --ours output/tokenizer_32k \
        --qwen3 output/qwen3_next_tokenizer -v  # noqa: E501

    # 指定测试文件
    python scripts/verify_tokenizer.py --ours output/tokenizer_32k \
        --qwen3 output/qwen3_next_tokenizer --test-file test.txt  # noqa: E501
        """,
    )
    parser.add_argument(
        "--ours",
        type=Path,
        default=Path("output/tokenizer_32k"),
        help="我们的 tokenizer 目录 (默认: output/tokenizer_32k)",
    )
    parser.add_argument(
        "--qwen3",
        type=Path,
        default=Path("output/qwen3_next_tokenizer"),
        help="qwen3 tokenizer 目录 (默认: output/qwen3_next_tokenizer)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="用于压缩率测试的文本文件",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="显示详细结果",
    )

    args = parser.parse_args()

    exit_code = main(
        ours_dir=args.ours,
        qwen3_dir=args.qwen3,
        test_file=args.test_file,
        verbose=args.verbose,
    )

    exit(exit_code)
