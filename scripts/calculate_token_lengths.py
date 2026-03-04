#!/usr/bin/env python3
"""计算 Token 长度分布。"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Calculate token length distribution")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="output/tokenizer_36k")

    args = parser.parse_args()

    print("🚧 此脚本尚未实现")
    print(f"输入目录: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"Tokenizer: {args.tokenizer}")


if __name__ == "__main__":
    main()
