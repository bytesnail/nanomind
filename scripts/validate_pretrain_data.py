#!/usr/bin/env python3
"""验证预训练数据质量。"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Validate pretrain data")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tokenizer", type=str, default="output/tokenizer_36k")

    args = parser.parse_args()

    print("🚧 此脚本尚未实现")
    print(f"输入目录: {args.input}")
    print(f"验证全部: {args.all}")
    print(f"Tokenizer: {args.tokenizer}")


if __name__ == "__main__":
    main()
