#!/usr/bin/env python3
"""按 Token 长度分桶重组文档。"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Bucket documents by token length")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lengths", type=str, required=True)
    parser.add_argument("--buckets", type=str, default="512,1024,2048,4096")

    args = parser.parse_args()

    print("🚧 此脚本尚未实现")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"长度文件: {args.lengths}")
    print(f"分桶边界: {args.buckets}")


if __name__ == "__main__":
    main()
