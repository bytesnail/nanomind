import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm

from src.data_processing.bucket_config import (
    get_all_bucket_configs,
    get_bucket_config,
    get_bucket_names,
)

logger = logging.getLogger(__name__)
REQUIRED_FIELDS = {"id", "text", "score"}


def _validate_file(path: Path, bucket_name: str) -> list[str]:
    errors = []
    try:
        table = pq.read_table(path)
        if path.stat().st_size == 0:
            errors.append("文件大小为0")
        if table.num_rows == 0:
            errors.append("文件不包含任何记录")

        cols = set(table.column_names)
        if not REQUIRED_FIELDS.issubset(cols):
            errors.append(f"缺少必需字段: {REQUIRED_FIELDS - cols}")

        if "score" in cols:
            bucket = get_bucket_config(bucket_name)
            scores = table.column("score").to_pylist()
            bad = sum(1 for s in scores if not bucket.contains(s))
            if bad:
                mx = "+∞" if bucket.max_score is None else str(bucket.max_score)
                errors.append(f"{bad}条记录不在范围[{bucket.min_score}, {mx})")
    except Exception as e:
        errors.append(f"读取失败: {e}")

    return errors


def _collect_stats(files: list[Path]) -> dict[str, Any]:
    total_size = sum(f.stat().st_size for f in files)
    record_count = 0
    for f in files:
        try:
            record_count += pq.ParquetFile(f).metadata.num_rows
        except (OSError, ValueError) as e:
            logger.debug(f"读取文件{f}失败: {e}")
    return {
        "file_count": len(files),
        "record_count": record_count,
        "total_size_gb": total_size / (1024**3),
    }


def _validate_bucket(path: Path, name: str) -> dict:
    result = {
        "bucket_name": name,
        "valid": True,
        "file_count": 0,
        "error_count": 0,
        "errors": [],
        "stats": {},
    }
    if not path.exists():
        return {**result, "valid": False, "errors": [f"桶目录不存在: {path}"]}

    files = list(path.rglob("*.parquet"))
    result["file_count"] = len(files)
    if not files:
        return {**result, "valid": False, "errors": ["桶中没有parquet文件"]}

    for f in tqdm(files, desc=f"验证桶 {name}"):
        errs = _validate_file(f, name)
        if errs:
            result["valid"] = False
            result["error_count"] += len(errs)
            result["errors"].extend(f"{f}: {e}" for e in errs)

    result["stats"] = _collect_stats(files)
    return result


def _validate_all(base: Path, bucket_name: str | None = None) -> dict:
    buckets = (
        [get_bucket_config(bucket_name)] if bucket_name else get_all_bucket_configs()
    )
    results = {
        "valid": True,
        "buckets": {},
        "summary": {
            "total_files": 0,
            "total_records": 0,
            "total_size_gb": 0,
            "total_errors": 0,
        },
    }

    for b in buckets:
        r = _validate_bucket(base / b.name, b.name)
        results["buckets"][b.name] = r
        results["valid"] &= r["valid"]
        for key in ("total_files", "total_errors", "total_records"):
            results["summary"][key] += (
                r["file_count"]
                if key == "total_files"
                else r["error_count"]
                if key == "total_errors"
                else r["stats"].get("record_count", 0)
            )
        results["summary"]["total_size_gb"] += r["stats"].get("total_size_gb", 0)
    return results


def _print_report(
    results: dict, verbose: bool = False, title: str = "FineWeb-Edu 重组验证报告"
) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    for name, r in results["buckets"].items():
        print(
            f"\n桶 {name}:\n  状态: {'✅ 通过' if r['valid'] else '❌ 失败'}\n  文件数: {r['file_count']}"
        )
        s = r.get("stats", {})
        print(
            f"  记录数: {s.get('record_count', 0):,}\n  总大小: {s.get('total_size_gb', 0):.2f} GB"
        )
        if verbose and r["errors"]:
            print("  错误详情:")
            for e in r["errors"][:10]:
                print(f"    - {e}")
            if len(r["errors"]) > 10:
                print(f"    ... 还有 {len(r['errors']) - 10} 个错误")

    print(
        f"\n{'-' * 60}\n总计:\n  文件数: {results['summary']['total_files']}\n  记录数: {results['summary']['total_records']:,}"
    )
    print(
        f"  总大小: {results['summary']['total_size_gb']:.2f} GB\n  错误数: {results['summary']['total_errors']}"
    )
    print(f"\n{'✅ 验证通过' if results['valid'] else '❌ 验证失败'}\n{'=' * 60}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="验证 FineWeb-Edu 重组结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python scripts/validate_output.py --input data/datasets/fineweb/en
  python scripts/validate_output.py --input data/datasets/fineweb/en --bucket 3.0
  python scripts/validate_output.py --input data/datasets/fineweb/en --verbose""",
    )
    parser.add_argument("--input", type=Path, required=True, help="输入目录")
    parser.add_argument(
        "--bucket", type=str, choices=get_bucket_names(), help="只验证指定桶"
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细错误信息")
    parser.add_argument("--json", type=Path, help="将结果保存为 JSON 文件")

    args = parser.parse_args()
    if not args.input.exists():
        print(f"错误：输入目录不存在：{args.input}", file=sys.stderr)
        return 1

    results = _validate_all(args.input, args.bucket)
    _print_report(results, verbose=args.verbose)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.json}")

    return 0 if results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
