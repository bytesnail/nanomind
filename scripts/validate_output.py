"""FineWeb-Edu 重组结果验证。"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm

from src.data_processing.bucket_config import get_all_bucket_configs, get_bucket_config

logger = logging.getLogger(__name__)
REQUIRED_FIELDS = {"id", "text", "score"}


def _validate_schema(path: Path) -> tuple[bool, list[str]]:
    try:
        cols = set(pq.read_table(path).column_names)
        if not REQUIRED_FIELDS.issubset(cols):
            return False, [f"缺少必需字段: {REQUIRED_FIELDS - cols}"]
        return True, []
    except Exception as e:
        return False, [f"读取失败: {e}"]


def _validate_integrity(path: Path) -> tuple[bool, list[str]]:
    if path.stat().st_size == 0:
        return False, ["文件大小为 0"]
    try:
        t = pq.read_table(path)
        return (True, []) if t.num_rows > 0 else (False, ["文件不包含任何记录"])
    except Exception as e:
        return False, [f"文件损坏: {e}"]


def _validate_score_range(path: Path, name: str) -> tuple[bool, list[str]]:
    try:
        bucket = get_bucket_config(name)
        t = pq.read_table(path)
        if "score" not in t.column_names:
            return False, ["缺少 score 字段"]
        scores = t.column("score").to_pylist()
        bad = sum(1 for s in scores if not bucket.contains(s))
        if bad > 0:
            mx = "+∞" if bucket.max_score is None else str(bucket.max_score)
            return False, [f"{bad} 条记录不在范围 [{bucket.min_score}, {mx})"]
        return True, []
    except ValueError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"验证失败: {e}"]


def _collect_stats(path: Path) -> dict[str, Any]:
    cc_batches: set[str] = set()
    stats: dict[str, Any] = {
        "file_count": 0,
        "record_count": 0,
        "total_size_bytes": 0,
    }
    for f in path.rglob("*.parquet"):
        stats["file_count"] += 1
        stats["total_size_bytes"] += f.stat().st_size
        try:
            t = pq.read_table(f)
            stats["record_count"] += t.num_rows
            cc = f.parent.name
            if cc.startswith("CC-MAIN-"):
                cc_batches.add(cc)
        except (OSError, ValueError) as e:
            logger.debug(f"读取文件 {f} 失败: {e}")
    stats["cc_main_batches"] = sorted(cc_batches)
    stats["total_size_gb"] = stats["total_size_bytes"] / (1024**3)
    return stats


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
        result["valid"] = False
        result["errors"].append(f"桶目录不存在: {path}")
        return result

    files = list(path.rglob("*.parquet"))
    result["file_count"] = len(files)
    if not files:
        result["valid"] = False
        result["errors"].append("桶中没有 parquet 文件")
        return result

    for f in tqdm(files, desc=f"验证桶 {name}"):
        for validator, prefix in [
            (_validate_integrity, "完整性"),
            (_validate_schema, "Schema"),
            (lambda p: _validate_score_range(p, name), "评分范围"),
        ]:
            ok, errs = validator(f)
            if not ok:
                result["valid"] = False
                result["error_count"] += 1
                result["errors"].extend([f"{f} ({prefix}): {e}" for e in errs])
                break

    result["stats"] = _collect_stats(path)
    return result


def _validate_all(base: Path) -> dict:
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
    for b in get_all_bucket_configs():
        r = _validate_bucket(base / b.name, b.name)
        results["buckets"][b.name] = r
        if not r["valid"]:
            results["valid"] = False
        results["summary"]["total_files"] += r["file_count"]
        results["summary"]["total_errors"] += r["error_count"]
        results["summary"]["total_records"] += r["stats"].get("record_count", 0)
        results["summary"]["total_size_gb"] += r["stats"].get("total_size_gb", 0)
    return results


def _print_report(
    results: dict, verbose: bool = False, title: str = "FineWeb-Edu 重组验证报告"
) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for name, r in results["buckets"].items():
        print(f"\n桶 {name}:")
        print(f"  状态: {'✅ 通过' if r['valid'] else '❌ 失败'}")
        print(f"  文件数: {r['file_count']}")
        s = r.get("stats", {})
        print(f"  记录数: {s.get('record_count', 0):,}")
        print(f"  总大小: {s.get('total_size_gb', 0):.2f} GB")
        print(f"  CC-MAIN 批次数: {len(s.get('cc_main_batches', []))}")
        if verbose and r["errors"]:
            print("  错误详情:")
            for e in r["errors"][:10]:
                print(f"    - {e}")
            if len(r["errors"]) > 10:
                print(f"    ... 还有 {len(r['errors']) - 10} 个错误")

    print("\n" + "-" * 60)
    print("总计:")
    print(f"  文件数: {results['summary']['total_files']}")
    print(f"  记录数: {results['summary']['total_records']:,}")
    print(f"  总大小: {results['summary']['total_size_gb']:.2f} GB")
    print(f"  错误数: {results['summary']['total_errors']}")
    print(f"\n{'✅ 验证通过' if results['valid'] else '❌ 验证失败'}")
    print("=" * 60)


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
        "--bucket", type=str, choices=["2.8", "3.0", "3.5", "4.0"], help="只验证指定桶"
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细错误信息")
    parser.add_argument("--json", type=Path, help="将结果保存为 JSON 文件")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入目录不存在：{args.input}", file=sys.stderr)
        return 1

    results = (
        _validate_all(args.input)
        if not args.bucket
        else {
            "valid": (r := _validate_bucket(args.input / args.bucket, args.bucket))[
                "valid"
            ],
            "buckets": {args.bucket: r},
            "summary": {
                "total_files": r["file_count"],
                "total_records": r["stats"].get("record_count", 0),
                "total_size_gb": r["stats"].get("total_size_gb", 0),
                "total_errors": r["error_count"],
            },
        }
    )

    _print_report(results, verbose=args.verbose)

    if args.json:
        for r in results["buckets"].values():
            if "cc_main_batches" in r.get("stats", {}):
                r["stats"]["cc_main_batches"] = list(r["stats"]["cc_main_batches"])
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.json}")

    return 0 if results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
