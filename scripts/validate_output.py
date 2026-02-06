"""FineWeb-Edu 重组结果验证脚本。"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from src.data_processing.bucket_config import get_all_bucket_configs, get_bucket_config


def validate_schema(file_path: Path) -> tuple[bool, list[str]]:
    """验证 parquet 文件的 schema。"""
    errors = []
    try:
        table = pq.read_table(file_path)
        columns = set(table.column_names)

        required = {"id", "text", "score"}
        if not required.issubset(columns):
            errors.append(f"缺少必需字段: {required - columns}")

        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"读取失败: {e}"]


def validate_file_integrity(file_path: Path) -> tuple[bool, list[str]]:
    """验证 parquet 文件完整性。"""
    if file_path.stat().st_size == 0:
        return False, ["文件大小为 0"]

    try:
        table = pq.read_table(file_path)
        if table.num_rows == 0:
            return False, ["文件不包含任何记录"]
        return True, []
    except Exception as e:
        return False, [f"文件损坏: {e}"]


def validate_score_range(file_path: Path, bucket_name: str) -> tuple[bool, list[str]]:
    """验证文件中的评分是否在对应桶的范围内。"""
    errors = []

    try:
        bucket = get_bucket_config(bucket_name)
    except ValueError:
        return False, [f"未知桶: {bucket_name}"]

    try:
        table = pq.read_table(file_path)

        if "score" not in table.column_names:
            return False, ["缺少 score 字段"]

        scores = table.column("score").to_pylist()
        out_of_range = sum(1 for score in scores if not bucket.contains(score))

        if out_of_range > 0:
            max_str = "+∞" if bucket.max_score is None else str(bucket.max_score)
            errors.append(
                f"{out_of_range} 条记录不在范围 [{bucket.min_score}, {max_str})"
            )

    except Exception as e:
        errors.append(f"验证失败: {e}")

    return len(errors) == 0, errors


def collect_bucket_stats(bucket_path: Path) -> dict:
    """收集评分桶的统计信息。"""
    stats = {
        "file_count": 0,
        "record_count": 0,
        "total_size_bytes": 0,
        "cc_main_batches": set(),
    }

    for parquet_file in bucket_path.rglob("*.parquet"):
        stats["file_count"] += 1
        stats["total_size_bytes"] += parquet_file.stat().st_size

        try:
            table = pq.read_table(parquet_file)
            stats["record_count"] += table.num_rows

            cc_main = parquet_file.parent.name
            if cc_main.startswith("CC-MAIN-"):
                stats["cc_main_batches"].add(cc_main)
        except Exception:
            pass

    stats["cc_main_batches"] = sorted(stats["cc_main_batches"])
    stats["total_size_gb"] = stats["total_size_bytes"] / (1024**3)

    return stats


def validate_bucket(bucket_path: Path, bucket_name: str) -> dict:
    """验证单个评分桶。"""
    result = {
        "bucket_name": bucket_name,
        "valid": True,
        "file_count": 0,
        "error_count": 0,
        "errors": [],
        "stats": {},
    }

    if not bucket_path.exists():
        result["valid"] = False
        result["errors"].append(f"桶目录不存在: {bucket_path}")
        return result

    parquet_files = list(bucket_path.rglob("*.parquet"))
    result["file_count"] = len(parquet_files)

    if not parquet_files:
        result["valid"] = False
        result["errors"].append("桶中没有 parquet 文件")
        return result

    for file_path in tqdm(parquet_files, desc=f"验证桶 {bucket_name}"):
        for validator, error_prefix in [
            (validate_file_integrity, "完整性"),
            (validate_schema, "Schema"),
            (lambda p: validate_score_range(p, bucket_name), "评分范围"),
        ]:
            valid, errors = validator(file_path)
            if not valid:
                result["valid"] = False
                result["error_count"] += 1
                result["errors"].extend(
                    [f"{file_path} ({error_prefix}): {e}" for e in errors]
                )
                break

    result["stats"] = collect_bucket_stats(bucket_path)
    return result


def validate_all_buckets(base_path: Path) -> dict:
    """验证所有评分桶。"""
    buckets = get_all_bucket_configs()
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

    for bucket in buckets:
        bucket_path = base_path / bucket.name
        bucket_result = validate_bucket(bucket_path, bucket.name)
        results["buckets"][bucket.name] = bucket_result

        if not bucket_result["valid"]:
            results["valid"] = False

        results["summary"]["total_files"] += bucket_result["file_count"]
        results["summary"]["total_errors"] += bucket_result["error_count"]
        results["summary"]["total_records"] += bucket_result["stats"].get(
            "record_count", 0
        )
        results["summary"]["total_size_gb"] += bucket_result["stats"].get(
            "total_size_gb", 0
        )

    return results


def print_report(results: dict, verbose: bool = False) -> None:
    """打印验证报告。"""
    print("\n" + "=" * 60)
    print("FineWeb-Edu 重组验证报告")
    print("=" * 60)

    for bucket_name, bucket_result in results["buckets"].items():
        print(f"\n桶 {bucket_name}:")
        print(f"  状态: {'✅ 通过' if bucket_result['valid'] else '❌ 失败'}")
        print(f"  文件数: {bucket_result['file_count']}")

        stats = bucket_result.get("stats", {})
        print(f"  记录数: {stats.get('record_count', 0):,}")
        print(f"  总大小: {stats.get('total_size_gb', 0):.2f} GB")
        print(f"  CC-MAIN 批次数: {len(stats.get('cc_main_batches', []))}")

        if verbose and bucket_result["errors"]:
            print("  错误详情:")
            for error in bucket_result["errors"][:10]:
                print(f"    - {error}")
            if len(bucket_result["errors"]) > 10:
                print(f"    ... 还有 {len(bucket_result['errors']) - 10} 个错误")

    print("\n" + "-" * 60)
    print("总计:")
    print(f"  文件数: {results['summary']['total_files']}")
    print(f"  记录数: {results['summary']['total_records']:,}")
    print(f"  总大小: {results['summary']['total_size_gb']:.2f} GB")
    print(f"  错误数: {results['summary']['total_errors']}")
    print(f"\n{'✅ 验证通过' if results['valid'] else '❌ 验证失败'}")
    print("=" * 60)


def main() -> int:
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description="验证 FineWeb-Edu 重组结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 验证所有桶
  python scripts/validate_output.py --input data/datasets/fineweb/en

  # 验证指定桶
  python scripts/validate_output.py --input data/datasets/fineweb/en --bucket 3.0

  # 详细输出
  python scripts/validate_output.py --input data/datasets/fineweb/en --verbose
        """,
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

    if args.bucket:
        bucket_path = args.input / args.bucket
        bucket_result = validate_bucket(bucket_path, args.bucket)
        results = {
            "valid": bucket_result["valid"],
            "buckets": {args.bucket: bucket_result},
            "summary": {
                "total_files": bucket_result["file_count"],
                "total_records": bucket_result["stats"].get("record_count", 0),
                "total_size_gb": bucket_result["stats"].get("total_size_gb", 0),
                "total_errors": bucket_result["error_count"],
            },
        }
    else:
        results = validate_all_buckets(args.input)

    print_report(results, verbose=args.verbose)

    if args.json:
        for bucket_result in results["buckets"].values():
            if "cc_main_batches" in bucket_result.get("stats", {}):
                bucket_result["stats"]["cc_main_batches"] = list(
                    bucket_result["stats"]["cc_main_batches"]
                )

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.json}")

    return 0 if results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
