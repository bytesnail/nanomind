"""FineWeb-Edu 重组结果验证脚本。

验证输出数据的完整性、正确性和质量指标。
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from src.data_processing.bucket_config import get_bucket_config


def validate_schema(file_path: Path) -> tuple[bool, list[str]]:
    """验证 parquet 文件的 schema。

    Args:
        file_path: parquet 文件路径

    Returns:
        tuple[bool, list[str]]: (是否通过, 错误信息列表)
    """
    errors = []
    try:
        table = pq.read_table(file_path)
        columns = set(table.column_names)

        # 检查必需字段：id、text、score 都必须在顶层
        required_top_columns = {"id", "text", "score"}
        if not required_top_columns.issubset(columns):
            missing = required_top_columns - columns
            errors.append(f"缺少必需顶层字段: {missing}")

        return len(errors) == 0, errors
    except Exception as e:
        errors.append(f"读取文件失败: {e}")
        return False, errors


def validate_file_integrity(file_path: Path) -> tuple[bool, list[str]]:
    """验证 parquet 文件完整性。

    Args:
        file_path: parquet 文件路径

    Returns:
        tuple[bool, list[str]]: (是否通过, 错误信息列表)
    """
    errors = []

    if file_path.stat().st_size == 0:
        errors.append("文件大小为 0")
        return False, errors

    try:
        table = pq.read_table(file_path)
        if table.num_rows == 0:
            errors.append("文件不包含任何记录")
            return False, errors
    except Exception as e:
        errors.append(f"文件损坏或无法读取: {e}")
        return False, errors

    return True, []


def validate_score_range(file_path: Path, bucket_name: str) -> tuple[bool, list[str]]:
    """验证文件中的评分是否在对应桶的范围内。

    Args:
        file_path: parquet 文件路径
        bucket_name: 评分桶名称

    Returns:
        tuple[bool, list[str]]: (是否通过, 错误信息列表)
    """
    errors = []

    try:
        bucket = get_bucket_config(bucket_name)
    except ValueError:
        errors.append(f"未知桶名称: {bucket_name}")
        return False, errors

    try:
        table = pq.read_table(file_path)
        columns = table.column_names

        # score 现在是顶层字段，直接从顶层读取
        if "score" not in columns:
            errors.append("缺少顶层 score 字段")
            return False, errors

        scores = table.column("score").to_pylist()

        out_of_range = 0
        for score in scores:
            if not bucket.contains(score):
                out_of_range += 1

        if out_of_range > 0:
            max_score_str = "+∞" if bucket.max_score is None else str(bucket.max_score)
            errors.append(
                f"有 {out_of_range} 条记录的评分不在范围内 [{bucket.min_score}, {max_score_str})"
            )

    except Exception as e:
        errors.append(f"验证评分范围失败: {e}")

    return len(errors) == 0, errors


def collect_bucket_stats(bucket_path: Path) -> dict:
    """收集评分桶的统计信息。

    Args:
        bucket_path: 评分桶目录路径

    Returns:
        dict: 统计信息字典
    """
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

            # 提取 CC-MAIN 批次
            cc_main = parquet_file.parent.name
            if cc_main.startswith("CC-MAIN-"):
                stats["cc_main_batches"].add(cc_main)
        except Exception:
            pass

    stats["cc_main_batches"] = sorted(stats["cc_main_batches"])
    stats["total_size_gb"] = stats["total_size_bytes"] / (1024**3)

    return stats


def validate_bucket(bucket_path: Path, bucket_name: str) -> dict:
    """验证单个评分桶的所有文件。

    Args:
        bucket_path: 评分桶目录路径
        bucket_name: 评分桶名称

    Returns:
        dict: 验证结果
    """
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

    if len(parquet_files) == 0:
        result["valid"] = False
        result["errors"].append("桶中没有 parquet 文件")
        return result

    for file_path in tqdm(parquet_files, desc=f"验证桶 {bucket_name}"):
        # 验证文件完整性
        valid, errors = validate_file_integrity(file_path)
        if not valid:
            result["valid"] = False
            result["error_count"] += 1
            result["errors"].extend([f"{file_path}: {e}" for e in errors])
            continue

        # 验证 schema
        valid, errors = validate_schema(file_path)
        if not valid:
            result["valid"] = False
            result["error_count"] += 1
            result["errors"].extend([f"{file_path}: {e}" for e in errors])
            continue

        # 验证评分范围
        valid, errors = validate_score_range(file_path, bucket_name)
        if not valid:
            result["valid"] = False
            result["error_count"] += 1
            result["errors"].extend([f"{file_path}: {e}" for e in errors])

    # 收集统计信息
    result["stats"] = collect_bucket_stats(bucket_path)

    return result


def validate_all_buckets(base_path: Path) -> dict:
    """验证所有评分桶。

    Args:
        base_path: 输出基础目录（包含 en/ 子目录）

    Returns:
        dict: 完整验证结果
    """
    from src.data_processing.bucket_config import get_all_bucket_configs

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


def print_validation_report(results: dict, verbose: bool = False) -> None:
    """打印验证报告。

    Args:
        results: 验证结果字典
        verbose: 是否打印详细信息
    """
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
            for error in bucket_result["errors"][:10]:  # 只显示前 10 个错误
                print(f"    - {error}")
            if len(bucket_result["errors"]) > 10:
                print(f"    ... 还有 {len(bucket_result['errors']) - 10} 个错误")

    print("\n" + "-" * 60)
    print("总计:")
    print(f"  文件数: {results['summary']['total_files']}")
    print(f"  记录数: {results['summary']['total_records']:,}")
    print(f"  总大小: {results['summary']['total_size_gb']:.2f} GB")
    print(f"  错误数: {results['summary']['total_errors']}")

    if results["valid"]:
        print("\n✅ 验证通过")
    else:
        print("\n❌ 验证失败")

    print("=" * 60)


def main() -> int:
    """主入口函数。

    Returns:
        int: 退出码（0=成功，1=失败）
    """
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

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入目录（包含评分桶子目录）",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        choices=["2.8", "3.0", "3.5", "4.0"],
        help="只验证指定评分桶",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细错误信息",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="将结果保存为 JSON 文件",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入目录不存在：{args.input}", file=sys.stderr)
        return 1

    if args.bucket:
        bucket_path = args.input / args.bucket
        results = {
            "valid": True,
            "buckets": {args.bucket: validate_bucket(bucket_path, args.bucket)},
            "summary": {
                "total_files": 0,
                "total_records": 0,
                "total_size_gb": 0,
                "total_errors": 0,
            },
        }
        results["valid"] = results["buckets"][args.bucket]["valid"]
        results["summary"]["total_files"] = results["buckets"][args.bucket][
            "file_count"
        ]
        results["summary"]["total_errors"] = results["buckets"][args.bucket][
            "error_count"
        ]
        stats = results["buckets"][args.bucket].get("stats", {})
        results["summary"]["total_records"] = stats.get("record_count", 0)
        results["summary"]["total_size_gb"] = stats.get("total_size_gb", 0)
    else:
        results = validate_all_buckets(args.input)

    print_validation_report(results, verbose=args.verbose)

    if args.json:
        # 将 set 转换为 list 以便 JSON 序列化
        for bucket_result in results["buckets"].values():
            if "stats" in bucket_result and "cc_main_batches" in bucket_result["stats"]:
                bucket_result["stats"]["cc_main_batches"] = list(
                    bucket_result["stats"]["cc_main_batches"]
                )

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.json}")

    return 0 if results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
