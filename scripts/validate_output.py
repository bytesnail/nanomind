import json
import logging
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_processing.bucket_config import (
    find_bucket_for_score,
    get_all_bucket_configs,
)
from src.data_processing.config_loader import get_dataset_configs

logger = logging.getLogger(__name__)
REQUIRED_FIELDS = {"id", "text", "score"}


def validate_file(path: Path, bucket_name: str, dataset_key: str) -> list[str]:
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
            scores = table.column("score").to_pylist()
            bad = 0
            for s in scores:
                b = find_bucket_for_score(s, dataset_key)
                if b is None or b.name != bucket_name:
                    bad += 1
            if bad:
                bucket = get_all_bucket_configs(dataset_key)
                target = next((b for b in bucket if b.name == bucket_name), None)
                if target:
                    mx = "+∞" if target.max_score is None else str(target.max_score)
                    errors.append(f"{bad}条记录不在范围[{target.min_score}, {mx})")
    except Exception as e:
        errors.append(f"读取失败: {e}")

    return errors


def collect_stats(files: list[Path]) -> dict[str, Any]:
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


def validate_bucket(path: Path, name: str, dataset_key: str) -> dict:
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
        errs = validate_file(f, name, dataset_key)
        if errs:
            result["valid"] = False
            result["error_count"] += len(errs)
            result["errors"].extend(f"{f}: {e}" for e in errs)

    result["stats"] = collect_stats(files)
    return result


def validate_all_buckets(base: Path, dataset_key: str) -> dict:
    buckets = get_all_bucket_configs(dataset_key)
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
        r = validate_bucket(base / b.name, b.name, dataset_key)
        results["buckets"][b.name] = r
        results["valid"] &= r["valid"]
        results["summary"]["total_files"] += r["file_count"]
        results["summary"]["total_errors"] += r["error_count"]
        results["summary"]["total_records"] += r["stats"].get("record_count", 0)
        results["summary"]["total_size_gb"] += r["stats"].get("total_size_gb", 0)
    return results


def print_report(
    results: dict, verbose: bool = False, title: str = "FineWeb-Edu 重组验证报告"
) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    for name, r in results["buckets"].items():
        status_icon = "✅ 通过" if r["valid"] else "❌ 失败"
        print(f"\n桶 {name}:\n  状态: {status_icon}\n  文件数: {r['file_count']}")
        s = r.get("stats", {})
        print(
            f"  记录数: {s.get('record_count', 0):,}\n"
            f"  总大小: {s.get('total_size_gb', 0):.2f} GB"
        )
        if verbose and r["errors"]:
            print("  错误详情:")
            for e in r["errors"][:10]:
                print(f"    - {e}")
            if len(r["errors"]) > 10:
                print(f"    ... 还有 {len(r['errors']) - 10} 个错误")

    print(
        f"\n{'-' * 60}\n总计:\n"
        f"  文件数: {results['summary']['total_files']}\n"
        f"  记录数: {results['summary']['total_records']:,}"
    )
    print(
        f"  总大小: {results['summary']['total_size_gb']:.2f} GB\n"
        f"  错误数: {results['summary']['total_errors']}"
    )
    print(f"\n{'✅ 验证通过' if results['valid'] else '❌ 验证失败'}\n{'=' * 60}")


def validate(
    input_dir: Path,
    dataset_key: str,
    verbose: bool = False,
    json_output: Path | None = None,
) -> int:
    if not input_dir.exists():
        print(f"错误：输入目录不存在：{input_dir}", file=sys.stderr)
        return 1

    results = validate_all_buckets(input_dir, dataset_key)
    print_report(
        results, verbose=verbose, title=f"FineWeb-Edu 验证报告 [{dataset_key}]"
    )

    if json_output:
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {json_output}")

    return 0 if results["valid"] else 1


def validate_all(verbose: bool = False, json_output: Path | None = None) -> int:
    dataset_configs = get_dataset_configs()
    overall_valid = True
    all_results = {}

    for dataset_key, dataset_config in dataset_configs.items():
        input_dir = Path(dataset_config.get("output_dir", ""))
        print(f"\n{'=' * 60}")
        print(f"验证数据集: {dataset_key}")
        print(f"{'=' * 60}")

        if not input_dir.exists():
            print(f"警告：输出目录不存在，跳过 {dataset_key}: {input_dir}")
            continue

        result = validate_all_buckets(input_dir, dataset_key)
        all_results[dataset_key] = result
        overall_valid &= result["valid"]

        print_report(result, verbose=verbose, title=f"验证报告 [{dataset_key}]")

    if json_output:
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n所有结果已保存到: {json_output}")

    return 0 if overall_valid else 1


def main() -> int:
    return validate_all()


if __name__ == "__main__":
    sys.exit(main())
