import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_processing import print_report, validate_all_buckets
from src.data_processing.config_loader import DEFAULT_LOG_FORMAT, get_dataset_configs

logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 输出验证工具 - 验证分桶结果的正确性"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="输入目录路径 (验证单个目录，覆盖配置中的 output_dir)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="en",
        help="数据集键 (如 'en', 'zh')，默认: en",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细错误信息",
    )
    parser.add_argument(
        "--json-output",
        "-j",
        type=Path,
        default=None,
        help="将验证结果保存为 JSON 文件",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="验证所有配置的数据集",
    )

    args = parser.parse_args()

    if args.all:
        return validate_all(verbose=args.verbose, json_output=args.json_output)
    elif args.input:
        return validate(
            input_dir=args.input,
            dataset_key=args.dataset,
            verbose=args.verbose,
            json_output=args.json_output,
        )
    else:
        return validate_all(verbose=args.verbose, json_output=args.json_output)


if __name__ == "__main__":
    sys.exit(main())
