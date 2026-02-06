"""FineWeb-Edu 数据重组试运行脚本。"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_dataset(
    source_dir: Path,
    output_dir: Path,
    max_files: int = 5,
    max_rows_per_file: int = 2000,
) -> dict:
    """创建测试数据集。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(source_dir.rglob("*.parquet"))
    selected_files = parquet_files[:max_files]

    logger.info(f"选择 {len(selected_files)} 个文件用于测试")

    stats = {"total_files": 0, "total_rows": 0, "score_distribution": {}}

    for i, file_path in enumerate(selected_files):
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas().head(max_rows_per_file)

            if "score" in df.columns:
                for score in df["score"]:
                    from src.data_processing.bucket_config import find_bucket_for_score

                    bucket = find_bucket_for_score(score)
                    bucket_name = bucket.name if bucket else "<2.8"
                    stats["score_distribution"][bucket_name] = (
                        stats["score_distribution"].get(bucket_name, 0) + 1
                    )

            relative_path = file_path.relative_to(source_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_parquet(output_path, compression="zstd")

            stats["total_files"] += 1
            stats["total_rows"] += len(df)

            logger.info(
                f"  [{i + 1}/{len(selected_files)}] {relative_path}: {len(df)} 行"
            )

        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {e}")

    logger.info(f"测试数据集已创建: {output_dir}")
    logger.info(f"总计: {stats['total_files']} 个文件, {stats['total_rows']} 行")
    logger.info(f"评分分布: {stats['score_distribution']}")

    return stats


def run_trial_processing(
    input_dir: Path,
    output_dir: Path,
    workers: int = 2,
    random_seed: int = 42,
) -> dict:
    """运行试运行处理。"""
    from src.data_processing.fineweb_reorganizer import process_all_buckets

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始试运行处理")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Workers: {workers}")
    logger.info(f"随机种子: {random_seed}")
    logger.info("=" * 60)

    results = process_all_buckets(
        input_path=input_dir,
        output_base=output_dir,
        workers_per_bucket=workers,
        random_seed=random_seed,
        parallel_buckets=1,
        compression="zstd",
        max_file_size=128 * 1024 * 1024,
    )

    logger.info(f"处理完成: {results}")
    return {"processed_buckets": results}


def validate_trial_results(output_dir: Path) -> dict:
    """验证试运行结果。"""
    from scripts.validate_output import validate_all_buckets

    logger.info("=" * 60)
    logger.info("开始验证结果")
    logger.info("=" * 60)

    results = validate_all_buckets(output_dir)

    print("\n" + "=" * 60)
    print("试运行验证报告")
    print("=" * 60)

    for bucket_name, bucket_result in results["buckets"].items():
        print(f"\n桶 {bucket_name}:")
        print(f"  状态: {'✅ 通过' if bucket_result['valid'] else '❌ 失败'}")
        print(f"  文件数: {bucket_result['file_count']}")

        stats = bucket_result.get("stats", {})
        print(f"  记录数: {stats.get('record_count', 0):,}")
        print(f"  总大小: {stats.get('total_size_gb', 0):.2f} GB")
        print(f"  CC-MAIN 批次数: {len(stats.get('cc_main_batches', []))}")

    print("\n" + "-" * 60)
    print("总计:")
    print(f"  文件数: {results['summary']['total_files']}")
    print(f"  记录数: {results['summary']['total_records']:,}")
    print(f"  总大小: {results['summary']['total_size_gb']:.2f} GB")
    print(f"  错误数: {results['summary']['total_errors']}")
    print(f"\n{'✅ 验证通过' if results['valid'] else '❌ 验证失败'}")
    print("=" * 60)

    return results


def analyze_sampling_accuracy(input_dir: Path, output_dir: Path) -> dict:
    """分析采样准确性。"""
    logger.info("=" * 60)
    logger.info("分析采样准确性")
    logger.info("=" * 60)

    input_scores = {"2.8": 0, "3.0": 0, "3.5": 0, "4.0": 0}
    output_counts = {"2.8": 0, "3.0": 0, "3.5": 0, "4.0": 0}

    for parquet_file in tqdm(list(input_dir.rglob("*.parquet"))[:10], desc="统计输入"):
        try:
            table = pq.read_table(parquet_file, columns=["score"])
            for score in table.column("score").to_pylist():
                if 2.8 <= score < 3.0:
                    input_scores["2.8"] += 1
                elif 3.0 <= score < 3.5:
                    input_scores["3.0"] += 1
                elif 3.5 <= score < 4.0:
                    input_scores["3.5"] += 1
                elif score >= 4.0:
                    input_scores["4.0"] += 1
        except Exception as e:
            logger.warning(f"读取文件 {parquet_file} 失败: {e}")

    for bucket_name in ["2.8", "3.0", "3.5", "4.0"]:
        bucket_dir = output_dir / bucket_name
        if bucket_dir.exists():
            for parquet_file in bucket_dir.rglob("*.parquet"):
                try:
                    table = pq.read_table(parquet_file)
                    output_counts[bucket_name] += table.num_rows
                except Exception as e:
                    logger.warning(f"读取文件 {parquet_file} 失败: {e}")

    sampling_rates = {"2.8": 0.30, "3.0": 0.60, "3.5": 0.80, "4.0": 1.0}

    print("\n采样准确性分析:")
    print("-" * 60)

    for bucket_name in ["2.8", "3.0", "3.5", "4.0"]:
        input_count = input_scores[bucket_name]
        output_count = output_counts[bucket_name]
        expected_rate = sampling_rates[bucket_name]

        if input_count > 0:
            actual_rate = output_count / input_count
            error_rate = abs(actual_rate - expected_rate) / expected_rate * 100

            print(f"\n桶 {bucket_name}:")
            print(f"  输入记录数: {input_count:,}")
            print(f"  输出记录数: {output_count:,}")
            print(f"  期望采样率: {expected_rate:.0%}")
            print(f"  实际采样率: {actual_rate:.2%}")
            print(f"  误差: {error_rate:.2f}%")
            print(f"  状态: {'✅ 通过' if error_rate < 5.0 else '⚠️ 偏差较大'}")
        else:
            print(f"\n桶 {bucket_name}: 无输入数据")

    print("-" * 60)

    return {"input_scores": input_scores, "output_counts": output_counts}


def main() -> int:
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 数据重组试运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整试运行
  python scripts/trial_run.py

  # 使用已有测试数据
  python scripts/trial_run.py --skip-create-test

  # 指定 workers
  python scripts/trial_run.py --workers 4
        """,
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/datasets/HuggingFaceFW/fineweb-edu/data"),
    )
    parser.add_argument(
        "--test-input", type=Path, default=Path("data/datasets/test_fineweb_input")
    )
    parser.add_argument(
        "--test-output", type=Path, default=Path("data/datasets/test_fineweb_output")
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--skip-create-test", action="store_true")
    parser.add_argument("--skip-processing", action="store_true")
    parser.add_argument("--analyze-sampling", action="store_true")
    parser.add_argument("--json", type=Path)

    args = parser.parse_args()

    results: dict = {
        "test_data_stats": {},
        "processing_results": {},
        "validation_results": {},
        "sampling_analysis": {},
    }

    try:
        if not args.skip_create_test:
            if not args.source.exists():
                logger.error(f"源数据目录不存在: {args.source}")
                return 1

            if args.test_input.exists():
                logger.info(f"清理旧的测试输入数据: {args.test_input}")
                shutil.rmtree(args.test_input)

            results["test_data_stats"] = create_test_dataset(
                source_dir=args.source,
                output_dir=args.test_input,
                max_files=args.max_files,
                max_rows_per_file=args.max_rows,
            )
        else:
            logger.info("跳过创建测试数据")

        if not args.skip_processing:
            if args.test_output.exists():
                logger.info(f"清理旧的测试输出数据: {args.test_output}")
                shutil.rmtree(args.test_output)

            results["processing_results"] = run_trial_processing(
                input_dir=args.test_input,
                output_dir=args.test_output,
                workers=args.workers,
                random_seed=args.seed,
            )
        else:
            logger.info("跳过处理步骤")

        if args.test_output.exists():
            results["validation_results"] = validate_trial_results(args.test_output)

            if args.analyze_sampling:
                results["sampling_analysis"] = analyze_sampling_accuracy(
                    input_dir=args.test_input,
                    output_dir=args.test_output,
                )
        else:
            logger.warning(f"输出目录不存在，跳过验证: {args.test_output}")

        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"结果已保存到: {args.json}")

        if results["validation_results"] and not results["validation_results"]["valid"]:
            return 1

        return 0

    except Exception as e:
        logger.exception(f"试运行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
