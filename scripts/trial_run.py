"""FineWeb-Edu 数据重组试运行。"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from src.data_processing.bucket_config import (
    find_bucket_for_score,
    get_bucket_names,
    get_sampling_rates,
)
from src.data_processing.config_loader import get_config
from src.data_processing.fineweb_reorganizer import process_all_buckets
from scripts.validate_output import _print_report, _validate_all as validate_all_buckets

_cfg = get_config()
_processing_cfg = _cfg.processing
_paths_cfg = _cfg.paths

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _create_test_dataset(
    source: Path, out: Path, max_files: int = 5, max_rows: int = 2000
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = list(source.rglob("*.parquet"))[:max_files]
    logger.info(f"选择 {len(files)} 个文件用于测试")

    stats = {"total_files": 0, "total_rows": 0, "score_distribution": {}}
    for i, f in enumerate(files):
        try:
            df = pq.read_table(f).to_pandas().head(max_rows)
            if "score" in df.columns:
                for score in df["score"]:
                    b = find_bucket_for_score(score)
                    name = b.name if b else "<2.8"
                    stats["score_distribution"][name] = (
                        stats["score_distribution"].get(name, 0) + 1
                    )
            rel = f.relative_to(source)
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dest, compression="zstd")
            stats["total_files"] += 1
            stats["total_rows"] += len(df)
            logger.info(f"  [{i + 1}/{len(files)}] {rel}: {len(df)} 行")
        except Exception as e:
            logger.error(f"处理文件 {f} 失败: {e}")

    logger.info(f"测试数据集已创建: {out}")
    logger.info(f"总计: {stats['total_files']} 个文件, {stats['total_rows']} 行")
    logger.info(f"评分分布: {stats['score_distribution']}")
    return stats


def _run_processing(
    input_dir: Path, output_dir: Path, workers: int = 2, seed: int = 42
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("开始试运行处理")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Workers: {workers}")
    logger.info(f"随机种子: {seed}")
    logger.info("=" * 60)

    _trial_cfg = _processing_cfg.get("trial", {})
    compression = _processing_cfg.get("compression", "zstd")
    max_size = _trial_cfg.get("max_file_size_bytes", 128 * 1024 * 1024)
    results = process_all_buckets(
        input_dir, output_dir, workers, seed, 1, compression, max_size
    )
    logger.info(f"处理完成: {results}")
    return {"processed_buckets": results}


def _validate_results(out: Path) -> dict:
    logger.info("开始验证结果")
    results = validate_all_buckets(out)
    _print_report(results, title="试运行验证报告")
    return results


def _analyze_sampling(input_dir: Path, output_dir: Path) -> dict:
    logger.info("=" * 60)
    logger.info("分析采样准确性")
    logger.info("=" * 60)

    bucket_names = get_bucket_names()
    sampling_rates = get_sampling_rates()
    input_cnt = {b: 0 for b in bucket_names}
    out_cnt = {b: 0 for b in bucket_names}

    for f in tqdm(list(input_dir.rglob("*.parquet"))[:10], desc="统计输入"):
        try:
            scores = pq.read_table(f, columns=["score"]).column("score").to_pylist()
            for s in scores:
                b = find_bucket_for_score(s)
                if b:
                    input_cnt[b.name] += 1
        except Exception as e:
            logger.warning(f"读取文件 {f} 失败: {e}")

    for b in bucket_names:
        d = output_dir / b
        if d.exists():
            for f in d.rglob("*.parquet"):
                try:
                    out_cnt[b] += pq.read_table(f).num_rows
                except Exception as e:
                    logger.warning(f"读取文件 {f} 失败: {e}")

    print("\n采样准确性分析:")
    print("-" * 60)
    for b in bucket_names:
        ic, oc = input_cnt[b], out_cnt[b]
        exp = sampling_rates[b]
        if ic > 0:
            actual = oc / ic
            err = abs(actual - exp) / exp * 100
            status = "✅ 通过" if err < 5.0 else "⚠️ 偏差较大"
            print(f"\n桶 {b}:\n  输入记录数: {ic:,}\n  输出记录数: {oc:,}")
            print(f"  期望采样率: {exp:.0%}\n  实际采样率: {actual:.2%}")
            print(f"  误差: {err:.2f}%\n  状态: {status}")
        else:
            print(f"\n桶 {b}: 无输入数据")
    print("-" * 60)
    return {"input_scores": input_cnt, "output_counts": out_cnt}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 数据重组试运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python scripts/trial_run.py
  python scripts/trial_run.py --skip-create-test
  python scripts/trial_run.py --workers 4""",
    )

    _trial_cfg = _processing_cfg.get("trial", {})
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            _paths_cfg.get("input_dir", "data/datasets/HuggingFaceFW/fineweb-edu")
        )
        / "data",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=Path(
            _paths_cfg.get("trial_input_dir", "data/datasets/test_fineweb_input")
        ),
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path(
            _paths_cfg.get("trial_output_dir", "data/datasets/test_fineweb_output")
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--seed", type=int, default=_processing_cfg.get("random_seed", 42)
    )
    parser.add_argument("--max-files", type=int, default=_trial_cfg.get("max_files", 5))
    parser.add_argument(
        "--max-rows", type=int, default=_trial_cfg.get("max_rows", 2000)
    )
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
            results["test_data_stats"] = _create_test_dataset(
                args.source, args.test_input, args.max_files, args.max_rows
            )
        else:
            logger.info("跳过创建测试数据")

        if not args.skip_processing:
            if args.test_output.exists():
                logger.info(f"清理旧的测试输出数据: {args.test_output}")
                shutil.rmtree(args.test_output)
            results["processing_results"] = _run_processing(
                args.test_input, args.test_output, args.workers, args.seed
            )
        else:
            logger.info("跳过处理步骤")

        if args.test_output.exists():
            results["validation_results"] = _validate_results(args.test_output)
            if args.analyze_sampling:
                results["sampling_analysis"] = _analyze_sampling(
                    args.test_input, args.test_output
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
