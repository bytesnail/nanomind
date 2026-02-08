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
    get_all_bucket_configs,
)
from src.data_processing.config_loader import get_config
from src.data_processing.fineweb_reorganizer import process_all_buckets
from scripts.validate_output import _print_report, _validate_all

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
            table = pq.read_table(f)
            table = table.slice(0, min(max_rows, table.num_rows))

            if "score" in table.column_names:
                for score in table.column("score").to_pylist():
                    b = find_bucket_for_score(score)
                    name = b.name if b else "<2.8"
                    stats["score_distribution"][name] = (
                        stats["score_distribution"].get(name, 0) + 1
                    )

            rel = f.relative_to(source)
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest, compression="zstd")

            stats["total_files"] += 1
            stats["total_rows"] += table.num_rows
            logger.info(f"  [{i + 1}/{len(files)}] {rel}: {table.num_rows} 行")
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
    logger.info(f"开始试运行: 输入={input_dir}, 输出={output_dir}, workers={workers}")

    trial_cfg = _processing_cfg.get("trial", {})
    results = process_all_buckets(
        input_dir,
        output_dir,
        workers,
        workers,  # tasks = workers
        seed,
        _processing_cfg.get("compression", "zstd"),
        trial_cfg.get("max_file_size_bytes", 128 * 1024 * 1024),
    )
    logger.info(f"处理完成: {results}")
    return {"processed_buckets": results}


def _analyze_sampling(input_dir: Path, output_dir: Path) -> dict:
    logger.info("分析采样准确性")

    bucket_configs = get_all_bucket_configs()
    bucket_names = [b.name for b in bucket_configs]
    input_cnt: dict[str, int] = {}
    out_cnt: dict[str, int] = {}

    for f in tqdm(list(input_dir.rglob("*.parquet"))[:10], desc="统计输入"):
        try:
            scores = pq.read_table(f, columns=["score"]).column("score").to_pylist()
            for s in scores:
                b = find_bucket_for_score(s)
                if b:
                    input_cnt[b.name] = input_cnt.get(b.name, 0) + 1
        except Exception as e:
            logger.warning(f"读取文件 {f} 失败: {e}")

    for b in bucket_names:
        d = output_dir / b
        if d.exists():
            cnt = sum(
                pq.read_table(f).num_rows for f in d.rglob("*.parquet") if f.is_file()
            )
            out_cnt[b] = cnt

    print("\n采样准确性分析:\n" + "-" * 60)
    for b in bucket_configs:
        ic, oc = input_cnt.get(b.name, 0), out_cnt.get(b.name, 0)
        if ic > 0:
            actual = oc / ic
            err = abs(actual - b.sampling_rate) / b.sampling_rate * 100
            status = "✅ 通过" if err < 5.0 else "⚠️ 偏差较大"
            print(f"\n桶 {b.name}:\n  输入记录数: {ic:,}\n  输出记录数: {oc:,}")
            print(f"  期望采样率: {b.sampling_rate:.0%}\n  实际采样率: {actual:.2%}")
            print(f"  误差: {err:.2f}%\n  状态: {status}")
        else:
            print(f"\n桶 {b.name}: 无输入数据")
    print("-" * 60)
    return {"input_scores": input_cnt, "output_counts": out_cnt}


def _parse_args() -> argparse.Namespace:
    trial_cfg = _processing_cfg.get("trial", {})
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 数据重组试运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python scripts/trial_run.py
  python scripts/trial_run.py --skip-create-test
  python scripts/trial_run.py --workers 4""",
    )
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
    parser.add_argument("--max-files", type=int, default=trial_cfg.get("max_files", 5))
    parser.add_argument("--max-rows", type=int, default=trial_cfg.get("max_rows", 2000))
    parser.add_argument("--skip-create-test", action="store_true")
    parser.add_argument("--skip-processing", action="store_true")
    parser.add_argument("--analyze-sampling", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
                shutil.rmtree(args.test_input)
            results["test_data_stats"] = _create_test_dataset(
                args.source, args.test_input, args.max_files, args.max_rows
            )

        if not args.skip_processing:
            if args.test_output.exists():
                shutil.rmtree(args.test_output)
            results["processing_results"] = _run_processing(
                args.test_input, args.test_output, args.workers, args.seed
            )

        if args.test_output.exists():
            results["validation_results"] = _validate_all(args.test_output)
            _print_report(results["validation_results"], title="试运行验证报告")
            if args.analyze_sampling:
                results["sampling_analysis"] = _analyze_sampling(
                    args.test_input, args.test_output
                )

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
