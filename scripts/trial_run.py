import logging
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from validate_output import print_report, validate_all_buckets
from src.data_processing.bucket_config import (
    find_bucket_for_score,
    get_all_bucket_configs,
)
from src.data_processing.config_loader import (
    get_dataset_config,
    get_dataset_configs,
    get_paths_config,
    get_processing_config,
)
from src.data_processing.fineweb_reorganizer import process_single_dataset

_processing_cfg = get_processing_config()
_paths_cfg = get_paths_config()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _create_test_dataset(
    source: Path,
    out: Path,
    dataset_key: str,
    max_files: int = 5,
    max_rows: int = 2000,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = list(source.rglob("*.parquet"))[:max_files]
    logger.info(f"[{dataset_key}] 选择 {len(files)} 个文件用于测试")

    # 获取数据集根标记以保留完整路径结构
    dataset_config = get_dataset_config(dataset_key)
    root_marker = dataset_config.get("root_marker", "")
    stats = {"total_files": 0, "total_rows": 0, "score_distribution": {}}

    for i, f in enumerate(files):
        try:
            table = pq.read_table(f)
            table = table.slice(0, min(max_rows, table.num_rows))

            if "score" in table.column_names:
                norm_config = dataset_config.get("score_normalization")
                for raw_score in table.column("score").to_pylist():
                    score = (
                        raw_score * norm_config.get("multiplier", 1.0)
                        if norm_config and norm_config.get("enabled")
                        else raw_score
                    )
                    b = find_bucket_for_score(score, dataset_key)
                    name = b.name if b else "below_range"
                    stats["score_distribution"][name] = (
                        stats["score_distribution"].get(name, 0) + 1
                    )

            # 保留包含数据集标记的完整路径结构，以便适配器正确识别
            if root_marker and root_marker in str(f):
                rel = Path(str(f).split(root_marker, 1)[1].lstrip("/"))
                dest = out / root_marker / rel
            else:
                rel = f.relative_to(source)
                dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest, compression="zstd")

            stats["total_files"] += 1
            stats["total_rows"] += table.num_rows
            logger.info(f"  [{i + 1}/{len(files)}] {rel}: {table.num_rows} 行")
        except Exception as e:
            logger.error(f"处理文件 {f} 失败: {e}")

    logger.info(f"[{dataset_key}] 测试数据集已创建: {out}")
    logger.info(f"总计: {stats['total_files']} 个文件, {stats['total_rows']} 行")
    logger.info(f"评分分布: {stats['score_distribution']}")
    return stats


def _run_processing(
    input_dir: Path,
    output_dir: Path,
    dataset_key: str,
    workers: int = 2,
    seed: int = 42,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理 datatrove 日志目录以避免缓存问题
    log_dir = output_dir.parent / "logs"
    if log_dir.exists():
        import shutil

        shutil.rmtree(log_dir)

    logger.info(
        f"[{dataset_key}] 开始试运行: 输入={input_dir}, "
        f"输出={output_dir}, workers={workers}"
    )

    trial_cfg = _processing_cfg.get("trial", {})
    buckets = get_all_bucket_configs(dataset_key)
    if not buckets:
        logger.error(f"[{dataset_key}] 未找到评分桶配置")
        return {"processed_buckets": []}

    results = process_single_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        buckets=buckets,
        workers=workers,
        tasks=workers,
        random_seed=seed,
        compression=_processing_cfg.get("compression", "zstd"),
        max_size=trial_cfg.get("max_file_size_bytes", 128 * 1024 * 1024),
    )
    logger.info(f"[{dataset_key}] 处理完成: {results}")
    return {"processed_buckets": results}


def _analyze_sampling(input_dir: Path, output_dir: Path, dataset_key: str) -> dict:
    logger.info(f"[{dataset_key}] 分析采样准确性")

    bucket_configs = get_all_bucket_configs(dataset_key)
    bucket_names = [b.name for b in bucket_configs]
    input_cnt: dict[str, int] = {}
    out_cnt: dict[str, int] = {}

    for f in tqdm(
        list(input_dir.rglob("*.parquet"))[:10], desc=f"[{dataset_key}] 统计输入"
    ):
        try:
            scores = pq.read_table(f, columns=["score"]).column("score").to_pylist()
            for s in scores:
                b = find_bucket_for_score(s, dataset_key)
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

    print(f"\n[{dataset_key}] 采样准确性分析:\n" + "-" * 60)
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


def run_trial(
    dataset_key: str = "en",
    source: Path | None = None,
    test_input: Path | None = None,
    test_output: Path | None = None,
    workers: int = 2,
    seed: int = 42,
    max_files: int = 5,
    max_rows: int = 2000,
    skip_create_test: bool = False,
    skip_processing: bool = False,
    analyze_sampling: bool = False,
) -> int:
    results: dict = {
        "dataset_key": dataset_key,
        "test_data_stats": {},
        "processing_results": {},
        "validation_results": {},
        "sampling_analysis": {},
    }

    dataset_config = get_dataset_config(dataset_key)
    if not dataset_config:
        logger.error(f"未找到数据集配置: {dataset_key}")
        return 1

    source = source or Path(dataset_config.get("input_dir", ""))
    test_input = test_input or Path(
        _paths_cfg.get(
            "trial_input_dir", f"data/datasets/test_fineweb_{dataset_key}_input"
        )
    )
    test_output = test_output or Path(
        _paths_cfg.get(
            "trial_output_dir", f"data/datasets/test_fineweb_{dataset_key}_output"
        )
    )

    try:
        if not skip_create_test:
            if not source.exists():
                logger.error(f"源数据目录不存在: {source}")
                return 1
            if test_input.exists():
                shutil.rmtree(test_input)
            results["test_data_stats"] = _create_test_dataset(
                source, test_input, dataset_key, max_files, max_rows
            )

        if not skip_processing:
            if test_output.exists():
                shutil.rmtree(test_output)
            results["processing_results"] = _run_processing(
                test_input, test_output, dataset_key, workers, seed
            )

        if test_output.exists():
            results["validation_results"] = validate_all_buckets(
                test_output, dataset_key
            )
            print_report(
                results["validation_results"],
                title=f"试运行验证报告 [{dataset_key}]",
            )
            if analyze_sampling:
                results["sampling_analysis"] = _analyze_sampling(
                    test_input, test_output, dataset_key
                )

        if results["validation_results"] and not results["validation_results"]["valid"]:
            return 1
        return 0
    except Exception as e:
        logger.exception(f"试运行失败: {e}")
        return 1


def run_trial_all(
    workers: int = 2,
    seed: int = 42,
    max_files: int = 5,
    max_rows: int = 2000,
    analyze_sampling: bool = False,
) -> int:
    """Run trial for all configured datasets."""
    dataset_configs = get_dataset_configs()
    overall_success = True

    for dataset_key in dataset_configs:
        print(f"\n{'=' * 60}")
        print(f"处理数据集: {dataset_key}")
        print(f"{'=' * 60}")

        result = run_trial(
            dataset_key=dataset_key,
            workers=workers,
            seed=seed,
            max_files=max_files,
            max_rows=max_rows,
            analyze_sampling=analyze_sampling,
        )
        if result != 0:
            overall_success = False

    return 0 if overall_success else 1


def main() -> int:
    trial_cfg = _processing_cfg.get("trial", {})
    return run_trial_all(
        max_files=trial_cfg.get("max_files", 5),
        max_rows=trial_cfg.get("max_rows", 2000),
    )


if __name__ == "__main__":
    sys.exit(main())
