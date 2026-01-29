"""FineWeb-Edu 统一命令行入口。

提供统一的命令行接口用于：
- explore: 完整数据集探索
- quick: 快速统计
- demo: 运行演示
- test: 运行测试

运行示例:
    python experiments/run_fineweb.py --help
    python experiments/run_fineweb.py explore --config configs/fineweb.yaml
    python experiments/run_fineweb.py quick --limit 10000
    python experiments/run_fineweb.py demo
    python experiments/run_fineweb.py test -v
"""

import argparse
import logging
import os
import sys
import subprocess
from typing import Optional, List


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    Returns:
        配置好的ArgumentParser实例
    """
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu 统一命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python experiments/run_fineweb.py --help
  python experiments/run_fineweb.py explore --config configs/fineweb.yaml
  python experiments/run_fineweb.py quick --limit 10000
  python experiments/run_fineweb.py demo
  python experiments/run_fineweb.py test -v
        """,
    )

    # 全局参数
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fineweb.yaml",
        help="配置文件路径 (默认: configs/fineweb.yaml)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出模式（DEBUG日志级别）",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="演示模式，仅打印配置不实际执行",
    )

    # 子命令
    subparsers = parser.add_subparsers(
        dest="subcommand",
        title="子命令",
        description="可用的子命令",
        required=True,
    )

    # explore 子命令
    explore_parser = subparsers.add_parser(
        "explore",
        help="完整数据集探索",
        description="完整数据集探索（使用datatrove）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python experiments/run_fineweb.py explore
  python experiments/run_fineweb.py explore --config configs/fineweb.yaml --workers 16
  python experiments/run_fineweb.py explore --data-dir data/datasets/finewed-edu/data
        """,
    )

    explore_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据集路径 (默认: 从配置文件读取)",
    )

    explore_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: 从配置文件读取)",
    )

    explore_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行worker数量 (默认: 从配置文件读取)",
    )

    explore_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批量大小 (默认: 从配置文件读取)",
    )

    explore_parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别 (默认: 从配置文件读取)",
    )

    # quick 子命令
    quick_parser = subparsers.add_parser(
        "quick",
        help="快速统计",
        description="快速统计（限制处理文档数）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python experiments/run_fineweb.py quick
  python experiments/run_fineweb.py quick --limit 10000
  python experiments/run_fineweb.py quick --limit 5000 --verbose
        """,
    )

    quick_parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="限制处理文档数 (默认: 10000)",
    )

    quick_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据集路径 (默认: 从配置文件读取)",
    )

    quick_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: 从配置文件读取)",
    )

    quick_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行worker数量 (默认: 从配置文件读取)",
    )

    quick_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批量大小 (默认: 从配置文件读取)",
    )

    # demo 子命令
    demo_parser = subparsers.add_parser(
        "demo",
        help="运行演示",
        description="运行FinewebEduStatsCollector演示（100个文档）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python experiments/run_fineweb.py demo
        """,
    )

    # test 子命令
    test_parser = subparsers.add_parser(
        "test",
        help="运行测试",
        description="运行项目测试（使用pytest）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python experiments/run_fineweb.py test
  python experiments/run_fineweb.py test -- -v
  python experiments/run_fineweb.py test -- --cov
        """,
    )

    # 添加pytest参数（使用parse_known_args来支持任意pytest参数）
    test_parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="传递给pytest的参数（例如: -v, --cov, -k test_name）",
    )

    return parser


def setup_logging(verbose: bool) -> logging.Logger:
    """配置日志系统。

    Args:
        verbose: 是否启用详细模式

    Returns:
        配置好的日志记录器
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("run_fineweb")


def run_explore(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """执行explore子命令。

    调用fineweb_explore.main()进行完整数据集探索。

    Args:
        args: 命令行参数
        logger: 日志记录器
    """
    import sys

    logger.info("执行explore子命令")

    # 构建命令行参数
    cmd_args = ["experiments/fineweb_explore.py"]

    # 添加配置参数
    cmd_args.extend(["--config", args.config])

    # 添加explore特定参数
    if args.data_dir:
        cmd_args.extend(["--data-dir", args.data_dir])

    if args.output_dir:
        cmd_args.extend(["--output-dir", args.output_dir])

    if args.workers:
        cmd_args.extend(["--workers", str(args.workers)])

    if args.batch_size:
        cmd_args.extend(["--batch-size", str(args.batch_size)])

    if args.log_level:
        cmd_args.extend(["--log-level", args.log_level])
    elif args.verbose:
        cmd_args.extend(["--log-level", "DEBUG"])

    if args.dry_run:
        cmd_args.append("--dry-run")

    logger.debug(f"执行命令: python {' '.join(cmd_args)}")

    # Dry run模式
    if args.dry_run:
        print("\n" + "=" * 60)
        print("  Dry run: explore子命令")
        print("=" * 60)
        print(f"  配置文件: {args.config}")
        print(f"  数据目录: {args.data_dir or '(从配置文件读取)'}")
        print(f"  输出目录: {args.output_dir or '(从配置文件读取)'}")
        print(f"  Worker数量: {args.workers or '(从配置文件读取)'}")
        print(f"  批量大小: {args.batch_size or '(从配置文件读取)'}")
        print(f"  日志级别: {args.log_level or ('DEBUG' if args.verbose else 'INFO')}")
        print("=" * 60)
        print("  Dry run: configuration loaded")
        print("=" * 60 + "\n")
        return

    # 导入并执行
    try:
        import importlib.util

        # 动态导入fineweb_explore模块
        spec = importlib.util.spec_from_file_location(
            "fineweb_explore", "experiments/fineweb_explore.py"
        )
        if spec is None:
            raise ImportError("无法加载fineweb_explore模块")
        fineweb_explore = importlib.util.module_from_spec(spec)
        sys.modules["fineweb_explore"] = fineweb_explore
        if spec.loader is None:
            raise ImportError("fineweb_explore模块没有loader")
        spec.loader.exec_module(fineweb_explore)

        # 构造模拟的sys.argv
        original_argv = sys.argv
        sys.argv = cmd_args
        try:
            fineweb_explore.main()
        finally:
            sys.argv = original_argv
    except ImportError as e:
        logger.error(f"无法导入fineweb_explore模块: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"explore子命令执行失败: {e}")
        sys.exit(1)


def run_quick(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """执行quick子命令。

    调用fineweb_explore.main()并限制处理文档数。

    Args:
        args: 命令行参数
        logger: 日志记录器
    """
    import sys

    logger.info(f"执行quick子命令，限制处理文档数: {args.limit}")

    # 构建命令行参数
    cmd_args = ["experiments/fineweb_explore.py"]

    # 添加配置参数
    cmd_args.extend(["--config", args.config])

    # 添加limit参数
    cmd_args.extend(["--limit", str(args.limit)])

    # 添加其他参数
    if args.data_dir:
        cmd_args.extend(["--data-dir", args.data_dir])

    if args.output_dir:
        cmd_args.extend(["--output-dir", args.output_dir])

    if args.workers:
        cmd_args.extend(["--workers", str(args.workers)])

    if args.batch_size:
        cmd_args.extend(["--batch-size", str(args.batch_size)])

    if args.verbose:
        cmd_args.extend(["--log-level", "DEBUG"])

    if args.dry_run:
        cmd_args.append("--dry-run")

    logger.debug(f"执行命令: python {' '.join(cmd_args)}")

    # Dry run模式
    if args.dry_run:
        print("\n" + "=" * 60)
        print("  Dry run: quick子命令")
        print("=" * 60)
        print(f"  配置文件: {args.config}")
        print(f"  限制文档数: {args.limit}")
        print(f"  数据目录: {args.data_dir or '(从配置文件读取)'}")
        print(f"  输出目录: {args.output_dir or '(从配置文件读取)'}")
        print(f"  Worker数量: {args.workers or '(从配置文件读取)'}")
        print(f"  批量大小: {args.batch_size or '(从配置文件读取)'}")
        print(f"  日志级别: {'DEBUG' if args.verbose else 'INFO'}")
        print("=" * 60)
        print("  Dry run: configuration loaded")
        print("=" * 60 + "\n")
        return

    # 导入并执行
    try:
        import sys
        import importlib.util

        # 动态导入fineweb_explore模块
        spec = importlib.util.spec_from_file_location(
            "fineweb_explore", "experiments/fineweb_explore.py"
        )
        if spec is None:
            raise ImportError("无法加载fineweb_explore模块")
        fineweb_explore = importlib.util.module_from_spec(spec)
        sys.modules["fineweb_explore"] = fineweb_explore
        if spec.loader is None:
            raise ImportError("fineweb_explore模块没有loader")
        spec.loader.exec_module(fineweb_explore)

        # 构造模拟的sys.argv
        original_argv = sys.argv
        sys.argv = cmd_args
        try:
            fineweb_explore.main()
        finally:
            sys.argv = original_argv
    except ImportError as e:
        logger.error(f"无法导入fineweb_explore模块: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"quick子命令执行失败: {e}")
        sys.exit(1)


def run_demo(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """执行demo子命令。

    调用fineweb_demo.main()运行演示。

    Args:
        args: 命令行参数
        logger: 日志记录器
    """
    import sys

    logger.info("执行demo子命令")

    # Dry run模式
    if args.dry_run:
        print("\n" + "=" * 60)
        print("  Dry run: demo子命令")
        print("=" * 60)
        print(f"  配置文件: {args.config}")
        print(f"  日志级别: {'DEBUG' if args.verbose else 'INFO'}")
        print("=" * 60)
        print("  Dry run: configuration loaded")
        print("=" * 60 + "\n")
        return

    # 导入并执行
    try:
        import sys
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "fineweb_demo", "experiments/fineweb_demo.py"
        )
        if spec is None:
            raise ImportError("无法加载fineweb_demo模块")
        fineweb_demo = importlib.util.module_from_spec(spec)
        sys.modules["fineweb_demo"] = fineweb_demo
        if spec.loader is None:
            raise ImportError("fineweb_demo模块没有loader")
        spec.loader.exec_module(fineweb_demo)

        fineweb_demo.run_fineweb_stats_collection()
    except ImportError as e:
        logger.error(f"无法导入fineweb_demo模块: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"demo子命令执行失败: {e}")
        sys.exit(1)


def run_test(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """执行test子命令。

    调用pytest运行测试。

    Args:
        args: 命令行参数
        logger: 日志记录器
    """
    logger.info("执行test子命令")

    # 构建pytest命令
    cmd = ["python", "-m", "pytest"]

    # 添加pytest参数（过滤掉--分隔符）
    if args.pytest_args:
        pytest_params = [arg for arg in args.pytest_args if arg != "--"]
        cmd.extend(pytest_params)
    else:
        # 默认添加-v参数
        cmd.append("-v")

    logger.debug(f"执行命令: {' '.join(cmd)}")

    # Dry run模式
    if args.dry_run:
        print("\n" + "=" * 60)
        print("  Dry run: test子命令")
        print("=" * 60)
        pytest_params = (
            [arg for arg in args.pytest_args if arg != "--"]
            if args.pytest_args
            else ["-v"]
        )
        print(f"  pytest参数: {' '.join(pytest_params)}")
        print("=" * 60)
        print("  Dry run: configuration loaded")
        print("=" * 60 + "\n")
        return

    # 执行pytest
    try:
        result = subprocess.run(cmd, cwd="/mnt/usr/projects/nanomind")
        sys.exit(result.returncode)
    except Exception as e:
        logger.error(f"test子命令执行失败: {e}")
        sys.exit(1)


def main() -> None:
    """主函数，执行命令行入口流程。"""
    # 解析命令行参数
    parser = create_parser()
    args = parser.parse_args()

    # 设置日志
    logger = setup_logging(args.verbose)

    # 根据子命令执行相应操作
    if args.subcommand == "explore":
        run_explore(args, logger)
    elif args.subcommand == "quick":
        run_quick(args, logger)
    elif args.subcommand == "demo":
        run_demo(args, logger)
    elif args.subcommand == "test":
        run_test(args, logger)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
