"""实验 000: 环境配置验证。

运行命令：python experiments/000/exp_000_environment_check.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.resolve()
exp_dir = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.utils.paths import setup_experiment_paths

setup_experiment_paths(__file__)

from experiments.utils.common import print_separator
from experiments.utils.constants import SEPARATOR_WIDTH

from system_info import get_system_info, get_cpu_info, get_memory_info
from torch_info import get_torch_info, get_cuda_info, test_pytorch_operations

try:
    import torch
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
    sys.exit(1)

logger = logging.getLogger("exp_000")


def print_section(title: str, data: dict) -> None:
    """打印部分信息。

    Args:
        title: 部分标题。
        data: 要显示的数据字典。
    """
    print_separator()
    print(f"  {title}")
    print_separator()
    for key, value in data.items():
        print(f"  {key:25s}: {value}")


def setup_logging() -> None:
    """设置日志配置。"""
    logger.setLevel(logging.INFO)


def main() -> None:
    """主入口函数。"""
    setup_logging()

    print_separator()
    print("  实验 000: 环境配置验证")
    print_separator()
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_section("系统信息", get_system_info())
    print_section("CPU 信息", get_cpu_info())
    print_section("内存信息", get_memory_info())
    print_section("PyTorch 信息", get_torch_info())
    print_section("CUDA 和 GPU 信息", get_cuda_info())

    test_pytorch_operations()

    print_separator()
    print("  总结")
    print_separator()

    if torch.cuda.is_available():
        print("  ✅ 环境配置正确，CUDA 可用")
        print("  ✅ 可使用 GPU 加速训练")
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
    else:
        print("  ⚠️  CUDA 不可用，将使用 CPU 进行计算")
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
        print("  💡 建议: 如需 GPU 加速，请安装 CUDA 版本的 PyTorch")

    print(f"\n{'=' * SEPARATOR_WIDTH}\n")


if __name__ == "__main__":
    main()
