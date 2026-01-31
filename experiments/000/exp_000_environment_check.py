"""实验 000: 环境配置验证。

运行命令：python -m experiments.000
"""

from experiments.utils.paths import setup_experiment_paths

setup_experiment_paths(__file__)

import sys
from datetime import datetime

try:
    import torch
except ImportError as e:
    torch = None
    print(f"❌ PyTorch 导入失败: {e}")
    sys.exit(1)

from experiments.utils.common import print_section, print_separator

from system_info import get_system_info, get_cpu_info, get_memory_info
from torch_info import get_torch_info, get_cuda_info, test_pytorch_operations


def main() -> None:
    print_section("  实验 000: 环境配置验证")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_section("系统信息", "", get_system_info())
    print_section("CPU 信息", "", get_cpu_info())
    print_section("内存信息", "", get_memory_info())
    print_section("PyTorch 信息", "", get_torch_info())
    print_section("CUDA 和 GPU 信息", "", get_cuda_info())

    test_pytorch_operations()

    print_section("  总结")

    if torch.cuda.is_available():
        print("  ✅ 环境配置正确，CUDA 可用")
        print("  ✅ 可使用 GPU 加速训练")
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
    else:
        print("  ⚠️  CUDA 不可用，将使用 CPU 进行计算")
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
        print("  💡 建议: 如需 GPU 加速，请安装 CUDA 版本的 PyTorch")

    print_separator()


if __name__ == "__main__":
    main()
