"""
实验 000: 环境配置验证

目的：
- 获取本地硬件配置信息
- 验证 PyTorch 安装
- 验证 CUDA 支持

运行命令：
    python experiments/exp_000_environment_check.py

预期输出：
- 完整的系统硬件信息
- Python 和 PyTorch 版本信息
- CUDA 和 GPU 信息
"""

import os
import sys
import platform
from datetime import datetime
from typing import Dict, Any

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
    TORCH_AVAILABLE = False
    sys.exit(1)


def get_system_info() -> Dict[str, Any]:
    """获取系统基本信息。"""
    info = {
        "操作系统": platform.system(),
        "内核版本": platform.release(),
        "Python 版本": f"{sys.version.split()[0]} ({sys.version_info.releaselevel})",
        "Python 实现": platform.python_implementation(),
        "CPU 架构": platform.machine(),
        "主机名": platform.node(),
    }
    return info


def get_cpu_info() -> Dict[str, Any]:
    """获取 CPU 信息。"""
    logical_cores = os.cpu_count()
    info = {"逻辑处理器数": logical_cores or "未知"}

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()

                physical_ids = set()
                cpu_model = None
                cpu_freq = None
                current_physical_id = None
                core_info = {}

                for line in cpuinfo.split("\n"):
                    if line.startswith("physical id"):
                        current_physical_id = line.split(":")[1].strip()
                        physical_ids.add(current_physical_id)
                    elif line.startswith("model name") and cpu_model is None:
                        cpu_model = line.split(":")[1].strip()
                    elif line.startswith("cpu MHz") and cpu_freq is None:
                        freq_mhz = float(line.split(":")[1].strip())
                        cpu_freq = f"{freq_mhz / 1000:.2f} GHz"
                    elif (
                        line.startswith("cpu cores") and current_physical_id is not None
                    ):
                        cores = int(line.split(":")[1].strip())
                        core_info[current_physical_id] = cores

                info["物理 CPU 数量"] = len(physical_ids)
                info["CPU 型号"] = cpu_model or "未知"
                info["CPU 频率"] = cpu_freq or "未知"

                if len(core_info) > 0:
                    cores_per_socket = list(core_info.values())[0]
                    total_physical_cores = sum(core_info.values())
                    info["每颗 CPU 核心数"] = f"{cores_per_socket} 核"
                    info["总物理核心数"] = f"{total_physical_cores} 核"
                    if logical_cores is not None:
                        info["超线程"] = (
                            "启用" if logical_cores > total_physical_cores else "未启用"
                        )
        except Exception as e:
            print(f"⚠️  CPU 信息获取失败: {e}")

    return info


def get_memory_info() -> Dict[str, Any]:
    """获取内存信息。"""
    info = {}

    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
                mem_data = {}
                for line in meminfo.split("\n"):
                    if ":" in line:
                        key, value = line.split(":")
                        mem_data[key.strip()] = value.strip().split()[0]

                total_kb = int(mem_data.get("MemTotal", 0))
                info["总内存"] = f"{total_kb / (1024 * 1024):.2f} GB"

                available_kb = int(mem_data.get("MemAvailable", 0))
                info["可用内存"] = f"{available_kb / (1024 * 1024):.2f} GB"

                usage_percent = (
                    (1 - available_kb / total_kb) * 100 if total_kb > 0 else 0
                )
                info["内存使用率"] = f"{usage_percent:.1f}%"
        except Exception as e:
            print(f"⚠️  内存信息获取失败: {e}")
            info["内存信息"] = "获取失败"
    else:
        info["内存信息"] = "仅在 Linux 系统支持"

    return info


def get_torch_info() -> Dict[str, Any]:
    """获取 PyTorch 信息。"""
    if not TORCH_AVAILABLE:
        return {}

    info = {
        "PyTorch 版本": torch.__version__,
        "PyTorch 编译信息": (
            f"CUDA {torch.version.cuda if torch.version.cuda else '未支持'}"
        ),
    }

    return info


def get_cuda_info() -> Dict[str, Any]:
    """获取 CUDA 和 GPU 信息。"""
    info = {}

    if not TORCH_AVAILABLE:
        return info

    cuda_available = torch.cuda.is_available()
    info["CUDA 可用"] = "✅ 是" if cuda_available else "❌ 否"

    if cuda_available:
        info["CUDA 版本"] = torch.version.cuda
        info["GPU 数量"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)

            info[f"GPU {i} 型号"] = device_name
            info[f"GPU {i} 计算能力"] = f"{device_capability[0]}.{device_capability[1]}"

            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024**3)
            info[f"GPU {i} 显存"] = f"{total_memory_gb:.2f} GB"

        cudnn_available = torch.backends.cudnn.is_available()
        info["cuDNN 可用"] = "✅ 是" if cudnn_available else "❌ 否"

        if cudnn_available:
            info["cuDNN 版本"] = torch.backends.cudnn.version()
            info["cuDNN 优化"] = "启用" if torch.backends.cudnn.enabled else "禁用"
    else:
        info["建议"] = "请检查 CUDA 安装或使用 CPU 版本的 PyTorch"

    return info


def print_section(title: str, data: Dict[str, Any]) -> None:
    """格式化打印一个部分的信息。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for key, value in data.items():
        print(f"  {key:25s}: {value}")


def test_pytorch_operations() -> None:
    """测试基本的 PyTorch 操作。"""
    print(f"\n{'=' * 60}")
    print("  PyTorch 功能测试")
    print(f"{'=' * 60}")

    print("\n  1. 张量创建测试...")
    x = torch.randn(3, 4)
    print(f"     ✅ 成功创建张量 {x.shape}")

    print("\n  2. 矩阵运算测试...")
    y = torch.randn(4, 5)
    z = torch.matmul(x, y)
    print(f"     ✅ 成功进行矩阵乘法 {x.shape} @ {y.shape} = {z.shape}")

    if torch.cuda.is_available():
        print("\n  3. GPU 操作测试...")
        device = torch.device("cuda")
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        z_gpu = torch.matmul(x_gpu, y_gpu)
        print(f"     ✅ 成功在 GPU 上进行矩阵乘法")

        allocated = torch.cuda.memory_allocated() / (1024**2)
        cached = torch.cuda.memory_reserved() / (1024**2)
        print(f"     GPU 内存分配: {allocated:.2f} MB")
        print(f"     GPU 内存缓存: {cached:.2f} MB")
    else:
        print("\n  3. GPU 操作: ⚠️  CUDA 不可用，跳过 GPU 测试")

    print("\n  4. 自动求导测试...")
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    c = a * b
    c.backward()
    print(f"     ✅ a={a.item():.1f}, b={b.item():.1f}, c={c.item():.1f}")

    if a.grad is not None and b.grad is not None:
        print(f"     ✅ ∂c/∂a={a.grad.item():.1f}, ∂c/∂b={b.grad.item():.1f}")

    print("\n  🎉 所有 PyTorch 功能测试通过！")


def main() -> None:
    """主函数。"""
    print("\n" + "=" * 60)
    print("  实验 000: 环境配置验证")
    print("=" * 60)
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_section("系统信息", get_system_info())
    print_section("CPU 信息", get_cpu_info())
    print_section("内存信息", get_memory_info())
    print_section("PyTorch 信息", get_torch_info())
    print_section("CUDA 和 GPU 信息", get_cuda_info())

    test_pytorch_operations()

    print(f"\n{'=' * 60}")
    print("  总结")
    print(f"{'=' * 60}")

    if torch.cuda.is_available():
        print("  ✅ 环境配置正确，CUDA 可用")
        print(f"  ✅ 可使用 GPU 加速训练")
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
    else:
        print("  ⚠️  CUDA 不可用，将使用 CPU 进行计算")
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
        print(f"  💡 建议: 如需 GPU 加速，请安装 CUDA 版本的 PyTorch")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
