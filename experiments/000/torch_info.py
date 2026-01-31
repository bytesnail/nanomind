"""PyTorch 信息获取和测试模块。

提供 PyTorch 版本、CUDA 信息和基本操作测试的功能。
"""

import logging
from typing import Any, Dict, Optional, cast

try:
    import torch
    from torch.backends import cuda, cudnn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    cuda = None
    cudnn = None


def _get_torch() -> Any:
    if not TORCH_AVAILABLE:
        return None
    return torch


try:
    from experiments.utils.constants import SEPARATOR_WIDTH, GB_FACTOR, MB_FACTOR
except ImportError:
    # 回退到常量定义
    SEPARATOR_WIDTH = 60
    GB_FACTOR = 1024**3
    MB_FACTOR = 1024**2

logger = logging.getLogger(__name__)


def get_torch_info() -> Dict[str, Any]:
    """获取 PyTorch 信息。

    Returns:
        Dict[str, Any]: PyTorch 版本和编译信息字典。
    """
    if not TORCH_AVAILABLE:
        return {}

    t = cast(Any, _get_torch())

    info = {
        "PyTorch 版本": t.__version__,
        "PyTorch 编译信息": (f"CUDA {t.version.cuda if t.version.cuda else '未支持'}"),
    }

    return info


def get_cuda_info() -> Dict[str, Any]:
    """获取 CUDA 和 GPU 信息。

    Returns:
        Dict[str, Any]: CUDA 可用性、GPU 信息和显存信息字典。
    """
    info = {}

    if not TORCH_AVAILABLE:
        return info

    t = cast(Any, _get_torch())

    cuda_available = t.cuda.is_available()
    info["CUDA 可用"] = "✅ 是" if cuda_available else "❌ 否"

    if cuda_available:
        info["CUDA 版本"] = t.version.cuda
        info["GPU 数量"] = t.cuda.device_count()

        for i in range(t.cuda.device_count()):
            device_name = t.cuda.get_device_name(i)
            device_capability = t.cuda.get_device_capability(i)

            info[f"GPU {i} 型号"] = device_name
            info[f"GPU {i} 计算能力"] = f"{device_capability[0]}.{device_capability[1]}"

            props = t.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / GB_FACTOR
            info[f"GPU {i} 显存"] = f"{total_memory_gb:.2f} GB"

        cudnn_module = cast(Any, cudnn)
        cudnn_available = cudnn_module.is_available()
        info["cuDNN 可用"] = "✅ 是" if cudnn_available else "❌ 否"

        if cudnn_available:
            info["cuDNN 版本"] = cudnn_module.version()
            info["cuDNN 优化"] = "启用" if cudnn_module.enabled else "禁用"
    else:
        info["建议"] = "请检查 CUDA 安装或使用 CPU 版本的 PyTorch"

    return info


def test_pytorch_operations() -> None:
    """测试 PyTorch 基本操作。

    执行张量创建、矩阵运算、GPU 操作（如果可用）和自动求导测试。
    """
    if not TORCH_AVAILABLE:
        return

    t = cast(Any, torch)

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("  PyTorch 功能测试")
    print(f"{'=' * SEPARATOR_WIDTH}")

    print("\n  1. 张量创建测试...")
    x = t.randn(3, 4)
    logger.debug(f"成功创建张量 {x.shape}")
    print(f"     ✅ 成功创建张量 {x.shape}")

    print("\n  2. 矩阵运算测试...")
    y = t.randn(4, 5)
    z = t.matmul(x, y)
    logger.debug(f"成功进行矩阵乘法 {x.shape} @ {y.shape} = {z.shape}")
    print(f"     ✅ 成功进行矩阵乘法 {x.shape} @ {y.shape} = {z.shape}")

    if t.cuda.is_available():
        print("\n  3. GPU 操作测试...")
        device = t.device("cuda")
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        t.matmul(x_gpu, y_gpu)
        logger.debug("成功在 GPU 上进行矩阵乘法")
        print("     ✅ 成功在 GPU 上进行矩阵乘法")

        allocated = t.cuda.memory_allocated() / MB_FACTOR
        cached = t.cuda.memory_reserved() / MB_FACTOR
        print(f"     GPU 内存分配: {allocated:.2f} MB")
        print(f"     GPU 内存缓存: {cached:.2f} MB")
    else:
        print("\n  3. GPU 操作: ⚠️  CUDA 不可用，跳过 GPU 测试")
        logger.warning("CUDA 不可用，跳过 GPU 测试")

    print("\n  4. 自动求导测试...")
    a = t.tensor([2.0], requires_grad=True)
    b = t.tensor([3.0], requires_grad=True)
    c = a * b
    c.backward()
    logger.debug(f"a={a.item():.1f}, b={b.item():.1f}, c={c.item():.1f}")
    print(f"     ✅ a={a.item():.1f}, b={b.item():.1f}, c={c.item():.1f}")

    if a.grad is not None and b.grad is not None:
        logger.debug(f"∂c/∂a={a.grad.item():.1f}, ∂c/∂b={b.grad.item():.1f}")
