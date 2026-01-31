"""系统信息收集模块。

提供获取系统、CPU 和内存信息的函数。
"""

import logging
import os
import platform
import sys
from typing import Dict, Any

from experiments.utils.constants import MB_FACTOR

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """获取系统信息。

    Returns:
        Dict[str, Any]: 系统信息字典，包含操作系统、内核版本、Python 版本等。
    """
    info = {
        "操作系统": platform.system(),
        "内核版本": platform.release(),
        "Python 版本": f"{sys.version.split()[0]} ({sys.version_info.releaselevel})",
        "Python 实现": platform.python_implementation(),
        "CPU 架构": platform.machine(),
        "主机名": platform.node(),
    }
    return info


def _read_cpuinfo() -> str:
    """读取 /proc/cpuinfo 文件内容。

    Returns:
        CPU 信息文件的完整文本内容。
    """
    with open("/proc/cpuinfo", "r") as f:
        return f.read()


def _collect_cpu_physical_ids(cpuinfo: str) -> set:
    """收集所有物理 CPU ID。

    Args:
        cpuinfo (str): CPU 信息文件内容。

    Returns:
        set: 包含所有物理 CPU ID 的集合。
    """
    physical_ids = set()
    for line in cpuinfo.split("\n"):
        if line.startswith("physical id"):
            current_physical_id = line.split(":")[1].strip()
            physical_ids.add(current_physical_id)
    return physical_ids


def _get_cpu_model(cpuinfo: str) -> str | None:
    """提取 CPU 型号名称。

    Args:
        cpuinfo (str): CPU 信息文件内容。

    Returns:
        str | None: CPU 型号名称或 None。
    """
    for line in cpuinfo.split("\n"):
        if line.startswith("model name"):
            return line.split(":")[1].strip()
    return None


def _get_cpu_frequency(cpuinfo: str) -> str | None:
    """提取 CPU 频率。

    Args:
        cpuinfo (str): CPU 信息文件内容。

    Returns:
        str | None: 格式化的 CPU 频率字符串或 None。
    """
    for line in cpuinfo.split("\n"):
        if line.startswith("cpu MHz"):
            freq_mhz = float(line.split(":")[1].strip())
            return f"{freq_mhz / 1000:.2f} GHz"
    return None


def _get_core_counts(cpuinfo: str) -> dict:
    """获取核心数量信息。

    Args:
        cpuinfo (str): CPU 信息文件内容。

    Returns:
        dict: 核心 ID 到核心数的映射字典。
    """
    core_info = {}
    current_id = None

    for line in cpuinfo.split("\n"):
        if line.startswith("physical id"):
            current_id = line.split(":")[1].strip()
        elif line.startswith("cpu cores") and current_id:
            core_info[current_id] = int(line.split(":")[1].strip())

    return core_info


def _add_core_info(info: dict, core_info: dict) -> None:
    """添加核心信息到字典。

    Args:
        info (dict): 目标信息字典。
        core_info (dict): 核心信息字典。
    """
    logical_cores = os.cpu_count()
    cores_per_socket = list(core_info.values())[0]
    total_physical_cores = sum(core_info.values())
    info["每颗 CPU 核心数"] = f"{cores_per_socket} 核"
    info["总物理核心数"] = f"{total_physical_cores} 核"
    if logical_cores is not None:
        info["超线程"] = "启用" if logical_cores > total_physical_cores else "未启用"


def _build_base_info(
        cpu_model: str | None, cpu_freq: str | None, physical_ids: set
) -> dict:
    """构建基础 CPU 信息字典。

    Args:
        cpu_model (str | None): CPU 型号。
        cpu_freq (str | None): CPU 频率。
        physical_ids (set): 物理 CPU ID 集合。

    Returns:
        dict: 基础 CPU 信息字典。
    """
    return {
        "物理 CPU 数量": len(physical_ids),
        "CPU 型号": cpu_model or "未知",
        "CPU 频率": cpu_freq or "未知",
    }


def _build_cpu_info_dict(
        cpu_model: str | None, cpu_freq: str | None, core_info: dict, physical_ids: set
) -> dict:
    """构建 CPU 信息字典。

    Args:
        cpu_model (str | None): CPU 型号。
        cpu_freq (str | None): CPU 频率。
        core_info (dict): 核心信息字典。
        physical_ids (set): 物理 CPU ID 集合。

    Returns:
        dict: 完整的 CPU 信息字典。
    """
    info = _build_base_info(cpu_model, cpu_freq, physical_ids)
    if core_info:
        _add_core_info(info, core_info)
    return info


def get_linux_cpu_info() -> Dict[str, Any]:
    """获取 Linux 系统的 CPU 信息。

    解析 /proc/cpuinfo 文件，提取物理 CPU 数量、CPU 型号、频率等信息。

    Returns:
        Dict[str, Any]: CPU 信息字典，包含物理 CPU 数量、CPU 型号、频率、核心数等。
    """
    try:
        cpuinfo = _read_cpuinfo()
        physical_ids = _collect_cpu_physical_ids(cpuinfo)
        cpu_model = _get_cpu_model(cpuinfo)
        cpu_freq = _get_cpu_frequency(cpuinfo)
        core_info = _get_core_counts(cpuinfo)
        return _build_cpu_info_dict(cpu_model, cpu_freq, core_info, physical_ids)
    except (OSError, ValueError, IndexError) as e:
        logger.debug(f"获取 CPU 信息失败: {e}", exc_info=True)
        return {"CPU 信息": "获取失败"}


def get_linux_memory_info() -> Dict[str, Any]:
    """获取 Linux 系统的内存信息。

    解析 /proc/meminfo 文件，计算总内存、可用内存和使用率。

    Returns:
        Dict[str, Any]: 内存信息字典，包含总内存、可用内存和使用率。
    """
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
            mem_data = {}
            for line in meminfo.split("\n"):
                if ":" in line:
                    key, value = line.split(":")
                    mem_data[key.strip()] = value.strip().split()[0]

            total_kb = int(mem_data.get("MemTotal", 0))
            available_kb = int(mem_data.get("MemAvailable", 0))

            info = {
                "总内存": f"{total_kb / MB_FACTOR:.2f} GB",
                "可用内存": f"{available_kb / MB_FACTOR:.2f} GB",
            }

            usage_percent = (1 - available_kb / total_kb) * 100 if total_kb > 0 else 0
            info["内存使用率"] = f"{usage_percent:.1f}%"
    except (OSError, ValueError, IndexError):
        info = {"内存信息": "获取失败"}

    return info


def get_cpu_info() -> Dict[str, Any]:
    """获取 CPU 信息。

    根据操作系统类型调用相应的信息获取函数。

    Returns:
        Dict[str, Any]: CPU 信息字典。
    """
    logical_cores = os.cpu_count()
    info = {"逻辑处理器数": logical_cores or "未知"}

    if platform.system() == "Linux":
        linux_info = get_linux_cpu_info()
        info.update(linux_info)

    return info


def get_memory_info() -> Dict[str, Any]:
    """获取内存信息。

    根据操作系统类型调用相应的信息获取函数。

    Returns:
        Dict[str, Any]: 内存信息字典。
    """
    info = {}

    if platform.system() == "Linux":
        linux_info = get_linux_memory_info()
        info.update(linux_info)
    else:
        info["内存信息"] = "仅在 Linux 系统支持"

    return info
