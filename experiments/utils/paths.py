"""路径处理工具模块。

提供统一的路径处理功能，确保所有路径都使用相对路径。
本模块遵循项目的相对路径优先原则，不使用绝对路径。
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Union


def get_project_root() -> str:
    """获取项目根目录的相对路径。

    从当前文件位置向上查找项目根目录（包含 .git 的目录）。

    Returns:
        项目根目录的相对路径字符串。
    """
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    # 向上查找包含 .git 的目录
    project_root = current_dir
    while project_root.parent != project_root:
        if (project_root / ".git").exists():
            break
        project_root = project_root.parent

    # 转换为相对于当前工作目录的相对路径
    cwd = Path.cwd()
    try:
        rel_path = os.path.relpath(project_root, cwd)
        return rel_path if rel_path != "." else ""
    except ValueError:
        # 如果在不同驱动器上，返回空字符串
        return ""


def ensure_relative_path(path: Union[str, Path]) -> str:
    """确保路径为相对路径。

    如果传入绝对路径，会转换为相对于当前工作目录的相对路径。

    Args:
        path: 输入路径（字符串或 Path 对象）。

    Returns:
        相对路径字符串。
    """
    path_str = str(path)

    # 如果已经是相对路径，直接返回
    if not os.path.isabs(path_str):
        return path_str

    # 转换绝对路径为相对路径
    try:
        cwd = os.getcwd()
        rel_path = os.path.relpath(path_str, cwd)
        return rel_path
    except ValueError:
        # 如果在不同驱动器上，返回原始路径
        return path_str


def get_output_dir(experiment_name: str) -> str:
    """获取实验输出目录的相对路径。

    Args:
        experiment_name: 实验名称。

    Returns:
        输出目录的相对路径字符串。
    """
    # 确保输出目录路径是相对的
    output_path = ensure_relative_path(f"outputs/{experiment_name}")
    return output_path


def setup_experiment_paths(file_path: str) -> Tuple[Path, Path]:
    """为实验脚本设置统一的路径管理。

    计算项目根目录和实验目录，并将它们添加到 sys.path。

    Args:
        file_path: 调用脚本的 __file__ 路径。

    Returns:
        包含 (project_root, exp_dir) 的元组。
    """
    exp_dir = Path(file_path).resolve().parent
    project_root = exp_dir.parent.parent

    sys.path.insert(0, str(project_root))
    sys.path.insert(1, str(exp_dir))

    return project_root, exp_dir
