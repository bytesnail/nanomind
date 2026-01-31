"""实验 000 的主入口点。

提供使用 `python -m experiments.000` 运行实验的模块入口点。
"""

import sys
from pathlib import Path

from experiments.utils.paths import setup_experiment_paths

setup_experiment_paths(__file__)

from exp_000_environment_check import main

if __name__ == "__main__":
    main()
