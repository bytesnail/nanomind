"""实验 001 的主入口点。

提供使用 `python -m experiments.001` 运行实验的模块入口点。
"""

import sys
from pathlib import Path

from experiments.utils.paths import setup_experiment_paths

setup_experiment_paths(__file__)

from exp_001_datasets_stats import main

if __name__ == "__main__":
    main()
