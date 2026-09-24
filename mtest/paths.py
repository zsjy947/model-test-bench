"""路径解析工具。

仓库内的 ``configs/``、``data/`` 位于仓库根目录（包外），运行时可能从任意 CWD
调用 CLI，因此提供统一查找逻辑：环境变量 > 从 CWD 逐级向上 > 包安装位置的同级
目录。
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_REPO = "MTEST_REPO_ROOT"


def package_root() -> Path:
    """``mtest`` 包所在目录。"""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """推断仓库根目录（包含 configs/ 与 data/ 的目录）。"""
    env = os.environ.get(_ENV_REPO)
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p

    # 源码安装：包目录的上一级即仓库根
    candidate = package_root().parent
    if (candidate / "configs").is_dir() or (candidate / "data").is_dir():
        return candidate

    # 从 CWD 逐级向上找（支持在子目录中调用 CLI）
    cur = Path.cwd().resolve()
    for asc in [cur, *cur.parents]:
        if (asc / "configs").is_dir() and (asc / "data").is_dir():
            return asc

    return candidate


def find_defaults_config() -> Path:
    """定位 ``configs/defaults.yaml``。"""
    p = repo_root() / "configs" / "defaults.yaml"
    if not p.is_file():
        raise FileNotFoundError(
            f"未找到 defaults.yaml（期望位于 {p}）。"
            f"可用环境变量 {_ENV_REPO} 指定仓库根目录。"
        )
    return p


def models_config_dir() -> Path:
    return repo_root() / "configs" / "models"


def results_dir() -> Path:
    return repo_root() / "results"


def data_dir() -> Path:
    return repo_root() / "data"


def resolve_data(path: str | Path) -> Path:
    """解析数据路径：绝对路径直接用；相对路径先按 CWD，再按仓库根。"""
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    fallback = data_dir() / p
    return fallback if fallback.exists() else p.resolve()
