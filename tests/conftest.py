"""pytest 配置：仓库根加入 sys.path（未安装也可跑）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
