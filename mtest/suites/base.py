"""测试套件统一接口。

所有套件输出统一 :class:`SuiteResult`（结构化指标 + 明细行 + 阶段时间戳），
pipeline 汇总交给 report 层渲染。套件注册点见 ``suites/__init__.py``
（预留 accuracy / stability 扩展）。
"""

from __future__ import annotations

import abc
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from ..client import BenchClient
from ..config import BenchConfig
from ..monitor import NPUMonitor

SUITE_PASSED = "passed"
SUITE_FAILED = "failed"
SUITE_ERROR = "error"
SUITE_SKIPPED = "skipped"


@dataclass
class SuiteResult:
    """套件执行结果（结构统一，供 metrics.json 与 summary.md 渲染）。"""

    name: str
    status: str = SUITE_PASSED
    started_at: str = ""
    ended_at: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)   # 标签 → 文件路径
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @property
    def duration_s(self) -> float | None:
        if not self.started_at or not self.ended_at:
            return None
        t0 = datetime.fromisoformat(self.started_at)
        t1 = datetime.fromisoformat(self.ended_at)
        return (t1 - t0).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "details": self.details,
            "artifacts": self.artifacts,
            "error": self.error,
            "warnings": self.warnings,
        }


@dataclass
class RunContext:
    """一次 run 的共享上下文（pipeline 构造，注入各套件）。"""

    cfg: BenchConfig
    run_dir: Path
    client: BenchClient
    served_model: str
    console: Any = None                       # rich Console
    monitor: NPUMonitor | None = None
    env_info: dict[str, Any] = field(default_factory=dict)
    startup_seconds: float | None = None
    skip_serve: bool = False

    def say(self, msg: str) -> None:
        if self.console is not None:
            self.console.print(msg)

    def phase(self, label: str):
        """阶段标记（写入 NPU 采样时间轴）。"""
        if self.monitor is not None:
            return self.monitor.phase(label)
        return nullcontext()


class Suite(abc.ABC):
    """套件基类。子类设置 ``name`` / ``applies_to`` 并实现 :meth:`run`。"""

    name: ClassVar[str] = "base"
    applies_to: ClassVar[tuple[str, ...]] = ()

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.cfg = ctx.cfg

    @abc.abstractmethod
    async def run(self) -> SuiteResult: ...

    def new_result(self) -> SuiteResult:
        result = SuiteResult(name=self.name, started_at=SuiteResult._now())
        return result

    def finish(self, result: SuiteResult, status: str = SUITE_PASSED) -> SuiteResult:
        result.status = status
        result.ended_at = SuiteResult._now()
        return result
