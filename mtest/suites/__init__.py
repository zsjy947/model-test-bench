"""套件注册表：统一 Suite 接口，预留扩展点（accuracy / stability）。"""

from __future__ import annotations

from typing import Type

from ..config import BenchConfig
from .base import RunContext, Suite, SuiteResult, SUITE_ERROR, SUITE_FAILED, SUITE_PASSED, SUITE_SKIPPED  # noqa: F401

# 执行顺序：functional 冒烟先行（快速失败），再压测类
_ORDER = ["functional", "perf", "longctx", "embedding", "ocr"]
_REGISTRY: dict[str, Type[Suite]] = {}


def register(cls: Type[Suite]) -> Type[Suite]:
    """注册套件（扩展点：accuracy / stability 等以此接入）。"""
    _REGISTRY[cls.name] = cls
    return cls


def _load_builtin() -> None:
    # 延迟导入避免循环依赖
    from .embedding import EmbeddingSuite
    from .functional import FunctionalSuite
    from .longctx import LongctxSuite
    from .ocr import OcrSuite
    from .perf import PerfSuite

    for cls in (FunctionalSuite, PerfSuite, LongctxSuite, EmbeddingSuite, OcrSuite):
        register(cls)


def registered_suites() -> dict[str, Type[Suite]]:
    if not _REGISTRY:
        _load_builtin()
    return dict(_REGISTRY)


def suites_for_config(cfg: BenchConfig) -> list[Type[Suite]]:
    """按执行顺序返回启用且适用于该模型类型的套件类。"""
    out: list[Type[Suite]] = []
    reg = registered_suites()
    for name in _ORDER + [n for n in reg if n not in _ORDER]:
        if name not in reg:
            continue
        suite_cfg = getattr(cfg.tests, name, None)
        if suite_cfg is not None and not getattr(suite_cfg, "enabled", True):
            continue
        cls = reg[name]
        if cfg.model.type not in cls.applies_to:
            continue
        out.append(cls)
    return out


async def run_suite(cls: Type[Suite], ctx: RunContext) -> SuiteResult:
    """执行单个套件并兜底异常（异常转为 error 状态结果）。"""
    suite = cls(ctx)
    ctx.say(f"[bold cyan]════ 套件 {suite.name} 开始 ════[/bold cyan]")
    try:
        result = await suite.run()
    except Exception as exc:  # noqa: BLE001 - 套件异常不阻断后续套件
        result = SuiteResult(name=suite.name, status=SUITE_ERROR,
                             error=f"{type(exc).__name__}: {exc}")
        result.started_at = SuiteResult._now()
        result.ended_at = result.started_at
        ctx.say(f"[red]套件 {suite.name} 异常: {exc}[/red]")
    color = {"passed": "green", "failed": "red", "error": "red", "skipped": "yellow"}.get(
        result.status, "white")
    ctx.say(f"[bold {color}]════ 套件 {suite.name} {result.status} "
            f"（{result.duration_s if result.duration_s is not None else 0:.0f}s）════[/bold {color}]")
    return result
