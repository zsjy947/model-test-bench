"""编排层：加载合并配置 → 服务管理 → 健康检查 → 逐套件执行（带阶段标记）→
NPU 采样伴随 → 停服务 → 聚合报告（设计 §2）。"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .client import BenchClient
from .config import BenchConfig, apply_cli_overrides, build_serve_command
from .errors import MTestError, ServeError
from .monitor import NPUMonitor, summarize_npu
from .paths import results_dir
from .report import build_conclusions, collect_environment, write_reports
from .serve.launcher import create_launcher
from .suites import run_suite, suites_for_config
from .suites.base import RunContext, SuiteResult

_UNSAFE_FS = re.compile(r"[^0-9A-Za-z._-]+")


@dataclass
class RunOptions:
    """CLI 运行选项。"""

    suites: list[str] | None = None
    skip_serve: bool = False
    keep_alive: bool = False
    dry_run: bool = False
    concurrency: list[int] | None = None
    config_path: str | None = None


@dataclass
class RunSummary:
    run_id: str
    run_dir: Path | None
    ok: bool
    suite_results: list[SuiteResult] = field(default_factory=list)
    error: str | None = None
    dry_run: bool = False
    summary_path: Path | None = None


def make_run_id(cfg: BenchConfig) -> str:
    safe = _UNSAFE_FS.sub("-", cfg.model.name).strip("-") or "model"
    return f"{time.strftime('%Y%m%d-%H%M')}_{safe}"


def _unique_run_dir(cfg: BenchConfig) -> Path:
    base = results_dir()
    run_id = make_run_id(cfg)
    run_dir = base / run_id
    n = 2
    while run_dir.exists():
        run_dir = base / f"{run_id}-{n}"
        n += 1
    return run_dir


async def run_pipeline(cfg: BenchConfig, opts: RunOptions, console) -> RunSummary:
    """执行完整闭环；套件异常不互相阻断，服务异常自动收割诊断包。"""
    apply_cli_overrides(cfg, suites=opts.suites, concurrency=opts.concurrency)

    run_dir = _unique_run_dir(cfg)
    run_id = run_dir.name

    # ---- dry-run：只打印将要执行的命令与套件 ---------------------------- #
    if opts.dry_run:
        console.print(f"[bold]dry-run（不启动服务）[/bold]，run_id 将为 {run_id}")
        console.print(f"  启动命令: bash -lc {build_serve_command(cfg)!r}")
        if cfg.serve.command:
            console.print("  （serve.command 逃生舱生效，以上为整体替代命令）")
        console.print(f"  启用套件: {cfg.enabled_suites()}")
        for w in cfg.warnings:
            console.print(f"  [yellow]警告: {w}[/yellow]")
        return RunSummary(run_id=run_id, run_dir=None, ok=True, dry_run=True)

    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    # 落盘生效配置（复现依据）
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(exclude={"warnings"}), allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    for w in cfg.warnings:
        console.print(f"[yellow]配置警告: {w}[/yellow]")

    monitor: NPUMonitor | None = None
    if cfg.monitor.enabled:
        monitor = NPUMonitor(interval=cfg.monitor.npu_interval,
                             csv_path=run_dir / "npu_samples.csv", console=console)
        monitor.start()

    launcher = create_launcher(cfg, run_dir, console)
    manage_service = (not opts.skip_serve) and cfg.serve.mode != "external"

    startup_seconds: float | None = None
    suite_results: list[SuiteResult] = []
    error: str | None = None
    env_info: dict[str, Any] = {}

    try:
        # ---- 起服务 + 健康检查 ---------------------------------------- #
        try:
            if manage_service:
                with _phase(monitor, "serve.startup"):
                    await launcher.start()
            with _phase(monitor, "serve.wait_ready"):
                startup_seconds = await launcher.wait_ready()
            console.print(f"[green]服务就绪（{startup_seconds:.0f}s）[/green]")
        except (ServeError, MTestError):
            await launcher.collect_diagnostics()
            error = "服务启动/健康检查失败，诊断包已收割（serve_failure/）"
            raise

        # ---- 建客户端 + 校验 served-model-name ------------------------- #
        max_cc = max([*cfg.tests.perf.concurrency, *cfg.tests.longctx.concurrency,
                      *cfg.tests.embedding.concurrency, *cfg.tests.ocr.concurrency, 64])
        async with BenchClient(cfg.client.base_url, api_key=cfg.client.api_key,
                               request_timeout=cfg.client.request_timeout,
                               max_concurrency=max_cc) as client:
            env_info = await collect_environment(client)
            try:
                models = await client.list_models()
                served = cfg.model.resolved_served_name()
                if models and served not in models:
                    console.print(f"[yellow]警告: served-model-name {served!r} 不在 /v1/models "
                                  f"列表 {models}，请求可能 404[/yellow]")
            except Exception as exc:  # noqa: BLE001 - 列表失败不阻断
                console.print(f"[yellow]/v1/models 拉取失败: {exc}[/yellow]")

            ctx = RunContext(cfg=cfg, run_dir=run_dir, client=client,
                             served_model=cfg.model.resolved_served_name(),
                             console=console, monitor=monitor,
                             env_info=env_info, startup_seconds=startup_seconds,
                             skip_serve=opts.skip_serve)

            # ---- 逐套件执行 -------------------------------------------- #
            for cls in suites_for_config(cfg):
                suite_results.append(await run_suite(cls, ctx))
    except Exception as exc:  # noqa: BLE001 - 顶层兜底（服务失败/致命错误）
        if error is None:
            error = f"{type(exc).__name__}: {exc}"
        console.print(f"[red]运行中止: {error}[/red]")
    finally:
        # ---- 停服务（keep_alive / skip_serve / external 除外） ---------- #
        if manage_service and not opts.keep_alive:
            with _phase(monitor, "serve.stop"):
                await launcher.stop()
        elif manage_service and opts.keep_alive:
            console.print("[yellow]--keep-alive：服务保持运行，请手动停止"
                          "（mtest serve down 或 docker stop）[/yellow]")
        if monitor is not None:
            monitor.stop_and_join()

    total_duration = time.monotonic() - started
    ok = error is None and all(r.status == "passed" for r in suite_results) and bool(suite_results)

    # ---- 聚合报告 ---------------------------------------------------- #
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mtest_version": __version__,
        "total_duration_s": total_duration,
        "config": {
            "model": cfg.model.model_dump(),
            "serve": {"mode": cfg.serve.mode, "args": cfg.serve.args,
                      "docker": cfg.serve.docker.model_dump()},
            "client": cfg.client.model_dump(),
            "enabled_suites": [r.name for r in suite_results],
            "source_file": opts.config_path,
        },
        "serve": {
            "mode": cfg.serve.mode,
            "managed": manage_service,
            "startup_seconds": startup_seconds,
            "command_preview": cfg.serve.command or build_serve_command(cfg),
            "keep_alive": opts.keep_alive,
        },
        "environment": env_info,
        "suites": [r.to_dict() for r in suite_results],
        "npu": ({
            "interval_s": cfg.monitor.npu_interval,
            "degraded": monitor.degraded,
            "samples": len(monitor.samples),
            "summary": summarize_npu(monitor.samples),
            "csv": str(run_dir / "npu_samples.csv"),
        } if monitor is not None else None),
        "warnings": [*cfg.warnings, *(w for r in suite_results for w in r.warnings)],
        "error": error,
    }
    payload["conclusions"] = build_conclusions(payload)
    try:
        summary_path = write_reports(run_dir, payload)
        console.print(f"[bold green]报告已生成: {summary_path}[/bold green]")
    except Exception as exc:  # noqa: BLE001 - 报告失败不影响退出码语义
        summary_path = None
        console.print(f"[red]报告写出失败: {exc}[/red]")

    return RunSummary(run_id=run_id, run_dir=run_dir, ok=ok,
                      suite_results=suite_results, error=error,
                      summary_path=summary_path)


class _null:
    """无 monitor 时的空上下文。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _phase(monitor: NPUMonitor | None, name: str):
    return monitor.phase(name) if monitor is not None else _null()


def run_sync(cfg: BenchConfig, opts: RunOptions, console) -> RunSummary:
    """同步入口（CLI 用）。"""
    return asyncio.run(run_pipeline(cfg, opts, console))
