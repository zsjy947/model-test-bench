"""多实例并行压测调度（§13 扩展，dev 分支实现）。

``mtest batch -c a.yaml -c b.yaml [--parallel N] [--auto-port]``：

- 顺序（默认 parallel=1）：逐个完整闭环，互不干扰
- 并行：asyncio.Semaphore 限流同时运行的 run 数
  - 端口冲突检查：process/docker 模式下重复 serve.port 直接报错，
    ``--auto-port`` 自动为冲突配置递增分配端口（同步改写 client.base_url）
  - 提示：并行时每个 run 各自采样 npu-smi（重复开销小）；同机资源竞争会
    反映在指标里，严谨对比请用顺序模式

汇总产物：``results/batch-<ts>/batch_summary.md``。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import BenchConfig
from .paths import results_dir
from .pipeline import RunOptions, RunSummary, run_pipeline


@dataclass
class BatchOutcome:
    config: str
    model: str
    run_id: str | None
    run_dir: str | None
    ok: bool
    error: str | None = None
    suites: list[dict] = field(default_factory=list)


def resolve_port_conflicts(cfgs: list[BenchConfig], paths: list[str],
                           *, auto_port: bool, console=None) -> None:
    """检查/修正并行运行的服务端口冲突（原地修改）。"""
    seen: dict[int, str] = {}
    for cfg, path in zip(cfgs, paths):
        if cfg.serve.mode == "external":
            continue  # external 不起服务
        port = cfg.serve.port
        if port in seen:
            if not auto_port:
                raise ValueError(
                    f"端口冲突：{path} 与 {seen[port]} 都使用 serve.port={port}；"
                    f"使用 --auto-port 自动分配或修改配置")
            new_port = port
            while new_port in seen or new_port <= 1024:
                new_port += 1
            cfg.serve.port = new_port
            # 同步 base_url（保持 host 与路径后缀 /v1）
            base = cfg.client.base_url.rstrip("/")
            suffix = "/v1" if base.endswith("/v1") else ""
            host_part = base[: -len(suffix)] if suffix else base
            # 替换尾部的 host:port
            if "://" in host_part:
                scheme, _, rest = host_part.partition("://")
                hostonly = rest.rsplit(":", 1)[0] if ":" in rest else rest
                cfg.client.base_url = f"{scheme}://{hostonly}:{new_port}{suffix}"
            if console is not None:
                console.print(f"[yellow]{path}: 端口 {port} 冲突，自动改用 {new_port}"
                              f"（client.base_url={cfg.client.base_url}）[/yellow]")
            seen[new_port] = path
        else:
            seen[port] = path


def _batch_summary_md(outcomes: list[BatchOutcome], parallel: int) -> str:
    lines = [f"# mtest batch 汇总", "",
             f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 配置数：{len(outcomes)}（并行度 {parallel}）", "",
             "| 配置 | 模型 | run_id | 结果 | 套件 | 错误 |",
             "|---|---|---|---|---|---|"]
    for o in outcomes:
        suites = ", ".join(f"{s['name']}:{s['status']}" for s in o.suites) or "-"
        err = (o.error or "").replace("|", "\\|")[:80]
        lines.append(f"| {Path(o.config).name} | {o.model} | {o.run_id or '-'} "
                     f"| {'✅ 通过' if o.ok else '❌ 未通过'} | {suites} | {err} |")
    n_ok = sum(1 for o in outcomes if o.ok)
    lines += ["", f"通过 {n_ok}/{len(outcomes)}。各 run 详情见对应 results/<run_id>/。"]
    return "\n".join(lines)


async def run_batch(config_paths: list[str], opts: RunOptions, console,
                    *, parallel: int = 1, auto_port: bool = False) -> int:
    """执行批量运行；返回进程退出码（0=全部通过）。"""
    from .config import load_config

    cfgs = [load_config(p) for p in config_paths]
    if parallel > 1:
        resolve_port_conflicts(cfgs, config_paths, auto_port=auto_port, console=console)

    semaphore = asyncio.Semaphore(max(1, parallel))
    outcomes: list[BatchOutcome | None] = [None] * len(cfgs)

    async def _run_one(idx: int, cfg: BenchConfig, path: str) -> None:
        async with semaphore:
            console.print(f"[bold cyan]══ batch[{idx + 1}/{len(cfgs)}] {path} 开始 ══[/bold cyan]")
            try:
                summary: RunSummary = await run_pipeline(cfg, opts, console)
                outcomes[idx] = BatchOutcome(
                    config=path, model=cfg.model.name, run_id=summary.run_id,
                    run_dir=str(summary.run_dir), ok=summary.ok,
                    error=summary.error,
                    suites=[{"name": r.name, "status": r.status}
                            for r in summary.suite_results])
            except Exception as exc:  # noqa: BLE001 - 单配置失败不阻断批次
                console.print(f"[red]batch[{path}] 失败: {exc}[/red]")
                outcomes[idx] = BatchOutcome(config=path, model=cfg.model.name,
                                             run_id=None, run_dir=None, ok=False,
                                             error=f"{type(exc).__name__}: {exc}")

    await asyncio.gather(*[_run_one(i, c, p) for i, (c, p)
                           in enumerate(zip(cfgs, config_paths))])

    final = [o for o in outcomes if o is not None]
    batch_dir = results_dir() / f"batch-{time.strftime('%Y%m%d-%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary_path = batch_dir / "batch_summary.md"
    summary_path.write_text(_batch_summary_md(final, parallel), encoding="utf-8")
    console.print(f"[bold green]batch 汇总: {summary_path}[/bold green]")

    return 0 if all(o.ok for o in final) else 1
