"""CLI 命令行入口（typer，设计 §7.2）。

```
mtest run    -c <yaml> [--suite perf,functional] [--skip-serve] [--keep-alive] [--dry-run]
mtest serve  up|down|status|logs -c <yaml>
mtest list
mtest validate -c <yaml>
mtest report show   <run_id>
mtest report compare <a> <b>
```
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from . import __version__
from .config import (BenchConfig, build_docker_command, build_process_command,
                     build_serve_command, load_config)
from .errors import ConfigError, MTestError
from .paths import models_config_dir, results_dir
from .pipeline import RunOptions, run_sync
from .report import compare_runs, load_metrics
from .serve.launcher import ServeController

app = typer.Typer(
    name="mtest",
    help="模型一键测试平台（Ascend 910B + vllm-ascend）：起服务 → 测试 → 报告。",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
serve_app = typer.Typer(help="服务单独管理（跨命令持久）", no_args_is_help=True)
report_app = typer.Typer(help="查看 / 对比历史运行报告", no_args_is_help=True)
app.add_typer(serve_app, name="serve")
app.add_typer(report_app, name="report")

console = Console()


def _load(ctx_config: Path) -> BenchConfig:
    try:
        return load_config(ctx_config)
    except (ConfigError, FileNotFoundError) as exc:
        console.print(f"[red]配置错误: {exc}[/red]")
        raise typer.Exit(code=2) from exc


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

@app.command()
def run(
    config: Path = typer.Option(..., "-c", "--config", exists=True, readable=True,
                                help="模型配置 yaml（configs/models/<model>.yaml）"),
    suite: str = typer.Option(None, "--suite",
                              help="只执行指定套件（逗号分隔）：perf,longctx,functional,embedding,ocr"),
    skip_serve: bool = typer.Option(False, "--skip-serve",
                                     help="不管理服务起停（复用已启动服务）"),
    keep_alive: bool = typer.Option(False, "--keep-alive", help="测试后保持服务运行"),
    dry_run: bool = typer.Option(False, "--dry-run", help="打印将执行的命令，不启动服务"),
    concurrency: str = typer.Option(None, "--concurrency", help="覆盖 perf 并发档（逗号分隔）"),
):
    """一键闭环：起服务 → 测试套件 → NPU 采样 → 报告 → 停服务。"""
    cfg = _load(config)
    opts = RunOptions(
        suites=[s.strip() for s in suite.split(",")] if suite else None,
        skip_serve=skip_serve,
        keep_alive=keep_alive,
        dry_run=dry_run,
        concurrency=[int(c) for c in concurrency.split(",")] if concurrency else None,
        config_path=str(config),
    )
    try:
        summary = run_sync(cfg, opts, console)
    except MTestError as exc:
        console.print(f"[red]运行失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if summary.dry_run:
        return
    if not summary.ok:
        console.print(f"[red]运行未完全成功（{summary.error or '存在失败套件'}），"
                      f"详见 {summary.run_dir}[/red]")
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #

@serve_app.command("up")
def serve_up(
    config: Path = typer.Option(..., "-c", "--config", exists=True),
):
    """按配置启动服务（CLI 退出后服务保持运行）。"""
    import asyncio

    cfg = _load(config)
    controller = ServeController(cfg, console)
    try:
        asyncio.run(controller.up())
    except MTestError as exc:
        console.print(f"[red]启动失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@serve_app.command("down")
def serve_down(
    config: Path = typer.Option(..., "-c", "--config", exists=True),
):
    """停止按配置启动的服务。"""
    import asyncio

    cfg = _load(config)
    asyncio.run(ServeController(cfg, console).down())


@serve_app.command("status")
def serve_status(
    config: Path = typer.Option(..., "-c", "--config", exists=True),
):
    """查看服务状态（记录 + 实体 + 健康检查）。"""
    import asyncio

    cfg = _load(config)
    st = asyncio.run(ServeController(cfg, console).status())
    if not st["recorded"]:
        console.print("[yellow]无服务状态记录[/yellow]")
        return
    table = Table(title=f"serve status: {cfg.model.name}")
    table.add_column("项")
    table.add_column("值")
    for key, val in st.items():
        table.add_row(key, str(val))
    console.print(table)


@serve_app.command("logs")
def serve_logs(
    config: Path = typer.Option(..., "-c", "--config", exists=True),
    tail: int = typer.Option(80, "--tail", min=1, max=2000, help="尾部行数"),
):
    """查看服务日志尾部。"""
    import asyncio

    cfg = _load(config)
    text = asyncio.run(ServeController(cfg, console).logs(tail))
    console.print(text)


# --------------------------------------------------------------------------- #
# list / validate
# --------------------------------------------------------------------------- #

@app.command("list")
def list_models():
    """列出可用模型配置。"""
    directory = models_config_dir()
    if not directory.is_dir():
        console.print(f"[red]模型配置目录不存在: {directory}[/red]")
        raise typer.Exit(code=2)
    table = Table(title=f"模型配置（{directory}）")
    for col in ("配置文件", "名称", "类型", "TP", "max-model-len", "端口", "套件", "状态"):
        table.add_column(col)
    found = False
    for path in sorted(directory.glob("*.y*ml")):
        if path.name.startswith("_"):
            continue
        found = True
        try:
            cfg = load_config(path)
            suites = ",".join(cfg.enabled_suites())
            table.add_row(path.name, cfg.model.name, cfg.model.type,
                          str(cfg.serve.tensor_parallel_size()),
                          str(cfg.serve.max_model_len() or "-"),
                          str(cfg.serve.port), suites, "[green]有效[/green]")
        except Exception as exc:  # noqa: BLE001 - 单个坏配置不阻断列表
            table.add_row(path.name, "-", "-", "-", "-", "-", "-", f"[red]无效: {exc}[/red]")
    if not found:
        console.print("[yellow]（无模型配置，复制 _template.yaml 开始）[/yellow]")
        return
    console.print(table)


@app.command()
def validate(
    config: Path = typer.Option(..., "-c", "--config", exists=True),
):
    """配置校验 + dry-run 打印生成的 vllm 命令。"""
    cfg = _load(config)

    console.print(f"[bold]模型[/bold]: {cfg.model.name}（{cfg.model.type}），"
                  f"served-name={cfg.model.resolved_served_name()}")
    console.print(f"[bold]服务[/bold]: mode={cfg.serve.mode} port={cfg.serve.port} "
                  f"startup_timeout={cfg.serve.startup_timeout}s")
    if cfg.serve.mode == "docker":
        console.print(f"[bold]docker 命令[/bold]:\n  {' '.join(build_docker_command(cfg))}")
    else:
        console.print(f"[bold]process 命令[/bold]:\n  {build_process_command(cfg, 'vllm.log')}")
    if cfg.serve.command:
        console.print("[yellow]serve.command 逃生舱生效（上述命令为整体替代命令）[/yellow]")
    console.print(f"[bold]客户端[/bold]: {cfg.client.base_url}")
    console.print(f"[bold]启用套件[/bold]: {cfg.enabled_suites()}")
    if cfg.warnings:
        for w in cfg.warnings:
            console.print(f"[yellow]警告: {w}[/yellow]")
    else:
        console.print("[green]校验通过，无警告[/green]")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

@report_app.command("show")
def report_show(
    run_id: str = typer.Argument(..., help="运行 ID（results/ 下目录名）"),
):
    """渲染查看一次运行的 summary.md。"""
    path = results_dir() / run_id / "summary.md"
    if not path.is_file():
        console.print(f"[red]未找到 {path}[/red]")
        raise typer.Exit(code=2)
    console.print(Markdown(path.read_text(encoding="utf-8")))


@report_app.command("compare")
def report_compare(
    a: str = typer.Argument(..., help="基准运行 ID"),
    b: str = typer.Argument(..., help="对比运行 ID"),
    out: Path = typer.Option(None, "--out", help="另存为 markdown 文件"),
):
    """关键指标并排 + 差异百分比。"""
    try:
        ma, mb = load_metrics(a), load_metrics(b)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    md = compare_runs(ma, mb)
    if out:
        out.write_text(md, encoding="utf-8")
        console.print(f"[green]已写出 {out}[/green]")
    console.print(Markdown(md))


@app.callback()
def _main(
    version: bool = typer.Option(False, "--version", help="显示版本"),
):
    if version:
        console.print(f"mtest {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
