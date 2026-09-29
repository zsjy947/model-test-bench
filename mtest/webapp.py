"""Web 管理台（§13 扩展，dev 分支实现）。

复用 aiohttp + jinja2（已是核心依赖，零新增）的只读结果浏览器：

- ``GET /``            运行列表（run_id / 模型 / 套件状态 / 耗时）
- ``GET /run/{id}``    单次运行详情（summary.md + 关键指标）
- ``GET /api/runs``    运行列表 JSON
- ``GET /api/run/{id}`` metrics.json 原文
- ``GET /compare?a=&b=`` 并排对比（复用 report.compare_runs）

启动：``mtest web --host 127.0.0.1 --port 8765``。引擎库形态不变，
Web 台只是 results/ 目录的另一个消费者。
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

from aiohttp import web

from .paths import results_dir
from .report import compare_runs

# Whitelist for run ids used in path construction (blocks "/", "\", ".." path
# escapes). run ids are generated as "%Y%m%d-%H%M_<sanitized-model-name>"; the
# sanitized model name may legitimately contain dots (e.g. "qwen2.5-7b"), so
# dots are allowed after the first character — a leading dot (and thus "..")
# is still rejected.
_RUN_ID_RE = re.compile(r"[A-Za-z0-9_\-][A-Za-z0-9._\-]*")


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise web.HTTPBadRequest(text=f"invalid run_id: {run_id!r}")

_STYLE = """
body { font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 2rem; color: #1f2328; background: #fafbfc; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 1.5rem; }
table { border-collapse: collapse; margin: 0.8rem 0; background: #fff; }
th, td { border: 1px solid #d0d7de; padding: 6px 12px; font-size: 0.9rem; text-align: left; }
th { background: #f0f3f6; }
.pass { color: #1a7f37; font-weight: 600; } .fail { color: #cf222e; font-weight: 600; }
.warn { color: #9a6700; } pre { background: #0d1117; color: #e6edf3; padding: 1rem;
       border-radius: 6px; overflow-x: auto; font-size: 0.82rem; }
a { color: #0969da; text-decoration: none; } a:hover { text-decoration: underline; }
.meta { color: #656d76; font-size: 0.85rem; }
"""

_INDEX_TPL = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>mtest 管理台</title>
<style>{style}</style></head><body>
<h1>mtest 运行记录（{root}）</h1>
<p class="meta">{n} 次运行 · 生成于 {now}</p>
<table><tr><th>run_id</th><th>模型</th><th>类型</th><th>套件状态</th><th>总耗时</th><th>创建时间</th></tr>
{rows}
</table></body></html>"""

_RUN_TPL = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>{run_id} - mtest</title>
<style>{style}</style></head><body>
<h1>{run_id}</h1>
<p class="meta"><a href="/">← 返回列表</a> · 目录 {run_dir}</p>
<h2>summary.md</h2>
<pre>{summary}</pre>
<h2>metrics.json（节选：套件状态）</h2>
<table><tr><th>套件</th><th>状态</th><th>耗时</th></tr>{suite_rows}</table>
</body></html>"""

_COMPARE_TPL = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>对比 - mtest</title>
<style>{style}</style></head><body>
<h1>运行对比</h1>
<p class="meta"><a href="/">← 返回列表</a></p>
<pre>{md}</pre></body></html>"""


def scan_runs() -> list[dict]:
    """扫描 results/ 下所有含 metrics.json 的目录（新→旧）。"""
    root = results_dir()
    runs: list[dict] = []
    if not root.is_dir():
        return runs
    for run_dir in sorted(root.iterdir(), reverse=True):
        metrics_path = run_dir / "metrics.json"
        if not (run_dir.is_dir() and metrics_path.is_file()):
            continue
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        suites = data.get("suites", [])
        runs.append({
            "run_id": data.get("run_id", run_dir.name),
            "model": (data.get("config", {}).get("model", {}) or {}).get("name", "-"),
            "type": (data.get("config", {}).get("model", {}) or {}).get("type", "-"),
            "suites": [{"name": s.get("name"), "status": s.get("status"),
                        "duration_s": s.get("duration_s")} for s in suites],
            "all_passed": bool(suites) and all(s.get("status") == "passed" for s in suites),
            "total_duration_s": data.get("total_duration_s"),
            "created_at": data.get("created_at", ""),
            "dir": str(run_dir),
        })
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs


def _suite_rows_html(suites: list[dict]) -> str:
    rows = []
    for s in suites:
        css = "pass" if s.get("status") == "passed" else "fail"
        dur = s.get("duration_s")
        rows.append(f"<tr><td>{html.escape(str(s.get('name')))}</td>"
                    f"<td class='{css}'>{html.escape(str(s.get('status')))}</td>"
                    f"<td>{f'{dur:.0f}s' if dur is not None else '-'}</td></tr>")
    return "".join(rows)


def create_app() -> web.Application:
    async def index(_request: web.Request) -> web.Response:
        rows = []
        for r in scan_runs():
            css = "pass" if r["all_passed"] else "fail"
            suites_txt = ", ".join(
                f"{html.escape(str(s['name']))}:{html.escape(str(s['status']))}"
                for s in r["suites"]) or "-"
            dur = r.get("total_duration_s")
            rows.append(
                f"<tr><td><a href='/run/{r['run_id']}'>{html.escape(r['run_id'])}</a></td>"
                f"<td>{html.escape(str(r['model']))}</td><td>{html.escape(str(r['type']))}</td>"
                f"<td class='{css}'>{suites_txt}</td>"
                f"<td>{f'{dur / 60:.1f}min' if dur is not None else '-'}</td>"
                f"<td>{html.escape(str(r['created_at']))}</td></tr>")
        body = _INDEX_TPL.format(
            style=_STYLE, root=results_dir(), n=len(rows), rows="".join(rows),
            now=datetime.now().isoformat(timespec="seconds"))
        return web.Response(text=body, content_type="text/html")

    def _load_run(run_id: str) -> tuple[Path, dict] | None:
        _validate_run_id(run_id)
        run_dir = results_dir() / run_id
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            return None
        try:
            return run_dir, json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    async def run_detail(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        loaded = _load_run(run_id)
        if loaded is None:
            raise web.HTTPNotFound(text=f"run not found: {run_id}")
        run_dir, data = loaded
        summary_path = run_dir / "summary.md"
        summary = summary_path.read_text(encoding="utf-8") if summary_path.is_file() \
            else "(summary.md 不存在)"
        body = _RUN_TPL.format(style=_STYLE, run_id=html.escape(run_id),
                               run_dir=html.escape(str(run_dir)),
                               summary=html.escape(summary),
                               suite_rows=_suite_rows_html(data.get("suites", [])))
        return web.Response(text=body, content_type="text/html")

    async def api_runs(_request: web.Request) -> web.Response:
        return web.json_response(scan_runs())

    async def api_run(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        loaded = _load_run(run_id)
        if loaded is None:
            raise web.HTTPNotFound(text=f"run not found: {run_id}")
        return web.json_response(loaded[1])

    async def compare(request: web.Request) -> web.Response:
        a, b = request.query.get("a", ""), request.query.get("b", "")
        if not (a and b):
            raise web.HTTPBadRequest(text="need ?a=<run_id>&b=<run_id>")
        _validate_run_id(a)
        _validate_run_id(b)
        try:
            ma = json.loads((results_dir() / a / "metrics.json").read_text(encoding="utf-8"))
            mb = json.loads((results_dir() / b / "metrics.json").read_text(encoding="utf-8"))
            md = compare_runs(ma, mb)
        except (OSError, json.JSONDecodeError) as exc:
            raise web.HTTPBadRequest(text=f"读取运行记录失败: {exc}") from exc
        body = _COMPARE_TPL.format(style=_STYLE, md=html.escape(md))
        return web.Response(text=body, content_type="text/html")

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/run/{run_id}", run_detail)
    app.router.add_get("/api/runs", api_runs)
    app.router.add_get("/api/run/{run_id}", api_run)
    app.router.add_get("/compare", compare)
    return app


def run_web(host: str, port: int, console=None) -> None:
    if console is not None:
        console.print(f"[green]mtest web 管理台: http://{host}:{port}[/green]"
                      "（Ctrl-C 退出）")
    web.run_app(create_app(), host=host, port=port, print=None)
