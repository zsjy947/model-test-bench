"""报告层：metrics.json（schema 固定）+ summary.md（人读摘要）+ compare。

原始数据全量落 JSON（供 compare 与二次分析），人读摘要落 Markdown。
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from . import __version__
from .metrics import safe_div

_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# 环境信息
# --------------------------------------------------------------------------- #

def _pip_show_version(pkg: str) -> str | None:
    try:
        cp = subprocess.run([sys.executable, "-m", "pip", "show", pkg],
                            capture_output=True, text=True, timeout=30,
                            encoding="utf-8", errors="replace")
        for line in (cp.stdout or "").splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


async def collect_environment(client=None) -> dict[str, Any]:
    """机型 / Python / vllm 与 vllm-ascend 版本 / NPU 概要。"""
    from .monitor import collect_npu_env

    env: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "mtest_version": __version__,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    env["vllm_version_pip"] = _pip_show_version("vllm")
    env["vllm_ascend_version_pip"] = _pip_show_version("vllm-ascend")
    if client is not None:
        env["server_version"] = await _fetch_server_version(client)
    env["npu"] = collect_npu_env()
    return env


async def _fetch_server_version(client) -> str | None:
    from .serve.health import root_url

    try:
        import aiohttp

        url = f"{root_url(client.base_url)}/version"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    payload = await resp.json(content_type=None)
                    v = payload.get("version") if isinstance(payload, dict) else None
                    return str(v) if v else None
    except Exception:  # noqa: BLE001 - 版本探测失败不影响主流程
        return None
    return None


# --------------------------------------------------------------------------- #
# 渲染辅助
# --------------------------------------------------------------------------- #

def _fmt(v, unit: str = "", nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


def _pct(v) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def _dist(d: dict | None, key: str) -> str:
    if not d:
        return "-"
    return _fmt(d.get(key), "s")


def _kv_table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| 项 | 值 |", "|---|---|"]
    lines += [f"| {k} | {v if v is not None else '-'} |" for k, v in rows]
    return "\n".join(lines)


def _perf_section(result: dict) -> str:
    m = result.get("metrics", {})
    matrix = m.get("matrix", {})
    if not matrix:
        return "（无数据）"
    lines = ["| input_len | 并发 | 成功率 | RPS | 输出 tok/s | TTFT p50 | TTFT p99 | "
             "TPOT | e2e p99 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(matrix, key=_matrix_key):
        c = matrix[key]
        t = c.get("ttft_s") or {}
        e = c.get("e2e_s") or {}
        p = c.get("tpot_s") or {}
        parts = key.split("x")
        lines.append(
            f"| {parts[0]} | {parts[1]} | {_pct(c.get('success_rate'))} "
            f"| {_fmt(c.get('rps'), nd=2)} | {_fmt(c.get('output_tps'), nd=1)} "
            f"| {_dist(t, 'p50')} | {_dist(t, 'p99')} "
            f"| {_dist(p, 'mean')} | {_dist(e, 'p99')} |")
    text = "\n".join(lines)
    cal = m.get("calibration") or {}
    if cal:
        text += "\n\n输入长度校准（目标/实际 prompt tokens）：\n\n" + _kv_table(
            [(k, f"{v.get('target_tokens')} / {v.get('actual_prompt_tokens')}"
              f"（{'✓' if v.get('calibrated') else '未达标'}）")
             for k, v in cal.items()])
    return text


def _longctx_section(result: dict) -> str:
    m = result.get("metrics", {})
    matrix = m.get("matrix", {})
    if not matrix:
        return "（无数据）"
    lines = ["| input_len | 并发 | 成功率 | TTFT p50 | TTFT p99 | prefill tok/s | "
             "decode tok/s | e2e p99 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(matrix, key=_matrix_key):
        c = matrix[key]
        t = c.get("ttft_s") or {}
        e = c.get("e2e_s") or {}
        parts = key.split("x")
        lines.append(
            f"| {parts[0]} | {parts[1]} | {_pct(c.get('success_rate'))} "
            f"| {_dist(t, 'p50')} | {_dist(t, 'p99')} "
            f"| {_fmt(c.get('prefill_tps_mean'), nd=1)} | {_fmt(c.get('decode_tps_mean'), nd=1)} "
            f"| {_dist(e, 'p99')} |")
    return "\n".join(lines)


def _embedding_section(result: dict) -> str:
    m = result.get("metrics", {})
    matrix = m.get("matrix", {})
    q = m.get("quality", {})
    lines = ["| batch | 并发 | 成功率 | sent/s | 延迟 p50 | 延迟 p99 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(matrix, key=_matrix_key):
        c = matrix[key]
        lat = c.get("latency_s") or {}
        lines.append(
            f"| {c.get('batch')} | {c.get('concurrency')} | {_pct(c.get('success_rate'))} "
            f"| {_fmt(c.get('sent_per_s'), nd=1)} | {_dist(lat, 'p50')} | {_dist(lat, 'p99')} |")
    text = "\n".join(lines)
    disc = q.get("discrimination") or {}
    if disc.get("status") == "done":
        sim, uns = disc.get("similar") or {}, disc.get("unrelated") or {}
        text += ("\n\n区分度 sanity（两组分布）：\n\n"
                 f"- 相似对 cosine：n={sim.get('n')} mean={_fmt(sim.get('mean'), nd=4)} "
                 f"p50={_fmt(sim.get('p50'), nd=4)} min={_fmt(sim.get('min'), nd=4)}\n"
                 f"- 无关对 cosine：n={uns.get('n')} mean={_fmt(uns.get('mean'), nd=4)} "
                 f"p50={_fmt(uns.get('p50'), nd=4)} max={_fmt(uns.get('max'), nd=4)}\n"
                 f"- 间隔（similar−unrelated）：{_fmt(disc.get('separation'), nd=4)}"
                 f"（阈值 {disc.get('min_separation')}，"
                 f"{'达标' if disc.get('pass') else '未达标（sanity 参考）'}）\n")
    text += (f"\n\n维度：实际 {q.get('dim')} / 声明 {q.get('dim_declared') or '-'}；"
             f"同文本复现 cosine：{_fmt(q.get('repeat_consistency_cosine'), nd=8)}")
    return text


def _ocr_section(result: dict) -> str:
    m = result.get("metrics", {})
    matrix = m.get("matrix", {})
    q = m.get("quality", {})
    lines = ["| 并发 | 成功率 | img/s | 延迟 p50 | 延迟 p99 |", "|---:|---:|---:|---:|---:|"]
    for key in sorted(matrix, key=_matrix_key):
        c = matrix[key]
        lat = c.get("latency_s") or {}
        lines.append(
            f"| {c.get('concurrency')} | {_pct(c.get('success_rate'))} "
            f"| {_fmt(c.get('img_per_s'), nd=2)} | {_dist(lat, 'p50')} | {_dist(lat, 'p99')} |")
    text = "\n".join(lines)
    cer_s = q.get("cer_summary")
    if cer_s:
        text += (f"\n\nCER（有标注文件）：n={cer_s.get('n')} mean={_fmt(cer_s.get('mean'), nd=4)} "
                 f"p50={_fmt(cer_s.get('p50'), nd=4)} max={_fmt(cer_s.get('max'), nd=4)}\n")
        rows = ["| 图片 | 分辨率档 | CER |", "|---|---|---:|"]
        for f in q.get("files", []):
            if f.get("cer") is not None:
                rows.append(f"| {f['image']} | {f['bucket']} | {_fmt(f['cer'], nd=4)} |")
        text += "\n" + "\n".join(rows)
    for fc in m.get("fault_cases", []) or []:
        text += (f"\n\n容错[{fc.get('case')}]：graceful={fc.get('graceful')} "
                 f"status={fc.get('status')} 服务存活={fc.get('service_alive_after')}")
    return text


def _functional_section(result: dict) -> str:
    details = result.get("details", [])
    if not details:
        return "（无用例）"
    lines = ["| 用例 | 类型 | 结果 | 备注 |", "|---|---|---|---|"]
    for row in details:
        mark = "✅ pass" if row.get("status") == "passed" else f"❌ {row.get('status')}"
        note = (row.get("note") or "").replace("|", "\\|")[:120]
        lines.append(f"| {row['id']} | {row['type']} | {mark} | {note} |")
    return "\n".join(lines)


_SECTIONS = {
    "perf": ("perf 性能压测（矩阵：input_len × 并发）", _perf_section),
    "longctx": ("longctx 长序列专项", _longctx_section),
    "embedding": ("embedding 专项", _embedding_section),
    "ocr": ("ocr 专项", _ocr_section),
    "functional": ("functional 功能冒烟", _functional_section),
}


def _matrix_key(key: str):
    try:
        a, b = key.replace("b", "").replace("c", "").split("x")
        return int(a), int(b)
    except ValueError:
        return (0, 0)


def build_conclusions(payload: dict) -> list[str]:
    """规则式结论与建议。"""
    out: list[str] = []
    suites = payload.get("suites", [])
    for s in suites:
        if s.get("status") == "error":
            out.append(f"套件 {s['name']} 执行异常：{s.get('error')}，检查服务日志与配置")
        elif s.get("status") == "failed":
            out.append(f"套件 {s['name']} 未通过，详见上方明细与告警")
    perf = next((s for s in suites if s["name"] == "perf"), None)
    if perf:
        matrix = perf.get("metrics", {}).get("matrix", {})
        worst = None
        for key, c in matrix.items():
            sr = c.get("success_rate")
            if sr is not None and (worst is None or sr < worst[1]):
                worst = (key, sr)
        if worst and worst[1] < 0.99:
            out.append(f"perf 最差成功率 {worst[1]:.1%}（{worst[0]}），关注错误类别"
                       f"{[c.get('errors') for k, c in matrix.items() if k == worst[0]]}")
        hi_cc = [(k, c) for k, c in matrix.items() if int(k.split('x')[-1]) >= 8]
        for k, c in hi_cc:
            ttft = (c.get("ttft_s") or {}).get("p99")
            if ttft and ttft > 30:
                out.append(f"高并发档 {k} TTFT p99 {ttft:.1f}s，调度排队明显，"
                           "可评估是否需调整 max-num-seqs / gpu-memory-utilization")
                break
    npu = (payload.get("npu") or {}).get("summary", {}).get("overall")
    if npu and npu.get("hbm_used_max_mb") and npu.get("hbm_total_mb"):
        usage = npu["hbm_used_max_mb"] / npu["hbm_total_mb"]
        if usage > 0.95:
            out.append(f"NPU HBM 峰值占用 {usage:.0%}，接近上限，长序列/高并发注意 OOM")
    client = (payload.get("npu") or {}).get("client", {})
    cpu_count = payload.get("environment", {}).get("cpu_count")
    load_max = client.get("loadavg_1m_max")
    if load_max is not None and cpu_count and load_max > cpu_count * 0.8:
        out.append(f"客户端 1m 负载峰值 {load_max:.1f}（CPU {cpu_count} 核的 "
                   f"{load_max / cpu_count:.0%}），压测端可能接近瓶颈，"
                   "高并发档指标存疑（可换机压测验证）")
    proc_cpu = client.get("proc_cpu_pct_max")
    if proc_cpu is not None and proc_cpu > 90:
        out.append(f"客户端进程 CPU 峰值 {proc_cpu:.0f}%（单核口径），"
                   "SSE 解析占用偏高，关注压测端瓶颈")
    if payload.get("environment", {}).get("npu"):
        out.append(f"NPU：{payload['environment']['npu'].get('npu_count')} 卡，"
                   f"单卡 HBM {payload['environment']['npu'].get('hbm_total_mb')} MB")
    if not out:
        out.append("各启用套件均通过，未发现明显异常")
    return out


# --------------------------------------------------------------------------- #
# 写出与对比
# --------------------------------------------------------------------------- #

def write_reports(run_dir: str | Path, payload: dict) -> Path:
    """写 metrics.json + summary.md；返回 summary 路径。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload.setdefault("schema_version", _SCHEMA_VERSION)
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary = render_summary(payload)
    path = run_dir / "summary.md"
    path.write_text(summary, encoding="utf-8")
    return path


def render_summary(payload: dict) -> str:
    env = Environment(loader=PackageLoader("mtest", "templates"),
                      autoescape=select_autoescape([]), trim_blocks=True,
                      lstrip_blocks=True, keep_trailing_newline=True)
    template = env.get_template("summary.md.j2")

    model = payload.get("config", {}).get("model", {})
    suites = payload.get("suites", [])
    sections = []
    for s in suites:
        title, builder = _SECTIONS.get(s["name"], (s["name"], lambda r: "（未适配渲染）"))
        sections.append({"name": s["name"], "title": title,
                         "status": s.get("status"), "md": builder(s).rstrip() + "\n"})
    npu_summary = (payload.get("npu") or {}).get("summary") or {}
    npu_rows = []
    for phase, agg in sorted(npu_summary.items()):
        npu_rows.append({
            "phase": phase,
            "aicore_max": _fmt(agg.get("aicore_util_max"), "%", 1),
            "hbm_max": _fmt(agg.get("hbm_used_max_mb"), " MB", 0),
            "power_max": _fmt(agg.get("power_max_w"), " W", 1),
            "temp_max": _fmt(agg.get("temp_max_c"), " ℃", 0),
        })
    return template.render(payload=payload, model=model, suites=suites,
                           sections=sections, npu_rows=npu_rows,
                           conclusions=payload.get("conclusions", []),
                           fmt=_fmt, pct=_pct, env_info=payload.get("environment", {}))


def load_metrics(run_id: str, results_root: str | Path | None = None) -> dict:
    from .paths import results_dir

    root = Path(results_root) if results_root else results_dir()
    path = root / run_id / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"未找到运行记录: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _diff_pct(a, b) -> str:
    if a is None or b is None or a == 0:
        return "-"
    delta = safe_div(b - a, abs(a))
    return "-" if delta is None else f"{delta * 100:+.1f}%"


def compare_runs(a: dict, b: dict) -> str:
    """两份 metrics.json 关键指标并排 + 差异百分比（Markdown）。"""
    ra = a.get("run_id"), b.get("run_id")
    lines = [f"# 对比：{ra[0]} vs {ra[1]}", ""]

    def get_suite(payload, name):
        return next((s for s in payload.get("suites", []) if s.get("name") == name), None)

    for suite_name, columns in (
        ("perf", [("rps", "RPS"), ("output_tps", "输出 tok/s")]),
        ("embedding", [("sent_per_s", "sent/s")]),
        ("ocr", [("img_per_s", "img/s")]),
    ):
        sa, sb = get_suite(a, suite_name), get_suite(b, suite_name)
        if not sa or not sb:
            continue
        ma = sa.get("metrics", {}).get("matrix", {})
        mb = sb.get("metrics", {}).get("matrix", {})
        common = sorted(set(ma) & set(mb), key=_matrix_key)
        if not common:
            continue
        lines.append(f"## {suite_name}")
        header = "| 档位 | 指标 | A | B | 差异 |"
        lines += [header, "|---|---|---:|---:|---:|"]
        for key in common:
            for col, label in columns:
                va, vb = ma[key].get(col), mb[key].get(col)
                lines.append(f"| {key} | {label} | {_fmt(va, nd=2)} | {_fmt(vb, nd=2)} "
                             f"| {_diff_pct(va, vb)} |")
        # TTFT p99 对比（三类套件通用延迟口径）
        lat_key = {"perf": "ttft_s", "embedding": "latency_s", "ocr": "latency_s"}[suite_name]
        lines += ["", "| 档位 | 延迟(A p99) | 延迟(B p99) | 差异 |", "|---|---:|---:|---:|"]
        for key in common:
            da = (ma[key].get(lat_key) or {}).get("p99")
            db = (mb[key].get(lat_key) or {}).get("p99")
            lines.append(f"| {key} | {_fmt(da, 's')} | {_fmt(db, 's')} | {_diff_pct(da, db)} |")
        lines.append("")
    if len(lines) <= 2:
        lines.append("无可对比的公共指标（两次运行需包含相同套件与档位）")
    return "\n".join(lines)
