"""统计口径与聚合工具（设计 §5.1 指标定义在此落地）。"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


def safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """线性插值分位数（q ∈ [0, 100]）；空序列返回 None。"""
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    rank = (len(vals) - 1) * (q / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(vals[int(rank)])
    frac = rank - lo
    return float(vals[lo] * (1 - frac) + vals[hi] * frac)


def dist_summary(values: Iterable[float]) -> dict | None:
    """延迟分布摘要：n / mean / std / min / max / p50 / p90 / p99。"""
    vals = [float(v) for v in values if v is not None and v >= 0]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
    }


def linreg_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """最小二乘斜率（稳定性趋势检测用）。"""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


# --------------------------------------------------------------------------- #
# 单请求记录（perf / longctx 共用）
# --------------------------------------------------------------------------- #

@dataclass
class ChatRecord:
    """一次 chat 请求的结果记录（含计时）。"""

    suite: str
    input_len: int
    concurrency: int
    round_idx: int
    seq: int
    ok: bool
    status: int | None = None
    error: str | None = None      # 错误类别（连接/超时/截断/HTTP/解析）
    ttft: float | None = None
    e2e: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    itl_mean: float | None = None
    itl_max: float | None = None
    truncated: bool = False

    def tpot(self) -> float | None:
        """TPOT = (e2e − TTFT) / (n_out − 1)。"""
        if self.e2e is None or self.ttft is None or not self.completion_tokens \
                or self.completion_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.completion_tokens - 1)

    def to_row(self) -> dict:
        d = asdict(self)
        d["tpot"] = self.tpot()
        return d


def records_to_rows(records: Iterable[ChatRecord]) -> list[dict]:
    return [r.to_row() for r in records]


def error_categories(records: Iterable[ChatRecord]) -> dict[str, int]:
    return dict(Counter(r.error for r in records if not r.ok))


def aggregate_chat_stats(records: Sequence[ChatRecord], wall_time: float) -> dict:
    """聚合一格（input_len × concurrency，可能跨 rounds）的指标。

    - rps = 成功请求数 / 墙钟；output_tps = 总输出 token / 墙钟
    - 延迟分位来自全部成功请求（跨 rounds 汇总）
    """
    ok = [r for r in records if r.ok]
    total_out = sum(r.completion_tokens or 0 for r in ok)
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    e2es = [r.e2e for r in ok if r.e2e is not None]
    tpots = [t for t in (r.tpot() for r in ok) if t is not None]
    itls = [r.itl_mean for r in ok if r.itl_mean is not None]
    prompt_total = sum(r.prompt_tokens or 0 for r in ok)
    return {
        "requests": len(records),
        "ok": len(ok),
        "success_rate": safe_div(len(ok), len(records)),
        "wall_s": wall_time,
        "rps": safe_div(len(ok), wall_time),
        "output_tps": safe_div(total_out, wall_time),
        "input_tps": safe_div(prompt_total, wall_time),
        "total_output_tokens": total_out,
        "ttft_s": dist_summary(ttfts),
        "e2e_s": dist_summary(e2es),
        "tpot_s": dist_summary(tpots),
        "itl_mean_s": dist_summary(itls),
        "errors": error_categories(records),
    }


def mean_of_cells(cells: Sequence[dict]) -> dict:
    """多轮（rounds）单元格均值合并：标量取均值，延迟分布合并重算分位。"""
    if not cells:
        return {}
    if len(cells) == 1:
        return cells[0]
    out: dict = {}
    keys_scalar = ("success_rate", "rps", "output_tps", "input_tps")
    for k in keys_scalar:
        vals = [c[k] for c in cells if c.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    out["requests"] = sum(c.get("requests", 0) for c in cells)
    out["ok"] = sum(c.get("ok", 0) for c in cells)
    out["wall_s"] = sum(c.get("wall_s", 0) for c in cells)
    out["total_output_tokens"] = sum(c.get("total_output_tokens", 0) for c in cells)
    for k in ("ttft_s", "e2e_s", "tpot_s", "itl_mean_s"):
        out[k] = dist_summary(_flatten_dist(cells, k))
    errors: Counter = Counter()
    for c in cells:
        errors.update(c.get("errors", {}))
    out["errors"] = dict(errors)
    return out


def _flatten_dist(cells: Sequence[dict], key: str) -> list[float]:
    """把多格延迟摘要近似还原为样本集（均值×n 展开，用于合并分位）。"""
    out: list[float] = []
    for c in cells:
        d = c.get(key)
        if not d:
            continue
        n = max(1, int(d.get("n", 1)))
        out.extend([d["mean"]] * n)
    return out


def write_csv(path: str | Path, rows: Sequence[dict]) -> Path:
    """明细 CSV 落盘（列取所有行的键并集，按首行排序）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return p
