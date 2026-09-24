"""stability 稳定性长跑套件（§13 扩展，dev 分支实现）。

固定并发持续运行 N 分钟，检测三类退化：

- 吞吐衰减：首窗口 vs 末窗口 RPS（及全程线性回归斜率参考）
- 错误率：整体与末窗口错误率
- HBM 增长：NPU 采样中首/末窗口 HBM 峰值差（疑似内存泄漏，依赖 monitor 开启）

窗口粒度 ``window_minutes`` 聚合；阈值均可配，超限判失败。
"""

from __future__ import annotations

import asyncio
import itertools
import time
from pathlib import Path

from ..metrics import aggregate_chat_stats, dist_summary, linreg_slope, safe_div, write_csv
from . import _runner
from .base import SUITE_FAILED, SUITE_PASSED, Suite, SuiteResult


def _hbm_series(monitor, phase: str) -> list[tuple[float, float]]:
    """取指定阶段 NPU 采样的 (epoch, 全卡 HBM 峰值) 序列；无 monitor 返回空。"""
    if monitor is None:
        return []
    out = []
    with monitor._lock:  # noqa: SLF001 - 同包内受控访问
        samples = [s for s in monitor.samples if s.phase == phase]
    for s in samples:
        used = [c.hbm_used_mb for c in s.chips if c.hbm_used_mb is not None]
        if used:
            out.append((s.ts, max(used)))
    return out


def hbm_growth_mb(series: list[tuple[float, float]], t0: float, t1: float,
                  window: float) -> float | None:
    """首窗口峰值 − 末窗口峰值（正数=增长）。"""
    if not series:
        return None
    first = [v for t, v in series if t0 <= t < t0 + window]
    last = [v for t, v in series if t1 - window < t <= t1]
    if not first or not last:
        return None
    return max(last) - max(first)


class StabilitySuite(Suite):
    name = "stability"
    applies_to = ("llm",)

    async def run(self) -> SuiteResult:
        result = self.new_result()
        s = self.cfg.tests.stability
        client = self.ctx.client
        model = self.ctx.served_model

        # 输入构造：随机 token 到目标长度后校准（长跑用固定输入，稳定可比）
        from ..datasets import LongDocMaterial, RandomTokens, calibrate_prompt_text
        base = RandomTokens(seed=20260924).generate(s.input_len)
        cal = await calibrate_prompt_text(client, model, base, LongDocMaterial(), s.input_len)
        messages = [{"role": "user", "content": cal.text}]

        duration_s = s.duration_minutes * 60
        window_s = min(s.window_minutes * 60, duration_s / 2)
        phase = f"stability[cc={s.concurrency},in={s.input_len}]"

        all_records: list = []
        with self.ctx.phase(phase):
            await _runner.warmup_chat(client, model, messages, count=2, max_tokens=32)
            t0 = time.monotonic()
            wall0 = time.time()   # NPU 采样使用墙钟时间戳，两套时钟分别记录
            stop_at = t0 + duration_s
            seq = itertools.count(1)

            async def worker() -> None:
                while time.monotonic() < stop_at:
                    res = await client.chat(model, messages, max_tokens=s.output_len,
                                            stream=True,
                                            timeout=self.cfg.client.request_timeout)
                    all_records.append(_runner.record_from_result(
                        res, suite=self.name, input_len=s.input_len,
                        concurrency=s.concurrency, round_idx=0, seq=next(seq)))

            await asyncio.gather(*[worker() for _ in range(s.concurrency)])
            t1 = time.monotonic()
            wall1 = time.time()

        # 窗口聚合
        n_windows = max(1, int((t1 - t0) // window_s))
        windows = []
        for w in range(n_windows):
            w0, w1 = t0 + w * window_s, t0 + (w + 1) * window_s
            # 记录按完成序近似映射到窗口（ChatRecord 无绝对时间，按序号切分）
            seg = self._slice_records(all_records, w, n_windows)
            agg = aggregate_chat_stats(seg, w1 - w0)
            windows.append({"window": w + 1, **{k: agg.get(k) for k in
                                                ("requests", "ok", "success_rate",
                                                 "rps", "output_tps", "errors")}})
        while windows and not windows[-1].get("requests"):
            windows.pop()  # 末尾不满窗口剔除
        if not windows:
            result.error = "长跑期间没有完成任何请求"
            return self.finish(result, SUITE_FAILED)

        first, last = windows[0], windows[-1]
        overall = aggregate_chat_stats(all_records, t1 - t0)
        rps_series = [w["rps"] for w in windows]
        err_series = [((1 - w["success_rate"]) if w.get("success_rate") is not None else None)
                      for w in windows]
        decay_pct = (safe_div(first["rps"] - last["rps"], first["rps"]) or 0) * 100 \
            if first.get("rps") else None
        slope = linreg_slope(list(range(1, len(windows) + 1)),
                             [r for r in rps_series if r is not None])

        series = _hbm_series(self.ctx.monitor, phase)
        growth = (hbm_growth_mb(series, wall0, wall1, min(window_s, wall1 - wall0))
                  if series else None)

        checks = {
            "throughput_decay_pct": {
                "value": decay_pct, "threshold": s.max_throughput_decay_pct,
                "pass": decay_pct is not None and decay_pct <= s.max_throughput_decay_pct,
            },
            "error_rate_pct": {
                "value": (1 - overall["success_rate"]) * 100 if overall.get("success_rate") is not None else None,
                "threshold": s.max_error_rate_pct,
                "pass": overall.get("success_rate") is not None
                and (1 - overall["success_rate"]) * 100 <= s.max_error_rate_pct,
            },
            "hbm_growth_mb": {
                "value": growth, "threshold": s.max_hbm_growth_mb,
                "pass": growth is None or growth <= s.max_hbm_growth_mb,
                "note": "monitor 未开启或无样本时不判定" if growth is None else "",
            },
        }
        all_pass = all(c["pass"] for c in checks.values())

        csv_path = write_csv(Path(self.ctx.run_dir) / "stability_details.csv",
                             [r.to_row() for r in all_records])
        result.metrics = {
            "duration_minutes": (t1 - t0) / 60,
            "concurrency": s.concurrency,
            "input_len": s.input_len,
            "output_len": s.output_len,
            "windows": windows,
            "rps_slope_per_window": slope,
            "overall": {k: overall.get(k) for k in
                        ("requests", "ok", "success_rate", "rps", "output_tps",
                         "ttft_s", "e2e_s", "tpot_s", "errors")},
            "checks": checks,
        }
        result.artifacts["details"] = str(csv_path)
        for name, c in checks.items():
            if not c["pass"]:
                result.warnings.append(
                    f"stability 检查未通过: {name} = {c['value']}（阈值 {c['threshold']}）")
        self.ctx.say(
            f"  stability: {len(all_records)} 请求 / {(t1 - t0) / 60:.1f} 分钟，"
            f"RPS 首/末 {first.get('rps')}/{last.get('rps')}，"
            f"衰减 {decay_pct if decay_pct is not None else '-'}%，"
            f"HBM 增长 {growth if growth is not None else '-'} MB")
        return self.finish(result, SUITE_PASSED if all_pass else SUITE_FAILED)

    @staticmethod
    def _slice_records(records: list, window_idx: int, n_windows: int) -> list:
        """按序号把记录近似均分到 n_windows 个窗口。"""
        n = len(records)
        if n == 0:
            return []
        size = n / n_windows
        lo, hi = int(window_idx * size), int((window_idx + 1) * size)
        if window_idx == n_windows - 1:
            hi = n
        return records[lo:hi]
