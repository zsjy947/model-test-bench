"""NPU 监控：后台线程周期执行 ``npu-smi info`` 并解析（设计 §6）。

- 按卡记录：HBM 已用/总量、AICore 利用率、功耗、温度 → ``npu_samples.csv``
- 解析器容忍不同 CANN 版本格式差异（正则多形态匹配）
- 采样失败降级为告警、不阻断测试（连续失败标记 degraded，成功后自动恢复）
- 与压测时间轴对齐：套件执行写阶段标记（phase），报告输出各阶段 NPU 峰值/均值
"""

from __future__ import annotations

import csv
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .errors import NpuMonitorError

# 主表两行组：芯片行 A（健康/功耗/温度）+ 行 B（Bus-Id/AICore/HBM）
_ROW_A = re.compile(
    r"^\|\s*(?P<npu>\d+)\s+\S+\s+\|\s*\w+\s+\|\s*(?P<power>[\d.]+)\s+(?P<temp>\d+)\s"
)
_ROW_B = re.compile(
    r"^\|\s*(?P<npu>\d+)\s+\S+\s+\|\s*[\d:A-Fa-f.\-]+\s+\|\s*(?P<aicore>\d+)\s+"
    r"(?P<used>\d+)\s*/\s*(?P<total>\d+)\s"
)
# ``npu-smi info -t usages`` 简表形态
_ROW_USAGE = re.compile(
    r"^\s*(?P<npu>\d+)\s+(?P<aicore>\d+)\s+(?P<used>\d+)\s*/\s*(?P<total>\d+)\s*$"
)


@dataclass
class ChipSample:
    npu: int
    aicore_util: float | None = None
    hbm_used_mb: float | None = None
    hbm_total_mb: float | None = None
    power_w: float | None = None
    temp_c: float | None = None


@dataclass
class NPUSample:
    ts: float                      # unix 时间戳
    phase: str
    chips: list[ChipSample] = field(default_factory=list)


def parse_npu_smi_output(text: str) -> list[ChipSample]:
    """容忍式解析 ``npu-smi info`` 主表（或 -t usages 简表）。

    无法解析出任何芯片行时抛 :class:`NpuMonitorError`。
    """
    lines = text.splitlines()
    chips: list[ChipSample] = []
    i = 0
    while i < len(lines) - 1:
        ma = _ROW_A.match(lines[i])
        if ma:
            mb = _ROW_B.match(lines[i + 1])
            if mb and ma.group("npu") == mb.group("npu"):
                chips.append(ChipSample(
                    npu=int(ma.group("npu")),
                    aicore_util=float(mb.group("aicore")),
                    hbm_used_mb=float(mb.group("used")),
                    hbm_total_mb=float(mb.group("total")),
                    power_w=float(ma.group("power")),
                    temp_c=float(ma.group("temp")),
                ))
                i += 2
                continue
        i += 1
    if not chips:
        for line in lines:
            m = _ROW_USAGE.match(line)
            if m:
                chips.append(ChipSample(
                    npu=int(m.group("npu")),
                    aicore_util=float(m.group("aicore")),
                    hbm_used_mb=float(m.group("used")),
                    hbm_total_mb=float(m.group("total")),
                ))
    if not chips:
        raise NpuMonitorError(f"npu-smi 输出无法解析（{len(lines)} 行）")
    return chips


def collect_npu_env() -> dict[str, Any] | None:
    """一次性采集 NPU 环境信息（版本头 + 卡列表），失败返回 None。"""
    try:
        cp = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                            timeout=20, encoding="utf-8", errors="replace")
        if cp.returncode != 0:
            return None
        out = cp.stdout or ""
        header = next((ln.strip() for ln in out.splitlines() if "npu-smi" in ln), "")
        chips = parse_npu_smi_output(out)
        return {
            "npu_smi_header": header,
            "npu_count": len(chips),
            "hbm_total_mb": chips[0].hbm_total_mb if chips else None,
        }
    except (OSError, subprocess.SubprocessError, NpuMonitorError):
        return None


class NPUMonitor:
    """后台采样线程；``phase()`` 上下文管理器写入阶段标记。"""

    def __init__(self, interval: float = 5.0, csv_path: str | Path | None = None,
                 console=None, command: Sequence[str] = ("npu-smi", "info"),
                 extra_samplers: list[Callable[[], dict]] | None = None):
        self.interval = max(1.0, float(interval))
        self.csv_path = Path(csv_path) if csv_path else None
        self.console = console
        self.command = list(command)
        self.extra_samplers = extra_samplers or []
        self.samples: list[NPUSample] = []
        self.degraded = False
        self.failure_count = 0
        self._phase = "idle"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._csv_fh = None
        self._csv_writer = None

    # -- 生命周期 ------------------------------------------------------- #
    def start(self) -> None:
        if self._thread is not None:
            return
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_fh)
            self._csv_writer.writerow(
                ["timestamp", "epoch", "phase", "npu", "aicore_util_pct",
                 "hbm_used_mb", "hbm_total_mb", "power_w", "temp_c"])
        self._thread = threading.Thread(target=self._loop, name="npu-monitor", daemon=True)
        self._thread.start()

    def stop_and_join(self, timeout: float = 15.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def _sample_once(self) -> None:
        try:
            cp = subprocess.run(self.command, capture_output=True, text=True, timeout=15,
                                encoding="utf-8", errors="replace")
            chips = parse_npu_smi_output(cp.stdout or cp.stderr or "")
            with self._lock:
                phase = self._phase
            sample = NPUSample(ts=time.time(), phase=phase, chips=chips)
            extra_cols: dict[str, Any] = {}
            for sampler in self.extra_samplers:
                try:
                    extra_cols.update(sampler())
                except Exception:  # noqa: BLE001 - 附加采样器失败不影响主采样
                    pass
            with self._lock:
                self.samples.append(sample)
                self._last_extra = extra_cols
            if self._csv_writer is not None:
                ts_str = datetime.fromtimestamp(sample.ts).isoformat(timespec="seconds")
                for chip in chips:
                    self._csv_writer.writerow([
                        ts_str, f"{sample.ts:.1f}", phase, chip.npu,
                        chip.aicore_util, chip.hbm_used_mb, chip.hbm_total_mb,
                        chip.power_w, chip.temp_c])
                if self._csv_fh is not None:
                    self._csv_fh.flush()
            if self.degraded:
                self._say("[green]NPU 采样已恢复[/green]")
            self.degraded = False
            self.failure_count = 0
        except Exception as exc:  # noqa: BLE001 - 采样失败降级
            self.failure_count += 1
            if self.failure_count == 1 or self.failure_count % 10 == 0:
                self._say(f"[yellow]NPU 采样失败（第 {self.failure_count} 次，已降级）: "
                          f"{type(exc).__name__}: {exc}[/yellow]")
            if self.failure_count >= 3:
                self.degraded = True

    def _say(self, msg: str) -> None:
        if self.console is not None:
            self.console.print(msg)

    # -- 阶段标记 ------------------------------------------------------- #
    def set_phase(self, name: str) -> None:
        with self._lock:
            self._phase = name

    @contextmanager
    def phase(self, name: str):
        prev = self._phase
        self.set_phase(name)
        try:
            yield
        finally:
            self.set_phase(prev)

    @property
    def last_extra(self) -> dict[str, Any]:
        with self._lock:
            return getattr(self, "_last_extra", {})


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #

def summarize_npu(samples: Sequence[NPUSample]) -> dict[str, Any]:
    """按阶段汇总 NPU 指标（峰值/均值）；含 overall 全程汇总。"""
    if not samples:
        return {}
    phases: dict[str, list[NPUSample]] = {}
    for s in samples:
        phases.setdefault(s.phase, []).append(s)

    def _summarize(group: list[NPUSample]) -> dict[str, Any]:
        aicore = [c.aicore_util for s in group for c in s.chips if c.aicore_util is not None]
        hbm = [c.hbm_used_mb for s in group for c in s.chips if c.hbm_used_mb is not None]
        power = [c.power_w for s in group for c in s.chips if c.power_w is not None]
        temp = [c.temp_c for s in group for c in s.chips if c.temp_c is not None]
        total = next((c.hbm_total_mb for s in group for c in s.chips
                      if c.hbm_total_mb is not None), None)
        return {
            "n_samples": len(group),
            "npu_count": max((len(s.chips) for s in group), default=0),
            "hbm_total_mb": total,
            "hbm_used_max_mb": max(hbm) if hbm else None,
            "hbm_used_avg_mb": (sum(hbm) / len(hbm)) if hbm else None,
            "aicore_util_max": max(aicore) if aicore else None,
            "aicore_util_avg": (sum(aicore) / len(aicore)) if aicore else None,
            "power_max_w": max(power) if power else None,
            "temp_max_c": max(temp) if temp else None,
        }

    out = {phase: _summarize(group) for phase, group in sorted(phases.items())}
    out["overall"] = _summarize(list(samples))
    return out
