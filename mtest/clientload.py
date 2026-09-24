"""客户端侧负载采样（§12 风险 R5：压测端自身瓶颈监控）。

同机压测时客户端与服务端抢占 CPU，报告需记录客户端负载以便归因。
采样器注入 ``NPUMonitor.extra_samplers``，随 NPU 周期落 ``client_samples.csv``。

- loadavg：``os.getloadavg`` / ``/proc/loadavg``（POSIX；Windows 降级跳过）
- proc_cpu_pct：mtest 进程 CPU 占用（% 单核口径；psutil 可选增强，缺失时用
  ``os.times()`` 差分）
- threads：进程线程数
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

try:  # psutil 为可选增强，不进入依赖清单
    import psutil
except ImportError:  # pragma: no cover - 环境差异
    psutil = None  # type: ignore[assignment]


def loadavg_sampler() -> dict:
    def _sample() -> dict:
        try:
            one, five, _fifteen = os.getloadavg()
            return {"loadavg_1m": round(one, 3), "loadavg_5m": round(five, 3)}
        except (OSError, AttributeError):
            try:  # Linux 上 getloadavg 不可用时的兜底
                parts = open("/proc/loadavg", encoding="ascii").read().split()
                return {"loadavg_1m": float(parts[0]), "loadavg_5m": float(parts[1])}
            except (OSError, ValueError, IndexError):
                return {}
    return _sample


def proc_cpu_sampler() -> dict:
    """mtest 进程 CPU（% 单核口径，多线程可 >100）。"""
    if psutil is not None:
        proc = psutil.Process()

        def _sample() -> dict:
            try:
                return {"proc_cpu_pct": round(proc.cpu_percent(interval=None), 2)}
            except Exception:  # noqa: BLE001 - 进程退出等边缘
                return {}
        # 首次调用建立基线
        proc.cpu_percent(interval=None)
        return _sample

    last = {"t": None, "cpu": None}

    def _sample_os_times() -> dict:
        t = os.times()
        now = time.monotonic()
        out: dict = {}
        if last["t"] is not None:
            wall = now - last["t"]
            cpu = (t.user + t.system) - last["cpu"]
            if wall > 0:
                out["proc_cpu_pct"] = round(max(0.0, cpu) / wall * 100, 2)
        last["t"] = now
        last["cpu"] = t.user + t.system
        return out
    _sample_os_times()  # 建立基线
    return _sample_os_times


def threads_sampler() -> dict:
    def _sample() -> dict:
        return {"threads": threading.active_count()}
    return _sample


def make_client_samplers() -> list[Callable[[], dict]]:
    """构造本机可用的客户端采样器集合。"""
    samplers: list[Callable[[], dict]] = []
    if hasattr(os, "getloadavg") or os.path.exists("/proc/loadavg"):
        samplers.append(loadavg_sampler())
    samplers.append(proc_cpu_sampler())
    samplers.append(threads_sampler())
    return samplers
