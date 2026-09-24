"""stability 套件测试：窗口切分 / HBM 增长 / 判定阈值（fake client）。"""

import asyncio
import time
from pathlib import Path

from mtest.client import ChatResult
from mtest.config import BenchConfig, ModelCfg
from mtest.monitor import ChipSample, NPUSample
from mtest.suites.base import RunContext
from mtest.suites.stability import StabilitySuite, hbm_growth_mb


def test_hbm_growth():
    now = time.time()
    series = [(now + i * 10, 20000 + i * 100) for i in range(12)]  # 持续增长
    growth = hbm_growth_mb(series, now, now + 120, 30)
    assert growth > 0
    flat = [(now + i * 10, 20000.0) for i in range(12)]
    assert hbm_growth_mb(flat, now, now + 120, 30) == 0.0
    assert hbm_growth_mb([], now, now + 120, 30) is None


def test_slice_records():
    recs = list(range(100))
    segs = [StabilitySuite._slice_records(recs, i, 4) for i in range(4)]
    assert [len(s) for s in segs] == [25, 25, 25, 25]
    assert sum(segs, []) == recs


class FakeStabClient:
    """带 2ms 延迟的成功响应；fail_from 起注入 500（控制测试请求量）。"""

    def __init__(self, fail_from: int | None = None):
        self.calls = 0
        self.fail_from = fail_from

    async def chat(self, model, messages, *, max_tokens=None, temperature=None,
                   stream=True, timeout=None, **kw):
        self.calls += 1
        await asyncio.sleep(0.002)
        if self.fail_from is not None and self.calls >= self.fail_from:
            return ChatResult(ok=False, status=500, error="http:500:boom",
                              e2e=0.01)
        return ChatResult(ok=True, status=200, output="ok", finish_reason="stop",
                          prompt_tokens=50, completion_tokens=16,
                          ttft=0.01, e2e=0.05)


def _make_suite(tmp_path: Path, client, monitor=None) -> StabilitySuite:
    cfg = BenchConfig(model=ModelCfg(name="t", path="/m/t"))
    cfg.finalize()
    cfg.tests.stability.enabled = True
    cfg.tests.stability.duration_minutes = 0.1   # 6 秒（测试用）
    cfg.tests.stability.window_minutes = 0.1
    cfg.tests.stability.concurrency = 4
    ctx = RunContext(cfg=cfg, run_dir=tmp_path, client=client, served_model="t",
                     monitor=monitor)
    return StabilitySuite(ctx)


_PHASE = "stability[cc=4,in=1024]"


class FakeMonitor:
    """动态生成样本的 monitor 替身：按访问时刻对齐 6s 运行窗口。

    首窗口（now-5.5 / now-5.0）低 HBM，末窗口（now-2.5 / now-2.0）高 HBM。
    """

    def __init__(self, first_mb: float, last_mb: float):
        self.first_mb = first_mb
        self.last_mb = last_mb
        self._lock = _FakeLock()

    @property
    def samples(self):
        now = time.time()
        low = [ChipSample(npu=0, hbm_used_mb=self.first_mb, hbm_total_mb=65536)]
        high = [ChipSample(npu=0, hbm_used_mb=self.last_mb, hbm_total_mb=65536)]
        return [NPUSample(ts=now - 5.5, phase=_PHASE, chips=low),
                NPUSample(ts=now - 5.0, phase=_PHASE, chips=low),
                NPUSample(ts=now - 2.5, phase=_PHASE, chips=high),
                NPUSample(ts=now - 2.0, phase=_PHASE, chips=high)]

    def phase(self, label):
        from contextlib import nullcontext
        return nullcontext()


class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_stability_pass(tmp_path: Path):
    suite = _make_suite(tmp_path, FakeStabClient())
    result = asyncio.run(suite.run())
    assert result.status == "passed"
    m = result.metrics
    assert m["requests_ok"] if False else m["overall"]["ok"] > 0
    assert m["checks"]["throughput_decay_pct"]["pass"]
    assert m["checks"]["error_rate_pct"]["pass"]
    assert m["checks"]["hbm_growth_mb"]["pass"]  # 无 monitor → 不判定即 pass


def test_stability_error_rate_fail(tmp_path: Path):
    suite = _make_suite(tmp_path, FakeStabClient(fail_from=10))
    result = asyncio.run(suite.run())
    assert result.status == "failed"
    check = result.metrics["checks"]["error_rate_pct"]
    assert not check["pass"]
    assert check["value"] > 5.0


def test_stability_hbm_growth_detected(tmp_path: Path):
    # 运行窗口约 6s（duration_minutes=0.1）：首窗口 20000 → 末窗口 60000
    suite = _make_suite(tmp_path, FakeStabClient(), monitor=FakeMonitor(20000, 60000))
    result = asyncio.run(suite.run())
    check = result.metrics["checks"]["hbm_growth_mb"]
    assert check["value"] == 40000.0
    assert not check["pass"]
    assert result.status == "failed"


def test_stability_hbm_flat_passes(tmp_path: Path):
    suite = _make_suite(tmp_path, FakeStabClient(), monitor=FakeMonitor(30000, 30500))
    result = asyncio.run(suite.run())
    check = result.metrics["checks"]["hbm_growth_mb"]
    assert check["value"] == 500.0  # 低于阈值 2048
    assert check["pass"]
