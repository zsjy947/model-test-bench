"""longctx 自适应降档测试（§12 R4）。"""

import asyncio
from pathlib import Path

from mtest.client import ChatResult
from mtest.config import BenchConfig, ModelCfg
from mtest.suites.base import RunContext
from mtest.suites.longctx import LongctxSuite, is_length_limit_error


def res(ok=True, error=None, status=200) -> ChatResult:
    return ChatResult(ok=ok, status=status, error=error, output="x", finish_reason="stop")


def test_length_limit_error_detection():
    assert is_length_limit_error(res(False, "http:400:This model's maximum context length is 32768 tokens"))
    assert is_length_limit_error(res(False, "http:400:prompt is too long"))
    assert is_length_limit_error(res(False, "http:500:torch out of memory"))
    assert is_length_limit_error(res(False, "http:400:exceeds the maximum number of tokens"))
    # 非长度类不触发
    assert not is_length_limit_error(res(False, "http:404:model not found"))
    assert not is_length_limit_error(res(False, "connect:Fail"))
    assert not is_length_limit_error(res(True))
    assert not is_length_limit_error(res(False, "timeout"))


class DegradeFakeClient:
    """前 fail_first 次**探测请求**（非校准的 max_tokens=1）返回长度超限 400。"""

    def __init__(self, fail_first: int, ratio: float = 0.5):
        self.fail_first = fail_first
        self.probes = 0
        self.ratio = ratio

    async def chat(self, model, messages, *, max_tokens=None, stream=False, **kw):
        text = messages[0]["content"]
        if max_tokens != 1:  # 显式探测
            self.probes += 1
            if self.probes <= self.fail_first:
                return res(False, "http:400:This model's maximum context length is 8192 tokens")
            return res(True)
        return ChatResult(ok=True, status=200, output="", finish_reason="length",
                          prompt_tokens=max(1, int(len(text) * self.ratio)),
                          completion_tokens=1)


def _make_suite(tmp_path: Path, client) -> LongctxSuite:
    cfg = BenchConfig(model=ModelCfg(name="t", path="/m/t"))
    cfg.finalize()
    cfg.tests.longctx.input_lens = [16384]
    ctx = RunContext(cfg=cfg, run_dir=tmp_path, client=client, served_model="t")
    return LongctxSuite(ctx)


def test_prepare_adaptive_degrades(tmp_path: Path):
    suite = _make_suite(tmp_path, DegradeFakeClient(fail_first=2))
    final, cal, degrades = asyncio.run(suite._prepare_adaptive(16384))
    # 16384 → 8192 → 4096（两次降档后探测通过）
    assert degrades == [16384, 8192]
    assert final == 4096
    assert cal.prompt_tokens is not None


def test_prepare_adaptive_no_degrade_on_success(tmp_path: Path):
    suite = _make_suite(tmp_path, DegradeFakeClient(fail_first=0))
    final, _cal, degrades = asyncio.run(suite._prepare_adaptive(16384))
    assert degrades == []
    assert final == 16384


def test_prepare_adaptive_caps_degrades(tmp_path: Path):
    suite = _make_suite(tmp_path, DegradeFakeClient(fail_first=99))
    final, _cal, degrades = asyncio.run(suite._prepare_adaptive(16384))
    assert len(degrades) == 2          # 最多降 2 次
    assert final == 4096
