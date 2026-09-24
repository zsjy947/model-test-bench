"""套件注册表与闭环负载发生器测试（fake client，无真实网络）。"""

import asyncio
from dataclasses import dataclass, field

from mtest.client import ChatResult
from mtest.config import BenchConfig, ModelCfg
from mtest.suites import registered_suites, register, run_suite, suites_for_config
from mtest.suites._runner import closed_loop_chat
from mtest.suites.base import RunContext, Suite, SuiteResult


@dataclass
class FakeClient:
    delay: float = 0.0
    fail_every: int = 0            # 每 N 个请求失败一次（0=从不）
    calls: int = 0
    chat_results: list = field(default_factory=list)

    async def chat(self, model, messages, *, max_tokens=None, temperature=None,
                   stream=False, timeout=None, **kw):
        self.calls += 1
        fail = self.fail_every and self.calls % self.fail_every == 0
        res = ChatResult(ok=not fail, status=200 if not fail else 500,
                         output="hi" * 4, finish_reason=None if fail else "stop",
                         error=None if not fail else "http:500",
                         prompt_tokens=10, completion_tokens=8,
                         ttft=0.01, e2e=0.05 + self.delay)
        self.chat_results.append(res)
        if self.delay:
            await asyncio.sleep(self.delay)
        return res


def test_registry_contains_builtins():
    reg = registered_suites()
    assert {"functional", "perf", "longctx", "embedding", "ocr"} <= set(reg)


def test_registry_extension_point():
    @register
    class AccuracySuite(Suite):
        name = "accuracy-test"
        applies_to = ("llm",)

        async def run(self) -> SuiteResult:
            r = self.new_result()
            return self.finish(r)

    assert "accuracy-test" in registered_suites()
    cfg = BenchConfig(model=ModelCfg(name="t", path="/m/t"))
    cfg.finalize()
    names = [c.name for c in suites_for_config(cfg)]
    assert "accuracy-test" in names  # 无对应 tests.<name> 配置时默认启用


def test_closed_loop_respects_num_requests():
    client = FakeClient()
    records, wall = asyncio.run(closed_loop_chat(
        client, "m", [{"role": "user", "content": "x"}],
        suite="perf", input_len=128, concurrency=8,
        duration_s=30, num_requests=50, max_tokens=16))
    assert len(records) == 50
    assert client.calls == 50
    assert wall >= 0
    assert all(r.ok for r in records)


def test_closed_loop_respects_duration():
    client = FakeClient(delay=0.05)
    records, wall = asyncio.run(closed_loop_chat(
        client, "m", [{"role": "user", "content": "x"}],
        suite="perf", input_len=128, concurrency=4,
        duration_s=0.3, num_requests=10000, max_tokens=16))
    # 0.3s 内 4 并发 × ~0.05s/请求 ≈ 24 个，远小于 10000
    assert len(records) < 100
    assert wall < 2.0


def test_closed_loop_failure_recording():
    client = FakeClient(fail_every=3)
    records, _ = asyncio.run(closed_loop_chat(
        client, "m", [{"role": "user", "content": "x"}],
        suite="perf", input_len=1, concurrency=2, duration_s=5, num_requests=9))
    failed = [r for r in records if not r.ok]
    assert len(failed) == 3
    assert failed[0].error == "http"


def test_run_suite_error_wrapped(tmp_path):
    class BoomSuite(Suite):
        name = "boom"
        applies_to = ("llm",)

        async def run(self):
            raise RuntimeError("炸了")

    cfg = BenchConfig(model=ModelCfg(name="t", path="/m/t"))
    ctx = RunContext(cfg=cfg, run_dir=tmp_path, client=FakeClient(), served_model="t")
    result = asyncio.run(run_suite(BoomSuite, ctx))
    assert result.status == "error"
    assert "炸了" in result.error
