"""闭环（closed-loop）负载发生器：perf / longctx / 稳定性类套件共用。

维持固定在途请求数（worker 数 = 并发数），一个完成立即补位；``duration`` 与
``num_requests`` 双上限，先到为准（停止发起新请求，在途请求自然收尾）。
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ..client import BenchClient, ChatResult
from ..metrics import ChatRecord

T = TypeVar("T")


def record_from_result(res: ChatResult, *, suite: str, input_len: int, concurrency: int,
                       round_idx: int, seq: int) -> ChatRecord:
    """ChatResult → ChatRecord；成功率口径：HTTP 200 且 finish_reason ∈ {stop,length}。

    error 保留完整错误串（如 ``http:400:This model's maximum context length...``），
    便于 longctx 自适应降档等模式识别；统计类别时按 ``:`` 前缀归并。
    """
    ok = res.ok and res.finish_reason in ("stop", "length")
    return ChatRecord(
        suite=suite,
        input_len=input_len,
        concurrency=concurrency,
        round_idx=round_idx,
        seq=seq,
        ok=ok,
        status=res.status,
        error=None if ok else (res.error or f"finish:{res.finish_reason}"),
        ttft=res.ttft,
        e2e=res.e2e,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        itl_mean=res.itl_mean,
        itl_max=res.itl_max,
        truncated=res.truncated,
    )


async def closed_loop_chat(
    client: BenchClient,
    model: str,
    messages: list[dict],
    *,
    suite: str,
    input_len: int,
    concurrency: int,
    duration_s: float,
    num_requests: int,
    round_idx: int = 0,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stream: bool = True,
    request_timeout: float | None = None,
) -> tuple[list[ChatRecord], float]:
    """闭环压测一格；返回 (记录列表, 墙钟秒)。"""
    records: list[ChatRecord] = []
    seq = itertools.count(1)
    started_at: list[float] = []
    stop_at = time.monotonic() + duration_s

    async def worker() -> None:
        while True:
            i = next(seq)
            if i > num_requests:
                return
            if time.monotonic() >= stop_at:
                return
            if not started_at:
                started_at.append(time.monotonic())
            res = await client.chat(model, messages, max_tokens=max_tokens,
                                    temperature=temperature, stream=stream,
                                    timeout=request_timeout)
            records.append(record_from_result(
                res, suite=suite, input_len=input_len, concurrency=concurrency,
                round_idx=round_idx, seq=i))

    workers = [asyncio.create_task(worker()) for _ in range(max(1, concurrency))]
    await asyncio.gather(*workers)
    wall = (time.monotonic() - started_at[0]) if started_at else 0.0
    return records, wall


async def warmup_chat(client: BenchClient, model: str, messages: list[dict], *,
                      count: int, max_tokens: int | None = None,
                      stream: bool = True) -> None:
    """预热请求（不记录；用短输出来兼顾预热效果与耗时）。"""
    for _ in range(max(0, count)):
        await client.chat(model, messages, max_tokens=max_tokens, stream=stream)


async def run_with_progress(coro_factory: Callable[[], Awaitable[T]], *,
                            console=None, label: str = "") -> T:
    """占位辅助：未来接入实时进度面板。当前直接执行。"""
    return await coro_factory()
