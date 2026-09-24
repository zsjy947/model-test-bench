"""健康检查：轮询 ``GET /health``，就绪后校验 ``/v1/models`` 注册情况。"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import aiohttp

from ..errors import ServeError, ServeTimeoutError


def root_url(base_url: str) -> str:
    """从 ``.../v1`` 形式的 base_url 推导服务根 URL（/health 所在）。"""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def _headers(api_key: str | None) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def probe_health(base_url: str, *, api_key: str | None = None,
                       timeout: float = 5.0) -> tuple[bool, int | None]:
    """单次探测；返回 (是否 2xx, HTTP 状态码)。连接错误返回 (False, None)。"""
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(f"{root_url(base_url)}/health",
                                   headers=_headers(api_key)) as resp:
                await resp.read()
                return 200 <= resp.status < 300, resp.status
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        return False, None


async def wait_for_health(
    base_url: str,
    timeout_s: float,
    *,
    interval: float = 2.0,
    api_key: str | None = None,
    console=None,
    is_dead: Callable[[], bool] | None = None,
    dead_detail: str = "",
) -> float:
    """轮询直到就绪；返回耗时（秒）。

    - ``is_dead``：可选回调，进程/容器已退出时立即失败（避免傻等超时）。
    - 超时抛 :class:`ServeTimeoutError`。
    """
    started = time.monotonic()
    deadline = started + timeout_s
    last_report = 0.0
    while True:
        ok, _status = await probe_health(base_url, api_key=api_key)
        if ok:
            return time.monotonic() - started
        if is_dead is not None and is_dead():
            raise ServeError(f"服务进程提前退出：{dead_detail}")
        if time.monotonic() >= deadline:
            raise ServeTimeoutError(
                f"健康检查在 {timeout_s:.0f}s 内未就绪（{root_url(base_url)}/health）"
            )
        now = time.monotonic()
        if console is not None and now - last_report >= 30:
            last_report = now
            console.print(f"  … 等待服务就绪 {now - started:.0f}s / {timeout_s:.0f}s")
        await asyncio.sleep(interval)


async def fetch_model_names(base_url: str, *, api_key: str | None = None,
                            timeout: float = 10.0) -> list[str]:
    """拉取 ``/v1/models`` 的模型 ID 列表。"""
    url = base_url.rstrip("/") + "/models"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.get(url, headers=_headers(api_key)) as resp:
            if resp.status != 200:
                raise ServeError(f"GET {url} 返回 {resp.status}: {(await resp.text())[:200]}")
            payload = await resp.json(content_type=None)
    try:
        return [item["id"] for item in payload.get("data", [])]
    except (KeyError, TypeError) as exc:
        raise ServeError(f"/v1/models 响应格式异常: {payload!r:.200}") from exc
