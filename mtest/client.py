"""异步压测客户端（aiohttp，OpenAI 兼容接口）。

自研而非包装 vllm benchmark_serving.py 的原因见设计 §8：指标口径、报告结构、
embedding/ocr 请求形态统一可控。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import aiohttp


@dataclass
class ChatResult:
    """一次 chat/completions 结果（流式含 chunk 级计时）。"""

    ok: bool
    status: int | None = None
    error: str | None = None            # 错误类别: connect/timeout/http/parse/truncated
    output: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft: float | None = None           # 发请求 → 首个内容 chunk
    e2e: float | None = None
    itl_mean: float | None = None       # 相邻内容 chunk 间隔均值
    itl_max: float | None = None
    content_chunks: int = 0
    truncated: bool = False             # 流异常中断（无 [DONE] / 无 usage）

    def error_kind(self) -> str:
        if self.ok:
            return ""
        if self.truncated:
            return "truncated"
        if self.status is not None and self.status >= 400:
            return "http"
        if self.error:
            kind = self.error.split(":", 1)[0]
            return kind if kind in ("connect", "timeout", "parse") else "error"
        return "error"


@dataclass
class EmbedResult:
    ok: bool
    status: int | None = None
    error: str | None = None
    vectors: list[list[float]] = field(default_factory=list)
    dim: int | None = None
    prompt_tokens: int | None = None
    e2e: float | None = None


class BenchClient:
    """OpenAI 兼容异步客户端（chat 流式/非流式 + embeddings）。

    压测端零重依赖：不加载 tokenizer，长度控制依赖 usage 反馈校准。
    """

    def __init__(self, base_url: str, *, api_key: str | None = None,
                 request_timeout: float = 600.0, max_concurrency: int = 256):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self._timeout = aiohttp.ClientTimeout(total=request_timeout, connect=30,
                                              sock_read=request_timeout)
        self._session: aiohttp.ClientSession | None = None
        self._max_concurrency = max_concurrency

    async def __aenter__(self) -> "BenchClient":
        await self.open()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def open(self) -> None:
        if self._session is None or self._session.closed:
            conn = aiohttp.TCPConnector(limit=max(256, self._max_concurrency),
                                        ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=self._timeout, connector=conn)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def request_health(self) -> tuple[bool, int | None]:
        from .serve import health

        assert self._session is not None
        try:
            async with self._session.get(
                f"{health.root_url(self.base_url)}/health", headers=self._headers()
            ) as resp:
                await resp.read()
                return 200 <= resp.status < 300, resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False, None

    async def list_models(self) -> list[str]:
        assert self._session is not None
        url = f"{self.base_url}/models"
        async with self._session.get(url, headers=self._headers()) as resp:
            payload = await resp.json(content_type=None)
        return [item.get("id") for item in payload.get("data", [])]

    # ------------------------------------------------------------------ #
    async def chat(
        self,
        model: str,
        messages: Sequence[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: Sequence[str] | None = None,
        stream: bool = False,
        include_usage: bool = True,
        extra_body: dict | None = None,
        timeout: float | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {"model": model, "messages": list(messages), "stream": stream}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop:
            payload["stop"] = list(stop)
        if stream and include_usage:
            payload.setdefault("stream_options", {})["include_usage"] = True
        if extra_body:
            for k, v in extra_body.items():
                payload[k] = v
        if stream:
            return await self._chat_stream(payload, timeout)
        return await self._chat_plain(payload, timeout)

    async def _chat_plain(self, payload: dict, timeout: float | None) -> ChatResult:
        t0 = time.perf_counter()
        try:
            assert self._session is not None
            async with self._session.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=timeout) if timeout else None,
            ) as resp:
                e2e = time.perf_counter() - t0
                if resp.status != 200:
                    return ChatResult(ok=False, status=resp.status, error=f"http:{resp.status}",
                                      e2e=e2e)
                data = await resp.json(content_type=None)
        except asyncio.TimeoutError:
            return ChatResult(ok=False, error="timeout", e2e=time.perf_counter() - t0)
        except aiohttp.ClientConnectionError as exc:
            return ChatResult(ok=False, error=f"connect:{type(exc).__name__}",
                              e2e=time.perf_counter() - t0)
        except aiohttp.ClientError as exc:
            return ChatResult(ok=False, error=f"error:{type(exc).__name__}",
                              e2e=time.perf_counter() - t0)
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            usage = data.get("usage") or {}
            return ChatResult(
                ok=True,
                status=200,
                output=msg.get("content") or "",
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                e2e=e2e,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return ChatResult(ok=False, status=200, error=f"parse:{exc}", e2e=e2e)

    async def _chat_stream(self, payload: dict, timeout: float | None) -> ChatResult:
        t0 = time.perf_counter()
        output_parts: list[str] = []
        gaps: list[float] = []
        finish_reason: str | None = None
        usage: dict | None = None
        first_content_at: float | None = None
        prev_content_at: float | None = None
        done = False
        status: int | None = None
        error: str | None = None
        try:
            assert self._session is not None
            async with self._session.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=timeout, sock_read=timeout) if timeout else None,
            ) as resp:
                status = resp.status
                if resp.status != 200:
                    return ChatResult(ok=False, status=resp.status,
                                      error=f"http:{resp.status}",
                                      e2e=time.perf_counter() - t0)
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        done = True
                        break
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError as exc:
                        error = f"parse:{exc}"
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            now = time.perf_counter()
                            if first_content_at is None:
                                first_content_at = now
                            elif prev_content_at is not None:
                                gaps.append(now - prev_content_at)
                            prev_content_at = now
                            output_parts.append(content)
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                    if isinstance(obj.get("usage"), dict) and obj["usage"]:
                        usage = obj["usage"]
        except asyncio.TimeoutError:
            return ChatResult(ok=False, status=status, error="timeout", truncated=True,
                              e2e=time.perf_counter() - t0,
                              output="".join(output_parts))
        except aiohttp.ClientConnectionError as exc:
            return ChatResult(ok=False, status=status, error=f"connect:{type(exc).__name__}",
                              truncated=True, e2e=time.perf_counter() - t0,
                              output="".join(output_parts))
        except aiohttp.ClientError as exc:
            return ChatResult(ok=False, status=status, error=f"error:{type(exc).__name__}",
                              truncated=True, e2e=time.perf_counter() - t0,
                              output="".join(output_parts))

        e2e = time.perf_counter() - t0
        truncated = not done
        if error and not output_parts:
            return ChatResult(ok=False, status=status, error=error, e2e=e2e)
        ok = done and error is None
        completion_tokens = (usage or {}).get("completion_tokens")
        if completion_tokens is None and ok:
            completion_tokens = len(output_parts)
        return ChatResult(
            ok=ok,
            status=status,
            error=None if ok else (error or "truncated"),
            output="".join(output_parts),
            finish_reason=finish_reason,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=completion_tokens,
            ttft=first_content_at - t0 if first_content_at is not None else None,
            e2e=e2e,
            itl_mean=(sum(gaps) / len(gaps)) if gaps else None,
            itl_max=max(gaps) if gaps else None,
            content_chunks=len(output_parts),
            truncated=truncated,
        )

    # ------------------------------------------------------------------ #
    async def embed(self, model: str, texts: Sequence[str], *,
                    timeout: float | None = None) -> EmbedResult:
        t0 = time.perf_counter()
        payload = {"model": model, "input": list(texts)}
        try:
            assert self._session is not None
            async with self._session.post(
                f"{self.base_url}/embeddings", json=payload, headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=timeout) if timeout else None,
            ) as resp:
                e2e = time.perf_counter() - t0
                if resp.status != 200:
                    body = (await resp.text())[:300]
                    return EmbedResult(ok=False, status=resp.status,
                                       error=f"http:{resp.status}", e2e=e2e)
                data = await resp.json(content_type=None)
        except asyncio.TimeoutError:
            return EmbedResult(ok=False, error="timeout", e2e=time.perf_counter() - t0)
        except aiohttp.ClientConnectionError as exc:
            return EmbedResult(ok=False, error=f"connect:{type(exc).__name__}",
                               e2e=time.perf_counter() - t0)
        except aiohttp.ClientError as exc:
            return EmbedResult(ok=False, error=f"error:{type(exc).__name__}",
                               e2e=time.perf_counter() - t0)
        try:
            vectors = [item["embedding"] for item in data["data"]]
            usage = data.get("usage") or {}
            return EmbedResult(ok=True, status=200, vectors=vectors,
                               dim=len(vectors[0]) if vectors else None,
                               prompt_tokens=usage.get("prompt_tokens"), e2e=e2e)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return EmbedResult(ok=False, status=200, error=f"parse:{exc}", e2e=e2e)
