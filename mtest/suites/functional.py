"""functional 功能冒烟套件（全模型类型，设计 §5.5）。

用例定义在 ``data/cases/*.yaml``（结构化断言，报告逐条 pass/fail），支持追加
自定义 yaml 用例。用例类型：

- ``chat``：单请求 + expect 断言（可选 repeats 一致性）
- ``stream_consistency``：流式与非流式输出一致性
- ``embedding_basic``：维度 / 非空 / 重复编码一致性
- ``ocr_basic``：图片请求 + 输出非空 / 无乱码 / CER（有 GT 时）
- ``ocr_fault``：损坏/超大图片容错（期望 4xx 或优雅报错而非服务崩溃）

expect 支持的键见 :func:`evaluate_chat_expect` / 各类型 runner。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..client import ChatResult
from ..datasets import LongDocMaterial, detect_template_residue
from ..paths import resolve_data
from .base import SUITE_FAILED, SUITE_PASSED, Suite, SuiteResult

_CONTROL_CHARS = set(chr(c) for c in range(32)) - {"\n", "\t", "\r"}


@dataclass
class CheckOutcome:
    name: str
    ok: bool
    note: str = ""


def mojibake_ratio(text: str) -> float:
    """乱码启发式：替换符/控制字符占比。"""
    if not text:
        return 0.0
    bad = sum(1 for ch in text if ch == "\ufffd" or ch in _CONTROL_CHARS)
    return bad / len(text)


# --------------------------------------------------------------------------- #
# 断言引擎
# --------------------------------------------------------------------------- #

def _as_list(v) -> list:
    return v if isinstance(v, list) else [v]


def evaluate_chat_expect(expect: dict, results: list[ChatResult]) -> list[CheckOutcome]:
    """对 chat 结果列表执行结构化断言。"""
    checks: list[CheckOutcome] = []
    res = results[0]

    if "status" in expect:
        want = _as_list(expect["status"])
        checks.append(CheckOutcome("status", res.status in want, f"got {res.status}"))

    if "status_lte" in expect:
        ok = res.status is not None and res.status <= expect["status_lte"]
        checks.append(CheckOutcome("status_lte", ok,
                                   f"got {res.status} <= {expect['status_lte']}"))

    if "finish_reason" in expect:
        want = _as_list(expect["finish_reason"])
        checks.append(CheckOutcome("finish_reason", res.finish_reason in want,
                                   f"got {res.finish_reason!r}"))

    if "non_empty" in expect:
        ok = bool((res.output or "").strip()) == bool(expect["non_empty"])
        checks.append(CheckOutcome("non_empty", ok, f"len={len(res.output or '')}"))

    if "contains" in expect:
        missing = [s for s in _as_list(expect["contains"]) if s not in (res.output or "")]
        checks.append(CheckOutcome("contains", not missing, f"missing={missing}"))

    if "not_contains" in expect:
        found = [s for s in _as_list(expect["not_contains"]) if s in (res.output or "")]
        checks.append(CheckOutcome("not_contains", not found, f"found={found}"))

    if expect.get("no_template_residue"):
        hits = detect_template_residue(res.output or "")
        checks.append(CheckOutcome("no_template_residue", not hits, f"residue={hits}"))

    if "completion_tokens_lte" in expect:
        n = res.completion_tokens
        ok = n is not None and n <= expect["completion_tokens_lte"]
        checks.append(CheckOutcome("completion_tokens_lte", ok,
                                   f"got {n} <= {expect['completion_tokens_lte']}"))

    if "completion_tokens_gte" in expect:
        n = res.completion_tokens
        ok = n is not None and n >= expect["completion_tokens_gte"]
        checks.append(CheckOutcome("completion_tokens_gte", ok,
                                   f"got {n} >= {expect['completion_tokens_gte']}"))

    if expect.get("not_truncated"):
        checks.append(CheckOutcome("not_truncated", not res.truncated,
                                   f"truncated={res.truncated}"))

    if expect.get("repeats_consistent"):
        outs = [r.output for r in results]
        ok = len(set(outs)) == 1
        sample = (outs[0][:40] if outs else "") if ok else f"{outs[0][:40]!r} vs {outs[-1][:40]!r}"
        checks.append(CheckOutcome("repeats_consistent", ok, sample))

    return checks


def _image_content(path: Path, prompt: str) -> list[dict]:
    import base64

    ext = path.suffix.lstrip(".").lower()
    mime = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "bmp": "bmp", "webp": "webp"}.get(
        ext, "png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "image_url",
         "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]


# --------------------------------------------------------------------------- #
# 套件
# --------------------------------------------------------------------------- #

class FunctionalSuite(Suite):
    name = "functional"
    applies_to = ("llm", "embedding", "multimodal")

    def __init__(self, ctx):
        super().__init__(ctx)
        self.material = LongDocMaterial()

    # -- 用例加载 ------------------------------------------------------- #
    def load_cases(self) -> list[dict]:
        spec = self.cfg.tests.functional.cases
        base = resolve_data(spec)
        paths: list[Path] = []
        if base.is_dir():
            paths = sorted(list(base.glob("*.yaml")) + list(base.glob("*.yml")))
        elif base.is_file():
            paths = [base]
        cases: list[dict] = []
        for path in paths:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
            if isinstance(data, dict):  # 允许顶层带 cases: 包装
                data = data.get("cases", [])
            for idx, case in enumerate(data):
                case.setdefault("id", f"{path.stem}-{idx + 1}")
                case["_source"] = str(path)
                cases.append(case)
        mtype = self.cfg.model.type
        return [c for c in cases if not c.get("applies_to") or mtype in c["applies_to"]]

    # -- 执行 ----------------------------------------------------------- #
    async def run(self) -> SuiteResult:
        result = self.new_result()
        cases = self.load_cases()
        if not cases:
            result.warnings.append("未找到适用的功能用例（检查 tests.functional.cases）")
            return self.finish(result, SUITE_FAILED)

        for case in cases:
            row = await self._run_case(case)
            result.details.append(row)
            mark = "✓" if row["status"] == "passed" else "✗"
            self.ctx.say(f"  {mark} [{case['type']:<18}] {row['id']}"
                         + (f"  ({row['note']})" if row.get("note") else ""))

        passed = sum(1 for r in result.details if r["status"] == "passed")
        result.metrics = {
            "total": len(result.details),
            "passed": passed,
            "failed": len(result.details) - passed,
            "failed_ids": [r["id"] for r in result.details if r["status"] != "passed"],
        }
        return self.finish(result, SUITE_PASSED if passed == len(result.details) else SUITE_FAILED)

    async def _run_case(self, case: dict) -> dict:
        t0 = time.perf_counter()
        ctype = case.get("type", "chat")
        try:
            handler = getattr(self, f"_case_{ctype}", None)
            if handler is None:
                raise ValueError(f"未知用例类型: {ctype}")
            checks = await handler(case)
        except Exception as exc:  # noqa: BLE001 - 单用例异常计为 error
            return {"id": case["id"], "type": ctype, "status": "error",
                    "note": f"{type(exc).__name__}: {exc}",
                    "checks": [], "duration_s": time.perf_counter() - t0}
        failed = [c for c in checks if not c.ok]
        note = "; ".join(f"{c.name}: {c.note}" for c in failed) if failed else ""
        return {"id": case["id"], "type": ctype,
                "status": "passed" if not failed else "failed",
                "note": note,
                "checks": [{"name": c.name, "ok": c.ok, "note": c.note} for c in checks],
                "duration_s": time.perf_counter() - t0}

    # -- 类型 runners ---------------------------------------------------- #
    async def _case_chat(self, case: dict) -> list[CheckOutcome]:
        req = case.get("request", {})
        expect = case.get("expect", {})
        messages = self._build_messages(req)
        # repeats_consistent: True → default 3 repeats; positive int → explicit
        # repeat count; otherwise fall back to request.repeats (or a single shot).
        # NOTE: check `rc is True` first — isinstance(True, int) is True.
        rc = expect.get("repeats_consistent")
        if rc is True:
            repeats = 3
        elif isinstance(rc, int) and rc > 0:
            repeats = rc
        else:
            repeats = int(req.get("repeats") or 1)
        results = []
        for _ in range(repeats):
            res = await self.ctx.client.chat(
                self.ctx.served_model, messages,
                max_tokens=req.get("max_tokens"),
                temperature=req.get("temperature", 0.0 if repeats > 1 else None),
                stop=req.get("stop"),
                stream=bool(req.get("stream", False)))
            results.append(res)
        return evaluate_chat_expect(expect, results)

    async def _case_stream_consistency(self, case: dict) -> list[CheckOutcome]:
        req = case.get("request", {})
        expect = case.get("expect", {})
        messages = self._build_messages(req)
        plain = await self.ctx.client.chat(self.ctx.served_model, messages,
                                           max_tokens=req.get("max_tokens"),
                                           temperature=req.get("temperature", 0.0),
                                           stream=False)
        streamed = await self.ctx.client.chat(self.ctx.served_model, messages,
                                              max_tokens=req.get("max_tokens"),
                                              temperature=req.get("temperature", 0.0),
                                              stream=True)
        checks = evaluate_chat_expect(expect, [plain, streamed])
        a = " ".join((plain.output or "").split())
        b = " ".join((streamed.output or "").split())
        same = bool(a) and a == b
        checks.append(CheckOutcome(
            "stream_equals_plain", same,
            "" if same else f"plain={a[:50]!r} stream={b[:50]!r}"))
        return checks

    def _build_messages(self, req: dict) -> list[dict]:
        """构造 messages；overlong 场景按 max-model-len 填充超长文本。"""
        messages = [dict(m) for m in req.get("messages", [])]
        if req.get("overlong_fill_tokens"):
            mml = self.cfg.serve.max_model_len() or 32768
            want = int(mml * 1.2)  # 明显超过上限
            text = self.material.fill(want * 2)
            messages = [{"role": "user", "content": text}]
        return messages

    async def _case_embedding_basic(self, case: dict) -> list[CheckOutcome]:
        req = case.get("request", {})
        expect = case.get("expect", {})
        texts = req.get("texts") or [req.get("text") or "mtest embedding 冒烟文本。"]
        res = await self.ctx.client.embed(self.ctx.served_model, texts)
        checks = [
            CheckOutcome("ok", res.ok, res.error or ""),
            CheckOutcome("non_empty", bool(res.vectors), f"n={len(res.vectors)}"),
        ]
        if expect.get("dim") or self.cfg.tests.embedding.dim:
            want = expect.get("dim") or self.cfg.tests.embedding.dim
            checks.append(CheckOutcome("dim", res.dim == want, f"got {res.dim}, want {want}"))
        if expect.get("repeat_consistent"):
            res2 = await self.ctx.client.embed(self.ctx.served_model, texts)
            same = (res.ok and res2.ok and res.vectors == res2.vectors)
            checks.append(CheckOutcome("repeat_consistent", same, ""))
        return checks

    async def _case_ocr_basic(self, case: dict) -> list[CheckOutcome]:
        req = case.get("request", {})
        expect = case.get("expect", {})
        image = resolve_data(req["image"])
        content = _image_content(image, req.get("prompt", "识别图中所有文字并原样输出。"))
        res = await self.ctx.client.chat(
            self.ctx.served_model, [{"role": "user", "content": content}],
            max_tokens=req.get("max_tokens", 512),
            temperature=req.get("temperature", 0.0))
        checks = [
            CheckOutcome("ok", res.ok, res.error or f"status={res.status}"),
            CheckOutcome("non_empty", bool((res.output or "").strip()),
                         f"len={len(res.output or '')}"),
        ]
        if expect.get("no_mojibake"):
            ratio = mojibake_ratio(res.output or "")
            checks.append(CheckOutcome("no_mojibake", ratio < 0.05, f"ratio={ratio:.3f}"))
        if expect.get("contains"):
            missing = [s for s in _as_list(expect["contains"]) if s not in (res.output or "")]
            checks.append(CheckOutcome("contains", not missing, f"missing={missing}"))
        return checks

    async def _case_ocr_fault(self, case: dict) -> list[CheckOutcome]:
        """容错用例：期望 4xx/优雅报错而非服务崩溃，随后正常请求确认服务仍健康。"""
        req = case.get("request", {})
        image = resolve_data(req["image"])
        content = _image_content(image, req.get("prompt", "识别图中文字"))
        res = await self.ctx.client.chat(
            self.ctx.served_model, [{"role": "user", "content": content}],
            max_tokens=req.get("max_tokens", 256))
        graceful = res.ok or (res.status is not None and 400 <= res.status < 500)
        checks = [CheckOutcome(
            "graceful_reject", graceful,
            res.error or f"status={res.status}") ]

        healthy, _ = await self.ctx.client.request_health()
        checks.append(CheckOutcome("service_alive_after", healthy, ""))
        probe = await self.ctx.client.chat(
            self.ctx.served_model,
            [{"role": "user", "content": "你好"}], max_tokens=8)
        checks.append(CheckOutcome("normal_request_after", probe.ok, probe.error or ""))
        return checks
