"""embedding 专项套件（设计 §5.3）。

- 性能：批量梯度 × 并发梯度（闭环），延迟分布 / sent/s / 批量扩展性
- 质量：维度校验、同文本复现一致性、区分度 sanity（相似对/无关对余弦分布，
  阈值可配，报告给出两组分布而非只给 pass/fail）
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from pathlib import Path

from ..datasets import PromptPool
from ..metrics import dist_summary, safe_div
from ..paths import data_dir, resolve_data
from .base import SUITE_FAILED, SUITE_PASSED, Suite, SuiteResult

_PAIRS_FILE = "cases/embedding_pairs.yaml"


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingSuite(Suite):
    name = "embedding"
    applies_to = ("embedding",)

    def __init__(self, ctx):
        super().__init__(ctx)
        pool = PromptPool()
        self.texts = [e.text for e in pool.entries]

    # ------------------------------------------------------------------ #
    async def _perf_grid(self) -> dict:
        e = self.cfg.tests.embedding
        client = self.ctx.client
        model = self.ctx.served_model
        matrix: dict[str, dict] = {}
        text_iter = itertools.cycle(self.texts)

        for batch in e.batch_sizes:
            for cc in e.concurrency:
                label = f"embedding[batch={batch},cc={cc}]"
                with self.ctx.phase(label):
                    for _ in range(max(0, e.warmup)):
                        texts = [next(text_iter) for _ in range(batch)]
                        await client.embed(model, texts)
                    lat: list[float] = []
                    errors: dict[str, int] = {}
                    ok = total = 0
                    seq = itertools.count(1)
                    stop_at = time.monotonic() + e.duration
                    started_at: list[float] = []

                    async def worker() -> None:
                        nonlocal ok, total
                        while True:
                            if next(seq) > e.num_requests:
                                return
                            if time.monotonic() >= stop_at:
                                return
                            if not started_at:
                                started_at.append(time.monotonic())
                            texts = [next(text_iter) for _ in range(batch)]
                            res = await client.embed(model, texts)
                            total += 1
                            if res.ok:
                                ok += 1
                                lat.append(res.e2e or 0.0)
                            else:
                                key = (res.error or f"http:{res.status}").split(":", 1)[0]
                                errors[key] = errors.get(key, 0) + 1

                    await asyncio.gather(*[worker() for _ in range(max(1, cc))])
                    wall = (time.monotonic() - started_at[0]) if started_at else 0.0
                cell = {
                    "batch": batch,
                    "concurrency": cc,
                    "requests": total,
                    "ok": ok,
                    "success_rate": safe_div(ok, total),
                    "wall_s": wall,
                    "req_per_s": safe_div(ok, wall),
                    "sent_per_s": safe_div(ok * batch, wall),
                    "latency_s": dist_summary(lat),
                    "errors": errors,
                }
                matrix[f"b{batch}c{cc}"] = cell
                self.ctx.say(
                    f"  embedding[batch={batch:>3} x cc{cc:<3}] ok {ok}/{total}  "
                    f"{cell['sent_per_s'] or 0:8.1f} sent/s  "
                    f"lat p50/p99 {(cell['latency_s'] or {}).get('p50') or 0:6.3f}/"
                    f"{(cell['latency_s'] or {}).get('p99') or 0:6.3f}s")
        return matrix

    # ------------------------------------------------------------------ #
    async def _quality(self) -> tuple[dict, list[str], list[str]]:
        """维度 / 一致性 / 区分度。返回 (quality, failures, warnings)。"""
        e = self.cfg.tests.embedding
        client = self.ctx.client
        model = self.ctx.served_model
        failures: list[str] = []
        warnings: list[str] = []

        probe_text = "mtest embedding 一致性与维度校验文本。"
        r1 = await client.embed(model, [probe_text])
        r2 = await client.embed(model, [probe_text])
        if not r1.ok or not r2.ok:
            return {"error": r1.error or r2.error}, ["探针请求失败"], []
        dim = r1.dim
        dim_ok = (e.dim is None) or (e.dim == dim)
        if not dim_ok:
            failures.append(f"维度不符：配置 {e.dim}，实际 {dim}")
        consistency = cosine(r1.vectors[0], r2.vectors[0])
        consistent = consistency >= 1 - 1e-6
        if not consistent:
            failures.append(f"同文本复现不一致：cosine={consistency}")

        # 区分度 sanity
        pairs_path = resolve_data(_PAIRS_FILE) if (data_dir() / _PAIRS_FILE).exists() \
            else data_dir() / _PAIRS_FILE
        discrimination: dict = {"status": "skipped",
                                "reason": f"未找到 {pairs_path}"}
        if pairs_path.is_file():
            import yaml

            payload = yaml.safe_load(pairs_path.read_text(encoding="utf-8")) or {}
            pairs = payload.get("pairs", [])
            if pairs:
                sims: list[tuple[str, float]] = []
                for pair in pairs:
                    ra = await client.embed(model, [pair["a"]])
                    rb = await client.embed(model, [pair["b"]])
                    if ra.ok and rb.ok:
                        sims.append((pair.get("label", "similar"),
                                     cosine(ra.vectors[0], rb.vectors[0])))
                similar = [v for lab, v in sims if lab == "similar"]
                unrelated = [v for lab, v in sims if lab != "similar"]
                sep = (sum(similar) / len(similar) - sum(unrelated) / len(unrelated)) \
                    if similar and unrelated else None
                discrimination = {
                    "status": "done",
                    "n_pairs": len(sims),
                    "similar": dist_summary(similar) or {"n": 0},
                    "unrelated": dist_summary(unrelated) or {"n": 0},
                    "separation": sep,
                    "min_separation": e.similarity_min_separation,
                    "pass": sep is not None and sep >= e.similarity_min_separation,
                }
                if not discrimination["pass"]:
                    warnings.append(
                        f"区分度不足（sanity 参考，不判失败）：相似对-无关对间隔 "
                        f"{sep} < {e.similarity_min_separation}")
        quality = {
            "dim": dim,
            "dim_declared": e.dim,
            "dim_ok": dim_ok,
            "repeat_consistency_cosine": consistency,
            "repeat_consistent": consistent,
            "discrimination": discrimination,
        }
        return quality, failures, warnings

    # ------------------------------------------------------------------ #
    async def run(self) -> SuiteResult:
        result = self.new_result()
        matrix = await self._perf_grid()
        quality, failures, warnings = await self._quality()
        result.metrics = {
            "matrix": matrix,
            "quality": quality,
        }
        result.warnings.extend(warnings)
        any_ok = any(c.get("ok") for c in matrix.values())
        if not any_ok:
            result.error = "全部 embedding 请求失败"
            return self.finish(result, SUITE_FAILED)
        return self.finish(result, SUITE_PASSED if not failures else SUITE_FAILED)
