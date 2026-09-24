"""perf 性能压测套件（llm，设计 §5.1）。

- 请求：OpenAI 兼容 ``/v1/chat/completions``，流式（SSE）+ include_usage
- 闭环并发：固定在途请求数，duration / num_requests 双上限先到为准
- 输入长度：探测校准法（usage 反馈迭代至目标 ±10% 后锁定复用）
- 输出：矩阵（input_len × concurrency）+ 明细 CSV
"""

from __future__ import annotations

import random
from pathlib import Path

from ..client import BenchClient
from ..datasets import (CalibratedPrompt, CustomJsonlPrompts, LongDocMaterial, PromptPool,
                        RandomTokens, calibrate_prompt_text)
from ..metrics import aggregate_chat_stats, mean_of_cells, records_to_rows, write_csv
from ..paths import resolve_data
from . import _runner
from .base import SUITE_FAILED, SUITE_PASSED, RunContext, Suite, SuiteResult

_WARMUP_MAX_TOKENS = 32  # 预热用短输出，兼顾预热效果与耗时


class PerfSuite(Suite):
    name = "perf"
    applies_to = ("llm",)

    def __init__(self, ctx: RunContext):
        super().__init__(ctx)
        p = self.cfg.tests.perf
        self.dataset_kind, self.custom_path = self._resolve_dataset(p.dataset)
        self.pool = PromptPool() if self.dataset_kind == "prompts" else None
        self.random = RandomTokens() if self.dataset_kind == "random" else None
        self.custom = (CustomJsonlPrompts(resolve_data(self.custom_path))
                       if self.dataset_kind == "custom" else None)
        self.material = LongDocMaterial()
        self._rng = random.Random(20260924)

    @staticmethod
    def _resolve_dataset(spec: str) -> tuple[str, str | None]:
        if spec in ("prompts", "random"):
            return spec, None
        path = resolve_data(spec)
        if not path.is_file():
            raise FileNotFoundError(f"perf.dataset 自定义 jsonl 不存在: {spec}")
        return "custom", str(path)

    # ------------------------------------------------------------------ #
    async def _prepare_messages(self, target_len: int) -> tuple[list[dict], dict]:
        """按数据集类型构造目标长度 messages（校准一次并锁定）。"""
        model = self.ctx.served_model
        if self.dataset_kind == "custom":
            messages = self.custom.next_messages()
            return messages, {"dataset": "custom", "calibrated": False}
        if self.dataset_kind == "random":
            base = self.random.generate(target_len)
        else:
            base = self.pool.pick_for_target(target_len, self._rng)
        cal: CalibratedPrompt = await calibrate_prompt_text(
            self.ctx.client, model, base, self.material, target_len)
        note = {
            "dataset": self.dataset_kind,
            "calibrated": cal.within_tolerance,
            "iterations": cal.iterations,
            "target_tokens": target_len,
            "actual_prompt_tokens": cal.prompt_tokens,
            "chars": len(cal.text),
        }
        return [{"role": "user", "content": cal.text}], note

    # ------------------------------------------------------------------ #
    async def run(self) -> SuiteResult:
        result = self.new_result()
        p = self.cfg.tests.perf
        client: BenchClient = self.ctx.client
        model = self.ctx.served_model

        matrix: dict[str, dict] = {}
        calibration: dict[str, dict] = {}
        all_records = []
        round_aggs: dict[tuple[int, int], list[dict]] = {}

        for input_len in p.input_lens:
            messages, cal_note = await self._prepare_messages(input_len)
            calibration[str(input_len)] = cal_note
            for cc in p.concurrency:
                label = f"perf[in={input_len},cc={cc}]"
                with self.ctx.phase(label):
                    await _runner.warmup_chat(client, model, messages, count=p.warmup,
                                              max_tokens=_WARMUP_MAX_TOKENS, stream=p.stream)
                    for round_idx in range(p.rounds):
                        records, wall = await _runner.closed_loop_chat(
                            client, model, messages,
                            suite=self.name, input_len=input_len, concurrency=cc,
                            duration_s=p.duration, num_requests=p.num_requests,
                            round_idx=round_idx, max_tokens=p.output_len,
                            stream=p.stream, request_timeout=self.cfg.client.request_timeout)
                        all_records.extend(records)
                        round_aggs.setdefault((input_len, cc), []).append(
                            aggregate_chat_stats(records, wall))
                cell = mean_of_cells(round_aggs[(input_len, cc)])
                matrix[f"{input_len}x{cc}"] = cell
                self._print_cell(input_len, cc, cell)

        csv_path = write_csv(Path(self.ctx.run_dir) / "perf_details.csv",
                             records_to_rows(all_records))
        result.metrics = {
            "matrix": matrix,
            "calibration": calibration,
            "rounds": p.rounds,
            "dataset": p.dataset,
            "total_requests": len(all_records),
            "total_ok": sum(1 for r in all_records if r.ok),
        }
        result.artifacts["details"] = str(csv_path)

        if not any(c.get("ok") for c in matrix.values()):
            result.error = "全部请求失败（服务不可用或配置错误）"
            return self.finish(result, SUITE_FAILED)
        for key, cell in matrix.items():
            sr = cell.get("success_rate")
            if sr is not None and sr < 0.99:
                result.warnings.append(f"[{key}] 成功率 {sr:.1%} < 99%")
        return self.finish(result, SUITE_PASSED)

    # ------------------------------------------------------------------ #
    def _print_cell(self, input_len: int, cc: int, cell: dict) -> None:
        ttft = cell.get("ttft_s") or {}
        e2e = cell.get("e2e_s") or {}
        tpot = cell.get("tpot_s") or {}
        self.ctx.say(
            f"  perf[{input_len:>6} x cc{cc:<3}] "
            f"ok {cell.get('ok', 0)}/{cell.get('requests', 0)} "
            f"RPS {cell.get('rps') or 0:6.2f}  out {cell.get('output_tps') or 0:8.1f} tok/s  "
            f"TTFT p50/p99 {ttft.get('p50') or 0:6.3f}/{ttft.get('p99') or 0:6.3f}s  "
            f"TPOT {tpot.get('mean') or 0:6.4f}s  e2e p99 {e2e.get('p99') or 0:7.3f}s"
        )
