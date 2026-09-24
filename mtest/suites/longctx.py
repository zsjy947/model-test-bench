"""longctx 长序列专项套件（llm，设计 §5.2）。

- 档位已在配置层按 ``max-model-len − output_len`` 封顶裁剪
- 用例构造：内置长文本素材循环填充 + 探测校准法
- 指标：TTFT（prefill 等待）、prefill 吞吐 ≈ prompt_tokens/TTFT、decode 吞吐、
  并发 1/8 下 e2e 分布、成功率（重点关注长序列下的超时/截断/段错误）
"""

from __future__ import annotations

from pathlib import Path

from ..datasets import LongDocMaterial, calibrate_prompt_text
from ..metrics import (aggregate_chat_stats, dist_summary, mean_of_cells, records_to_rows,
                       safe_div, write_csv)
from . import _runner
from .base import SUITE_FAILED, SUITE_PASSED, RunContext, Suite, SuiteResult


def _prefill_decode_tps(records) -> dict:
    prefill = [r.prompt_tokens / r.ttft for r in records
               if r.ok and r.prompt_tokens and r.ttft]
    decode = [r.completion_tokens / (r.e2e - r.ttft) for r in records
              if r.ok and r.completion_tokens and r.e2e and r.ttft
              and (r.e2e - r.ttft) > 0]
    return {
        "prefill_tps": dist_summary(prefill),
        "decode_tps": dist_summary(decode),
        "prefill_tps_mean": safe_div(sum(prefill), len(prefill)) if prefill else None,
        "decode_tps_mean": safe_div(sum(decode), len(decode)) if decode else None,
    }


class LongctxSuite(Suite):
    name = "longctx"
    applies_to = ("llm",)

    def __init__(self, ctx: RunContext):
        super().__init__(ctx)
        self.material = LongDocMaterial()

    async def run(self) -> SuiteResult:
        result = self.new_result()
        c = self.cfg.tests.longctx
        client = self.ctx.client
        model = self.ctx.served_model

        matrix: dict[str, dict] = {}
        calibration: dict[str, dict] = {}
        all_records = []
        round_aggs: dict[tuple[int, int], list[dict]] = {}

        for input_len in c.input_lens:
            # 素材填充到约目标长度后再校准（校准会按 usage 反馈微调）
            est_chars = int(input_len * 1.1)
            base = self.material.fill(est_chars)
            cal = await calibrate_prompt_text(client, model, base, self.material, input_len)
            messages = [{"role": "user", "content": cal.text}]
            calibration[str(input_len)] = {
                "calibrated": cal.within_tolerance,
                "iterations": cal.iterations,
                "target_tokens": input_len,
                "actual_prompt_tokens": cal.prompt_tokens,
            }
            for cc in c.concurrency:
                label = f"longctx[in={input_len},cc={cc}]"
                with self.ctx.phase(label):
                    await _runner.warmup_chat(client, model, messages, count=c.warmup,
                                              max_tokens=16, stream=c.stream)
                    for round_idx in range(c.rounds):
                        records, wall = await _runner.closed_loop_chat(
                            client, model, messages,
                            suite=self.name, input_len=input_len, concurrency=cc,
                            duration_s=c.duration, num_requests=c.num_requests,
                            round_idx=round_idx, max_tokens=c.output_len,
                            stream=c.stream, request_timeout=self.cfg.client.request_timeout)
                        all_records.extend(records)
                        agg = aggregate_chat_stats(records, wall)
                        agg.update(_prefill_decode_tps(records))
                        round_aggs.setdefault((input_len, cc), []).append(agg)
                cell = mean_of_cells(round_aggs[(input_len, cc)])
                # 分位合并（mean_of_cells 不了解 prefill/decode 键）
                cell.update(_prefill_decode_tps(
                    [r for r in all_records
                     if r.input_len == input_len and r.concurrency == cc]))
                matrix[f"{input_len}x{cc}"] = cell
                self._print_cell(input_len, cc, cell)

        csv_path = write_csv(Path(self.ctx.run_dir) / "longctx_details.csv",
                             records_to_rows(all_records))
        result.metrics = {
            "matrix": matrix,
            "calibration": calibration,
            "total_requests": len(all_records),
            "total_ok": sum(1 for r in all_records if r.ok),
        }
        result.artifacts["details"] = str(csv_path)

        if not any(c.get("ok") for c in matrix.values()):
            result.error = "全部请求失败"
            return self.finish(result, SUITE_FAILED)
        for key, cell in matrix.items():
            sr = cell.get("success_rate")
            if sr is not None and sr < 1.0:
                result.warnings.append(
                    f"[{key}] 成功率 {sr:.1%}，失败明细 {cell.get('errors')}")
        return self.finish(result, SUITE_PASSED)

    def _print_cell(self, input_len: int, cc: int, cell: dict) -> None:
        ttft = cell.get("ttft_s") or {}
        e2e = cell.get("e2e_s") or {}
        self.ctx.say(
            f"  longctx[{input_len:>6} x cc{cc:<3}] "
            f"ok {cell.get('ok', 0)}/{cell.get('requests', 0)}  "
            f"TTFT p50/p99 {ttft.get('p50') or 0:7.3f}/{ttft.get('p99') or 0:7.3f}s  "
            f"prefill {cell.get('prefill_tps_mean') or 0:8.1f} tok/s  "
            f"decode {cell.get('decode_tps_mean') or 0:7.1f} tok/s  "
            f"e2e p99 {e2e.get('p99') or 0:8.3f}s"
        )
