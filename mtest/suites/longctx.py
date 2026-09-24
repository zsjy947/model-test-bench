"""longctx 长序列专项套件（llm，设计 §5.2）。

- 档位已在配置层按 ``max-model-len − output_len`` 封顶裁剪
- 用例构造：内置长文本素材循环填充 + 探测校准法
- 指标：TTFT（prefill 等待）、prefill 吞吐 ≈ prompt_tokens/TTFT、decode 吞吐、
  并发 1/8 下 e2e 分布、成功率（重点关注长序列下的超时/截断/段错误）
- §12 风险 R4 缓解：探测请求命中"长度超限/OOM"类错误时自动降档重试（最多 2 次）
"""

from __future__ import annotations

import re
from pathlib import Path

from ..client import ChatResult
from ..datasets import CalibratedPrompt, LongDocMaterial, calibrate_prompt_text
from ..metrics import (aggregate_chat_stats, dist_summary, mean_of_cells, records_to_rows,
                       safe_div, write_csv)
from . import _runner
from .base import SUITE_FAILED, SUITE_PASSED, RunContext, Suite, SuiteResult

# 命中即触发自适应降档的错误模式（服务端对超长/OOM 的常见表述）
_LIMIT_PATTERNS = re.compile(
    r"maximum context length|longer than the maximum|too long|exceeds? the "
    r"(maximum|context|model)|out of memory|OOM|CUDA out of memory|kv.?cache", re.IGNORECASE)
_MIN_ADAPTIVE_LEN = 1024  # 降档下限，再低失去长序列意义
_MAX_DEGRADES = 2


def is_length_limit_error(res: ChatResult) -> bool:
    """HTTP 4xx 且错误体匹配长度超限/OOM 模式。"""
    if res.ok or not res.error:
        return False
    if not (res.error.startswith("http:4") or res.error.startswith("http:5")):
        return False
    return bool(_LIMIT_PATTERNS.search(res.error))


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

    async def _prepare_adaptive(self, target_len: int) -> tuple[int, CalibratedPrompt, list[int]]:
        """校准 + 探测；命中长度超限/OOM 错误时降档重试（§12 R4）。

        返回 (最终目标长度, 校准结果, 降档轨迹)。
        """
        degrades: list[int] = []
        target = target_len
        while True:
            base = self.material.fill(int(target * 1.1))
            cal = await calibrate_prompt_text(self.ctx.client, self.ctx.served_model,
                                              base, self.material, target)
            probe = await self.ctx.client.chat(
                self.ctx.served_model,
                [{"role": "user", "content": cal.text}], max_tokens=8, stream=False)
            if probe.ok or not is_length_limit_error(probe) or len(degrades) >= _MAX_DEGRADES:
                return target, cal, degrades
            new_target = max(target // 2, _MIN_ADAPTIVE_LEN)
            if new_target >= target:
                return target, cal, degrades
            self.ctx.say(
                f"[yellow]longctx 档位 {target} 命中长度/OOM 限制"
                f"（{(probe.error or '')[:120]}），自动降档为 {new_target} 重试[/yellow]")
            degrades.append(target)
            target = new_target

    async def run(self) -> SuiteResult:
        result = self.new_result()
        c = self.cfg.tests.longctx
        client = self.ctx.client
        model = self.ctx.served_model

        matrix: dict[str, dict] = {}
        calibration: dict[str, dict] = {}
        adaptive: dict[str, dict] = {}
        all_records = []
        round_aggs: dict[tuple[int, int], list[dict]] = {}

        for input_len in c.input_lens:
            final_len, cal, degrades = await self._prepare_adaptive(input_len)
            messages = [{"role": "user", "content": cal.text}]
            calibration[str(input_len)] = {
                "calibrated": cal.within_tolerance,
                "iterations": cal.iterations,
                "target_tokens": final_len,
                "actual_prompt_tokens": cal.prompt_tokens,
            }
            if degrades:
                adaptive[str(input_len)] = {
                    "original": input_len, "final": final_len, "degrades": degrades}
                result.warnings.append(
                    f"longctx 档位 {input_len} 命中长度/OOM 限制，自动降档至 {final_len}（§12 R4）")
            label_len = final_len
            for cc in c.concurrency:
                label = f"longctx[in={label_len},cc={cc}]"
                with self.ctx.phase(label):
                    await _runner.warmup_chat(client, model, messages, count=c.warmup,
                                              max_tokens=16, stream=c.stream)
                    for round_idx in range(c.rounds):
                        records, wall = await _runner.closed_loop_chat(
                            client, model, messages,
                            suite=self.name, input_len=label_len, concurrency=cc,
                            duration_s=c.duration, num_requests=c.num_requests,
                            round_idx=round_idx, max_tokens=c.output_len,
                            stream=c.stream, request_timeout=self.cfg.client.request_timeout)
                        all_records.extend(records)
                        agg = aggregate_chat_stats(records, wall)
                        agg.update(_prefill_decode_tps(records))
                        round_aggs.setdefault((label_len, cc), []).append(agg)
                cell = mean_of_cells(round_aggs[(label_len, cc)])
                # 分位合并（mean_of_cells 不了解 prefill/decode 键）
                cell.update(_prefill_decode_tps(
                    [r for r in all_records
                     if r.input_len == label_len and r.concurrency == cc]))
                matrix[f"{label_len}x{cc}"] = cell
                self._print_cell(label_len, cc, cell)

        csv_path = write_csv(Path(self.ctx.run_dir) / "longctx_details.csv",
                             records_to_rows(all_records))
        result.metrics = {
            "matrix": matrix,
            "calibration": calibration,
            "adaptive": adaptive,
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
