"""accuracy 精度评测套件（§13 扩展，dev 分支实现）。

- 数据集：GSM8K 风格数学题固定子集（data/accuracy/gsm8k_mini.jsonl，自研题库）
  与客观选择题子集（mcq_mini.jsonl）；支持自定义 jsonl（type: math / mcq）。
- 评分：
  - math：从输出提取最后一个数字（容忍千分位逗号/单位），与答案数值比较
  - mcq：提取选项字母（优先"答案是X"类表述，其次最后一个独立 A-D）
- 指标：分数据集准确率 + 总体准确率；低于 pass_threshold 判失败。
- 注意：固定小样本（各 20 题）适合冒烟级精度回归筛查，不能替代完整
  GSM8K / C-Eval 评测（见 docs/EXTENSIONS.md 边界说明）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..paths import resolve_data
from .base import SUITE_FAILED, SUITE_PASSED, Suite, SuiteResult

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MCQ_EXPLICIT_RE = re.compile(r"答案[是：:\s]*([A-D])|[选择答]\s*[:：]?\s*\(?([A-D])\)?", )
_MCQ_LETTER_RE = re.compile(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)")

_MATH_PROMPT_SUFFIX = "\n请逐步推理，并在最后一行给出最终数字答案。"
_MCQ_PROMPT_SUFFIX = "\n请直接回答正确选项的字母（A/B/C/D）。"


def extract_number(text: str) -> float | None:
    """提取输出中最后一个数字（容忍千分位逗号）。"""
    matches = _NUM_RE.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def extract_choice(text: str) -> str | None:
    """提取选项字母：优先显式'答案是X'，退化为最后一个独立 A-D。"""
    for pattern in (_MCQ_EXPLICIT_RE, _MCQ_LETTER_RE):
        hits = [g for m in pattern.finditer(text or "") for g in m.groups() if g]
        if hits:
            return hits[-1]
    return None


def _format_mcq_question(item: dict) -> str:
    choices = item.get("choices", {})
    lines = [item["question"]]
    for key in sorted(choices):
        lines.append(f"{key}. {choices[key]}")
    return "\n".join(lines) + _MCQ_PROMPT_SUFFIX


class AccuracySuite(Suite):
    name = "accuracy"
    applies_to = ("llm", "multimodal")

    def load_items(self) -> list[dict]:
        base = resolve_data(self.cfg.tests.accuracy.datasets)
        paths = ([base] if base.is_file()
                 else sorted(list(base.glob("*.jsonl"))) if base.is_dir() else [])
        items: list[dict] = []
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("type") in ("math", "mcq") and item.get("question"):
                    item["_source"] = path.stem
                    items.append(item)
        return items

    @staticmethod
    def grade(item: dict, output: str) -> tuple[bool, str]:
        """返回 (是否正确, 提取到的答案)。"""
        if item["type"] == "math":
            got = extract_number(output)
            want = extract_number(str(item.get("answer", "")))
            if got is None or want is None:
                return False, "-"
            return abs(got - want) < 1e-6, str(got)
        got = extract_choice(output or "")
        want = str(item.get("answer", "")).strip().upper()
        if got is None:
            return False, "-"
        return got.upper() == want, got.upper()

    async def run(self) -> SuiteResult:
        result = self.new_result()
        a = self.cfg.tests.accuracy
        items = self.load_items()
        if not items:
            result.error = f"无可用精度题集: {a.datasets}"
            return self.finish(result, SUITE_FAILED)

        by_source: dict[str, list[dict]] = {}
        for item in items:
            if item["type"] == "math":
                prompt = item["question"] + _MATH_PROMPT_SUFFIX
            else:
                prompt = _format_mcq_question(item)
            res = await self.ctx.client.chat(
                self.ctx.served_model, [{"role": "user", "content": prompt}],
                max_tokens=a.max_tokens, temperature=0.0, stream=False)
            ok, got = self.grade(item, res.output) if res.ok else (False, "-")
            note = "" if res.ok else (res.error or f"http:{res.status}")
            row = {"id": item.get("id", "?"), "dataset": item["_source"],
                   "type": item["type"], "expected": str(item.get("answer", "")),
                   "got": got, "passed": ok, "note": note}
            result.details.append(row)
            by_source.setdefault(item["_source"], []).append(row)
            mark = "✓" if ok else "✗"
            self.ctx.say(f"  {mark} [{item['type']:<4}] {row['id']} "
                         f"want={row['expected']} got={got}")

        total = len(result.details)
        passed = sum(1 for r in result.details if r["passed"])
        overall = passed / total
        result.metrics = {
            "total": total,
            "passed": passed,
            "accuracy": overall,
            "pass_threshold": a.pass_threshold,
            "by_dataset": {
                src: {"total": len(rows),
                      "passed": sum(1 for r in rows if r["passed"]),
                      "accuracy": sum(1 for r in rows if r["passed"]) / len(rows)}
                for src, rows in sorted(by_source.items())},
            "failed_ids": [r["id"] for r in result.details if not r["passed"]],
        }
        result.warnings.append(
            "固定小样本（各 20 题）适合冒烟级精度回归筛查，不能替代完整 GSM8K/C-Eval 评测")
        status = SUITE_PASSED if overall >= a.pass_threshold else SUITE_FAILED
        return self.finish(result, status)
