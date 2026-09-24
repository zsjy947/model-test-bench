"""数据集与输入长度校准（设计 §5.1 / §5.2）。

压测端不加载本地 tokenizer：先按字符启发式估算选基底 prompt，再用
``max_tokens=1`` 探测请求的 ``usage.prompt_tokens`` 反馈迭代拼接/裁剪至目标
长度 ±10%，然后锁定复用。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .client import BenchClient
from .paths import data_dir

_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
    (0x3000, 0x303F), (0xFF00, 0xFFEF),
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """字符启发式 token 估算：CJK ≈ 1 token/字，其余 ≈ 1 token/4 字符。"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, round(cjk + other / 4.0))


# --------------------------------------------------------------------------- #
# 数据源
# --------------------------------------------------------------------------- #

@dataclass
class PromptEntry:
    text: str
    lang: str = "zh"
    est_tokens: int = 0


class PromptPool:
    """内置中英 prompt 池（data/prompts/{zh,en}.jsonl，按长度分桶）。"""

    def __init__(self, paths: Sequence[Path] | None = None):
        if paths is None:
            paths = [data_dir() / "prompts" / "zh.jsonl",
                     data_dir() / "prompts" / "en.jsonl"]
        self.entries: list[PromptEntry] = []
        for path in paths:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or ""
                if text:
                    self.entries.append(
                        PromptEntry(text=text, lang=obj.get("lang", "zh"),
                                    est_tokens=estimate_tokens(text)))
        if not self.entries:
            raise FileNotFoundError(f"prompt 池为空：{[str(p) for p in paths]}")
        self.entries.sort(key=lambda e: e.est_tokens)

    def pick_for_target(self, target_tokens: int, rng: random.Random | None = None) -> str:
        """取估算长度不超过目标的最大桶（同桶内随机）；无更小桶则取最小桶。"""
        smaller = [e for e in self.entries if e.est_tokens <= target_tokens]
        if smaller:
            best = smaller[-1].est_tokens
            bucket = [e for e in smaller if e.est_tokens == best]
        else:
            best = self.entries[0].est_tokens
            bucket = [e for e in self.entries if e.est_tokens == best]
        pick = (rng or random).choice(bucket)
        return pick.text


_ZH_VOCAB = (
    "模型 推理 服务 性能 延迟 吞吐 并发 请求 响应 序列 向量 矩阵 参数 权重 显存 "
    "缓存 调度 批处理 流式 接口 计算 优化 部署 监控 日志 指标 基准 测试 报告 "
    "系统 数据 结构 算法 训练 微调 量化 精度 上下文 长度 分块 编码 解码 生成"
).split()

_EN_VOCAB = (
    "model inference server performance latency throughput concurrency request "
    "response sequence vector matrix parameter weight memory cache scheduler "
    "batch streaming api compute optimization deploy monitor metric benchmark "
    "test report system data structure algorithm training finetune quantize "
    "precision context length chunk encode decode generate"
).split()


class RandomTokens:
    """随机 token 数据集（seed 可复现）。"""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, target_tokens: int) -> str:
        words: list[str] = []
        est = 0
        while est < target_tokens:
            zh = self._rng.random() < 0.5
            word = self._rng.choice(_ZH_VOCAB if zh else _EN_VOCAB)
            words.append(word)
            est += max(1, round(len(word) / (1.0 if any(_is_cjk(c) for c in word) else 4.0)))
        return " ".join(words)


class LongDocMaterial:
    """长序列填充素材（data/longdoc/*.txt 循环填充）。"""

    def __init__(self, directory: Path | None = None):
        directory = directory or (data_dir() / "longdoc")
        parts: list[str] = []
        for path in sorted(directory.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                parts.append(text)
        if not parts:
            raise FileNotFoundError(f"长文本素材目录为空: {directory}")
        self.text = "\n\n".join(parts)

    def fill(self, target_chars: int) -> str:
        if target_chars <= 0:
            return ""
        reps = target_chars // len(self.text) + 1
        return (self.text * reps)[:target_chars]


class CustomJsonlPrompts:
    """自定义 jsonl 数据集（messages 格式，循环取用）。"""

    def __init__(self, path: Path):
        self.messages_list: list[list[dict]] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if msgs:
                self.messages_list.append(msgs)
        if not self.messages_list:
            raise ValueError(f"自定义 jsonl 中无有效 messages: {path}")
        self._idx = 0

    def next_messages(self) -> list[dict]:
        msgs = self.messages_list[self._idx % len(self.messages_list)]
        self._idx += 1
        return msgs


# --------------------------------------------------------------------------- #
# 校准法
# --------------------------------------------------------------------------- #

@dataclass
class CalibratedPrompt:
    text: str
    prompt_tokens: int | None
    iterations: int
    target_tokens: int
    within_tolerance: bool


async def calibrate_prompt_text(
    client: BenchClient,
    model: str,
    base_text: str,
    material: LongDocMaterial,
    target_tokens: int,
    *,
    tol: float = 0.10,
    max_iter: int = 5,
    min_chars: int = 32,
) -> CalibratedPrompt:
    """探测校准：按 usage.prompt_tokens 反馈迭代拼接/裁剪至目标 ±tol。

    每次迭代发 ``max_tokens=1`` 的探测请求，假设 tokens 与字符数近似线性，
    按比例调整字符数（基底不足时用长文本素材补齐）。
    """
    text = _adjust_chars(base_text, max(min_chars, int(target_tokens * 0.5)), material)
    best_text, best_tokens, best_gap = text, None, float("inf")
    iterations = 0
    for i in range(max_iter):
        iterations = i + 1
        res = await client.chat(model, [{"role": "user", "content": text}],
                                max_tokens=1, stream=False)
        if not res.ok or res.prompt_tokens is None:
            break  # 服务端未返回 usage，退化为字符估算
        got = res.prompt_tokens
        gap = abs(got - target_tokens)
        if gap < best_gap:
            best_text, best_tokens, best_gap = text, got, gap
        if gap <= tol * target_tokens:
            return CalibratedPrompt(text, got, iterations, target_tokens, True)
        # 字符数按比例调整
        chars_per_tok = max(len(text) / got, 0.5)
        want_chars = int(chars_per_tok * target_tokens)
        if abs(want_chars - len(text)) < 16:
            break  # 已无法显著调整
        text = _adjust_chars(text, want_chars, material)
    return CalibratedPrompt(best_text, best_tokens, iterations, target_tokens,
                            best_tokens is not None
                            and abs(best_tokens - target_tokens) <= tol * target_tokens)


def _adjust_chars(text: str, want_chars: int, material: LongDocMaterial) -> str:
    """把文本调整到约 want_chars 字符：截断或素材补齐。"""
    if want_chars <= len(text):
        return text[:want_chars]
    pad_chars = want_chars - len(text) + 64
    return f"{text}\n{material.fill(pad_chars)}"[:want_chars]


_TEMPLATE_RESIDUE = re.compile(
    r"<\|im_(start|end)\|>|<\|(begin|end)_of_text\|>|<\|endoftext\|>|<\|user\|>|"
    r"<\|assistant\|>|### (Instruction|Response):"
)


def detect_template_residue(text: str) -> list[str]:
    """检测输出中的 chat 模板残留标记（functional 冒烟用）。"""
    hits = []
    for m in _TEMPLATE_RESIDUE.finditer(text or ""):
        token = m.group(0)
        if token not in hits:
            hits.append(token)
    return hits
