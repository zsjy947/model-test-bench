"""datasets 测试：token 估算 / 池选取 / 校准法（fake client）/ 素材。"""

import asyncio
import random
from pathlib import Path

import pytest

from mtest.datasets import (CustomJsonlPrompts, LongDocMaterial, PromptPool,
                            RandomTokens, calibrate_prompt_text, detect_template_residue,
                            estimate_tokens)


def test_estimate_tokens_cjk_vs_ascii():
    assert estimate_tokens("一二三四五") == 5
    assert estimate_tokens("abcdefgh") == 2  # 8/4


def test_prompt_pool_buckets():
    pool = PromptPool()
    small = pool.pick_for_target(10, random.Random(1))
    large = pool.pick_for_target(4000, random.Random(1))
    assert estimate_tokens(large) > estimate_tokens(small)


def test_random_tokens_seeded_reproducible():
    a = RandomTokens(seed=7).generate(200)
    b = RandomTokens(seed=7).generate(200)
    assert a == b
    assert abs(estimate_tokens(a) - 200) <= 40


def test_longdoc_material_fill():
    mat = LongDocMaterial()
    text = mat.fill(5000)
    assert len(text) == 5000


def test_custom_jsonl_cycle(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"messages": [{"role": "user", "content": "a"}]}\n'
                 '{"messages": [{"role": "user", "content": "b"}]}\n', encoding="utf-8")
    c = CustomJsonlPrompts(p)
    assert c.next_messages()[0]["content"] == "a"
    assert c.next_messages()[0]["content"] == "b"
    assert c.next_messages()[0]["content"] == "a"  # 循环


class FakeChatClient:
    """prompt_tokens = len(text) // 2，用于验证校准收敛。"""

    async def chat(self, model, messages, *, max_tokens=None, stream=False, **kw):
        from mtest.client import ChatResult

        text = messages[0]["content"]
        return ChatResult(ok=True, status=200, output="", finish_reason="length",
                          prompt_tokens=max(1, len(text) // 2), completion_tokens=1)


def test_calibrate_converges():
    material = LongDocMaterial()
    client = FakeChatClient()
    cal = asyncio.run(calibrate_prompt_text(client, "m", "短基底", material, 200))
    assert cal.within_tolerance
    assert cal.prompt_tokens is not None
    assert abs(cal.prompt_tokens - 200) <= 20  # ±10%
    assert cal.iterations <= 5


def test_calibrate_falls_back_when_no_usage():
    class NoUsage:
        async def chat(self, *a, **kw):
            from mtest.client import ChatResult
            return ChatResult(ok=False, error="connect:x")

    cal = asyncio.run(calibrate_prompt_text(NoUsage(), "m", "基底", LongDocMaterial(), 100))
    assert not cal.within_tolerance


def test_template_residue_detection():
    assert detect_template_residue("正常输出<|im_end|>") == ["<|im_end|>"]
    assert detect_template_residue("正常输出") == []
    assert "<|endoftext|>" in detect_template_residue("x<|endoftext|>y")


def test_prompt_pool_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        PromptPool([tmp_path / "none.jsonl"])
