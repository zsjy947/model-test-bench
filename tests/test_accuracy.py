"""accuracy 套件测试：答案提取 / 评分 / 数据集加载 / 全流程（fake client）。"""

import asyncio
import json
from pathlib import Path

from mtest.config import BenchConfig, ModelCfg
from mtest.suites.accuracy import AccuracySuite, extract_choice, extract_number
from mtest.suites.base import RunContext


def test_extract_number():
    assert extract_number("答案是 42") == 42.0
    assert extract_number("所以共有 1,680 个") == 1680.0
    assert extract_number("先算 5*3=15，再加 5 等于 20") == 20.0  # 最后一个数
    assert extract_number("没有任何数字") is None
    assert extract_number("温度 2.5 度") == 2.5


def test_extract_choice():
    assert extract_choice("答案是 B") == "B"
    assert extract_choice("正确选项：C") == "C"
    assert extract_choice("A. 不对，我选 D") == "D"
    assert extract_choice("选 C") == "C"
    assert extract_choice("没有字母") is None


def test_grade_math():
    item = {"type": "math", "answer": "160"}
    assert AccuracySuite.grade(item, "打八折是 160 元")[0]
    assert not AccuracySuite.grade(item, "是 150 元")[0]
    assert not AccuracySuite.grade(item, "无数字")[0]


def test_grade_mcq():
    item = {"type": "mcq", "answer": "B"}
    assert AccuracySuite.grade(item, "答案是B")[0]
    assert not AccuracySuite.grade(item, "答案是 C")[0]


def _make_suite(tmp_path: Path, client) -> AccuracySuite:
    cfg = BenchConfig(model=ModelCfg(name="t", path="/m/t"))
    cfg.finalize()
    ctx = RunContext(cfg=cfg, run_dir=tmp_path, client=client, served_model="t")
    return AccuracySuite(ctx)


class FixedAnswerClient:
    """按题目 id 映射返回固定输出。"""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    async def chat(self, model, messages, *, max_tokens=None, temperature=None,
                   stream=False, **kw):
        from mtest.client import ChatResult

        text = messages[0]["content"]
        output = ""
        if "32 名学生" in text:
            output = self.mapping["gsm-001"]
        elif "书架" in text:
            output = self.mapping["gsm-002"]
        elif "KV 缓存" in text:
            output = self.mapping["mcq-001"]
        elif "429" in text:
            output = self.mapping["mcq-002"]
        return ChatResult(ok=True, status=200, output=output, finish_reason="stop",
                          completion_tokens=5)


def test_accuracy_run_with_custom_dataset(tmp_path: Path):
    ds = tmp_path / "mini.jsonl"
    items = [
        {"id": "gsm-001", "type": "math", "question": "一个班级有 32 名学生，其中一半是女生。女生有多少名？", "answer": "16"},
        {"id": "gsm-002", "type": "math", "question": "书架上有 5 层，每层放 12 本书。一共有多少本书？", "answer": "60"},
        {"id": "mcq-001", "type": "mcq", "question": "LLM 推理时 KV 缓存通常存放在哪种存储上？", "choices": {"A": "HBM 显存", "B": "机械硬盘", "C": "BIOS", "D": "CPU 寄存器"}, "answer": "A"},
        {"id": "mcq-002", "type": "mcq", "question": "HTTP 状态码 429 表示什么？", "choices": {"A": "未找到", "B": "请求过多", "C": "错误", "D": "未授权"}, "answer": "B"},
    ]
    ds.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n",
                  encoding="utf-8")
    client = FixedAnswerClient({
        "gsm-001": "女生有 16 名",       # ✓
        "gsm-002": "共 5*12=60 本",      # ✓
        "mcq-001": "答案是 A",           # ✓
        "mcq-002": "我选 C",             # ✗
    })
    suite = _make_suite(tmp_path, client)
    suite.cfg.tests.accuracy.datasets = str(ds)
    result = asyncio.run(suite.run())
    assert result.status == "failed"      # 3/4 = 75% < 80%
    assert result.metrics["total"] == 4
    assert result.metrics["passed"] == 3
    assert abs(result.metrics["accuracy"] - 0.75) < 1e-9
    assert result.metrics["failed_ids"] == ["mcq-002"]

    client2 = FixedAnswerClient({
        "gsm-001": "16", "gsm-002": "60", "mcq-001": "A", "mcq-002": "B"})
    suite2 = _make_suite(tmp_path, client2)
    suite2.cfg.tests.accuracy.datasets = str(ds)
    result2 = asyncio.run(suite2.run())
    assert result2.status == "passed"


def test_builtin_datasets_load(tmp_path: Path):
    suite = _make_suite(tmp_path, FixedAnswerClient({}))
    items = suite.load_items()
    math_n = sum(1 for i in items if i["type"] == "math")
    mcq_n = sum(1 for i in items if i["type"] == "mcq")
    assert math_n == 20 and mcq_n == 20
