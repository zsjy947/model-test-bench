"""functional 断言引擎与用例加载测试。"""

from mtest.client import ChatResult
from mtest.config import BenchConfig, ModelCfg
from mtest.suites.base import RunContext
from mtest.suites.functional import CheckOutcome, FunctionalSuite, evaluate_chat_expect


def ok_result(**kw) -> ChatResult:
    params = dict(ok=True, status=200, output="你好，我是助手。", finish_reason="stop",
                  completion_tokens=8)
    params.update(kw)
    return ChatResult(**params)


def outcomes(checks: list[CheckOutcome]) -> dict[str, bool]:
    return {c.name: c.ok for c in checks}


def test_expect_all_pass():
    checks = evaluate_chat_expect(
        {"status": 200, "finish_reason": ["stop"], "non_empty": True,
         "no_template_residue": True, "completion_tokens_lte": 16,
         "not_contains": ["ERROR"]},
        [ok_result()])
    assert all(c.ok for c in checks)


def test_expect_failures():
    checks = evaluate_chat_expect(
        {"status": [200], "finish_reason": ["stop"], "non_empty": True,
         "contains": ["缺失词"], "not_contains": ["助手"], "no_template_residue": True,
         "completion_tokens_lte": 4},
        [ok_result(output="你好，我是助手。<|im_end|>", completion_tokens=8)])
    omap = outcomes(checks)
    assert omap["contains"] is False
    assert omap["not_contains"] is False
    assert omap["no_template_residue"] is False
    assert omap["completion_tokens_lte"] is False


def test_repeats_consistent():
    checks = evaluate_chat_expect({"repeats_consistent": True},
                                  [ok_result(), ok_result()])
    assert outcomes(checks)["repeats_consistent"] is True
    checks = evaluate_chat_expect({"repeats_consistent": True},
                                  [ok_result(), ok_result(output="不同输出")])
    assert outcomes(checks)["repeats_consistent"] is False


def test_status_lte():
    checks = evaluate_chat_expect({"status_lte": 499}, [ok_result(status=400, ok=False)])
    assert outcomes(checks)["status_lte"] is True
    checks = evaluate_chat_expect({"status_lte": 499}, [ok_result(status=500, ok=False)])
    assert outcomes(checks)["status_lte"] is False


def _suite_ctx(tmp_path, model_type="llm") -> FunctionalSuite:
    cfg = BenchConfig(model=ModelCfg(name="t", path="/m/t", type=model_type))
    cfg.finalize()
    ctx = RunContext(cfg=cfg, run_dir=tmp_path, client=object(), served_model="t")
    return FunctionalSuite(ctx)


def test_load_cases_filters_by_type(tmp_path):
    suite = _suite_ctx(tmp_path, "embedding")
    cases = suite.load_cases()
    ids = [c["id"] for c in cases]
    assert all(c.get("applies_to") in (None, ["embedding"]) for c in cases)
    assert "embedding-basic" in ids
    assert "chat-basic-zh" not in ids


def test_load_cases_llm(tmp_path):
    suite = _suite_ctx(tmp_path, "llm")
    cases = suite.load_cases()
    ids = [c["id"] for c in cases]
    assert "chat-stream-consistency" in ids
    assert "chat-overlong-input" in ids
    assert not any(i.startswith("ocr") for i in ids)


def test_load_cases_from_single_file(tmp_path):
    suite = _suite_ctx(tmp_path)
    suite.cfg.tests.functional.cases = "data/cases/llm_smoke.yaml"
    cases = suite.load_cases()
    assert cases and all(c["_source"].endswith("llm_smoke.yaml") for c in cases)


def test_build_messages_overlong(tmp_path):
    suite = _suite_ctx(tmp_path)
    msgs = suite._build_messages({"messages": [{"role": "user", "content": "x"}],
                                  "overlong_fill_tokens": True})
    assert len(msgs[0]["content"]) > 32768 * 1.2 * 1.5  # 明显超过 max-model-len
