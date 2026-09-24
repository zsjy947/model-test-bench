"""metrics 统计测试。"""

from mtest.client import ChatResult
from mtest.metrics import (ChatRecord, aggregate_chat_stats, dist_summary, error_categories,
                           linreg_slope, mean_of_cells, percentile, records_to_rows, write_csv)


def test_percentile_basic():
    vals = [1, 2, 3, 4, 5]
    assert percentile(vals, 50) == 3
    assert percentile(vals, 0) == 1
    assert percentile(vals, 100) == 5
    assert percentile(vals, 90) == 4.6


def test_percentile_empty():
    assert percentile([], 50) is None


def test_dist_summary():
    d = dist_summary([1.0] * 10 + [2.0])
    assert d["n"] == 11
    assert d["min"] == 1.0
    assert d["max"] == 2.0
    assert abs(d["mean"] - (12 / 11)) < 1e-9


def test_linreg_slope():
    assert linreg_slope([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert linreg_slope([0, 1], [1, 1]) == 0.0
    assert linreg_slope([1], [1]) is None


def make_record(ok=True, ttft=0.1, e2e=1.0, ptok=100, ctok=32, error=None) -> ChatRecord:
    return ChatRecord(suite="perf", input_len=128, concurrency=8, round_idx=0, seq=1,
                      ok=ok, status=200 if ok else None, error=error,
                      ttft=ttft, e2e=e2e, prompt_tokens=ptok, completion_tokens=ctok,
                      itl_mean=0.028, itl_max=0.05)


def test_tpot():
    r = make_record(ttft=0.2, e2e=1.2, ctok=11)
    assert abs(r.tpot() - 0.1) < 1e-9
    assert make_record(ctok=1).tpot() is None


def test_aggregate_chat_stats():
    records = [make_record() for _ in range(10)] + [
        make_record(ok=False, error="connect:Fail")]
    agg = aggregate_chat_stats(records, wall_time=5.0)
    assert agg["requests"] == 11
    assert agg["ok"] == 10
    assert abs(agg["success_rate"] - 10 / 11) < 1e-9
    assert abs(agg["rps"] - 2.0) < 1e-9
    assert agg["total_output_tokens"] == 320
    assert abs(agg["output_tps"] - 64.0) < 1e-9
    assert agg["errors"] == {"connect": 1}
    assert agg["ttft_s"]["n"] == 10


def test_mean_of_cells():
    c1 = aggregate_chat_stats([make_record() for _ in range(10)], 5.0)
    c2 = aggregate_chat_stats([make_record() for _ in range(10)], 10.0)
    merged = mean_of_cells([c1, c2])
    assert merged["requests"] == 20
    assert abs(merged["rps"] - (2.0 + 1.0) / 2) < 1e-9


def test_error_categories_and_csv(tmp_path):
    records = [make_record(ok=False, error="timeout"),
               make_record(ok=False, error="timeout"),
               make_record()]
    assert error_categories(records) == {"timeout": 2}
    rows = records_to_rows(records)
    assert rows[0]["tpot"] is not None
    path = write_csv(tmp_path / "d.csv", rows)
    assert path.is_file()
    content = path.read_text(encoding="utf-8").splitlines()
    assert content[0].startswith("suite,input_len,concurrency,round_idx,seq,ok")


def test_record_from_result_success_criteria():
    from mtest.suites._runner import record_from_result
    ok_res = ChatResult(ok=True, status=200, finish_reason="stop",
                        prompt_tokens=10, completion_tokens=5, ttft=0.1, e2e=0.5)
    rec = record_from_result(ok_res, suite="perf", input_len=128, concurrency=1,
                             round_idx=0, seq=1)
    assert rec.ok
    bad = ChatResult(ok=True, status=200, finish_reason=None)  # 无 finish_reason
    assert not record_from_result(bad, suite="perf", input_len=128, concurrency=1,
                                  round_idx=0, seq=1).ok
    length = ChatResult(ok=True, status=200, finish_reason="length")
    assert record_from_result(length, suite="perf", input_len=1, concurrency=1,
                              round_idx=0, seq=1).ok
