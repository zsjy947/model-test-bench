"""报告渲染 / 对比 / 结论测试（离线，不依赖网络与 NPU）。"""

import json
from pathlib import Path

from mtest.report import build_conclusions, compare_runs, render_summary, write_reports


def make_payload(run_id: str = "20260924-1200_demo", perf_fail: bool = False) -> dict:
    return {
        "run_id": run_id,
        "created_at": "2026-09-24T12:00:00",
        "mtest_version": "0.1.0",
        "total_duration_s": 300.0,
        "config": {
            "model": {"name": "demo", "path": "/m/demo", "type": "llm",
                      "served_name": None, "trust_remote_code": True},
            "serve": {"mode": "process", "args": {}, "docker": {}},
            "client": {}, "enabled_suites": ["functional", "perf"],
        },
        "serve": {"mode": "process", "managed": True, "startup_seconds": 210,
                  "command_preview": "vllm serve /m/demo", "keep_alive": False},
        "environment": {
            "hostname": "h1", "platform": "linux", "python": "3.10", "cpu_count": 128,
            "vllm_version_pip": "0.9.2", "vllm_ascend_version_pip": "0.9.2",
            "server_version": "0.9.2",
            "npu": {"npu_count": 8, "hbm_total_mb": 65536, "npu_smi_header": "npu-smi 24.1"},
        },
        "suites": [
            {"name": "functional", "status": "passed", "duration_s": 12,
             "metrics": {"total": 8, "passed": 8, "failed": 0, "failed_ids": []},
             "details": [{"id": "chat-basic-zh", "type": "chat", "status": "passed",
                          "note": "", "duration_s": 1.2}],
             "artifacts": {}, "error": None, "warnings": [],
             "started_at": "2026-09-24T12:00:00", "ended_at": "2026-09-24T12:00:12"},
            {"name": "perf", "status": "failed" if perf_fail else "passed", "duration_s": 100,
             "metrics": {"matrix": {
                 "128x1": {"requests": 10, "ok": 9, "success_rate": 0.9, "rps": 9.5,
                           "output_tps": 2400.0, "ttft_s": {"p50": 0.02, "p99": 0.05},
                           "tpot_s": {"mean": 0.01}, "e2e_s": {"p99": 3.2}, "errors": {}},
                 "128x8": {"requests": 10, "ok": 10, "success_rate": 1.0, "rps": 19.5,
                           "output_tps": 4800.0, "ttft_s": {"p50": 0.04, "p99": 35.0},
                           "tpot_s": {"mean": 0.01}, "e2e_s": {"p99": 4.2}, "errors": {}},
             }, "calibration": {"128": {"target_tokens": 128,
                                        "actual_prompt_tokens": 125, "calibrated": True}}},
             "details": [], "artifacts": {"details": "perf_details.csv"},
             "error": None, "warnings": ["[128x1] 成功率 90.0% < 99%"],
             "started_at": "2026-09-24T12:00:12", "ended_at": "2026-09-24T12:02:00"},
        ],
        "npu": {"interval_s": 5, "degraded": False, "samples": 10,
                "summary": {"overall": {"n_samples": 10, "npu_count": 8,
                                        "hbm_total_mb": 65536, "hbm_used_max_mb": 65000,
                                        "hbm_used_avg_mb": 50000, "aicore_util_max": 98.0,
                                        "aicore_util_avg": 80.0, "power_max_w": 350.0,
                                        "temp_max_c": 60}},
                "csv": "npu_samples.csv"},
        "warnings": ["[128x1] 成功率 90.0% < 99%"],
        "error": None,
    }


def test_render_summary_contains_sections():
    md = render_summary(make_payload())
    assert "# mtest 测试报告" in md
    assert "perf 性能压测" in md
    assert "functional 功能冒烟" in md
    assert "128 | 1 | 90.0%" in md
    assert "NPU 指标" in md
    assert "结论与建议" in md


def test_write_reports(tmp_path: Path):
    payload = make_payload()
    summary = write_reports(tmp_path / "run", payload)
    assert summary.is_file()
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == 1
    assert metrics["run_id"] == payload["run_id"]
    assert len(metrics["suites"]) == 2


def test_compare_runs():
    a = make_payload("run-a")
    b = make_payload("run-b")
    # b 提速 10%
    for cell in b["suites"][1]["metrics"]["matrix"].values():
        cell["rps"] = cell["rps"] * 1.1
    md = compare_runs(a, b)
    assert "run-a vs run-b" in md
    assert "+10.0%" in md  # 差异百分比
    assert "| 128x1 | RPS |" in md or "128x1" in md


def test_compare_runs_no_common():
    a = make_payload()
    b = make_payload()
    b["suites"] = []
    md = compare_runs(a, b)
    assert "无可对比" in md


def test_conclusions_rules():
    payload = make_payload(perf_fail=True)
    conclusions = build_conclusions(payload)
    assert any("perf" in c and "未通过" in c for c in conclusions)
    assert any("HBM" in c for c in conclusions)          # 峰值 65000/65536 > 95%
    assert any("TTFT p99" in c for c in conclusions)     # 高并发 TTFT 35s
