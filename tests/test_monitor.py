"""NPU 监控测试：解析器（多形态）/ 采样线程（fake 命令）/ 汇总。"""

import time
from pathlib import Path

import pytest

from mtest.errors import NpuMonitorError
from mtest.monitor import NPUMonitor, NPUSample, ChipSample, parse_npu_smi_output, summarize_npu

FIXTURE_MAIN = """+------------------------------------------------------------------------------------------------------+
| npu-smi 23.3.3                   Version: 23.3.3                                                     |
+===========================+===============+=======================================================+
| NPU     Name              | Health        | Power(W)     Temp(C)           Hugepages-Usage(page)  |
| Chip    Device             | Bus-Id        | AICore(%)    Memory-Usage(MB)                        |
+===========================+===============+=======================================================+
| 0       910B3              | OK            | 67.5         42                0    / 0               |
| 0       NA                 | 0000:C1:00.0  | 60           5042  / 65536                            |
+===========================+===============+=======================================================+
| 1       910B3              | OK            | 70.1         44                0    / 0               |
| 1       NA                 | 0000:C2:00.0  | 85           6100  / 65536                            |
+===========================+===============+=======================================================+
| 7       910B4              | Warning       | 310.2        63                0    / 0               |
| 7       NA                 | 0000:C8:00.0  | 99           64000 / 65536                            |
+===========================+===============+=======================================================+
"""

# CANN 8.x 风格（列结构一致，间距不同）
FIXTURE_V2 = """+-----------------------------------------------------------------------------------------------------------+
| npu-smi 24.1.0                        Version: 24.1.0                                                   |
+=========================================+===============+=================================================+
| NPU     Name                            | Health        | Power(W)     Temp(C)            AI Core(%)   |
| Chip    Device                          | Bus-Id        | Memory-Usage(MB)                               |
+=========================================+===============+=================================================+
| 0       910B4                           | OK            | 95.0         46                0    / 0        |
| 0       NA                              | 0000:C1:00.0  | 32           4283  /  65536                     |
+=========================================+===============+=================================================+
"""

FIXTURE_USAGES = """
 NPU   AICore(%)   Memory-Usage(MB)
 0     45          5000  /  65536
 1     32          4800  /  65536
"""


def test_parse_main_table():
    chips = parse_npu_smi_output(FIXTURE_MAIN)
    assert len(chips) == 3
    assert chips[0].npu == 0 and chips[0].aicore_util == 60.0
    assert chips[0].hbm_used_mb == 5042 and chips[0].hbm_total_mb == 65536
    assert chips[0].power_w == 67.5 and chips[0].temp_c == 42
    assert chips[2].npu == 7 and chips[2].aicore_util == 99


def test_parse_v2_table():
    chips = parse_npu_smi_output(FIXTURE_V2)
    assert len(chips) == 1
    assert chips[0].hbm_used_mb == 4283


def test_parse_usages_table():
    chips = parse_npu_smi_output(FIXTURE_USAGES)
    assert len(chips) == 2
    assert chips[1].aicore_util == 32
    assert chips[1].power_w is None


def test_parse_failure_raises():
    with pytest.raises(NpuMonitorError):
        parse_npu_smi_output("garbage output\nnothing here")


def test_monitor_thread_and_csv(tmp_path: Path):
    fixture = tmp_path / "smi.txt"
    fixture.write_text(FIXTURE_MAIN, encoding="utf-8")
    csv_path = tmp_path / "samples.csv"
    cmd = ["python", "-c",
           f"print(open(r'{fixture}', encoding='utf-8').read(), end='')"]
    monitor = NPUMonitor(interval=1.0, csv_path=csv_path, command=cmd)
    monitor.start()
    with monitor.phase("perf[in=128,cc=1]"):
        time.sleep(2.5)
    monitor.stop_and_join(timeout=5)
    assert not monitor.degraded
    assert monitor.samples, "应至少采到一个样本"
    assert monitor.samples[-1].phase.startswith("perf[")
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp,epoch,phase,npu,aicore_util_pct")
    assert len(lines) >= 3  # header + 至少一行数据（每行一卡）


def test_summarize_npu():
    chips = [ChipSample(npu=0, aicore_util=50, hbm_used_mb=1000, hbm_total_mb=65536,
                        power_w=200, temp_c=40),
             ChipSample(npu=0, aicore_util=90, hbm_used_mb=2000, hbm_total_mb=65536,
                        power_w=300, temp_c=55)]
    samples = [NPUSample(ts=1.0, phase="perf", chips=chips[:1]),
               NPUSample(ts=2.0, phase="idle", chips=chips[1:])]
    summary = summarize_npu(samples)
    assert summary["perf"]["aicore_util_max"] == 50
    assert summary["idle"]["hbm_used_max_mb"] == 2000
    assert summary["overall"]["power_max_w"] == 300
