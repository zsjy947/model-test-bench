"""客户端负载采样测试（§12 R5）。"""

import time
from pathlib import Path

from mtest.clientload import make_client_samplers, proc_cpu_sampler, threads_sampler


def test_samplers_produce_dicts():
    samplers = make_client_samplers()
    assert samplers, "至少应有 proc_cpu 与 threads 采样器"
    for sampler in samplers:
        row = sampler()
        assert isinstance(row, dict)


def test_threads_sampler():
    sample = threads_sampler()
    assert sample()["threads"] >= 1


def test_proc_cpu_sampler_delta():
    sample = proc_cpu_sampler()  # 第一次建立基线
    # 消耗一点 CPU 后应有读数（可能为 0.x，但键存在）
    row = sample()
    assert "proc_cpu_pct" in row


def test_monitor_records_client_samples(tmp_path: Path):
    from mtest.monitor import NPUMonitor

    fixture = tmp_path / "smi.txt"
    fixture.write_text(
        "| 0       910B3              | OK            | 67.5         42                0    / 0               |\n"
        "| 0       NA                 | 0000:C1:00.0  | 60           5042  / 65536                            |\n",
        encoding="utf-8")
    csv_path = tmp_path / "npu_samples.csv"
    monitor = NPUMonitor(
        interval=1.0, csv_path=csv_path,
        command=["python", "-c", f"print(open(r'{fixture}', encoding='utf-8').read(), end='')"],
        extra_samplers=[lambda: {"proc_cpu_pct": 42.0, "threads": 8}])
    monitor.start()
    time.sleep(2.2)
    monitor.stop_and_join(timeout=5)

    assert monitor.client_samples, "客户端样本应被记录"
    row = monitor.client_samples[-1]
    assert row["proc_cpu_pct"] == 42.0
    client_csv = tmp_path / "client_samples.csv"
    assert client_csv.is_file()
    lines = client_csv.read_text(encoding="utf-8").splitlines()
    assert "timestamp" in lines[0] and "proc_cpu_pct" in lines[0]

    summary = monitor.client_load_summary()
    assert summary["proc_cpu_pct_max"] == 42.0
    assert summary["threads_max"] == 8
