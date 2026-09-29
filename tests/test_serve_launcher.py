"""serve launcher 超时清理 / 原子状态写入 / 占位锁测试（mock 子进程，不依赖 docker）。"""

import asyncio
import subprocess

import pytest

from mtest.config import BenchConfig, ModelCfg
from mtest.errors import ServeError
from mtest.serve import launcher as launcher_mod
from mtest.serve.launcher import DockerLauncher, ProcessLauncher, ServeController


def _cfg(mode: str = "docker") -> BenchConfig:
    cfg = BenchConfig(model=ModelCfg(name="demo", path="/m/demo"))
    cfg.finalize()
    cfg.serve.mode = mode
    if mode == "docker":
        cfg.serve.docker.image = "quay.io/ascend/vllm-ascend:latest"
    return cfg


class _FakeRun:
    """记录调用的 _run 替身；docker run 按需抛 TimeoutExpired。"""

    def __init__(self, timeout_on_run: bool = True):
        self.calls: list[list[str]] = []
        self.timeout_on_run = timeout_on_run

    def __call__(self, cmd, *, timeout: float = 30.0):
        self.calls.append(list(cmd))
        if self.timeout_on_run and cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, 120)
        return subprocess.CompletedProcess(cmd, 0, "", "")


# --------------------------------------------------------------------------- #
# DockerLauncher.start 超时清理
# --------------------------------------------------------------------------- #

def test_docker_start_timeout_cleans_container(tmp_path, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(launcher_mod, "_run", fake)
    lc = DockerLauncher(_cfg("docker"), tmp_path)
    with pytest.raises(ServeError, match="docker run 超时"):
        asyncio.run(lc.start())
    assert ["docker", "rm", "-f", lc.container] in fake.calls
    assert lc._started is False


def test_docker_start_timeout_cleanup_failure_still_raises(tmp_path, monkeypatch):
    def fake_run(cmd, *, timeout: float = 30.0):
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, 120)
        raise OSError("docker daemon gone")  # 清理本身也失败
    monkeypatch.setattr(launcher_mod, "_run", fake_run)
    lc = DockerLauncher(_cfg("docker"), tmp_path)
    with pytest.raises(ServeError, match="docker run 超时"):
        asyncio.run(lc.start())


# --------------------------------------------------------------------------- #
# ServeController.up：启动失败进入诊断+stop 清理；占位锁
# --------------------------------------------------------------------------- #

def _controller(tmp_path, monkeypatch) -> ServeController:
    monkeypatch.setattr(launcher_mod, "_serve_state_dir",
                        lambda name: tmp_path / "serve" / name)
    return ServeController(_cfg("docker"))


def test_up_failure_collects_diagnostics_and_stops(tmp_path, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(launcher_mod, "_run", fake)
    ctrl = _controller(tmp_path, monkeypatch)
    with pytest.raises(ServeError, match="docker run 超时"):
        asyncio.run(ctrl.up())
    # except 分支执行了诊断收集与容器回收
    assert (ctrl.state_dir / "serve_failure").is_dir()
    assert ["docker", "stop", "-t", "30", "mtest-demo"] in fake.calls
    # 失败后占位锁一定释放
    assert not ctrl.state_path.with_suffix(".lock").exists()


def test_up_locked_while_another_up_running(tmp_path, monkeypatch):
    ctrl = _controller(tmp_path, monkeypatch)
    lock = ctrl.state_path.with_suffix(".lock")
    ctrl.state_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text("", encoding="utf-8")
    with pytest.raises(ServeError, match="另一 serve up"):
        asyncio.run(ctrl.up())
    assert lock.exists()  # 他人持有的锁不可被破坏


def test_write_state_atomic_no_tmp_left(tmp_path, monkeypatch):
    ctrl = _controller(tmp_path, monkeypatch)
    ctrl._write_state({"mode": "process", "pid": 42})
    data = ctrl.read_state()
    assert data["pid"] == 42 and data["mode"] == "process"
    assert not ctrl.state_path.with_suffix(".json.tmp").exists()
    assert ctrl.state_path.is_file()


# --------------------------------------------------------------------------- #
# ProcessLauncher.stop：SIGKILL 后仍不退出 → 告警但句柄一定关闭
# --------------------------------------------------------------------------- #

class _StuckProc:
    """kill 后 wait 仍超时的假进程（D 态场景）。"""

    pid = 98765

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def poll(self) -> None:
        return None

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired(["bash"], timeout)


def test_stop_gives_up_but_closes_handles(tmp_path):
    lc = ProcessLauncher(_cfg("process"), tmp_path)
    lc._proc = _StuckProc()
    fh = lc._log_fh = open(tmp_path / "fh.log", "wb")
    asyncio.run(lc.stop())  # 不应抛 TimeoutExpired
    assert lc._proc is None
    assert lc._log_fh is None and fh.closed
