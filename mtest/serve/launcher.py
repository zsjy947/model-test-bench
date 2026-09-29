"""服务管理：process / docker / external 三模式统一接口（设计 §4）。

统一接口：``start / wait_ready / stop / status / log_tail / collect_diagnostics``。
``ServeController`` 提供 ``mtest serve up|down|status|logs`` 所需的跨进程持久管理
（状态文件 + pid / 容器名记账）。
"""

from __future__ import annotations

import abc
import asyncio
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from ..config import BenchConfig, build_docker_command, build_process_command, container_name_for
from ..errors import ServeError
from . import health

_ERROR_PATTERN = re.compile(r"ERROR|Traceback|CANN", re.IGNORECASE)
_STOP_GRACE = 30.0  # SIGTERM 后等待秒数，超时 SIGKILL


def _run(cmd: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                          errors="replace")


class BaseLauncher(abc.ABC):
    """三模式统一接口。"""

    mode: ClassVar[str] = "base"

    def __init__(self, cfg: BenchConfig, run_dir: Path, console=None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.console = console
        self.log_path = self.run_dir / "vllm.log"
        self.started_at: float | None = None

    # ------------------------------------------------------------------ #
    def _say(self, msg: str) -> None:
        if self.console is not None:
            self.console.print(msg)

    @abc.abstractmethod
    async def start(self) -> None: ...

    async def wait_ready(self) -> float:
        """等待健康检查通过；返回启动耗时（秒）。"""
        self.started_at = time.monotonic()
        elapsed = await health.wait_for_health(
            self.cfg.client.base_url,
            self.cfg.serve.startup_timeout,
            api_key=self.cfg.client.api_key,
            console=self.console,
            is_dead=self.is_dead,
            dead_detail=self.log_tail(30),
        )
        return elapsed

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    def is_dead(self) -> bool:
        """启动/运行实体是否已退出（用于健康等待期快速失败）。"""

    def status(self) -> dict[str, Any]:
        return {"mode": self.mode, "log": str(self.log_path)}

    def log_tail(self, n: int = 200) -> str:
        if not self.log_path.is_file():
            return ""
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    # ------------------------------------------------------------------ #
    async def collect_diagnostics(self) -> Path:
        """失败诊断包：vllm.log 尾部 200 行、npu-smi 快照、进程列表 → serve_failure/。"""
        out_dir = self.run_dir / "serve_failure"
        out_dir.mkdir(parents=True, exist_ok=True)

        # log_tail may shell out to `docker logs` and _run spawns subprocesses;
        # both block, so run them in a worker thread to keep the event loop free.
        tail = await asyncio.to_thread(self.log_tail, 200)
        (out_dir / "vllm_tail.log").write_text(tail, encoding="utf-8")

        for name, cmd in (
            ("npu_smi.txt", ["npu-smi", "info"]),
            ("process_list.txt", ["bash", "-c", "ps -ef | grep -Ei 'vllm|torch' | grep -v grep"]),
        ):
            try:
                cp = await asyncio.to_thread(_run, cmd, timeout=20)
                (out_dir / name).write_text(cp.stdout or cp.stderr, encoding="utf-8")
            except (OSError, subprocess.SubprocessError):
                pass  # 非 Linux / 无 npu-smi 环境下静默跳过

        # 终端直接打印 ERROR / Traceback / CANN 关键行
        hits = [ln for ln in tail.splitlines() if _ERROR_PATTERN.search(ln)]
        if hits:
            self._say(f"[red]vllm 日志中的错误关键行（{len(hits)} 行）:[/red]")
            for ln in hits[:20]:
                self._say(f"  [dim]{ln.strip()[:240]}[/dim]")
        self._say(f"诊断包已保存: {out_dir}")
        return out_dir


class ProcessLauncher(BaseLauncher):
    """``bash -lc "<env_init> && vllm serve …"``，进程组方式启动与回收。"""

    mode: ClassVar[str] = "process"

    def __init__(self, cfg: BenchConfig, run_dir: Path, console=None):
        super().__init__(cfg, run_dir, console)
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    async def start(self) -> None:
        if self._proc is not None:
            raise ServeError("process launcher 已启动")
        log_path = str(self.log_path.resolve())
        cmd = build_process_command(self.cfg, log_path)
        self._say(f"[cyan]启动 vllm（process 模式）:[/cyan] bash -lc {cmd!r}")
        self._log_fh = open(self.run_dir / "vllm.log", "ab")
        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True  # 独立进程组，便于整组回收
        try:
            self._proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                cwd=str(self.run_dir),
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                **popen_kwargs,
            )
        except OSError as exc:
            self._log_fh.close()
            self._log_fh = None
            raise ServeError(f"启动进程失败（缺少 bash?）: {exc}") from exc
        self._say(f"  pid={self._proc.pid}，日志: {self.log_path}")

    def is_dead(self) -> bool:
        return self._proc is not None and self._proc.poll() is not None

    async def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._say(f"停止 vllm 进程组 pid={proc.pid} …")
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=_STOP_GRACE)
            except subprocess.TimeoutExpired:
                self._say(f"  { _STOP_GRACE:.0f}s 内未退出，SIGKILL 强杀")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Even SIGKILL failed (uninterruptible D state?); give up on
                    # waiting but still release our handles below.
                    self._say("[yellow]  SIGKILL 后仍未退出（D 态？），放弃等待[/yellow]")
        else:  # 非 POSIX 开发环境（无进程组语义）
            proc.terminate()
            try:
                proc.wait(timeout=_STOP_GRACE)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._say("[yellow]  kill 后仍未退出（假死？），放弃等待[/yellow]")
        self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def status(self) -> dict[str, Any]:
        st = super().status()
        st["pid"] = self._proc.pid if self._proc else None
        st["running"] = bool(self._proc and self._proc.poll() is None)
        return st


class DockerLauncher(BaseLauncher):
    """``docker run`` + 昇腾设备映射；``docker stop`` 回收。"""

    mode: ClassVar[str] = "docker"

    def __init__(self, cfg: BenchConfig, run_dir: Path, console=None):
        super().__init__(cfg, run_dir, console)
        self.container = container_name_for(cfg)
        self._started = False

    async def start(self) -> None:
        argv = build_docker_command(self.cfg, self.container)
        self._say(f"[cyan]启动 vllm（docker 模式）:[/cyan] {' '.join(argv)}")
        try:
            cp = _run(argv, timeout=120)
        except subprocess.TimeoutExpired as exc:
            # docker run hung (e.g. stalled image pull): best-effort removal so
            # the half-created container does not outlive the failed launch.
            try:
                _run(["docker", "rm", "-f", self.container], timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
            raise ServeError(
                f"docker run 超时（120s），已尝试清理容器 {self.container}: {exc}") from exc
        if cp.returncode != 0:
            raise ServeError(f"docker run 失败: {cp.stderr.strip()[:500]}")
        self._started = True
        self._say(f"  容器 {self.container} 已创建，日志: docker logs {self.container}")

    def _docker_running(self) -> bool:
        # Stays synchronous: is_dead() is a sync callback polled inside
        # health.wait_for_health and status() is sync — cannot await here.
        cp = _run(["docker", "inspect", "-f", "{{.State.Running}}", self.container], timeout=15)
        return cp.returncode == 0 and cp.stdout.strip().lower() == "true"

    def is_dead(self) -> bool:
        return self._started and not self._docker_running()

    async def stop(self) -> None:
        self._say(f"停止容器 {self.container} …")
        cp = _run(["docker", "stop", "-t", str(int(_STOP_GRACE)), self.container], timeout=_STOP_GRACE + 60)
        if cp.returncode != 0:
            self._say(f"[yellow]docker stop 返回非零: {cp.stderr.strip()[:200]}[/yellow]")
            _run(["docker", "rm", "-f", self.container])
        self._started = False

    def log_tail(self, n: int = 200) -> str:
        # Stays synchronous: passed as the eagerly-evaluated `dead_detail`
        # argument of wait_for_health (str, not awaitable).
        if self._started:
            cp = _run(["docker", "logs", "--tail", str(n), self.container], timeout=20)
            return cp.stdout + cp.stderr
        return super().log_tail(n)

    def status(self) -> dict[str, Any]:
        st = super().status()
        st["container"] = self.container
        st["running"] = self._started and self._docker_running()
        return st


class ExternalLauncher(BaseLauncher):
    """对接已启动服务：只测不管起停（仍做健康检查）。"""

    mode: ClassVar[str] = "external"

    async def start(self) -> None:
        self._say("[cyan]external 模式：[/cyan]跳过服务启动，仅做健康检查")

    def is_dead(self) -> bool:
        return False

    async def stop(self) -> None:
        self._say("external 模式：不停服务")

    def status(self) -> dict[str, Any]:
        st = super().status()
        st["running"] = None  # 由健康检查判断
        return st


_LAUNCHERS = {"process": ProcessLauncher, "docker": DockerLauncher, "external": ExternalLauncher}


def create_launcher(cfg: BenchConfig, run_dir: Path, console=None) -> BaseLauncher:
    cls = _LAUNCHERS.get(cfg.serve.mode)
    if cls is None:  # pragma: no cover - pydantic 已约束 mode
        raise ServeError(f"未知 serve.mode: {cfg.serve.mode}")
    return cls(cfg, run_dir, console)


# --------------------------------------------------------------------------- #
# 跨 CLI 调用的持久服务管理（mtest serve up/down/status/logs）
# --------------------------------------------------------------------------- #

class ServeController:
    """以状态文件记账，支持 ``serve up`` 后 CLI 退出、稍后 ``serve down``。"""

    def __init__(self, cfg: BenchConfig, console=None):
        self.cfg = cfg
        self.console = console
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in cfg.model.name)
        self.state_dir = _serve_state_dir(safe)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"

    # -- 状态 ----------------------------------------------------------- #
    def read_state(self) -> dict | None:
        if not self.state_path.is_file():
            return None
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_state(self, data: dict) -> None:
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        # Atomic write: readers never observe a torn state.json (os.replace is
        # atomic on both POSIX and Windows).
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _pid_alive(self, pid: int | None) -> bool:
        if not pid:
            return False
        try:
            if os.name == "posix":
                os.kill(pid, 0)
            else:
                cp = _run(["bash", "-c", f"kill -0 {pid} 2>/dev/null"])
                return cp.returncode == 0
            return True
        except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError):
            return False

    # -- up / down / status / logs -------------------------------------- #
    async def up(self) -> None:
        # Cross-process placeholder lock: only one `serve up` per model at a
        # time (os.open with O_CREAT|O_EXCL is atomic on POSIX and Windows).
        lock_path = self.state_path.with_suffix(".lock")
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ServeError(f"另一 serve up 进行中（锁文件已存在: {lock_path}）") from None
        try:
            await self._up_locked()
        finally:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)

    async def _up_locked(self) -> None:
        state = self.read_state()
        if state and self._entity_alive(state):
            self._say(f"[yellow]服务已在运行（{state.get('mode')}，started {state.get('started_at')}）[/yellow]")
            return
        # create_launcher stays outside the try: if it fails there is no
        # launcher entity to clean up in the except branch below.
        launcher = create_launcher(self.cfg, self.state_dir, self.console)
        try:
            await launcher.start()
            if isinstance(launcher, ProcessLauncher):
                record = {"mode": "process", "pid": launcher._proc.pid if launcher._proc else None}
            elif isinstance(launcher, DockerLauncher):
                record = {"mode": "docker", "container": launcher.container}
            else:
                record = {"mode": "external"}
            record["started_at"] = datetime.now().isoformat(timespec="seconds")
            record["port"] = self.cfg.serve.port
            record["base_url"] = self.cfg.client.base_url
            self._write_state(record)
            elapsed = await health.wait_for_health(
                self.cfg.client.base_url, self.cfg.serve.startup_timeout,
                api_key=self.cfg.client.api_key, console=self.console,
                is_dead=launcher.is_dead, dead_detail=launcher.log_tail(30))
        except Exception:
            await launcher.collect_diagnostics()
            await launcher.stop()
            raise
        self._say(f"[green]服务就绪（{elapsed:.0f}s）：{self.cfg.client.base_url}[/green]")

    def _entity_alive(self, state: dict) -> bool:
        mode = state.get("mode")
        if mode == "process":
            return self._pid_alive(state.get("pid"))
        if mode == "docker":
            cp = _run(["docker", "inspect", "-f", "{{.State.Running}}", state.get("container", "")],
                      timeout=15)
            return cp.returncode == 0 and cp.stdout.strip().lower() == "true"
        if mode == "external":
            return False  # external 无实体，总是允许重新 up
        return False

    async def down(self) -> None:
        state = self.read_state()
        if not state:
            self._say("[yellow]未找到服务状态记录，无操作[/yellow]")
            return
        mode = state.get("mode")
        if mode == "process":
            pid = state.get("pid")
            if not self._pid_alive(pid):
                self._say("进程已不在，清理状态记录")
            else:
                self._say(f"停止 vllm 进程组 pid={pid} …")
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        deadline = time.time() + _STOP_GRACE
                        while time.time() < deadline and self._pid_alive(pid):
                            time.sleep(1.0)
                        if self._pid_alive(pid):
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                else:
                    _run(["bash", "-c", f"kill {pid}"], timeout=15)
        elif mode == "docker":
            self._say(f"停止容器 {state.get('container')} …")
            _run(["docker", "stop", "-t", str(int(_STOP_GRACE)), state.get("container", "")],
                 timeout=_STOP_GRACE + 60)
        else:
            self._say("external 模式：无服务实体可停止")
        self.state_path.unlink(missing_ok=True)
        self._say("[green]已停止[/green]")

    async def status(self) -> dict[str, Any]:
        state = self.read_state() or {}
        alive = self._entity_alive(state) if state else False
        healthy, code = await health.probe_health(self.cfg.client.base_url,
                                                  api_key=self.cfg.client.api_key)
        return {
            "recorded": bool(state),
            "mode": state.get("mode"),
            "started_at": state.get("started_at"),
            "entity_alive": alive,
            "health_ok": healthy,
            "base_url": self.cfg.client.base_url,
        }

    async def logs(self, tail: int = 80) -> str:
        state = self.read_state() or {}
        if state.get("mode") == "docker":
            cp = _run(["docker", "logs", "--tail", str(tail), state.get("container", "")], timeout=20)
            return cp.stdout + cp.stderr
        return ServeController._tail_file(self.state_dir / "vllm.log", tail)

    @staticmethod
    def _tail_file(path: Path, n: int) -> str:
        if not path.is_file():
            return "(日志文件不存在)"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])

    def _say(self, msg: str) -> None:
        if self.console is not None:
            self.console.print(msg)


def _serve_state_dir(safe_name: str) -> Path:
    from ..paths import results_dir
    return results_dir() / "serve" / safe_name
