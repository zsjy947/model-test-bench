"""doctor 预检：起服务前的环境体检（§12 风险 R1 落地）。

检查项覆盖：Python/bash/npu-smi 可用性、NPU 卡数与 TP 匹配、模型路径、端口占用、
docker 可用性、vllm 与 vllm-ascend 版本匹配（咨询性静态表）、内置数据完整性、
serve.args 已知参数 lint（提示式，不拦截透传）。
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import BenchConfig
from .monitor import collect_npu_env
from .paths import data_dir
from .report import _pip_show_version

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    hint: str = ""


def _package_data(name: str) -> Path:
    return Path(__file__).resolve().parent / "data" / name


def load_version_matrix() -> dict:
    return yaml.safe_load(_package_data("version_matrix.yaml").read_text(encoding="utf-8"))


def load_known_args() -> set[str]:
    payload = yaml.safe_load(_package_data("vllm_args.yaml").read_text(encoding="utf-8"))
    return set(payload.get("args", []))


# --------------------------------------------------------------------------- #
# 单项检查
# --------------------------------------------------------------------------- #

def check_runtime() -> CheckResult:
    ok_py = sys.version_info >= (3, 10)
    bash = shutil.which("bash")
    if ok_py and bash:
        return CheckResult("运行时", OK, f"Python {sys.version.split()[0]}，bash 可用")
    return CheckResult(
        "运行时", FAIL,
        f"Python {sys.version.split()[0]}（需 ≥3.10），bash={bash or '缺失'}",
        hint="process 模式依赖 bash 执行 env_init 与启动命令")


def check_npu(cfg: BenchConfig | None) -> CheckResult:
    if shutil.which("npu-smi") is None:
        return CheckResult("NPU", WARN, "未找到 npu-smi 命令",
                           hint="采样将降级；若目标机确有 NPU，检查 PATH 与驱动安装")
    info = collect_npu_env()
    if not info:
        return CheckResult("NPU", WARN, "npu-smi 存在但输出解析失败",
                           hint="可能 CANN 版本输出格式变化，见 mtest/monitor.py 解析器")
    n_npu = info.get("npu_count") or 0
    detail = f"{n_npu} 卡，单卡 HBM {info.get('hbm_total_mb')} MB（{info.get('npu_smi_header')}）"
    if cfg is not None:
        tp = cfg.serve.tensor_parallel_size()
        if tp is not None and tp > n_npu:
            return CheckResult("NPU", FAIL, detail,
                               hint=f"TP={tp} 超过可见卡数 {n_npu}，vllm 启动会失败")
    return CheckResult("NPU", OK, detail)


def check_model_path(cfg: BenchConfig | None) -> CheckResult | None:
    if cfg is None:
        return None
    path = Path(cfg.model.path)
    if path.is_dir():
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
        detail = str(path) + ("（含权重文件）" if has_weights else "（未见 safetensors/bin）")
        return CheckResult("模型路径", OK if has_weights else WARN, detail,
                           hint="未见权重文件：分片命名特殊或路径为占位")
    if cfg.serve.mode == "docker":
        return CheckResult("模型路径", WARN, f"{path}（宿主不可见）",
                           hint="docker 模式下 path 指容器内路径，请核对 mounts 挂载")
    return CheckResult("模型路径", FAIL, f"{path} 不存在",
                       hint="检查 model.path；docker 模式需为容器内路径")


def check_port(cfg: BenchConfig | None) -> CheckResult | None:
    if cfg is None:
        return None
    host, port = cfg.serve.host, cfg.serve.port
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with socket.create_connection((probe_host, port), timeout=1.0):
            occupied = True
    except OSError:
        occupied = False
    if cfg.serve.mode == "external":
        # external 期望服务已在运行
        return CheckResult(
            "端口", OK if occupied else FAIL,
            f"{probe_host}:{port} {'已有服务监听（external 期望如此）' if occupied else '无服务（external 模式要求先启动服务）'}")
    if occupied:
        return CheckResult("端口", FAIL, f"{probe_host}:{port} 已被占用",
                           hint="换 serve.port 或停掉占用进程（ss -lntp | grep %d）" % port)
    return CheckResult("端口", OK, f"{probe_host}:{port} 空闲")


def check_docker(cfg: BenchConfig | None) -> CheckResult | None:
    if cfg is None or cfg.serve.mode != "docker":
        return None
    if shutil.which("docker") is None:
        return CheckResult("docker", FAIL, "未找到 docker 命令")
    try:
        cp = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=15)
        if cp.returncode != 0:
            return CheckResult("docker", FAIL, f"docker info 失败: {cp.stderr[:120]}")
        image = cfg.serve.docker.image or ""
        return CheckResult("docker", OK, f"docker 可用，image={image or '（未配置!）'}",
                           hint="" if image else "serve.docker.image 未配置")
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult("docker", FAIL, f"docker 探测异常: {exc}")


def check_versions(cfg: BenchConfig | None) -> CheckResult:
    vllm = _pip_show_version("vllm")
    ascend = _pip_show_version("vllm-ascend")
    if vllm is None and ascend is None:
        return CheckResult("vllm 版本", WARN,
                           "当前环境未安装 vllm / vllm-ascend（mtest 独立 venv 属正常）",
                           hint="在 vllm 服务端环境执行 doctor，或以服务端 /version 为准")
    matrix = load_version_matrix()
    pairs = {tuple(p) for p in (matrix.get("pairs") or [])}
    detail = f"vllm={vllm or '-'} vllm-ascend={ascend or '-'}"
    if (vllm, ascend) in pairs:
        return CheckResult("vllm 版本", OK, detail + "（在已知匹配表中）")
    return CheckResult("vllm 版本", WARN, detail + "（不在静态匹配表）",
                       hint="以 vllm-ascend 官方 README 匹配表为准；不匹配是兼容问题第一嫌疑")


def check_data_integrity() -> CheckResult:
    required = [
        data_dir() / "prompts" / "zh.jsonl",
        data_dir() / "prompts" / "en.jsonl",
        data_dir() / "longdoc" / "zh.txt",
        data_dir() / "cases" / "llm_smoke.yaml",
        data_dir() / "images" / "ocr",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return CheckResult("内置数据", WARN, f"缺失: {missing}",
                           hint="重新运行 scripts/gen_data.py 生成")
    return CheckResult("内置数据", OK, "prompt 池 / 长文本素材 / 用例 / OCR 样例齐全")


def lint_serve_args(cfg: BenchConfig | None) -> CheckResult | None:
    if cfg is None or cfg.serve.command:
        return None  # 逃生舱整体替代，lint 无意义
    known = load_known_args()
    reserved = {"tensor-parallel-size", "max-model-len", "gpu-memory-utilization", "dtype"}
    unknown = [k for k in cfg.serve.args if k not in known and k not in reserved]
    if not unknown:
        return CheckResult("args lint", OK, f"{len(cfg.serve.args)} 个参数均在已知清单")
    return CheckResult(
        "args lint", WARN,
        f"不在已知参数清单: {sorted(unknown)}",
        hint="可能是 vllm-ascend 专有/新版本参数（透传不受影响），或拼写错误；"
             "用 mtest validate 预览生成命令肉眼确认")


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #

def run_doctor(cfg: BenchConfig | None, console) -> list[CheckResult]:
    """执行全部预检并打印；返回结果列表。"""
    checks = [check_runtime(), check_npu(cfg), check_model_path(cfg), check_port(cfg),
              check_docker(cfg), check_versions(cfg), check_data_integrity(),
              lint_serve_args(cfg)]
    checks = [c for c in checks if c is not None]
    for c in checks:
        style = {OK: "green", WARN: "yellow", FAIL: "red"}[c.status]
        mark = {OK: "✓", WARN: "!", FAIL: "✗"}[c.status]
        console.print(f"  [{style}]{mark} {c.name}[/{style}]  {c.detail}")
        if c.hint and c.status != OK:
            console.print(f"      [dim]→ {c.hint}[/dim]")
    n_fail = sum(1 for c in checks if c.status == FAIL)
    n_warn = sum(1 for c in checks if c.status == WARN)
    console.print(f"[bold]doctor: {len(checks) - n_fail - n_warn} ok / "
                  f"{n_warn} warn / {n_fail} fail[/bold]")
    return checks
