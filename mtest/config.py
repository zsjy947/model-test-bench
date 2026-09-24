"""配置系统：三层合并（defaults ← model yaml ← CLI 覆盖）+ pydantic 校验 +
vllm 启动命令生成。

合并规则（设计 §3.1）：
- 深合并：dict 递归合并；list 与标量整体覆盖（不逐项合并）；
- 支持 yaml 与 json 两种格式；
- 校验错误在起服务前暴露。
"""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import ConfigError

ModelType = Literal["llm", "embedding", "multimodal"]
ServeMode = Literal["process", "docker", "external"]


# --------------------------------------------------------------------------- #
# 深合并与文件加载
# --------------------------------------------------------------------------- #

def deep_merge(base: dict, override: dict) -> dict:
    """递归深合并：dict 合并，list/标量整体覆盖。不修改入参。"""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_structured_file(path: str | Path) -> dict:
    """加载 yaml / json 配置文件为 dict；空文件返回 {}。"""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"配置文件不存在: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - 文件系统错误
        raise ConfigError(f"配置文件读取失败: {p}: {exc}") from exc
    if not text.strip():
        return {}
    try:
        if p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except Exception as exc:
        raise ConfigError(f"配置文件解析失败: {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射（dict）: {p}")
    return data


# --------------------------------------------------------------------------- #
# pydantic 模型（字段清单见设计 §3.2）
# --------------------------------------------------------------------------- #

class ModelCfg(BaseModel):
    """模型三要素与元信息。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="模型标识（用于报告/目录命名）")
    path: str = Field(..., description="模型权重路径（本地目录）")
    type: ModelType = "llm"
    served_name: str | None = Field(None, description="vllm --served-model-name，缺省取 path 尾段")
    trust_remote_code: bool = True

    @field_validator("name", "path")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("不能为空")
        return v.strip()

    def resolved_served_name(self) -> str:
        """served-model-name；缺省取 path 尾段（兼容 HF id 与尾斜杠）。"""
        if self.served_name:
            return self.served_name
        segs = [s for s in self.path.replace("\\", "/").split("/") if s]
        if not segs:
            raise ConfigError(f"无法从 model.path 推断 served-model-name: {self.path!r}")
        return segs[-1]


class DockerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str | None = None
    devices: list[int] = Field(default_factory=lambda: list(range(8)))
    mounts: list[str] = Field(default_factory=list, description="host:container[:mode] 列表")
    extra_args: list[str] = Field(default_factory=list, description="docker run 附加参数")

    @field_validator("devices")
    @classmethod
    def _devices_range(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("docker.devices 不能为空（至少映射一张卡）")
        bad = [d for d in v if not 0 <= d <= 7]
        if bad:
            raise ValueError(f"设备号必须在 0-7 之间: {bad}")
        return v


class ServeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ServeMode = "process"
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1024, le=65535)
    startup_timeout: int = Field(1800, ge=10, description="健康检查超时（秒）")
    env_init: str | None = Field(None, description="启动前环境激活命令串")
    args: dict[str, Any] = Field(
        default_factory=lambda: {
            "tensor-parallel-size": 8,
            "max-model-len": 32768,
            "gpu-memory-utilization": 0.9,
            "dtype": "float16",
        },
        description="自动生成 vllm serve 命令行参数（透传）",
    )
    command: str | None = Field(None, description="逃生舱：非空则整体替代自动生成的启动命令")
    docker: DockerCfg = Field(default_factory=DockerCfg)

    def tensor_parallel_size(self) -> int | None:
        v = self.args.get("tensor-parallel-size")
        return int(v) if v is not None else None

    def max_model_len(self) -> int | None:
        v = self.args.get("max-model-len")
        return int(v) if v is not None else None


class ClientCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str | None = None
    request_timeout: float = Field(600, ge=1)


class PerfCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    dataset: str = "prompts"  # prompts | random | 自定义 jsonl 路径
    input_lens: list[int] = Field(default_factory=lambda: [128, 1024, 4096])
    output_len: int = Field(256, ge=1)
    concurrency: list[int] = Field(default_factory=lambda: [1, 8, 32, 64])
    duration: int = Field(120, ge=1, description="每档时长秒（与 num_requests 双上限，先到为准）")
    num_requests: int = Field(200, ge=1)
    rounds: int = Field(3, ge=1, description="每档重复次数，取均值")
    warmup: int = Field(2, ge=0, description="每档预热请求数")
    stream: bool = True


class LongctxCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    input_lens: list[int] = Field(default_factory=lambda: [16384, 32768])
    concurrency: list[int] = Field(default_factory=lambda: [1, 8])
    output_len: int = Field(128, ge=1)
    duration: int = Field(600, ge=1)
    num_requests: int = Field(32, ge=1)
    rounds: int = Field(1, ge=1)
    warmup: int = Field(1, ge=0)
    stream: bool = True


class FunctionalCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    cases: str = "data/cases/"  # 内置用例集，支持追加自定义 yaml 用例


class EmbeddingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 8, 32, 128])
    concurrency: list[int] = Field(default_factory=lambda: [1, 8, 32])
    dim: int | None = Field(None, description="声明维度用于校验；null 则取首次响应")
    duration: int = Field(60, ge=1)
    num_requests: int = Field(100, ge=1)
    rounds: int = Field(1, ge=1)
    warmup: int = Field(2, ge=0)
    similarity_min_separation: float = Field(
        0.15, description="区分度 sanity：相似对均值 - 无关对均值 的最小间隔"
    )


class OcrCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    image_dir: str = "data/images/ocr"
    ground_truth: str | None = None  # 可选 json（文件名→标准答案）
    concurrency: list[int] = Field(default_factory=lambda: [1, 8])
    duration: int = Field(120, ge=1)
    num_requests: int = Field(60, ge=1)
    rounds: int = Field(1, ge=1)
    warmup: int = Field(1, ge=0)
    fault_cases: bool = Field(True, description="是否执行损坏/超大图片容错用例")


class TestsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perf: PerfCfg = Field(default_factory=PerfCfg)
    longctx: LongctxCfg = Field(default_factory=LongctxCfg)
    functional: FunctionalCfg = Field(default_factory=FunctionalCfg)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    ocr: OcrCfg = Field(default_factory=OcrCfg)


class MonitorCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    npu_interval: int = Field(5, ge=1, description="采样间隔秒")


class BenchConfig(BaseModel):
    """全量生效配置（合并后的最终形态）。"""

    model_config = ConfigDict(extra="forbid")

    model: ModelCfg
    serve: ServeCfg = Field(default_factory=ServeCfg)
    client: ClientCfg = Field(default_factory=ClientCfg)
    tests: TestsCfg = Field(default_factory=TestsCfg)
    monitor: MonitorCfg = Field(default_factory=MonitorCfg)

    warnings: list[str] = Field(default_factory=list, description="校验/裁剪产生的告警")

    # ------------------------------------------------------------------ #
    # 跨字段校验与调整（设计 §3.4）
    # ------------------------------------------------------------------ #

    def finalize(self) -> list[str]:
        """跨字段校验与自动调整；返回告警列表并写入 self.warnings。

        - tensor-parallel-size ∈ {1,2,4,8}
        - input_len + output_len < max-model-len（越界自动裁剪并警告）
        - model.type 与套件匹配（不匹配的套件自动禁用并警告）
        """
        warns: list[str] = []

        tp = self.serve.tensor_parallel_size()
        if tp is not None and tp not in (1, 2, 4, 8):
            raise ConfigError(
                f"serve.args['tensor-parallel-size']={tp} 非法，必须 ∈ {{1,2,4,8}}"
            )

        # 类型-套件匹配（先于长度裁剪，避免对将禁用的套件产生冗余告警）
        if self.model.type == "llm":
            if self.tests.embedding.enabled:
                self.tests.embedding.enabled = False
                warns.append("embedding 套件仅适用于 embedding 模型，llm 已自动禁用")
            if self.tests.ocr.enabled:
                self.tests.ocr.enabled = False
                warns.append("ocr 套件仅适用于 multimodal 模型，llm 已自动禁用")
        if self.model.type == "embedding":
            if self.tests.perf.enabled:
                self.tests.perf.enabled = False
                warns.append("embedding 模型不支持 perf 压测套件，已自动禁用")
            if self.tests.longctx.enabled:
                self.tests.longctx.enabled = False
                warns.append("embedding 模型不支持 longctx 长序列套件，已自动禁用")
            if self.tests.ocr.enabled:
                self.tests.ocr.enabled = False
                warns.append("ocr 套件仅适用于 multimodal 模型，embedding 已自动禁用")
        if self.model.type == "multimodal":
            if self.tests.perf.enabled:
                self.tests.perf.enabled = False
                warns.append("multimodal 模型不走 perf 套件（请使用 ocr 套件），已自动禁用")
            if self.tests.longctx.enabled:
                self.tests.longctx.enabled = False
                warns.append("longctx 套件当前仅支持 llm 文本模型，multimodal 已自动禁用")
            if self.tests.embedding.enabled:
                self.tests.embedding.enabled = False
                warns.append("embedding 套件仅适用于 embedding 模型，multimodal 已自动禁用")

        mml = self.serve.max_model_len()
        if mml is not None:
            for suite_name in ("perf", "longctx"):
                suite = getattr(self.tests, suite_name)
                if not suite.enabled:
                    continue
                cap = int(mml) - suite.output_len - 64  # 64 token 模板/安全余量
                if cap <= 0:
                    warns.append(
                        f"[{suite_name}] max-model-len({mml}) - output_len({suite.output_len}) - 64 "
                        f"<= 0，套件自动禁用"
                    )
                    suite.enabled = False
                    continue
                clipped = []
                for length in suite.input_lens:
                    if length <= 0:
                        continue
                    if length > cap:
                        warns.append(
                            f"[{suite_name}] input_len {length} + output_len {suite.output_len} "
                            f"超过 max-model-len {mml}（含 64 token 余量），自动裁剪为 {cap}"
                        )
                        clipped.append(cap)
                    else:
                        clipped.append(length)
                dedup: list[int] = []
                for length in clipped:
                    if length not in dedup:
                        dedup.append(length)
                suite.input_lens = dedup
                if not dedup:
                    warns.append(f"[{suite_name}] 裁剪后无可用档位，套件自动禁用")
                    suite.enabled = False

        self.warnings = warns
        return warns

    def enabled_suites(self) -> list[str]:
        """按执行顺序返回启用的套件名。"""
        order = ["functional", "perf", "longctx", "embedding", "ocr"]
        return [n for n in order if getattr(self.tests, n).enabled]


# --------------------------------------------------------------------------- #
# 加载与 CLI 覆盖
# --------------------------------------------------------------------------- #

def load_config(
    model_yaml: str | Path | None = None,
    *,
    defaults_yaml: str | Path | None = None,
    overrides: dict | None = None,
) -> BenchConfig:
    """三层合并加载并校验。"""
    merged = load_merged_dict(model_yaml, defaults_yaml=defaults_yaml, overrides=overrides)
    try:
        cfg = BenchConfig(**merged)
    except Exception as exc:
        raise ConfigError(f"配置校验失败: {exc}") from exc
    cfg.finalize()
    return cfg


def load_merged_dict(
    model_yaml: str | Path | None = None,
    *,
    defaults_yaml: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """返回 defaults ← model ← overrides 深合并后的原始 dict（未做 pydantic 校验）。"""
    from .paths import find_defaults_config

    base = load_structured_file(defaults_yaml or find_defaults_config())
    if model_yaml:
        base = deep_merge(base, load_structured_file(model_yaml))
    if overrides:
        base = deep_merge(base, overrides)
    return base


def apply_cli_overrides(cfg: BenchConfig, *, suites: list[str] | None = None,
                        concurrency: list[int] | None = None) -> BenchConfig:
    """应用 CLI 覆盖项（--suite / --concurrency），原地修改并返回。"""
    known = {"functional", "perf", "longctx", "embedding", "ocr"}
    if suites is not None:
        unknown = [s for s in suites if s not in known]
        if unknown:
            raise ConfigError(f"未知套件: {unknown}，可选: {sorted(known)}")
        for name in known:
            getattr(cfg.tests, name).enabled = name in suites
    if concurrency is not None:
        if not concurrency or any(c < 1 for c in concurrency):
            raise ConfigError("--concurrency 必须为正整数列表")
        cfg.tests.perf.concurrency = sorted(set(concurrency))
    return cfg


# --------------------------------------------------------------------------- #
# vllm 命令生成（设计 §3.3）
# --------------------------------------------------------------------------- #

def build_vllm_args(cfg: BenchConfig) -> list[str]:
    """生成 ``vllm serve`` 参数列表（不含 env_init / 重定向）。

    - 布尔值 → 单 flag（true 时 ``--key``，false 时省略）
    - dict → ``--key '<紧凑json>'``（兼容 vllm-ascend additional-config）
    - list → 逗号连接
    - 其余 → ``--key value``
    """
    argv = ["vllm", "serve", cfg.model.path]
    if cfg.model.served_name and cfg.model.served_name != Path(cfg.model.path).name:
        argv += ["--served-model-name", cfg.model.served_name]
    if cfg.model.trust_remote_code:
        argv.append("--trust-remote-code")
    for key, val in cfg.serve.args.items():
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                argv.append(f"--{key}")
        elif isinstance(val, dict):
            argv += [f"--{key}", json.dumps(val, ensure_ascii=False, separators=(",", ":"))]
        elif isinstance(val, (list, tuple)):
            argv += [f"--{key}", ",".join(str(x) for x in val)]
        else:
            argv += [f"--{key}", str(val)]
    return argv


def build_serve_command(cfg: BenchConfig) -> str:
    """进程模式启动命令（bash -lc 的执行体；command 逃生舱优先）。"""
    if cfg.serve.command:
        return cfg.serve.command
    return shlex.join(build_vllm_args(cfg))


def build_process_command(cfg: BenchConfig, log_path: str = "vllm.log") -> str:
    """process 模式 ``bash -lc`` 执行的完整命令串（env_init + 命令 + 日志重定向）。"""
    cmd = build_serve_command(cfg)
    parts = []
    if cfg.serve.env_init:
        parts.append(f"({cfg.serve.env_init})")
    parts.append(cmd)
    joined = " && ".join(parts)
    return f"{joined} > {shlex.quote(str(log_path))} 2>&1"


_ASCEND_DEVICES = ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc")


def container_name_for(cfg: BenchConfig) -> str:
    """容器名：mtest-<model.name>（去除不安全字符）。"""
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in cfg.model.name)
    return f"mtest-{safe}".strip("-")


def build_docker_command(cfg: BenchConfig, container_name: str | None = None) -> list[str]:
    """docker 模式 ``docker run`` 参数列表（宿主 network=host，容器内执行 vllm serve）。

    昇腾设备映射：/dev/davinci<N> + /dev/davinci_manager + /dev/devmm_svm +
    /dev/hisi_hdc；模型目录等通过 ``serve.docker.mounts`` 挂载（容器内路径需与
    model.path 一致）。
    """
    d = cfg.serve.docker
    if not d.image:
        raise ConfigError("serve.mode=docker 需要配置 serve.docker.image")
    argv = ["docker", "run", "-d", "--rm", "--name", container_name or container_name_for(cfg),
            "--network", "host"]
    for dev in d.devices:
        argv += ["--device", f"/dev/davinci{dev}"]
    for dev in _ASCEND_DEVICES:
        argv += ["--device", dev]
    for mount in d.mounts:
        argv += ["-v", mount]
    argv += ["-e", f"ASCEND_RT_VISIBLE_DEVICES={','.join(str(x) for x in d.devices)}"]
    argv += d.extra_args
    argv.append(d.image)
    argv += build_vllm_args(cfg)
    return argv
