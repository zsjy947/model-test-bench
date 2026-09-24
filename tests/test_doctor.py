"""doctor 预检测试（不依赖真实 NPU / vllm）。"""

from pathlib import Path

from mtest.config import BenchConfig, ModelCfg
from mtest.doctor import (FAIL, OK, WARN, check_model_path, check_port, lint_serve_args,
                          load_known_args, load_version_matrix)


def make_cfg(tmp_path: Path, **model_kwargs) -> BenchConfig:
    params = {"name": "t", "path": str(tmp_path / "model")}
    params.update(model_kwargs)
    cfg = BenchConfig(model=ModelCfg(**params))
    cfg.finalize()
    return cfg


def test_model_path_missing_fails(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    check = check_model_path(cfg)
    assert check.status == FAIL


def test_model_path_with_weights_ok(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x")
    cfg = make_cfg(tmp_path)
    assert check_model_path(cfg).status == OK


def test_model_path_no_weights_warns(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    cfg = make_cfg(tmp_path)
    assert check_model_path(cfg).status == WARN


def test_model_path_none_when_no_cfg():
    assert check_model_path(None) is None


def test_port_free_vs_occupied(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    cfg.serve.host = "127.0.0.1"
    cfg.serve.port = 59999  # 基本空闲端口
    assert check_port(cfg).status == OK

    import socket

    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)  # backlog 足够大，允许多次探测连接排队
        port = srv.getsockname()[1]
        cfg.serve.port = port
        # 占用 + process 模式 → fail
        assert check_port(cfg).status == FAIL
        # 占用 + external 模式 → ok（期望服务在跑）
        cfg.serve.mode = "external"
        assert check_port(cfg).status == OK


def test_lint_args(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    check = lint_serve_args(cfg)
    assert check.status == OK  # defaults 中的参数均在清单

    cfg.serve.args["my-typo-argg"] = 1
    check = lint_serve_args(cfg)
    assert check.status == WARN
    assert "my-typo-argg" in check.detail

    cfg.serve.args.pop("my-typo-argg")
    cfg.serve.command = "python run.py"
    assert lint_serve_args(cfg) is None  # 逃生舱生效时不 lint


def test_version_matrix_and_known_args_load():
    matrix = load_version_matrix()
    assert ("0.9.2", "0.9.2") in {tuple(p) for p in matrix["pairs"]}
    known = load_known_args()
    assert {"tensor-parallel-size", "max-model-len", "additional-config",
            "limit-mm-per-prompt", "task"} <= known
