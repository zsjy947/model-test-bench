"""vllm 命令行生成测试（设计 §3.3，含 --port/--host 注入）。"""

from mtest.config import (BenchConfig, ModelCfg, build_docker_command,
                          build_process_command, build_serve_command, build_vllm_args,
                          container_name_for)


def make_cfg() -> BenchConfig:
    cfg = BenchConfig(model=ModelCfg(name="demo", path="/m/demo"))
    return cfg


def test_basic_args():
    argv = build_vllm_args(make_cfg())
    assert argv[0:3] == ["vllm", "serve", "/m/demo"]
    assert "--trust-remote-code" in argv
    assert "--tensor-parallel-size" in argv


def test_bool_flag():
    cfg = make_cfg()
    cfg.serve.args["disable-log-stats"] = True
    cfg.serve.args["enable-prefix-caching"] = False
    argv = build_vllm_args(cfg)
    assert "--disable-log-stats" in argv
    assert "--enable-prefix-caching" not in argv


def test_dict_arg_compact_json():
    cfg = make_cfg()
    cfg.serve.args["additional-config"] = {"torchair_graph_config": {"enabled": True}}
    argv = build_vllm_args(cfg)
    i = argv.index("--additional-config")
    assert argv[i + 1] == '{"torchair_graph_config":{"enabled":true}}'


def test_list_arg_joined():
    cfg = make_cfg()
    cfg.serve.args["allowed-local-media"] = ["/a", "/b"]
    argv = build_vllm_args(cfg)
    assert argv[argv.index("--allowed-local-media") + 1] == "/a,/b"


def test_command_escape_hatch():
    cfg = make_cfg()
    cfg.serve.command = "python launch_ocr_server.py --port 8000"
    assert build_serve_command(cfg) == "python launch_ocr_server.py --port 8000"
    assert "python launch_ocr_server.py" in build_process_command(cfg)


def test_process_command_env_init_and_redirect():
    cfg = make_cfg()
    cfg.serve.env_init = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    cmd = build_process_command(cfg, "/tmp/run/vllm.log")
    assert cmd.startswith("(source /usr/local/Ascend/ascend-toolkit/set_env.sh) && vllm serve")
    assert cmd.endswith("> /tmp/run/vllm.log 2>&1")


def test_docker_command_devices_and_mounts():
    cfg = make_cfg()
    cfg.serve.mode = "docker"
    cfg.serve.docker.image = "quay.io/ascend/vllm-ascend:latest"
    cfg.serve.docker.devices = [0, 1]
    cfg.serve.docker.mounts = ["/data/models:/data/models:ro"]
    argv = build_docker_command(cfg, "mtest-demo")
    assert argv[0:6] == ["docker", "run", "-d", "--rm", "--name", "mtest-demo"]
    assert "--device" in argv and "/dev/davinci0" in argv and "/dev/davinci1" in argv
    for dev in ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"):
        assert dev in argv
    assert "-v" in argv and "/data/models:/data/models:ro" in argv
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1" in argv
    assert argv[-1:] != [] and "vllm" in argv


def test_container_name_sanitized():
    cfg = BenchConfig(model=ModelCfg(name="Qwen/VL: 7B", path="/m/x"))
    name = container_name_for(cfg)
    assert all(ch.isalnum() or ch in "-_." for ch in name)
    assert name.startswith("mtest-")


def test_port_host_injected():
    cfg = make_cfg()
    cfg.serve.port = 8123
    argv = build_vllm_args(cfg)
    i = argv.index("--port")
    assert argv[i + 1] == "8123"
    assert argv.count("--port") == 1
    # default host "0.0.0.0" is injected too
    j = argv.index("--host")
    assert argv[j + 1] == "0.0.0.0"


def test_port_host_not_duplicated_when_in_args():
    cfg = make_cfg()
    cfg.serve.port = 8123
    cfg.serve.args["port"] = 9000
    cfg.serve.args["host"] = "127.0.0.1"
    argv = build_vllm_args(cfg)
    assert argv.count("--port") == 1
    assert argv[argv.index("--port") + 1] == "9000"
    assert argv.count("--host") == 1
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


def test_docker_command_includes_port_host():
    cfg = make_cfg()
    cfg.serve.mode = "docker"
    cfg.serve.docker.image = "quay.io/ascend/vllm-ascend:latest"
    cfg.serve.port = 8123
    argv = build_docker_command(cfg, "mtest-demo")
    i = argv.index("--port")
    assert argv[i + 1] == "8123"
    assert argv.count("--port") == 1
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
