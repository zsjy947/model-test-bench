"""config 系统测试：深合并 / 三层加载 / 校验裁剪 / CLI 覆盖。"""

from pathlib import Path

import pytest

from mtest.config import (BenchConfig, ModelCfg, apply_cli_overrides, deep_merge,
                          load_config, load_merged_dict, load_structured_file)
from mtest.errors import ConfigError

REPO = Path(__file__).resolve().parents[1]


def base_overrides() -> dict:
    return {"model": {"name": "t", "path": "/m/t"}}


# --------------------------------------------------------------------------- #
# 深合并
# --------------------------------------------------------------------------- #

def test_deep_merge_dict_recursive():
    out = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3, "z": 4}})
    assert out == {"a": {"x": 1, "y": 3, "z": 4}}


def test_deep_merge_list_overrides_wholesale():
    out = deep_merge({"a": [1, 2, 3]}, {"a": [9]})
    assert out == {"a": [9]}


def test_deep_merge_no_mutation():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    deep_merge(base, override)
    assert base == {"a": {"x": 1}}


# --------------------------------------------------------------------------- #
# 三层加载
# --------------------------------------------------------------------------- #

def test_load_config_defaults_then_model(tmp_path: Path):
    model_yaml = tmp_path / "m.yaml"
    model_yaml.write_text(
        "model:\n  name: demo\n  path: /m/demo\nserve:\n  args:\n"
        "    tensor-parallel-size: 2\n", encoding="utf-8")
    cfg = load_config(model_yaml)
    assert cfg.model.type == "llm"                      # 继承 defaults
    assert cfg.serve.args["tensor-parallel-size"] == 2  # 模型覆盖
    assert cfg.serve.port == 8000                       # 继承 defaults


def test_load_config_overrides_layer(tmp_path: Path):
    model_yaml = tmp_path / "m.yaml"
    model_yaml.write_text("model:\n  name: d\n  path: /m/d\n", encoding="utf-8")
    cfg = load_config(model_yaml, overrides={"serve": {"port": 9000}})
    assert cfg.serve.port == 9000


def test_load_json_config(tmp_path: Path):
    model_json = tmp_path / "m.json"
    model_json.write_text('{"model": {"name": "j", "path": "/m/j"}}', encoding="utf-8")
    cfg = load_config(model_json)
    assert cfg.model.name == "j"


def test_load_structured_file_missing_raises():
    with pytest.raises(ConfigError):
        load_structured_file(REPO / "nope.yaml")


def test_unknown_field_rejected(tmp_path: Path):
    model_yaml = tmp_path / "m.yaml"
    model_yaml.write_text("model:\n  name: d\n  path: /m/d\n  typo_field: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(model_yaml)


# --------------------------------------------------------------------------- #
# 校验与裁剪
# --------------------------------------------------------------------------- #

def make_cfg(**model_kwargs) -> BenchConfig:
    params = {"name": "t", "path": "/m/t"}
    params.update(model_kwargs)
    cfg = BenchConfig(model=ModelCfg(**params))
    cfg.finalize()
    return cfg


def test_tp_validation():
    cfg = BenchConfig(model=ModelCfg(name="t", path="/m/t"))
    cfg.serve.args["tensor-parallel-size"] = 3
    with pytest.raises(ConfigError):
        cfg.finalize()


def test_port_range():
    with pytest.raises(Exception):
        BenchConfig(model=ModelCfg(name="t", path="/m/t"),
                    **{"serve": {"port": 80}})


def test_len_clipping():
    cfg2 = BenchConfig(model=ModelCfg(name="t", path="/m/t"))
    cfg2.serve.args["max-model-len"] = 2048
    warns = cfg2.finalize()
    # perf: cap = 2048-256-64 = 1728 → 4096 裁剪，128/1024 保留
    assert cfg2.tests.perf.input_lens == [128, 1024, 1728]
    # longctx: cap = 2048-128-64 = 1856 → 两档均裁剪合并
    assert cfg2.tests.longctx.input_lens == [1856]
    assert any("自动裁剪" in w for w in warns)


def test_type_suite_matrix():
    assert make_cfg().enabled_suites() == ["functional", "perf", "longctx"]
    assert make_cfg(type="embedding").enabled_suites() == ["functional", "embedding"]
    assert make_cfg(type="multimodal").enabled_suites() == ["functional", "ocr"]


def test_served_name_defaults_to_path_tail():
    assert make_cfg().model.resolved_served_name() == "t"
    cfg = make_cfg()
    cfg.model.path = "/data/models/Qwen2.5-7B/"
    assert cfg.model.resolved_served_name() == "Qwen2.5-7B"


# --------------------------------------------------------------------------- #
# CLI 覆盖
# --------------------------------------------------------------------------- #

def test_cli_overrides_suites():
    cfg = make_cfg()
    apply_cli_overrides(cfg, suites=["perf"])
    assert cfg.enabled_suites() == ["perf"]


def test_cli_overrides_unknown_suite():
    with pytest.raises(ConfigError):
        apply_cli_overrides(make_cfg(), suites=["no-such-suite"])


def test_cli_overrides_concurrency():
    cfg = make_cfg()
    apply_cli_overrides(cfg, concurrency=[8, 1, 8])
    assert cfg.tests.perf.concurrency == [1, 8]


def test_load_merged_dict_layers():
    merged = load_merged_dict(overrides={"model": {"name": "o", "path": "/m/o"}})
    assert merged["model"]["name"] == "o"
    assert merged["serve"]["mode"] == "process"
