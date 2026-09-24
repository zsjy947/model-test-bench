"""web 管理台与 batch 调度测试（fake 数据，不依赖真实服务）。"""

import asyncio
import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from mtest.batch import BatchOutcome, _batch_summary_md, resolve_port_conflicts
from mtest.config import BenchConfig, ModelCfg
from mtest.webapp import create_app, scan_runs


def _make_run(root: Path, run_id: str, model: str = "m1", passed: bool = True) -> None:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run_id, "created_at": "2026-09-24T10:00:00",
        "total_duration_s": 60.0,
        "config": {"model": {"name": model, "type": "llm"}},
        "suites": [{"name": "functional", "status": "passed" if passed else "failed",
                    "duration_s": 5.0}],
    }), encoding="utf-8")
    (d / "summary.md").write_text(f"# summary {run_id}\n\n- hello", encoding="utf-8")


def test_scan_runs(tmp_path: Path, monkeypatch):
    import mtest.webapp as webapp_mod

    monkeypatch.setattr(webapp_mod, "results_dir", lambda: tmp_path)
    _make_run(tmp_path, "r1", "m1")
    _make_run(tmp_path, "r2", "m2", passed=False)
    (tmp_path / "not_a_run").mkdir()
    runs = scan_runs()
    assert {r["run_id"] for r in runs} == {"r1", "r2"}
    by_id = {r["run_id"]: r for r in runs}
    assert by_id["r1"]["all_passed"] is True
    assert by_id["r2"]["all_passed"] is False


def test_webapp_endpoints(tmp_path: Path, monkeypatch):
    import mtest.webapp as webapp_mod

    monkeypatch.setattr(webapp_mod, "results_dir", lambda: tmp_path)
    _make_run(tmp_path, "r1", "m1")
    _make_run(tmp_path, "r2", "m2", passed=False)

    async def _check() -> None:
        async with TestClient(TestServer(create_app())) as client:
            resp = await client.get("/")
            assert resp.status == 200
            text = await resp.text()
            assert "r1" in text and "r2" in text

            resp = await client.get("/run/r1")
            assert resp.status == 200
            assert "summary r1" in await resp.text()

            resp = await client.get("/run/nope")
            assert resp.status == 404

            resp = await client.get("/api/runs")
            assert resp.status == 200
            payload = await resp.json()
            assert len(payload) == 2

            resp = await client.get("/api/run/r2")
            assert (await resp.json())["config"]["model"]["name"] == "m2"

            resp = await client.get("/compare?a=r1&b=r2")
            assert resp.status == 200

    asyncio.run(_check())


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #

def _cfg(port: int, mode: str = "process") -> BenchConfig:
    cfg = BenchConfig(model=ModelCfg(name=f"m{port}", path=f"/m/{port}"))
    cfg.finalize()
    cfg.serve.mode = mode
    cfg.serve.port = port
    cfg.client.base_url = f"http://127.0.0.1:{port}/v1"
    return cfg


def test_port_conflict_rejected():
    cfgs = [_cfg(8000), _cfg(8000)]
    try:
        resolve_port_conflicts(cfgs, ["a.yaml", "b.yaml"], auto_port=False)
        assert False, "应抛出端口冲突"
    except ValueError as exc:
        assert "端口冲突" in str(exc)


def test_port_auto_assignment():
    cfgs = [_cfg(8000), _cfg(8000), _cfg(8000), _cfg(8001)]
    resolve_port_conflicts(cfgs, ["a.yaml", "b.yaml", "c.yaml", "d.yaml"], auto_port=True)
    ports = [c.serve.port for c in cfgs]
    # 顺序分配：先到先得，冲突者递增到未占用端口（d 的 8001 已被 b 占用 → 8003）
    assert ports == [8000, 8001, 8002, 8003]
    assert len(set(ports)) == 4
    # base_url 同步
    assert cfgs[1].client.base_url == "http://127.0.0.1:8001/v1"
    assert cfgs[3].client.base_url == "http://127.0.0.1:8003/v1"


def test_external_mode_no_conflict():
    cfgs = [_cfg(8000, mode="external"), _cfg(8000, mode="external")]
    resolve_port_conflicts(cfgs, ["a.yaml", "b.yaml"], auto_port=False)  # 不抛
    assert all(c.serve.port == 8000 for c in cfgs)


def test_batch_summary_md():
    outcomes = [
        BatchOutcome(config="/x/a.yaml", model="ma", run_id="r1", run_dir="/r1",
                     ok=True, suites=[{"name": "perf", "status": "passed"}]),
        BatchOutcome(config="/x/b.yaml", model="mb", run_id=None, run_dir=None,
                     ok=False, error="ConfigError: bad"),
    ]
    md = _batch_summary_md(outcomes, parallel=2)
    assert "通过 1/2" in md
    assert "a.yaml" in md and "b.yaml" in md
