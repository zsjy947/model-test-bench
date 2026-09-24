# AGENTS.md — 仓库协作约定

面向 AI 代理与人类贡献者的仓库工作约定。

## 项目一句话

Ascend 910B + vllm(-ascend) 平台的一键模型测试工具：起服务 → 测试（perf/longctx/functional/embedding/ocr）→ NPU 采样 → 报告。

## 运行环境约定

- **目标运行平台是 Linux Ubuntu（8 卡 Ascend 910B）**，开发机可能是任意平台。
- 单元测试 **不得依赖 NPU、vllm、真实网络服务**；所有外部交互（HTTP / npu-smi / docker / 子进程）在测试中必须 mock 或使用 fixture 文本。
- 压测端保持零重依赖：不得引入 torch / transformers / numpy 等重型依赖。

## 代码约定

- Python ≥ 3.10；类型标注齐全；公共 API 有 docstring。
- 配置一律经过 `mtest/config.py` 的 pydantic 校验；新增配置字段必须同步更新 `configs/defaults.yaml` 注释与 `docs/CONFIG_GUIDE.md`。
- 新增测试套件：继承 `mtest/suites/base.py::Suite`，在 `mtest/suites/__init__.py` 注册（预留扩展点：accuracy / stability）。
- 指标口径（TTFT/ITL/TPOT/RPS/成功率等）定义见 `docs/SUITES_GUIDE.md`，修改口径必须先改文档。
- 报告原始数据全量落 `metrics.json`（schema 固定，向后兼容只增不删），人读摘要落 `summary.md`。

## 提交约定

- Conventional Commits：`feat(scope): ...` / `fix(scope): ...` / `docs: ...` / `test: ...` / `chore: ...` / `refactor: ...`。
- scope 常用：config / serve / suites / perf / longctx / functional / embedding / ocr / client / monitor / report / pipeline / cli / data / docs。
- 每个提交应可独立运行（不破坏 import / 测试）。

## 文档约定

- `docs/` 下手册类文件以全大写命名（USAGE.md、CONFIG_GUIDE.md 等）。
- `docs/PROGRESS.md` 记录操作进度与后续计划，完成里程碑后更新。
- `docs/DESIGN.md` 为原始设计文档，非经确认不重写，只追加修订记录。

## 常用命令

```bash
pip install -e .[dev]
pytest                                   # 全部单测（无需 NPU）
mtest validate -c configs/models/<m>.yaml  # 配置校验 + dry-run
python -m mtest.cli --help
```
