# USAGE.md — CLI 使用手册

> 适用版本：mtest v0.1.x ｜ 目标平台：8 卡 Ascend 910B / Linux Ubuntu + vllm(-ascend)

## 1. 安装与环境

```bash
# 独立 venv（纯客户端轻依赖，与 vllm 环境隔离，不装 torch/transformers）
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # 或 pip install .
mtest --version
```

- 压测默认**同机**直连（`client.base_url` 指向 127.0.0.1），排除网络干扰。
- mtest 所在机器需要能访问：`bash`（起服务）、`npu-smi`（采样，缺失自动降级）、
  目标 `base_url`（HTTP）。
- 环境变量 `MTEST_REPO_ROOT` 可在任意 CWD 下指定仓库根（定位 configs/ 与 data/）。

## 2. 命令总览

```
mtest run     -c <yaml> [--suite ...] [--concurrency ...] [--skip-serve] [--keep-alive] [--dry-run]
mtest serve   up|down|status|logs -c <yaml> [--tail N]
mtest list
mtest validate -c <yaml>
mtest report  show <run_id>
mtest report  compare <run_id_a> <run_id_b> [--out file.md]
```

## 3. 典型工作流

### 3.1 一键全流程（默认 process 模式）

```bash
mtest run -c configs/models/qwen2.5-7b-instruct.yaml
```

执行顺序：加载合并配置 → 校验（错误在起服务前暴露）→ 启动 vllm（进程组）→
健康检查（/health 轮询，2s 间隔）→ 校验 /v1/models → 依次执行
functional → perf → longctx（llm 类型）→ NPU 伴随采样（5s 间隔）→ 停服务 →
生成 `results/<run_id>/` 报告。启动失败自动收割诊断包（见 TROUBLESHOOTING.md）。

### 3.2 只跑部分套件 / 临时改并发

```bash
mtest run -c configs/models/qwen2.5-7b-instruct.yaml --suite perf,functional
mtest run -c configs/models/qwen2.5-7b-instruct.yaml --concurrency 1,8
```

`--suite` 可选值：`perf,longctx,functional,embedding,ocr`（逗号分隔）；
`--concurrency` 只覆盖 perf 套件并发档。

### 3.3 复用已启动的服务（external 思路）

```bash
mtest serve up -c configs/models/qwen2.5-7b-instruct.yaml     # 起 + 等就绪，CLI 退出后服务保持
mtest run  -c configs/models/qwen2.5-7b-instruct.yaml --skip-serve   # 只测不管起停
mtest serve status -c configs/models/qwen2.5-7b-instruct.yaml
mtest serve logs   -c configs/models/qwen2.5-7b-instruct.yaml --tail 200
mtest serve down   -c configs/models/qwen2.5-7b-instruct.yaml
```

`serve up/down` 通过 `results/serve/<model>/state.json` 跨命令记账（pid / 容器名）。
也可以在模型配置里直接设 `serve.mode: external`，对接非 mtest 启动的服务。

### 3.4 新模型接入（三步）

```bash
cp configs/models/_template.yaml configs/models/<new-model>.yaml
vim configs/models/<new-model>.yaml     # 改 name / path / type 与 serve.args
mtest validate -c configs/models/<new-model>.yaml   # 校验 + 预览生成命令
```

### 3.5 报告查看与对比

```bash
mtest report show 20260924-1530_qwen2.5-7b-instruct
mtest report compare 20260924-1530_qwen2.5-7b-instruct 20260925-0900_qwen2.5-7b-instruct
mtest report compare <a> <b> --out compare.md
```

对比输出 perf/embedding/ocr 公共档位的关键指标并排 + 差异百分比。解读见
REPORT_GUIDE.md。

## 4. run 选项明细

| 选项 | 说明 |
|---|---|
| `-c / --config` | 模型配置 yaml（必填） |
| `--suite a,b` | 只执行指定套件（覆盖配置中的 enabled） |
| `--concurrency 1,8` | 覆盖 perf 并发档（升序去重） |
| `--skip-serve` | 不管理服务起停（要求服务已就绪） |
| `--keep-alive` | 测试结束后不停止服务（配合人工观察/复用） |
| `--dry-run` | 打印将生成的 vllm 命令与启用套件，不做任何启动 |

退出码：0=全部通过；1=存在失败套件或运行中止；2=配置/参数错误。

## 5. 三类模型的套件映射

| model.type | 执行套件（顺序） |
|---|---|
| `llm` | functional → perf → longctx |
| `embedding` | functional → embedding |
| `multimodal` | functional → ocr |

不匹配的套件在配置校验阶段自动禁用并给出警告（可安全地全部保持 enabled）。

## 6. 运行产物

`results/<run_id>/`（run_id = `YYYYmmdd-HHMM_<model_name>`）：

| 文件 | 内容 |
|---|---|
| `summary.md` | 人读摘要（环境/套件表/NPU 峰值/结论） |
| `metrics.json` | 全量原始指标（schema 固定，供 compare/二次分析） |
| `npu_samples.csv` | NPU 采样明细（时间戳+阶段+各卡指标） |
| `vllm.log` | 服务完整日志（process 模式） |
| `config.resolved.yaml` | 本次运行的生效配置（复现依据） |
| `perf_details.csv` / `longctx_details.csv` / `ocr_details.csv` | 套件明细行 |
| `serve_failure/` | （仅启动失败时）诊断包 |

## 7. 常见问题

- **压测端会不会成为瓶颈？** aiohttp 异步单进程足以支撑 64 并发流式请求；
  报告记录了客户端 CPU/环境信息。若怀疑，可在另一台机器压（改 `client.base_url`）。
- **能压远端服务吗？** 可以，`client.base_url` 指向远端 + `serve.mode: external`；
  但此时指标包含网络开销，仅反映端到端体验。
- **测试中途 Ctrl-C？** 服务会尽力回收（finally 停止）；若残留，用
  `mtest serve down -c <yaml>` 或 `pkill -f "vllm serve"` 清理。

更多：配置见 [CONFIG_GUIDE.md](CONFIG_GUIDE.md)，指标口径见
[SUITES_GUIDE.md](SUITES_GUIDE.md)，报告解读见 [REPORT_GUIDE.md](REPORT_GUIDE.md)，
故障排查见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。
