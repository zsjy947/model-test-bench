# 模型一键测试平台设计计划（model-test-bench / mtest）

> 版本：v1.0（计划稿）
> 日期：2026-09-24
> 目标平台：8 卡 Ascend 910B / Linux Ubuntu
> 后端：vllm（vllm-ascend 插件）
> 模型范围：≤32B 的 LLM（7B–27B 为主）、embedding 模型、OCR/多模态模型

---

## 0. 需求背景与已确认决策

**核心诉求**：对新模型能够在只改启动参数的情况下完成一键测试（起服务 → 测试 → 出报告全流程）。

已确认的四项决策：

| 决策项 | 结论 |
|---|---|
| 前端形态 | **CLI 命令行**（核心引擎做成库，为将来 Web 管理台留接口，本期不做） |
| 测试范围 | **性能压测 + 功能冒烟 + 长序列专项 + embedding/OCR 专项**；不做精度评测（GSM8K/C-Eval）与稳定性长跑（架构预留扩展点） |
| 服务管理 | **两种模式都支持**：`serve.mode: process / docker / external`（external = 对接已启动服务，只测不管起停；默认 process 全自动） |
| 压测位置 | **同机压测**（默认 localhost，排除网络干扰，指标纯反映 NPU 推理性能；base_url 可配置指向远端） |

---

## 1. 项目定位与目标

- **一键闭环**：`mtest run -c configs/models/<model>.yaml` 自动完成 加载配置 → 启动 vllm → 健康检查 → 按模型类型执行测试套件 → NPU 指标伴随采样 → 停服务 → 生成报告
- **一模型一文件**：新模型接入只需复制 `_template.yaml`，改 model 三要素（path / name / type）和 serve.args，其余全部继承 defaults.yaml
- **三类模型全覆盖**：
  - llm（7B–27B，≤32B）→ perf + longctx + functional
  - embedding → embedding + functional
  - multimodal/OCR → ocr + functional
- 套件架构预留扩展点（accuracy、stability、Web 管理台），本期不实现

---

## 2. 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│ CLI 层 (typer): run / serve up|down|status|logs / list /      │
│                 validate / report show|compare                │
├──────────────────────────────────────────────────────────────┤
│ 编排层 pipeline.py                                            │
│  加载合并配置 → 服务管理 → 健康检查 → 逐套件执行(带阶段标记)   │
│  → NPU采样伴随 → 停服务 → 聚合报告                             │
├────────────┬────────────┬─────────────┬─────────────────────┤
│ 配置层      │ 服务层      │ 套件层       │ 支撑组件             │
│ config.py  │ launcher   │ perf        │ client.py 异步客户端 │
│ defaults + │ (process/  │ longctx     │ monitor.py NPU采样   │
│ model yaml │  docker/   │ functional  │ metrics.py 统计      │
│ pydantic   │  external) │ embedding   │ datasets.py 数据集   │
│ 校验+深合并 │ health.py  │ ocr         │ report.py 报告       │
├────────────┴────────────┴─────────────┴─────────────────────┤
│ 产物层: results/<run_id>/ summary.md + metrics.json +        │
│         npu_samples.csv + vllm.log (+ serve_failure/)        │
└──────────────────────────────────────────────────────────────┘
```

数据流：所有套件输出统一 `SuiteResult`（结构化指标 + 明细行 + 阶段时间戳），pipeline 汇总交给 report 层渲染；原始数据全量落 JSON（metrics.json），人读摘要落 Markdown（summary.md）。

---

## 3. 配置系统设计

### 3.1 合并规则

三层合并：`configs/defaults.yaml` ← `configs/models/<model>.yaml` ← CLI 覆盖项（如 `--suite perf`、`--concurrency 1,8`）。

- 深合并（dict 递归合并，list 整体覆盖不逐项合并）
- 支持 yaml 与 json 两种格式
- pydantic 校验，配置错误在起服务前暴露（避免 30 分钟启动后才发现参数写错）

### 3.2 defaults.yaml 字段清单

```yaml
model:
  name:                  # 模型标识（用于报告/目录命名）
  path:                  # 模型权重路径
  type: llm              # llm | embedding | multimodal
  served_name:           # vllm --served-model-name，缺省取 path 尾段
  trust_remote_code: true

serve:
  mode: process          # process | docker | external
  host: 0.0.0.0
  port: 8000
  startup_timeout: 1800  # 秒，健康检查超时
  env_init:              # 启动前环境激活命令串
    # 例: source /usr/local/Ascend/ascend-toolkit/set_env.sh && conda activate vllm
  args:                  # 自动生成 vllm serve 命令行参数
    tensor-parallel-size: 8
    max-model-len: 32768
    gpu-memory-utilization: 0.9
    dtype: float16
  command: null          # 逃生舱：非空则整体替代自动生成的启动命令
  docker:
    image:               # 昇腾 vllm 镜像
    devices: [0,1,2,3,4,5,6,7]
    mounts: []           # 模型目录等挂载
    extra_args: []       # docker run 附加参数

client:
  base_url: http://127.0.0.1:8000/v1   # 同机默认；可指向远端
  api_key: null
  request_timeout: 600

tests:
  perf:
    enabled: true
    dataset: prompts     # prompts | random | 自定义 jsonl 路径
    input_lens: [128, 1024, 4096]
    output_len: 256
    concurrency: [1, 8, 32, 64]
    duration: 120        # 每档时长秒（与 num_requests 双上限，先到为准）
    num_requests: 200
    rounds: 3            # 每档重复次数，取均值
    warmup: 2            # 每档预热请求数
    stream: true
  longctx:
    enabled: true
    input_lens: [16384, 32768]
    concurrency: [1, 8]
    output_len: 128
  functional:
    enabled: true
    cases: data/cases/   # 内置用例集，支持追加自定义 yaml 用例
  embedding:
    enabled: true
    batch_sizes: [1, 8, 32, 128]
    concurrency: [1, 8, 32]
    dim: null            # 声明维度，用于校验；null 则取首次响应
  ocr:
    enabled: true
    image_dir: data/images/ocr
    ground_truth: null   # 可选 json（文件名→标准答案）
    concurrency: [1, 8]

monitor:
  enabled: true
  npu_interval: 5        # 采样间隔秒
```

### 3.3 args → vllm 命令行生成规则

- 布尔值 → 单 flag（`disable_log_stats: true` → `--disable-log-stats`）
- 嵌套 dict → `--additional-config '<json>'`（兼容 vllm-ascend 的 torchair 图模式等）
- 其余 键: 值 → `--key value`
- `command` 非空时整体替代（GOT-OCR 等特殊分支/容器内启动脚本场景）

### 3.4 校验规则

- tensor-parallel-size ∈ {1, 2, 4, 8}；port ∈ 1024–65535
- input_len + output_len < max-model-len（越界自动裁剪并警告）
- model.type 与套件匹配：embedding 模型禁用 perf/longctx；multimodal 禁用 perf（走 ocr 套件）

### 3.5 示例配置（4 + 1 份）

`_template.yaml`（注释齐全模板）、qwen2.5-7b-instruct（TP=2，llm）、qwen2.5-14b-instruct（TP=4，llm）、bge-m3（TP=1，`--task embed`）、qwen2.5-vl-7b（TP=4，multimodal）。

---

## 4. 服务管理层

三种模式接口统一（start / wait_ready / stop / status / collect_logs），配置切换：

| 模式 | 启动方式 | 停止方式 |
|---|---|---|
| **process（默认）** | `bash -lc "<env_init> && vllm serve <path> <args> > vllm.log 2>&1"`，subprocess 进程组 | 杀整进程组（SIGTERM → 超时 SIGKILL） |
| **docker** | `docker run --rm --name mtest-<model>` + 昇腾设备映射（/dev/davinci0-7、/dev/davinci_manager、/dev/devmm_svm、/dev/hisi_hdc）+ 模型目录挂载 | `docker stop` |
| **external** | 不启动 | 不停止，仅健康检查后直接测试 |

**健康检查**：轮询 `GET /health`（间隔 2s，超时 startup_timeout）；就绪后拉 `/v1/models` 校验 served_model_name 已注册。

**失败诊断包**：启动失败/超时自动收割 → vllm.log 尾部 200 行、`npu-smi info` 快照、进程列表，存 `results/<run_id>/serve_failure/`；终端直接打印日志中匹配 ERROR/Traceback/CANN 关键字的行。

---

## 5. 测试套件详细设计

### 5.1 perf 性能压测（llm）

- **请求方式**：OpenAI 兼容 `/v1/chat/completions`，流式（SSE），`stream_options.include_usage` 取真实 token 数
- **并发控制**：闭环（closed-loop）——维持固定在途请求数，一个完成立即补位；`duration` 与 `num_requests` 双上限，先到为准
- **输入长度控制**：不加载本地 tokenizer（压测端零重依赖），采用校准法——每档先发 max_tokens=1 探测请求，按 usage.prompt_tokens 反馈迭代拼接/裁剪 prompt 至目标长度 ±10%，然后锁定复用
- **数据集**：内置中英 prompt 池（按长度分桶）、随机 token（seed 可复现）、自定义 jsonl（messages 格式）
- **指标口径**：
  - TTFT：发请求 → 首个内容 chunk
  - ITL：相邻 chunk 间隔；TPOT = (e2e − TTFT) / (n_out − 1)
  - e2e 延迟；输出吞吐 = 总输出 token / 墙钟；请求吞吐 RPS
  - 成功率：HTTP 200 且 finish_reason ∈ {stop, length}；连接错误/超时/异常截断计失败
  - p50 / p90 / p99 分位数
  - 配合 monitor：各卡 AICore 利用率、HBM 占用峰值
- **输出**：矩阵表（input_len × concurrency，每格核心指标）+ 明细 CSV

### 5.2 longctx 长序列专项（llm）

- 档位自动按 `max-model-len − output_len` 封顶裁剪（如 32K 模型实际跑 16K / 31.8K）
- 指标：TTFT（反映 prefill 等待）、prefill 吞吐 ≈ prompt_tokens / TTFT、decode 吞吐、并发 1/8 下 e2e 分布、成功率（重点关注长序列下的超时/截断/段错误）
- 用例构造：长文档拼接（内置长文本素材循环填充）+ 探测校准法

### 5.3 embedding 专项

- `/v1/embeddings`；批量梯度（1/8/32/128）× 并发梯度
- 性能指标：延迟分布（p50/90/99）、吞吐 sent/s、批量扩展性曲线
- 质量指标：
  - 维度与配置声明一致
  - 同文本复现一致性（两次编码完全一致或误差 < 1e-6）
  - 区分度 sanity：内置 10 组相似对/无关对，校验相似对余弦距离显著小于无关对（阈值可配，报告给出两组分布而非只给 pass/fail）

### 5.4 ocr 专项（multimodal）

- 请求：`/v1/chat/completions` + `image_url`（本地图片 base64）；内置样例图集（中英文印刷体/手写/表格/多分辨率分档）+ `data/images/ocr/` 用户扩展
- 性能指标：单图延迟分布、吞吐 img/s（含并发档）、按分辨率分档延迟
- 质量指标：
  - 有 ground_truth.json 时算 CER = 编辑距离 / 参考长度（文本归一化：去空白、全半角统一）；可选关键字段抽取命中率
  - 无标注时仅冒烟（输出非空、无乱码启发式检查）
- 容错用例：损坏图片、超大图片 → 期望 4xx/优雅报错而非服务崩溃，随后跟一个正常请求确认服务仍健康

### 5.5 functional 功能冒烟（全类型）

LLM：chat 模板正确性（中文提问无模板残留乱码）、流式 chunk 完整性与拼装结果和非流式一致性、finish_reason 正确、max_tokens 遵守、stop 串生效、temperature=0 重复请求输出一致、边界输入（空输入、超长输入触发截断而非 500）。

embedding / ocr 各有对应冒烟子集（见 5.3 / 5.4）。用例定义在 `data/cases/*.yaml`，结构化断言，报告逐条 pass/fail。

---

## 6. NPU 监控

- monitor.py 后台线程，套件执行期间每 5s 执行 `npu-smi info` 解析
- 按卡记录：HBM 已用/总量、AICore 利用率、功耗、温度 → `npu_samples.csv`（带时间戳）
- 解析器容忍不同 CANN 版本格式差异（正则 + 多版本 fixture 单测），采样失败降级为告警、不阻断测试
- 与压测时间轴对齐（套件执行写阶段标记），报告输出各阶段 NPU 峰值/均值

---

## 7. 报告与 CLI

### 7.1 报告产物

`results/<run_id>/`（run_id = `YYYYmmdd-HHMM_<model_name>`）：

| 文件 | 内容 |
|---|---|
| summary.md | 环境信息（机型/CANN 与驱动版本/vllm 与 vllm-ascend 版本/卡数/TP）、服务启动耗时、各套件结果表、NPU 峰值、异常与结论建议 |
| metrics.json | 全量原始指标（schema 固定，供 compare 与二次分析） |
| npu_samples.csv | NPU 采样明细 |
| vllm.log | 服务完整日志 |
| serve_failure/ | （仅启动失败时）诊断包 |

### 7.2 CLI 命令集

```
mtest run    -c <yaml> [--suite perf,functional] [--skip-serve] [--keep-alive] [--dry-run]
mtest serve  up|down|status|logs -c <yaml>
mtest list                       # 列出可用模型配置
mtest validate -c <yaml>         # 配置校验 + dry-run 打印生成的 vllm 命令
mtest report show   <run_id>
mtest report compare <a> <b>     # 关键指标并排 + 差异百分比
```

---

## 8. 关键技术决策与理由

| 决策 | 理由 |
|---|---|
| 自研 aiohttp 压测客户端（不直接包 vllm benchmark_serving.py） | 指标口径、报告结构、embedding/ocr 请求形态统一可控；官方脚本跨版本不稳定，作参考不依赖 |
| 闭环并发 + 时长/请求数双上限 | 业界标准做法，结果与 vllm 官方 benchmark 可横向对照 |
| 不加载 HF tokenizer，用 usage 反馈校准长度 | 压测端零重依赖（不装 torch/transformers），长度精度 ±10% 足够压测分档用途 |
| vllm args 全量从 yaml 透传 + command 逃生舱 | vllm-ascend 各版本参数差异大（如 additional-config 的 torchair 图模式），透传保证兼容；GOT-OCR 等特殊模型走逃生舱 |
| pydantic 校验 + dry-run | 配置错误在起服务前暴露，避免长时间启动后才发现写错参数 |

---

## 9. 910B / vllm-ascend 平台注意事项

- 环境激活链：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` + venv/conda；容器部署时在容器内执行
- vllm 与 vllm-ascend 版本必须匹配（以官方 README 匹配表为准；报告自动记录 `pip show vllm vllm-ascend` 版本）
- TP 分卡建议：≤8B → 1/2/4 卡；14–27B → 4/8 卡；32B → 8 卡；embedding 小模型通常 1 卡
- embedding 需 `--task embed`；多模态 Qwen-VL 系列原生支持度依 vllm-ascend 版本，GOT-OCR 走 command 逃生舱 + ocr 套件
- 长序列注意 HBM 与 kv-cache 上限，longctx 档位自动封顶
- docker 模式设备映射清单固化在配置模板

---

## 10. 目录结构

```
model-test-bench/
├── README.md
├── AGENTS.md
├── docs/
│   ├── DESIGN.md            # 原始设计文档
│   ├── PROGRESS.md          # 当前操作进度、后续计划
│   └── …………                 # 以全大写命名的md文件，记录操作手册/特性说明等等必要的信息
├── pyproject.toml
├── mtest/
│   ├── cli.py / config.py / pipeline.py
│   ├──client.py / monitor.py / metrics.py / report.py / datasets.py
│   ├── serve/
│   │   ├── launcher.py      # process / docker / external 三模式
│   │   └── health.py
│   └── suites/
│       ├── perf.py / longctx.py / functional.py / embedding.py / ocr.py
│       └── __init__.py      # 统一 Suite 接口，预留注册点（accuracy/stability）
├── configs/
│   ├── defaults.yaml
│   └── models/              # _template.yaml + 4 份示例
├── data/
│   ├── prompts/{zh,en}.jsonl
│   ├── longdoc/             # 长序列填充素材
│   ├── images/ocr/          # 样例图 + ground_truth.json
│   └── cases/               # functional 用例 yaml
├── results/                 # 运行产物
└── tests/                   # 单测（不依赖 NPU）
```

依赖：Python 3.10+；typer / pydantic / PyYAML / aiohttp / jinja2 / rich（纯客户端轻依赖，与 vllm 环境隔离，可装在独立 venv）。

---

## 11. 实施阶段（里程碑）

| 阶段 | 内容 | 交付物 |
|---|---|---|
| M1 闭环骨架 | config + validate/dry-run + serve(process) + health + pipeline + report 雏形 | 一条命令跑通空套件闭环 |
| M2 压测核心 | perf 套件 + metrics 统计 + monitor NPU 采样 + summary.md / compare | 可用的 LLM 压测报告 |
| M3 套件补全 | functional → embedding → ocr → longctx | 三类模型全覆盖 |
| M4 收尾 | docker/external 模式完善、示例配置与内置数据集、DESIGN.md、单测补齐 | 完整项目 |

---

## 12. 风险与缓解

| 风险 | 缓解 |
|---|---|
| vllm-ascend 版本参数差异 | args 透传 + dry-run 提前发现 + command 逃生舱 |
| 多模态在昇腾后端支持不一 | 套件可单独关闭，functional 兜底；特殊模型走逃生舱 |
| npu-smi 输出格式随 CANN 变化 | 容忍式正则 + 多版本 fixture 单测 + 失败降级不阻断 |
| 长序列 OOM / 慢启动 | 档位自动封顶、startup_timeout 可配、失败诊断包自动收割 |
| 压测端自身成为瓶颈（同机 CPU 抢占） | 报告记录客户端 CPU/负载采样；aiohttp 异步单进程足以支撑 64 并发流式请求 |

---

## 13. 未来扩展点（本期不做，架构已留位）

- accuracy 精度评测（GSM8K / C-Eval 固定子集）
- stability 稳定性长跑（固定并发 N 小时，检测吞吐衰减/错误率上升/HBM 增长）
- Web 管理台（复用引擎库）
- 多实例并行压测调度
