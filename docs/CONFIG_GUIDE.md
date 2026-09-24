# CONFIG_GUIDE.md — 配置系统手册

> 三层合并 / 字段参考 / args→命令生成规则 / 校验规则 / 新模型接入指南

## 1. 三层合并模型

```
configs/defaults.yaml  ←  configs/models/<model>.yaml  ←  CLI 覆盖项
       （全局默认）              （一模型一文件）          （--suite / --concurrency）
```

- **深合并**：dict 递归合并；list 与标量**整体覆盖**（不逐项合并）。
  例如模型配置里写 `input_lens: [512]`，则 defaults 的 `[128, 1024, 4096]`
  被整体替换为 `[512]`，而不是追加。
- 支持 yaml 与 json 两种格式（按扩展名识别）。
- 合并结果经 pydantic 严格校验（未知字段直接报错，抓字段拼写错误），
  **校验错误在起服务前暴露**（`mtest validate` 或 run 的第一步）。

## 2. 字段参考

### model（模型三要素）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | str | 必填 | 模型标识，用于报告与目录命名（run_id、容器名） |
| `path` | str | 必填 | 模型权重本地路径（docker 模式下为容器内路径） |
| `type` | enum | `llm` | `llm` / `embedding` / `multimodal`，决定执行套件 |
| `served_name` | str | path 尾段 | vllm `--served-model-name`；显式设置且 ≠ path 尾段时才追加该参数 |
| `trust_remote_code` | bool | true | true 时追加 `--trust-remote-code` |

### serve（服务管理）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `mode` | enum | `process` | `process` / `docker` / `external` |
| `host` / `port` | str/int | `0.0.0.0` / `8000` | 监听地址；port ∈ 1024–65535 |
| `startup_timeout` | int(s) | `1800` | 健康检查超时（大模型加载慢，默认 30 分钟） |
| `env_init` | str | 空 | 启动前环境激活命令串（source set_env.sh、conda activate 等） |
| `args` | dict | 见 defaults | **vllm serve 参数透传**（规则见 §3） |
| `command` | str | null | **逃生舱**：非空则整体替代自动生成的启动命令 |
| `docker.image` | str | 空 | docker 模式必填（昇腾 vllm 镜像） |
| `docker.devices` | list[int] | `[0..7]` | 映射 `/dev/davinci<N>`（0–7） |
| `docker.mounts` | list[str] | `[]` | `host:container[:mode]` 挂载（模型目录等） |
| `docker.extra_args` | list[str] | `[]` | `docker run` 附加参数 |

### client（压测客户端）

| 字段 | 默认 | 说明 |
|---|---|---|
| `base_url` | `http://127.0.0.1:8000/v1` | OpenAI 兼容端点；同机默认，可指向远端 |
| `api_key` | null | Bearer 认证（服务开了鉴权时填） |
| `request_timeout` | 600 | 单请求超时（秒） |

### tests（各套件开关与档位）

| 套件 | 关键字段 | 默认 |
|---|---|---|
| perf | `dataset`（prompts/random/自定义 jsonl 路径）、`input_lens`、`output_len`、`concurrency`、`duration`、`num_requests`、`rounds`、`warmup`、`stream` | 见 defaults.yaml |
| longctx | `input_lens`（自动封顶）、`concurrency`、`output_len`（扩展：duration/num_requests/rounds/warmup） | `[16384,32768]` / `[1,8]` / `128` |
| functional | `cases`（内置用例目录或单个 yaml 文件） | `data/cases/` |
| embedding | `batch_sizes`、`concurrency`、`dim`（null=取首次响应；扩展 duration/num_requests/similarity_min_separation） | `[1,8,32,128]` / `[1,8,32]` / null |
| ocr | `image_dir`、`ground_truth`、`concurrency`（扩展 duration/num_requests/fault_cases） | `data/images/ocr` / null / `[1,8]` |

### monitor

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | 关闭后不做 NPU 采样 |
| `npu_interval` | 5 | 采样间隔（秒） |

## 3. serve.args → vllm 命令生成规则

| 值类型 | 生成 | 示例 |
|---|---|---|
| `true`（布尔） | 单 flag | `disable-log-stats: true` → `--disable-log-stats` |
| `false`（布尔） | 省略 | — |
| dict | `--key '<紧凑 JSON>'` | `additional-config: {torchair_graph_config: {enabled: true}}` → `--additional-config '{"torchair_graph_config":{"enabled":true}}'` |
| list | 逗号连接 | `allowed-local-media: [/a, /b]` → `--allowed-local-media /a,/b` |
| 其余 | `--key value` | `dtype: float16` → `--dtype float16` |

进程模式完整命令：

```
bash -lc "(<env_init>) && vllm serve <path> --trust-remote-code ... > <run_dir>/vllm.log 2>&1"
```

- `command` 逃生舱优先（GOT-OCR 等特殊分支 / 容器内启动脚本场景）。
- docker 模式自动追加：`--network host`、`--device /dev/davinci<N>`（按 devices）、
  `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`、`-v mounts`、
  `-e ASCEND_RT_VISIBLE_DEVICES`。

## 4. 校验与自动调整规则（finalize）

1. `tensor-parallel-size` ∈ {1,2,4,8}，否则报错。
2. `port` ∈ [1024, 65535]（pydantic 层）。
3. `input_len + output_len + 64（余量） < max-model-len`，越界**自动裁剪并警告**
   （longctx 档位封顶，如 32K 模型实际跑 32576）。裁剪后无档位则禁用套件。
4. 类型-套件匹配（先于裁剪执行，避免冗余警告）：
   - `llm`：禁用 embedding / ocr
   - `embedding`：禁用 perf / longctx / ocr
   - `multimodal`：禁用 perf / longctx（走 ocr） / embedding

所有自动调整都会以警告形式打印（validate / run / summary.md 告警区）。

## 5. 新模型接入指南

1. `cp configs/models/_template.yaml configs/models/<model>.yaml`
2. 改三要素：`model.name`（建议=目录名）、`model.path`、`model.type`
3. 按规模调 `serve.args.tensor-parallel-size`：

   | 模型规模 | 建议 TP |
   |---|---|
   | ≤8B（含 embedding 小模型） | 1 / 2 / 4（embedding 通常 1） |
   | 14–27B | 4 / 8 |
   | 32B | 8 |

4. embedding 模型记得加 `args.task: embed`；多模态按需 `limit-mm-per-prompt`。
5. `mtest validate -c ...` 预览生成命令，确认无误后 `mtest run`。

## 6. 配置复现

每次 run 会把**生效配置**（合并+裁剪后的最终形态）落盘到
`results/<run_id>/config.resolved.yaml`，可直接作为后续运行的输入复现实验。
