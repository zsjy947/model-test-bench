# SUITES_GUIDE.md — 测试套件原理与指标口径

> 指标口径以本文档为准；修改口径必须先改本文档（见 AGENTS.md）。

## 0. 通用约定

- **请求端点**：OpenAI 兼容 `/v1/chat/completions`（perf/longctx/ocr/functional）、
  `/v1/embeddings`（embedding）。
- **成功率口径**：HTTP 200 且 `finish_reason ∈ {stop, length}`；
  连接错误 / 超时 / 异常截断（流未收到 `[DONE]`）计失败，并按类别统计
  （connect / timeout / http / parse / truncated）。
- **闭环并发（closed-loop）**：维持固定在途请求数（worker=并发数），一个完成立即
  补位；`duration`（时长）与 `num_requests`（请求数）双上限，**先到为准**——
  到限后停止发起新请求，在途请求自然收尾并计入统计。
- **长度校准法**：压测端不加载 tokenizer。每档先按字符启发式选基底 prompt，
  再发 `max_tokens=1` 探测请求，按 `usage.prompt_tokens` 反馈迭代拼接/裁剪至
  目标长度 ±10%，然后**锁定复用**（同一档位所有轮次/并发共用同一 prompt）。
- **预热**：每档先发 `warmup` 个短输出请求（不计入统计），用于预热缓存/图模式。
- **多轮取均值**：`rounds>1` 时 RPS/吞吐/成功率取各轮均值；延迟分位数跨轮汇总。

## 1. perf 性能压测（llm）

**结构**：`input_lens × concurrency` 矩阵，每格 `rounds` 轮。

**请求**：流式 SSE（`stream_options.include_usage` 取真实 token 数）。

**指标定义**：

| 指标 | 口径 |
|---|---|
| TTFT | 发请求 → 首个**内容** chunk |
| ITL | 相邻内容 chunk 间隔（记录均值/最大值） |
| TPOT | (e2e − TTFT) / (n_out − 1)，n_out 取 usage 的 completion_tokens |
| e2e | 发请求 → 流结束 |
| RPS | 成功请求数 / 墙钟（墙钟=首发到最后一笔完成） |
| 输出吞吐 | 总输出 token / 墙钟 |
| p50/p90/p99 | 延迟分位数（线性插值） |

**数据集**（`tests.perf.dataset`）：

- `prompts`：内置中英 prompt 池（`data/prompts/{zh,en}.jsonl`，按长度分桶）+ 校准
- `random`：随机 token（seed 固定可复现，中英词表混合）+ 校准
- 自定义 jsonl 路径：`{"messages": [...]}` 格式循环取用，**不校准**
  （矩阵标签为目标 input_len，实际 prompt_tokens 见明细 CSV）

**产物**：`perf_details.csv`（逐请求：时间/长度/并发/轮次/状态/各计时/usage）。

## 2. longctx 长序列专项（llm）

- 档位在配置层按 `max-model-len − output_len − 64` **自动封顶**（如 32K 模型实际
  跑 32576；8K embedding 类裁到更低）。
- 用例构造：内置长文本素材（`data/longdoc/*.txt`）循环填充 + 校准法。
- 关注点：长序列下的超时 / 截断 / 段错误（失败明细逐类别列出）。

**指标**：

| 指标 | 口径 |
|---|---|
| TTFT | 反映 prefill 排队+计算等待 |
| prefill 吞吐 | ≈ prompt_tokens / TTFT（单请求） |
| decode 吞吐 | completion_tokens / (e2e − TTFT) |
| e2e 分布 | 并发 1/8 档位下的 p50/p99 |
| 成功率 | 通用口径，失败类别明细 |

## 3. embedding 专项

**性能**：`batch_sizes × concurrency` 网格（闭环），从内置 prompt 池循环取文本。

| 指标 | 口径 |
|---|---|
| sent/s | 成功请求数 × batch / 墙钟 |
| 延迟分布 | 单请求 e2e 的 p50/p90/p99 |
| 批量扩展性 | 相同并发下 sent/s 随 batch 的变化曲线（矩阵纵向对比） |

**质量**：

| 检查 | 口径 | 判定 |
|---|---|---|
| 维度校验 | 实际 dim vs `tests.embedding.dim`（null 则只记录） | 不符 → 套件失败 |
| 复现一致性 | 同文本两次编码 cosine ≥ 1−1e-6 | 不一致 → 套件失败 |
| 区分度 sanity | 内置 10 组相似对/无关对（`data/cases/embedding_pairs.yaml`），报告两组余弦分布 + 间隔 | 间隔 < `similarity_min_separation`（默认 0.15）→ 仅告警（sanity 参考） |

## 4. ocr 专项（multimodal）

**请求**：`/v1/chat/completions` + `image_url`（本地图片 base64 data URL），
内置样例图集（`data/images/ocr/`：中文/英文印刷体、表格、多分辨率分档），
用户可向目录投放图片扩展；标注文件 `ground_truth.json`（文件名→标准答案，
缺省自动取图片目录同级）。

**性能**：并发档扫描，指标 img/s、单图延迟 p50/p90/p99、**按分辨率分档延迟**
（0.3/1/2/9 MP 分桶，尺寸从 PNG/JPEG 头解析）。

**质量**：

- 有标注：**CER = 编辑距离 / 参考长度**，文本先归一化（去空白、NFKC 全半角
  统一、小写）；报告输出每文件 CER 与汇总分布。
- 无标注：仅冒烟（输出非空、乱码启发式——替换符/控制字符占比 < 5%）。

**容错用例**（`data/images/ocr_fault/`，可在配置关闭）：

- 损坏图片（合法 PNG 头 + 乱码体）、超大图片（6000×6000）→ 期望 **4xx 或优雅
  报错而非服务崩溃**；随后健康检查 + 正常请求确认服务仍存活。

## 5. functional 功能冒烟（全类型）

用例定义在 `data/cases/*.yaml`（支持追加自定义用例；`applies_to` 过滤模型类型），
结构化断言，报告逐条 pass/fail。内置覆盖（llm）：

- chat 模板正确性（中文提问无模板残留乱码）
- 流式 chunk 完整性与拼装结果 == 非流式
- finish_reason 正确、max_tokens 遵守（completion_tokens ≤ max_tokens）
- stop 串生效（finish=stop 且输出不含 stop 串）
- temperature=0 重复请求输出一致
- 边界输入：空输入、超长输入（按 max-model-len×1.2 填充）→ 截断或 4xx 而非 5xx

embedding / ocr 的冒烟子集见 `embedding_smoke.yaml` / `ocr_smoke.yaml`。

**expect 断言键**（chat 类）：

`status`（int/list）、`status_lte`、`finish_reason`（str/list）、`non_empty`、
`contains` / `not_contains`（list[str]）、`no_template_residue`、
`completion_tokens_lte` / `completion_tokens_gte`、`not_truncated`、
`repeats_consistent`（true=默认重复 3 次；正整数=显式次数；temperature=0
校验多次输出一致）。

**用例类型**：`chat`、`stream_consistency`、`embedding_basic`、`ocr_basic`、
`ocr_fault`。自定义示例：

```yaml
- id: my-case
  applies_to: [llm]
  type: chat
  request:
    messages: [{role: user, content: 请输出 OK}]
    max_tokens: 8
    temperature: 0
  expect:
    status: 200
    contains: ["OK"]
    completion_tokens_lte: 8
```

## 6. 套件扩展点

新套件继承 `mtest/suites/base.py::Suite`（设置 `name` / `applies_to`，实现
`run() -> SuiteResult`），在 `mtest/suites/__init__.py` 调用 `register()` 注册。
预留扩展：accuracy（精度评测）、stability（稳定性长跑）——见设计 §13 与
docs/EXTENSIONS.md（dev 分支）。
