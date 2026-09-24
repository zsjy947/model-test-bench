# EXTENSIONS.md — 未来扩展点规划与实现突破（对照设计 §13）

> 分支：`dev` ｜ 更新：2026-09-25
> 设计 §13 列出的四个"本期不做、架构已留位"的扩展，在 dev 分支全部完成
> **可用的功能突破**（非完整生产化），各条均给出：已实现内容 / 用法 /
> 与完整形态的差距和后续路线。

## 总览

| 扩展点 | 设计定位 | dev 实现状态 | 使用方式 |
|---|---|---|---|
| accuracy 精度评测 | 预留 | ✅ 套件可用（GSM8K 风格 20 题 + 客观题 20 题自研题库） | 配置 `tests.accuracy.enabled: true` 或 `--suite accuracy` |
| stability 稳定性长跑 | 预留 | ✅ 套件可用（吞吐衰减/错误率/HBM 增长三检查） | `tests.stability.*`（默认 120 分钟，阈值可配） |
| Web 管理台 | 预留 | ✅ 只读结果浏览器（运行列表/详情/对比/API） | `mtest web --port 8765` |
| 多实例并行压测调度 | 预留 | ✅ 批量调度（顺序/并行 + 端口冲突自动分配） | `mtest batch -c a.yaml -c b.yaml --parallel 2 --auto-port` |

---

## 1. accuracy 精度评测

### 已实现

- **套件**：`mtest/suites/accuracy.py`（注册名 `accuracy`，适用于 llm/multimodal，
  functional → accuracy → perf 顺序执行）。
- **题库**（自研原创，避免数据集许可问题）：
  - `data/accuracy/gsm8k_mini.jsonl`：20 道 GSM8K 风格数学应用题
    （`{"id","type":"math","question","answer"}`）
  - `data/accuracy/mcq_mini.jsonl`：20 道客观选择题
    （`{"id","type":"mcq","question","choices":{A..D},"answer"}`，覆盖推理系统/
    网络/数学/常识）
- **评分**：math 从输出提取最后一个数字（容忍千分位逗号），与答案数值比较；
  mcq 提取选项字母（优先"答案是X"显式表述，退化取最后一个独立 A-D）。
- **指标与判定**：总体准确率 + 分题集准确率 + 答错题号清单；低于
  `pass_threshold`（默认 0.8）判套件失败。
- **配置**：`tests.accuracy.{enabled, datasets, max_tokens, pass_threshold}`。

### 用法

```bash
mtest run -c configs/models/qwen2.5-7b-instruct.yaml --suite functional,accuracy
```

### 与完整形态的差距（如实标注）

- **样本量**：40 题是冒烟级回归筛查，统计置信度低（±8% 级噪声），
  **不能替代完整 GSM8K（1319 题）/ C-Eval 官方评测**。
- **C-Eval 未接官方数据**：当前 mcq 题库为自研通识题；接入 C-Eval 需下载官方
  数据集并遵守其许可，结构已兼容（同 mcq jsonl 格式，加 `few-shot` 支持即可）。
- **无 few-shot / CoT 评测模式**、无按学科维度细分。
- **后续路线**：① 官方 GSM8K 子集加载器（500 题分层抽样）② few-shot 模板
  ③ 逐题耗时交叉分析（精度-性能联合画像）。

## 2. stability 稳定性长跑

### 已实现

- **套件**：`mtest/suites/stability.py`（注册名 `stability`，llm，最后执行）。
- **执行形态**：固定并发（默认 8）持续 N 分钟（默认 120，支持小数便于测试），
  输入固定长度（默认 1024，随机 token + 校准锁定），流式请求。
- **三类退化检查**（窗口粒度聚合，阈值均可配）：
  1. **吞吐衰减**：首窗口 vs 末窗口 RPS 衰减百分比 > `max_throughput_decay_pct`
     （默认 20%）→ 失败；附逐窗口线性回归斜率。
  2. **错误率**：整体错误率 > `max_error_rate_pct`（默认 5%）→ 失败。
  3. **HBM 增长**：NPU 采样首/末窗口 HBM 峰值差 > `max_hbm_growth_mb`
     （默认 2048MB，疑似内存泄漏）→ 失败；monitor 未开启时不判定。
- **产物**：逐窗口指标序列、检查结论表、`stability_details.csv` 逐请求明细。

### 用法

```yaml
tests:
  stability:
    enabled: true
    duration_minutes: 120   # 或 30 分钟快速版
    concurrency: 8
```

### 与完整形态的差距

- 窗口内请求按序号均分近似（记录无绝对时间戳），窗口边界误差 ≤ 1 个请求。
- 无故障注入（kill -9 / 网络抖动）、无多负载阶段脚本（如阶梯加压）。
- **后续路线**：① 请求级时间戳（消除窗口近似）② 阶梯负载 profile
  ③ 故障注入钩子 ④ 报警集成（webhook）。

## 3. Web 管理台

### 已实现

- **零新增依赖**：复用 aiohttp + jinja2（核心引擎库形态不变，Web 台只是
  `results/` 目录的另一个消费者，印证了设计"核心引擎做成库"的决策）。
- **路由**：
  - `GET /` 运行列表（run_id/模型/类型/各套件状态/耗时）
  - `GET /run/{id}` 详情（summary.md 渲染 + 套件状态表）
  - `GET /compare?a=&b=` 并排对比（复用 `report.compare_runs`）
  - `GET /api/runs`、`GET /api/run/{id}` JSON API（metrics.json 原文）
- **启动**：`mtest web --host 127.0.0.1 --port 8765`

### 与完整形态的差距

- **只读**：不能从页面发起 run / 管理服务（需写操作 + 任务队列 + 鉴权，
  属于产品化范围）。
- 无数据库（直接扫 results/ 目录，量级到千次运行前够用）。
- summary 以 `<pre>` 呈现，未做 Markdown→HTML 富渲染。
- **后续路线**：① 发起 run 的任务队列（复用 pipeline）② SQLite 索引
  ③ markdown 富渲染与图表（ECharts 接 metrics.json）④ OIDC/Basic 鉴权。

## 4. 多实例并行压测调度

### 已实现

- **`mtest batch`**：多配置批量运行（`-c` 可重复），汇总
  `results/batch-<ts>/batch_summary.md`（配置/模型/run_id/结果/套件状态/错误）。
- **顺序模式**（默认）：逐个完整闭环，互不干扰（严谨对比推荐）。
- **并行模式**（`--parallel N`，asyncio.Semaphore 限流）：
  - 端口冲突硬检查：process/docker 模式重复 `serve.port` 直接报错
  - `--auto-port`：冲突配置自动递增分配端口并同步改写 `client.base_url`
- 透传 `--suite / --skip-serve / --keep-alive / --dry-run`。

### 用法

```bash
# 顺序回归三个模型
mtest batch -c configs/models/qwen2.5-7b-instruct.yaml \
            -c configs/models/qwen2.5-14b-instruct.yaml \
            -c configs/models/bge-m3.yaml

# 8 卡机上两实例并行（各占 4 卡，需配置不同 TP 与端口）
mtest batch -c a.yaml -c b.yaml --parallel 2 --auto-port
```

### 与完整形态的差距

- 并行时每个 run 各自采样 npu-smi（重复开销小但曲线混在一起看的是全卡）；
  **同机资源竞争会反映在指标里**——并行结果适合功能性回归，横向性能对比
  请用顺序模式（文档与帮助中均已提示）。
- 无跨机分布式调度、无失败自动重试、无依赖 DAG（如"14B 跑完再跑 VL"）。
- **后续路线**：① 按卡分组的采样隔离（npu-smi 按设备过滤）② 失败重试策略
  ③ 简单 DAG（config 间依赖）④ 跨机 SSH 分发。

---

## 架构验证小结

四个扩展均通过既有扩展点接入（`register()` 注册 / 引擎库复用 / results 目录
约定），**未改动核心管线的任何接口**——设计 §2/§13 预留的架构位置有效。

## 建议的 dev → main 合并策略

1. accuracy / stability 套件与 doctor / 自适应降档：纯增量，风险低，可直接合入。
2. web / batch：入口命令新增（`mtest web`、`mtest batch`），合入后在 USAGE.md
   补充说明即可。
3. 合并前在目标机（8×910B）完成一次真实闭环验证（本仓库单测不依赖 NPU，
   真机行为需实测）。
