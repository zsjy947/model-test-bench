# PROGRESS.md — 操作进度与后续计划

> 更新时间：2026-09-25 ｜ 分支：`main`（M1–M4 已完成）＋ `dev`（§12/§13 已完成）

## 里程碑状态（对照设计 §11）

| 阶段 | 内容 | 状态 | 提交 |
|---|---|---|---|
| M1 闭环骨架 | config + validate/dry-run + serve(process) + health + pipeline + report 雏形 | ✅ 完成 | 见提交历史 |
| M2 压测核心 | perf 套件 + metrics 统计 + monitor NPU 采样 + summary.md / compare | ✅ 完成 | |
| M3 套件补全 | functional → embedding → ocr → longctx | ✅ 完成 | |
| M4 收尾 | docker/external 模式、示例配置与内置数据集、文档、单测 | ✅ 完成 | |

## 完成清单

### 代码（mtest/）

- [x] `config.py` — 三层合并（defaults ← model ← CLI）+ pydantic 校验 +
      深合并；args→vllm 命令生成（布尔 flag / dict JSON / list / 逃生舱）；
      类型-套件匹配与长度自动裁剪（带警告）
- [x] `serve/launcher.py` — process（bash -lc + 进程组 SIGTERM→SIGKILL）/
      docker（昇腾设备映射）/ external 三模式统一接口 + 失败诊断包
      （vllm 尾部日志 + npu-smi 快照 + 进程列表，终端打印 ERROR 关键行）
- [x] `serve/health.py` — /health 轮询（2s）+ /v1/models 校验 served-name
- [x] `serve/launcher.py::ServeController` — `mtest serve up/down/status/logs`
      跨命令持久管理（state.json 记账）
- [x] `client.py` — aiohttp 异步客户端：chat 流式（TTFT/ITL/usage/[DONE] 检测）
      与非流式、embeddings；错误分类（connect/timeout/http/parse/truncated）
- [x] `metrics.py` — p50/p90/p99 分位、聚合、多轮均值合并、CSV 写出
- [x] `datasets.py` — 中英 prompt 池（分桶）、随机 token（seed 可复现）、
      长文本素材、自定义 jsonl；**校准法**（max_tokens=1 探测 + usage 反馈迭代 ±10%）
- [x] `monitor.py` — npu-smi 容忍式解析（主表两行组 + usages 简表）、后台线程采样
      （5s）、阶段标记对齐、CSV 落盘、失败降级不阻断、按阶段峰值汇总
- [x] `suites/` — 统一 Suite 接口 + 注册扩展点；perf（闭环 + 双上限 + 矩阵）、
      longctx（封顶 + prefill/decode 吞吐）、functional（用例断言引擎 5 类型）、
      embedding（批量×并发 + 维度/一致性/区分度）、ocr（CER + 分辨率分档 +
      容错用例）
- [x] `pipeline.py` — 全闭环编排（run_id 目录、config.resolved.yaml、阶段标记、
      keep_alive/skip_serve/dry-run、服务失败自动收割诊断）
- [x] `report.py` + `templates/summary.md.j2` — metrics.json（schema v1）+
      summary.md（jinja2）+ compare（并排 + 差异百分比）+ 规则式结论
- [x] `cli.py` — typer：run / serve up|down|status|logs / list / validate /
      report show|compare

### 配置与数据

- [x] `configs/defaults.yaml`（与设计 §3.2 对齐）
- [x] `configs/models/`：`_template.yaml` + qwen2.5-7b（TP2）/ qwen2.5-14b（TP4）/
      bge-m3（embedding --task embed）/ qwen2.5-vl-7b（multimodal）
- [x] `data/prompts/{zh,en}.jsonl`（13+13 条，多长度分桶）
- [x] `data/longdoc/{zh,en}.txt`（长序列素材 ~20K chars）
- [x] `data/cases/`：llm_smoke（8 用例）/ embedding_smoke / ocr_smoke /
      embedding_pairs（10 组相似/无关对）
- [x] `data/images/ocr/`：5 张样例图（中文/英文印刷、表格、多分辨率）+
      ground_truth.json；`data/images/ocr_fault/`（corrupt.png / huge.png 6000×6000）
- [x] `scripts/gen_data.py` — 数据集生成器（含纯 stdlib 超大白图 PNG 写出）

### 测试与验证

- [x] `tests/` 77 个单测全绿（无 NPU 依赖）：配置合并/校验/裁剪、命令生成、
      metrics 分位与聚合、校准法收敛（fake client）、npu-smi 多版本 fixture、
      CER/归一化/PNG-JPEG 头解析、functional 断言引擎、报告渲染/对比/结论、
      套件注册扩展点、闭环双上限与失败分类
- [x] CLI 冒烟：`mtest list` / `mtest validate`（4 份示例配置）/
      `mtest run --dry-run`

### 文档

- [x] USAGE.md（CLI 手册与工作流）
- [x] CONFIG_GUIDE.md（三层合并、字段参考、命令生成规则、TP 分卡建议）
- [x] SUITES_GUIDE.md（套件原理与指标口径）
- [x] REPORT_GUIDE.md（产物解读 + metrics.json schema v1）
- [x] TROUBLESHOOTING.md（启动失败诊断包等 10 类故障）

## 已知边界（与设计一致）

- 精度评测（GSM8K/C-Eval）与稳定性长跑本期不做，套件注册点已预留。
- Web 管理台不做；核心引擎为库形态（pipeline/suites 可被复用）。
- 压测端零重依赖：不装 torch/transformers，长度控制依赖 usage 反馈校准（±10%）。

## 后续计划（dev 分支）

1. **§12 风险缓解落地**：doctor 预检（vllm/vllm-ascend 版本匹配、设备、端口）、
   serve.args 已知参数 lint、npu-smi 更多版本 fixture、客户端负载采样、
   longctx 自适应降档。实现困难项在 `docs/RISKS.md` 标明。
2. **§13 扩展突破**：accuracy 套件（GSM8K 风格固定子集 + 客观题）、stability
   套件（固定并发长跑 + 吞吐衰减/错误率/HBM 增长检测）、Web 管理台（复用引擎库）、
   多实例并行压测调度。详见 `docs/EXTENSIONS.md`。

---

# dev 分支进度（2026-09-25 完成）

## §12 风险缓解（详见 docs/RISKS.md）

- [x] **R1**：`mtest doctor` 环境预检（运行时/NPU 卡数 vs TP/模型路径/端口/
      docker/数据完整性）+ `version_matrix.yaml` 静态匹配表（咨询性）+
      `mtest validate --lint-args` 参数提示式检查
- [x] **R3**：doctor 对 npu-smi 可用性与解析独立检查（多版本 fixture 单测在
      主分支已备）
- [x] **R4**：longctx 自适应降档（探测命中长度超限/OOM 错误 → 二分降档重试
      ≤2 次，轨迹入 metrics）
- [x] **R5**：客户端负载采样 `client_samples.csv`（loadavg/进程 CPU/线程数）
      + summary 瓶颈归因结论
- [x] 客户端 HTTP 错误保留响应体片段（供错误模式识别）

## §13 扩展突破（详见 docs/EXTENSIONS.md）

- [x] **accuracy 套件**：自研 GSM8K 风格 20 题 + 客观题 20 题题库、答案提取
      评分（数字/选项字母）、分题集准确率与阈值判定
- [x] **stability 套件**：固定并发长跑 + 窗口聚合，吞吐衰减/错误率/HBM 增长
      三项阈值检查（duration/window 支持小数分钟）
- [x] **Web 管理台**：`mtest web`（aiohttp 零新依赖）：运行列表/详情/对比/
      JSON API
- [x] **batch 调度**：`mtest batch` 多配置顺序/并行（信号量限流 + 端口冲突
      检查与 --auto-port 自动分配 + batch 汇总报告）

## dev 分支验证

- 单测 110 个全绿（新增 doctor/clientload/longctx-adaptive/accuracy/
  stability/web+batch 共 33 个）
- `mtest doctor`、`mtest web`（本机起服冒烟）、CLI dry-run 冒烟通过

## 剩余事项

- [ ] 在目标机（8×910B + vllm-ascend）完成真实闭环验证（单测不依赖 NPU）
- [ ] accuracy 接入官方 GSM8K/C-Eval 数据集（许可与加载器）
- [ ] dev → main 合并（建议策略见 EXTENSIONS.md 末节）
