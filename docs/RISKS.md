# RISKS.md — 风险与缓解落地状态（对照设计 §12）

> 分支：`dev` ｜ 更新：2026-09-25
> 结论先行：**5 项风险全部有代码级缓解落地**；其中 2 项（R1 版本匹配、
> R5 瓶颈归因）受客观条件限制只能做到"提示/归因"而非"消除"，详见各条。

## R1. vllm-ascend 版本参数差异

| 缓解手段 | 状态 | 说明 |
|---|---|---|
| args 全量从 yaml 透传 | ✅ 主分支已有 | 兼容各版本参数（含 additional-config JSON） |
| dry-run 提前发现 | ✅ 主分支已有 | `mtest validate` / `mtest run --dry-run` 预览生成命令 |
| command 逃生舱 | ✅ 主分支已有 | GOT-OCR 等特殊分支整体替代启动命令 |
| **doctor 环境预检** | ✅ dev 新增 | `mtest doctor [-c x.yaml]`：运行时/NPU 卡数 vs TP/模型路径/端口占用/docker/内置数据完整性 |
| **版本匹配表** | ⚠ dev 新增（咨询性） | `mtest/data/version_matrix.yaml` 静态配对表 + pip 探测，不在表中给 warn |
| **args lint** | ⚠ dev 新增（提示式） | `mtest validate --lint-args`：`serve.args` 不在已知清单（`mtest/data/vllm_args.yaml`）时提示，**不拦截**透传 |

**实现困难点（如实标注）**：

- 权威的 vllm↔vllm-ascend 匹配表在官方 README 且随版本演进，**离线工具无法
  自动同步**。当前做法是静态快照 + 明确标注"以官方 README 为准"。要彻底解决
  需要联网拉取官方匹配表（或定期人工更新），已留数据文件与告警通路。
- args 清单无法穷举（vllm 参数数百个且随版本增删），lint 定位是"拼写错误
  提示"而非"合法性校验"，误报率与漏报率只能取折中。

## R2. 多模态在昇腾后端支持不一

| 缓解手段 | 状态 |
|---|---|
| 套件可单独关闭（enabled 开关 + `--suite` 过滤） | ✅ 主分支已有 |
| functional 兜底（multimodal 必跑冒烟子集） | ✅ 主分支已有 |
| 特殊模型走 command 逃生舱 | ✅ 主分支已有（qwen2.5-vl 配置内有示例注释） |

**说明**：此项风险的根因在 vllm-ascend 各版本对 VL 系列的实现完成度，
工具侧能做的是"失败可见 + 可降级 + 可绕行"，均已具备。无进一步可消除空间，
归类为**外部依赖风险，已缓解至可运维水平**。

## R3. npu-smi 输出格式随 CANN 变化

| 缓解手段 | 状态 |
|---|---|
| 容忍式正则（主表两行组 + `-t usages` 简表双形态） | ✅ 主分支已有 |
| 多版本 fixture 单测（23.3 主表 / 24.1 变体 / usages 简表） | ✅ 主分支已有（`tests/test_monitor.py`） |
| 失败降级不阻断 | ✅ 主分支已有（连续失败≥3 标记 degraded，恢复自动清除） |
| doctor 对 npu-smi 可用性/解析做独立检查 | ✅ dev 新增 |

**实现困难点**：无法预知未来 CANN 的格式。当前解析器覆盖两种已知形态 +
doctor 独立探测；新格式出现时按 TROUBLESHOOTING.md §3 增补正则与 fixture
即可（约 10 行改动 + 1 个 fixture 测试）。

## R4. 长序列 OOM / 慢启动

| 缓解手段 | 状态 |
|---|---|
| 档位自动封顶（max-model-len − output_len − 64） | ✅ 主分支已有 |
| startup_timeout 可配 | ✅ 主分支已有 |
| 失败诊断包自动收割（日志尾部/npu-smi/进程列表） | ✅ 主分支已有 |
| **longctx 自适应降档** | ✅ dev 新增 | 探测请求命中"长度超限/OOM"类 4xx/5xx（正则匹配
  `maximum context length / too long / out of memory / kv-cache` 等）时自动
  二分降档重试（最多 2 次，下限 1024），降档轨迹记入 metrics 并在报告告警 |

**边界说明**：自适应只针对**服务端明确报长度/OOM 类错误**的场景；进程被
OOM-killer 直接杀死（连接重置类错误）时无法在请求层感知，依赖诊断包人工分析
——这是同机黑盒压测的固有边界。

## R5. 压测端自身成为瓶颈（同机 CPU 抢占）

| 缓解手段 | 状态 |
|---|---|
| aiohttp 异步单进程支撑 64 并发流式 | ✅ 主分支已有 |
| **客户端负载采样** | ✅ dev 新增 | `client_samples.csv`：loadavg(1m/5m)、mtest 进程
  CPU%（单核口径，psutil 可选增强 / os.times 差分兜底）、线程数；随 NPU 周期采样，
  独立于 npu-smi 成败 |
| **瓶颈归因结论** | ✅ dev 新增 | summary.md 自动输出：loadavg 峰值 > CPU 核数 80%
  或进程 CPU > 90% 时提示"压测端可能接近瓶颈，建议换机压测验证" |

**实现困难点（如实标注）**：

- 无 psutil 时进程 CPU 只有"本进程单核口径"，**拿不到整机各核分布**；Windows
  开发机上 loadavg 不可用（自动跳过该列）。psutil 未纳入依赖清单（保持零重依赖
  原则），目标 Linux 机上若需精确数据 `pip install psutil` 即增强。
- "客户端是否真是瓶颈"最终仍是归因判断而非确定性结论——工具提供证据
  （负载曲线 + 指标存疑标记），决策留给人。

## 汇总

| 风险 | 主分支基线 | dev 增强 | 残余风险 |
|---|---|---|---|
| R1 版本参数差异 | 透传+dry-run+逃生舱 | doctor / 版本表 / args lint | 匹配表需人工同步（联网自动同步未做） |
| R2 多模态支持 | 开关+兜底+逃生舱 | —（外部依赖） | vllm-ascend 实现完成度 |
| R3 npu-smi 格式 | 容忍解析+fixture+降级 | doctor 独立探测 | 未来格式未知 |
| R4 长序列 OOM | 封顶+超时+诊断包 | 自适应降档重试 | OOM-killer 级死亡无法请求层感知 |
| R5 压测端瓶颈 | 异步单进程 | 负载采样+归因结论 | 归因非确证；无 psutil 时精度受限 |
