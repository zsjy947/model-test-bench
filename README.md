# model-test-bench (mtest)

> 一键模型测试平台 —— 面向 **8 卡 Ascend 910B / vllm(-ascend)** 的模型测试闭环。
> 覆盖 ≤32B 的 LLM（7B–27B 为主）、embedding 模型、OCR/多模态模型。

`mtest run -c configs/models/<model>.yaml` 一条命令完成：**加载配置 → 启动 vllm → 健康检查 → 按模型类型执行测试套件 → NPU 指标伴随采样 → 停服务 → 生成报告**。

## 核心特性

- **一键闭环**：起服务、压测、采样、报告全自动；也支持对接已启动服务（`serve.mode: external`）只测不管起停。
- **一模型一文件**：新模型接入只需复制 `configs/models/_template.yaml`，改 model 三要素（path / name / type）和 `serve.args`，其余全部继承 `configs/defaults.yaml`。
- **三类模型全覆盖**：
  - `llm` → perf 性能压测 + longctx 长序列专项 + functional 功能冒烟
  - `embedding` → embedding 专项 + functional
  - `multimodal` → ocr 专项 + functional
- **同机压测**：默认 `localhost` 直连，排除网络干扰，指标纯反映 NPU 推理性能。
- **NPU 伴随监控**：测试期间周期采样 `npu-smi info`（各卡 AICore 利用率 / HBM / 功耗 / 温度），与套件执行时间轴对齐。
- **服务管理三模式**：`process`（默认，全自动）/ `docker`（昇腾容器）/ `external`（对接已有服务）。
- **压测端零重依赖**：不装 torch/transformers，不加载本地 tokenizer；输入长度通过 usage 反馈校准（±10%）。

## 安装

```bash
# 建议在独立 venv（与 vllm 环境隔离，纯客户端轻依赖）
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # 开发安装
# 或 pip install .

# 冒烟（不依赖 NPU）
mtest list
mtest validate -c configs/models/qwen2.5-7b-instruct.yaml
```

## 快速上手

```bash
# 1) 校验配置并预览将生成的 vllm 启动命令（dry-run，不起服务）
mtest validate -c configs/models/qwen2.5-7b-instruct.yaml

# 2) 全自动一键测试（起服务 → 测试 → 报告 → 停服务）
mtest run -c configs/models/qwen2.5-7b-instruct.yaml

# 3) 只跑部分套件 / 覆盖并发
mtest run -c configs/models/qwen2.5-7b-instruct.yaml --suite perf,functional --concurrency 1,8

# 4) 复用已启动的服务
mtest serve up -c configs/models/qwen2.5-7b-instruct.yaml
mtest run -c configs/models/qwen2.5-7b-instruct.yaml --skip-serve

# 5) 查看与对比报告
mtest report show   <run_id>
mtest report compare <run_id_a> <run_id_b>
```

报告产物位于 `results/<run_id>/`：`summary.md`（人读摘要）、`metrics.json`（全量指标）、`npu_samples.csv`、`vllm.log`，启动失败时另有 `serve_failure/` 诊断包。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | 原始设计文档 |
| [docs/USAGE.md](docs/USAGE.md) | CLI 使用手册与典型工作流 |
| [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md) | 配置系统：三层合并、字段参考、args→命令生成规则 |
| [docs/SUITES_GUIDE.md](docs/SUITES_GUIDE.md) | 测试套件原理与指标口径 |
| [docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md) | 报告产物解读与 metrics.json schema |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 常见故障排查（启动失败诊断包等） |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 当前进度与后续计划 |

## 目录结构

```
model-test-bench/
├── mtest/                  # 核心包（CLI/编排/配置/服务/套件/支撑组件）
├── configs/                # defaults.yaml + models/ 一模型一文件
├── data/                   # 内置 prompt 池、长文本素材、OCR 样例图、功能用例
├── results/                # 运行产物（gitignore）
└── tests/                  # 单测（不依赖 NPU）
```

## 定位说明

本期不做精度评测（GSM8K/C-Eval）与稳定性长跑（架构已预留扩展点），不做 Web 管理台（核心引擎为库形态，已留接口）。详见设计文档 §13。

## License

MIT
