# REPORT_GUIDE.md — 报告产物解读与 metrics.json Schema

## 1. 产物目录

`results/<run_id>/`（run_id = `YYYYmmdd-HHMM_<model_name>`，重名自动追加 `-2`）：

| 文件 | 内容 |
|---|---|
| `summary.md` | 人读摘要：环境信息、服务启动耗时、各套件结果表、NPU 峰值、异常与结论建议 |
| `metrics.json` | 全量原始指标（schema 固定，供 compare 与二次分析） |
| `npu_samples.csv` | NPU 采样明细 |
| `vllm.log` | 服务完整日志（process 模式） |
| `config.resolved.yaml` | 生效配置（合并+裁剪后），可复现实验 |
| `perf_details.csv` 等 | 各套件逐请求/逐文件明细 |
| `serve_failure/` | 仅启动失败时：vllm 尾部日志、npu-smi 快照、进程列表 |

## 2. summary.md 怎么读

- **环境信息**：机型/Python/CPU、vllm 与 vllm-ascend 版本（pip 探测 + 服务端
  `/version`）、NPU 卡数与单卡 HBM。版本不匹配是兼容问题的第一嫌疑。
- **模型与服务**：启动耗时（process 模式从 spawn 到健康检查通过）、实际使用的
  vllm 启动命令（含逃生舱）。
- **套件总览**：各套件状态（passed/failed/error）与耗时。
- **perf 矩阵**：每格（input_len × 并发）成功率、RPS、输出 tok/s、TTFT p50/p99、
  TPOT、e2e p99；附输入长度校准结果（目标/实际 prompt tokens）。
- **longctx**：TTFT（prefill 等待）、prefill/decode 吞吐、e2e 分布。
- **embedding**：批量×并发网格的 sent/s 与延迟；区分度两组分布。
- **ocr**：img/s、按分辨率分档延迟、每文件 CER、容错用例结果。
- **NPU 指标（按阶段）**：各阶段（套件/档位标记）AICore 峰值、HBM 峰值、
  功耗/温度峰值；与压测时间轴对齐（阶段标记写入采样行）。
- **结论与建议**：规则式提示（失败套件、成功率、高并发 TTFT 排队、HBM 接近
  上限等），最终判断仍需人工结合业务。

## 3. npu_samples.csv 字段

`timestamp`（ISO 时间）、`epoch`（unix 秒）、`phase`（阶段标记）、`npu`（卡号）、
`aicore_util_pct`、`hbm_used_mb`、`hbm_total_mb`、`power_w`、`temp_c`。

- 采样失败降级：连续失败 ≥3 次标记 degraded，恢复后自动清除；采样不可用
  （无 npu-smi / 非 Linux）不影响测试本身，summary 中会注明。

## 4. metrics.json Schema（v1）

```jsonc
{
  "schema_version": 1,
  "run_id": "20260924-1530_qwen2.5-7b-instruct",
  "created_at": "2026-09-24T15:30:00",
  "mtest_version": "0.1.0",
  "total_duration_s": 3600.0,

  "config": {
    "model":  { "name": "...", "path": "...", "type": "llm", "served_name": null,
                "trust_remote_code": true },
    "serve":  { "mode": "process", "args": { }, "docker": { } },
    "client": { "base_url": "...", "api_key": null, "request_timeout": 600 },
    "enabled_suites": ["functional", "perf", "longctx"],
    "source_file": "configs/models/qwen2.5-7b-instruct.yaml"
  },

  "serve": { "mode": "...", "managed": true, "startup_seconds": 210.0,
             "command_preview": "vllm serve ...", "keep_alive": false },

  "environment": {
    "hostname": "...", "platform": "...", "python": "3.10", "cpu_count": 128,
    "mtest_version": "0.1.0", "collected_at": "...",
    "vllm_version_pip": "0.9.2", "vllm_ascend_version_pip": "0.9.2",
    "server_version": "0.9.2",
    "npu": { "npu_smi_header": "npu-smi 24.1...", "npu_count": 8, "hbm_total_mb": 65536 }
  },

  "suites": [                            // SuiteResult 列表，按执行顺序
    {
      "name": "perf", "status": "passed",
      "started_at": "...", "ended_at": "...", "duration_s": 1200.0,
      "metrics": {
        "matrix": {                      // key: "<input_len>x<concurrency>"
          "128x8": {
            "requests": 200, "ok": 200, "success_rate": 1.0,
            "wall_s": 42.0, "rps": 4.76, "output_tps": 1218.0, "input_tps": 609.0,
            "total_output_tokens": 51200,
            "ttft_s":  { "n": 200, "mean": 0.03, "std": 0.01, "min": 0.02,
                         "max": 0.09, "p50": 0.03, "p90": 0.04, "p99": 0.06 },
            "e2e_s":   { "...": "同上结构" },
            "tpot_s":  { "...": "同上结构" },
            "itl_mean_s": { "...": "同上结构" },
            "errors": { "timeout": 0, "connect": 0 }
          }
        },
        "calibration": { "128": { "target_tokens": 128,
                                  "actual_prompt_tokens": 125,
                                  "calibrated": true, "iterations": 2 } },
        "rounds": 3, "dataset": "prompts", "total_requests": 7200, "total_ok": 7198
      },
      "details": [ ],                    // functional 为逐用例行；其余为空
      "artifacts": { "details": "results/<run>/perf_details.csv" },
      "error": null, "warnings": [ ]
    }
    // longctx 矩阵额外含 prefill_tps / decode_tps / prefill_tps_mean / decode_tps_mean
    // embedding 矩阵 key: "b<batch>c<cc>"，含 sent_per_s / latency_s / 批量扩展性
    //      metrics.quality 含 dim / repeat_consistency_cosine / discrimination
    // ocr 矩阵 key: "cc<N>"，含 img_per_s / bucket_latency_s；
    //      metrics.quality 含 files（含 cer）/ cer_summary；fault_cases 容错结果
  ],

  "npu": {
    "interval_s": 5, "degraded": false, "samples": 720,
    "summary": { "overall": { "hbm_used_max_mb": 64000, "aicore_util_max": 98.0,
                              "...": "..." },
                 "perf[in=4096,cc=64]": { "...": "同结构" } },
    "csv": "results/<run>/npu_samples.csv"
  },

  "warnings": [ ],          // 配置裁剪告警 + 各套件告警合并
  "error": null,            // 服务失败/致命错误（套件级错误在 suites[].error）
  "conclusions": [ ]        // 规则式结论
}
```

**兼容性承诺**：schema 向后兼容只增不删；`matrix` 的 key 约定
`<input_len>x<cc>`（perf/longctx）、`b<batch>c<cc>`（embedding）、`cc<N>`（ocr）。

## 5. compare 的对比口径

`mtest report compare A B`：取两次运行公共档位——

- perf：RPS、输出 tok/s、TTFT p99 并排 + 差异百分比（(B−A)/|A|）
- embedding：sent/s、延迟 p99
- ocr：img/s、延迟 p99

差异百分比为正表示 B 更优（吞吐类）或更慢（延迟类），读表时注意列语义。
