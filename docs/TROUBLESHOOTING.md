# TROUBLESHOOTING.md — 常见故障排查

## 1. 服务启动失败 / 健康检查超时

**现象**：`mtest run` 在等待就绪阶段失败，终端打印 ERROR/Traceback/CANN 关键行。

**自动动作**：mtest 已收割诊断包到 `results/<run_id>/serve_failure/`：

| 文件 | 内容 |
|---|---|
| `vllm_tail.log` | vllm.log 尾部 200 行 |
| `npu_smi.txt` | `npu-smi info` 快照（看是否有进程占卡） |
| `process_list.txt` | `ps -ef | grep vlll/torch` 进程列表 |

**排查顺序**：

1. `serve_failure/vllm_tail.log` 找 Traceback 常见原因：
   - `tpu/mlu/davinci device not found` → devices 数量与 TP 不符、驱动异常
   - `No such file or directory: ...safetensors` → model.path 错误
   - `Port already in use` → 端口占用（`ss -lntp | grep 8000`）
   - CANN/ACL 报错 → 环境未激活（检查 `serve.env_init` 是否 source 了
     `set_env.sh`）；vllm 与 vllm-ascend 版本不匹配
2. 残留进程占卡：`npu-smi info` 看 HBM 占用；`pkill -f "vllm serve"` 清理；
   或 `mtest serve down -c <yaml>`。
3. 大模型加载慢：`startup_timeout` 默认 1800s，不够就调大。

**预防**：先用 `mtest validate -c <yaml>` 预览生成的命令，肉眼确认参数。

## 2. 压测全部请求失败（connect refused / timeout）

- 服务未真正就绪就开测：确认 `mtest serve status` 健康检查为 true。
- `client.base_url` 与服务监听不符（host 0.0.0.0 时客户端应用 127.0.0.1）。
- 防火墙/SELinux 拦截本机端口（少见）。
- 请求超过 `request_timeout`（长序列档常见）：适当调大 `client.request_timeout`。

## 3. npu-smi 采样失败（降级告警）

- 现象：`NPU 采样失败（第 N 次，已降级）`。测试不受影响，报告缺 NPU 曲线。
- 原因：非 Linux / 无 npu-smi / 输出格式变化。
- 处理：`npu-smi info` 手动执行确认格式；若 CANN 版本输出格式与解析器不兼容，
  在 `mtest/monitor.py::parse_npu_smi_output` 增加正则形态并补充
  `tests/test_monitor.py` fixture（容忍式解析 + 多版本单测是设计要求）。

## 4. 长序列 OOM / 段错误

- 档位已自动封顶（max-model-len − output_len − 64），但仍可能因 kv-cache 不足
  在高并发长输入下 OOM。
- 处理顺序：降 `longctx.concurrency`；调低 `gpu-memory-utilization` 给系统留量；
  缩短 `max-model-len`；确认 `swap-space` 配置。
- 服务崩溃后会有残留：`mtest serve down` 或手动清理后重试。

## 5. 校准不达标（calibrated: false）

- 现象：summary 中输入长度校准 "未达标"，实际 prompt tokens 偏离目标 ±10%。
- 原因：服务端未返回 usage（部分后端/代理剥掉 usage）；prompt 模板 token 占比
  异常；目标长度超出模型词表表现。
- 影响：档位标签仍用目标值，横向对比同档位仍有效；明细 CSV 有真实
  prompt_tokens 可后处理。可在 `data/prompts/` 补充更长短语料改善初值。

## 6. OCR 结果乱码 / CER 偏高

- 先看 functional 的 `no_mojibake` 与具体输出（报告备注列）。
- 多模态在 vllm-ascend 各版本支持度不一：升级 vllm-ascend；特殊模型
  （GOT-OCR 等）走 `serve.command` 逃生舱 + 保持 `type: multimodal`。
- 图片过大触发预处理限制：查看 `limit-mm-per-prompt` 与服务端日志。

## 7. functional 超长输入用例返回 5xx

- 设计期望是截断（200）或 4xx 优雅拒绝；5xx 说明服务端对超长输入处理有缺陷，
  属于被测服务的问题——保留该失败作为测试结论，而不是绕过。

## 8. 压测端自身瓶颈

- 同机压测时客户端与推理服务抢 CPU：观察运行期间 `top`；64 并发流式 SSE 下
  aiohttp 单进程通常足够（设计 §12 风险表）。
- 若客户端 CPU 打满：换一台机器压（`client.base_url` 指向远端），指标含义
  变为端到端（含网络）。

## 9. Windows / 开发机上开发

- 代码目标平台是 Linux；开发机上可跑：`pytest`（全部单测不依赖 NPU）、
  `mtest list / validate / run --dry-run`。
- process 模式的 `bash -lc`、进程组、npu-smi、docker 设备映射仅目标平台有效。

## 10. 通用信息收集

报告生成失败或需要深挖时，手动收集：

```bash
npu-smi info                    # 芯片状态
cat /usr/local/Ascend/driver/version.info 2>/dev/null   # 驱动版本
pip show vllm vllm-ascend       # 服务端版本（在 vllm 环境）
results/<run_id>/vllm.log       # 完整服务日志
results/<run_id>/config.resolved.yaml   # 生效配置
```
