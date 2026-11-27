# TokenRouter × vLLM v0.33.0 — 8×A100 实测评估与 Codex 接管指南

本文档面向在 8×A100 Linux/CUDA 服务器上接管实测的 **Codex** 代理。
**你的核心目标**：在真实 GPU 与真实模型（Qwen3 / DeepSeek-R1 同词表大小模型对）上完成系统性验证、正确性核查与性能 Profiling，产出一份客观、可复现、数据详实的 `EVAL_REPORT.md`，交回主架构师分析并决定下一阶段优化方向。

---

## 1. 项目背景与代码全景

本项目参考 TokenRouter 论文（对 SGLang 的双模型协同改造）与 GlimpRouter 步级路由逻辑，在 **vLLM v0.33.0** 上实现了双模型异步协同推理服务：

- **双引擎独立进程**：SLM（入口模型，对外暴露唯一 OpenAI API）与 LLM（对端模型，支持 TP=1/2/4）各自运行独立 `EngineCoreProc`，独立持有 `Request` 与 KV Cache；两者共享相同词表和 tokenizer，跨引擎只传 `token_ids`。
- **同进程同步步级路由（GlimpRouter 逻辑）**：
  - 按 `"\n\n"` 划分推理步骤；
  - 每步开始，SLM 采样固定 `n`（`probe_tokens >= 1`）个探针 token（期间抑制客户端输出），利用 `SamplingParams(logprobs=20)` 返回的 top-k logprobs 计算平均或最大熵（`probe_agg = "mean" | "max"`）；
  - 熵 `< threshold`：确认并刷出探针 token 给客户端，SLM 续写本步直至 `"\n\n"`；
  - 熵 `>= threshold`：回滚全部 `n` 个探针 token（客户端不可见）、停放本地请求为 `PEER_WAIT`（保留 KV），通过 ZMQ `PUSH/PULL` 把前缀交给 LLM；
  - LLM 续写至完成 1 步（命中 `"\n\n"`）后保留全部生成（`discard_num_tokens=0`），停放并把增量后缀交回 SLM；SLM 利用 vLLM chunked prefill 补算增量 KV 后继续下一步探针；
  - 命中 `</think>` 后，携带 `finish_thinking=True` 将最终作答阶段整体交给 LLM 直至结束（EOS / `max_tokens`），不再交回 SLM；LLM 结束后广播 `finished`，SLM 释放停放 KV 并向客户端提交最终输出。

### 本分支相对官方 `vllm v0.33.0` 的全部改动文件（11 个核心文件 + 启动/验证套件）

| 类别 | 路径 | 职责 |
| --- | --- | --- |
| 原语状态 | `vllm/v1/request.py` | `PEER_WAIT` 非终止状态、handoff 序号/前沿、探针期输出抑制与缓冲 |
| 原语输出 | `vllm/v1/core/sched/output.py` | `CachedRequestData.peer_resumed_req_ids` |
| 原语调度 | `vllm/v1/core/sched/scheduler.py` | `suspend_requests(discard_num_tokens=...)`、`resume_requests`（含满批 `_peer_ready_queue` 排队）、`finish_parked_request`、`has_requests` 休眠排除、探针期输出抑制 |
| Worker 同步 | `vllm/v1/worker/gpu_model_runner.py` | 收到 `peer_resumed_req_ids` 时用 `all_token_ids` 刷新 `output_token_ids` 并重装 `InputBatch` |
| 路由协议 | `vllm/v1/core/sched/token_routing/peer_protocol.py` | `PeerReq`（`need` / `finished` / `abort`）与 `PeerSamplingParams` |
| 调度底座 | `vllm/v1/core/sched/token_routing/base_routing_scheduler.py` | `BaseTokenRoutingScheduler(Scheduler)`：`kvcache_loc` 追踪、双边 fence 校验、首次新建/后续增量恢复、入口输出汇总、非入口抑制、`abort` 广播 |
| 路由策略 | `vllm/v1/core/sched/token_routing/glimprouter_scheduler.py` | `GlimpRouterSLMScheduler` 与 `GlimpRouterLLMScheduler`（含 `_step_start_len` 防重入、`</think>` 跳过 `"\n\n"` 修复、`LogprobsLists` 第 0 列去重） |
| ZMQ 传输 | `vllm/v1/core/sched/token_routing/peer_transport.py` | `PeerTransportConfig`、`PeerTransportSender`（ZMQ PUSH）、`open_peer_pull_socket` / `drain_peer_pull_socket`（ZMQ PULL） |
| 引擎接线 | `vllm/v1/engine/__init__.py`、`vllm/v1/engine/core.py` | 新增 `EngineCoreRequestType.PEER_MSG`；在 `EngineCoreProc.process_input_sockets` 复用现有 `zmq.Poller` 监听 peer PULL 并唤醒 `input_queue`；step 与 abort 后刷新外发/即时客户端输出 |
| 启动与验证 | `tokenrouter_vllm/configs/glimprouter_qwen3.yaml`、`tokenrouter_vllm/scripts/launch_dual_engine.py`、`tokenrouter_vllm/scripts/verify_a100.py` | 双引擎 YAML 配置、一键拉起与健康检查、词表扫描与并发验证脚本 |

### 固定基线约束（第一版严禁擅自更改）

- 环境变量：`VLLM_USE_V2_MODEL_RUNNER=0`（固定 V1 ModelRunner）
- 启动参数：`--no-enable-prefix-caching`、`--no-async-scheduling`
- 并行配置：PP=1（SLM TP=1，LLM 支持 TP=1/2/4）
- 词表前提：SLM 与 LLM 必须为同系列同词表模型（例如 `Qwen3-0.6B/4B + Qwen3-32B` 或同系 DeepSeek-R1）

---

## 2. Codex 执行原则（必须严格遵守）

1. **实事求是，禁止“制造通过”**：不得通过删除/放宽断言、添加 `skip`/`xfail` 或篡改预期来让测试变绿。
2. **定位优先，克制修改**：
   - 如发现导致启动/运行中断的**明确小 Bug**（如导入错误、属性名/签名差异、小边界遗漏），允许做**最小局部修复**以推进后续评估，但必须在报告中完整记录：**原始报错栈、复现步骤、根因与你的 diff**；
   - **禁止擅自重构整体架构**（例如不要擅自改成外置路由、不要擅自引入 Delayed Batching、不要擅自改写 KV 分配器或启用 V2 runner/APC）。
3. **所有结论必须附带命令、原始关键日志与量化数据**，未执行项明确标注“未执行 + 原因”。

---

## 3. 分阶段评估清单（请按 E0 → E5 顺序执行）

### E0：环境与安装确认
1. 记录服务器环境：GPU 型号/显存（`nvidia-smi`）、CUDA 版本、PyTorch 版本、Python 版本、可用本地模型权重路径（SLM & LLM）。
2. 在本仓库根目录使用 `uv` 安装可编辑版 vLLM（优先预编译 wheel）：
   ```bash
   VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
   ```
3. 确认 `VLLM_USE_V2_MODEL_RUNNER=0` 下单卡原生 `vllm serve` 能正常加载 SLM 模型（排除基础环境问题）。

### E1：核心单测回归（原语层首次真实运行验收）
在仓库根目录运行：
```bash
VLLM_USE_V2_MODEL_RUNNER=0 pytest tests/v1/core/test_peer_handoff.py -q
VLLM_USE_V2_MODEL_RUNNER=0 pytest tests/v1/core/test_scheduler.py -q
VLLM_USE_V2_MODEL_RUNNER=0 pytest tests/v1/worker/test_gpu_model_runner.py -k peer_handoff -q
```
记录每组通过/失败数；如有失败，贴出完整失败用例名与 traceback。

### E2：底层关键假设核验（来自前期设计的专项待验证项）
1. **Qwen3 / DeepSeek-R1 词表 `"\n\n"` 合并现象核查**：
   ```bash
   python tokenrouter_vllm/scripts/verify_a100.py --check-tokenizer <SLM_MODEL_PATH> --skip-api
   ```
   记录：词表中包含 `"\n\n"` 的 token 总数、典型合并样本（如 `".\n\n"`、`":\n\n"`、`")\n\n"`）、以及 `encode(".\n\n")` 的实际分词结果。
2. **`ModelRunnerOutput.logprobs` 列布局与熵精度核查**：
   - 抽样打印 SLM 单步 `sliced.logprob_token_ids[0][:5]` 与 `sliced.logprobs[0][:5]`，确认第 0 列是采样 token、第 `1:1+top_k` 列是 top-k（以及去重前后熵值差异）；
   - 对比 `top_logprobs=20` 截断归一化熵与全词表精确香农熵（抽样 20 个不同确定度位置），给出最大误差与排序相关性。

### E3：单请求端到端正确性与数值等价性（核心验收）
修改 `tokenrouter_vllm/configs/glimprouter_qwen3.yaml` 指向服务器实际模型路径与空闲卡位，先 `--dry-run` 检查命令，再拉起双引擎：
```bash
python tokenrouter_vllm/scripts/launch_dual_engine.py --config tokenrouter_vllm/configs/glimprouter_qwen3.yaml
```
依次验证以下 4 个场景（均使用 `temperature=0.0` 贪心解码便于复现对比）：
1. **标准 GlimpRouter（`probe_tokens=1`, `threshold=0.5`）**：
   - 验证发生真实 SLM ↔ LLM 多次双向交接（记录交接次数、每次交接 `base_len` 与 `new_token_ids` 长度）；
   - 验证客户端拿到的完整输出文本语义连贯、**无重复 token、无被回滚的脏探针泄露、无丢失 LLM 后缀**。
2. **多 token 探针回滚等价性（`probe_tokens=3`, `probe_agg="mean"` 与 `"max"`）**：
   - 验证至少触发一次 `discard_num_tokens=3` 的回滚交接；
   - **关键数值验证**：抓取某次发生 `n=3` 回滚并由 LLM 续写返回后、SLM 增量补算并采出下一个 token 的结果，与“把相同最终前缀一次性送给干净 SLM 重新 prefill”采出的下一个 token 及 top-5 logits 对比（检查 `max_abs_diff` 是否在 bf16 正常误差内，确认原地覆写 KV 槽位无脏数据污染！）。
3. **连续两步交由 LLM（空后缀 `[]` 恢复）**：
   - 设低阈值（如 `threshold=0.0` 强制每次探针都交 LLM），验证 SLM 发回空后缀 `[]` 给 LLM 时，LLM 无报错且利用保留的未算尾 token 顺畅续写下一步，不重复 token、不卡死。
4. **`</think>` 交接与终态 KV 归零**：
   - 确认命中 `</think>` 后 SLM 带 `finish_thinking=True` 移交 LLM，LLM 后续即使出现 `"\n\n"` 也不再交回 SLM，直至自然 EOS 或 `max_tokens`；
   - 请求完成后检查 SLM 与 LLM 两侧 `kv_cache_usage` 与 `peer_waiting` 数量均回到 **0**。

### E4：多请求并发与健壮性测试
1. **并发冒烟与满批排队（`_peer_ready_queue`）**：
   - 跑 `verify_a100.py --concurrency 16`；
   - 再刻意将 SLM 和 LLM 的 `max_num_seqs` 调小（如 `4`），用 `--concurrency 12` 并发打压，确认触发满批恢复排队路径且全部请求正常完成、无 `ValueError`、无死锁。
2. **中途取消（Client Abort）清理**：
   - 在请求停放于 `PEER_WAIT` 期间主动断开客户端连接或调用 abort，确认 SLM 通过 `PeerReq(status="abort")` 通知 LLM，两侧 KV 块与 `requests` 字典均干净释放。
3. **空闲休眠与 CUDA Graph 命中检查**：
   - 无请求或全 `PEER_WAIT` 时观察 SLM/LLM 进程 CPU 占用（确认无忙等空转）；
   - 查看启动日志与运行日志，确认 `VLLM_USE_V2_MODEL_RUNNER=0` 下 CUDA Graph 捕获模式及是否频繁回退 eager mode。

### E5：性能基线与 Profiling（为二期优化提供量化依据）
在同一批 16–32 条数学/推理 prompt（`max_tokens=1024` 或 `2048`）下，对比 3 组基线在并发 `1 / 8 / 16 / 32` 下的表现：
- Baseline 1：纯 SLM 单引擎
- Baseline 2：纯 LLM 单引擎
- Ours：双模型 GlimpRouter（`probe_tokens=1`, `threshold=0.5` 和 `1.0`）

请采集并汇报：
1. **端到端指标**：TTFT（P50/P95）、TPOT（P50/P95）、整体输出吞吐（tok/s）、平均完成耗时。
2. **协同特征指标**：LLM 生成 token 占比、平均每请求交接次数、SLM 与 LLM 的平均运行 batch size、峰值 KV Cache 使用率。
3. **瓶颈 Profiling（决定二期该做什么优化）**：
   - SLM 侧 `logprobs=20`（`topk + rank + D2H`）占单步耗时比例（可对比 SLM 开/关 `logprobs=20` 的纯 decode TPOT）；
   - ZMQ 跨进程单次 `PeerReq` 传递延迟；
   - SLM `tokenizer.decode` 尾窗检查的 CPU 耗时占比；
   - LLM 侧是否出现明显“小 batch 低 GPU 利用率”现象（作为是否引入 Delayed Batching 的直接判据）；
   - 高并发（如 32/64 并发）下停放请求长期占着 KV 是否导致活跃请求频繁抢占或 KV 紧张（作为是否引入 Parked KV 驱逐/回收的直接判据）。

---

## 4. 交付报告格式（请保存为 `tokenrouter_vllm/EVAL_REPORT.md`）

请严格按以下结构输出最终报告，便于直接交回主架构师决策：

```markdown
# TokenRouter × vLLM v0.33.0 — 8×A100 实测评估报告

## 1. 一句话总评（Pass / Partial / Blocked + 核心结论）
## 2. 实验环境与模型配置（GPU / CUDA / vLLM / 模型路径 / 卡位 / TP）
## 3. E1–E4 功能与正确性结果汇总表（每项：Pass/Fail + 关键证据/数值误差）
## 4. 发现的问题与临时修复记录（若有：复现步骤 + 原始报错 + 根因 + 最小 diff）
## 5. E5 性能基线与 Profiling 对比表（纯 SLM vs 纯 LLM vs GlimpRouter）
## 6. 瓶颈诊断（量化回答：Sampler 熵开销 / Delayed Batching 必要性 / Parked KV 压力 / CUDA Graph 命中率）
## 7. 原始日志与复现命令索引
```
