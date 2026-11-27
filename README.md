# Collaboration-vllm

个人实验仓库：SLM + LLM 协同推理原型。小模型逐步生成推理步骤，在每个 step 边界用 top-k logprob 的归一化熵判断是否需要升级，超过阈值就把后续生成交给大模型。

> 个人测试用途，与任何上游开源项目无关，也不向上游提交任何内容。

## 思路

- SLM 与 LLM 各自独立起服务，请求由同一套路由调度器接管。
- SLM 每个推理步骤只生成固定数量的探测 token（`probe_tokens`），按 `mean` 或 `max` 聚合成归一化熵；`>= routing.threshold` 即在该 step 边界交接给 LLM。
- 交接后 LLM 接管生成，直到下一个 step 边界再次回判；`finished_str`（如 `</think>`）出现后由 LLM 完整输出剩余内容。
- 两个引擎之间走对等传输：单机多卡用 `ipc://`（最低延迟），跨机用 `tcp://IP:PORT`。

## 新增代码

| 路径 | 说明 |
| --- | --- |
| `tokenrouter_vllm/scripts/launch_dual_engine.py` | 启动器：读 YAML，按角色拼装命令、拉起两个引擎并注入调度器类 |
| `vllm/v1/core/sched/token_routing/base_routing_scheduler.py` | 路由调度基类与 `RouteDecision` |
| `vllm/v1/core/sched/token_routing/glimprouter_scheduler.py` | SLM / LLM 两个调度器（熵判定与交接逻辑） |
| `vllm/v1/core/sched/token_routing/peer_protocol.py` | 引擎间消息与请求协议 |
| `vllm/v1/core/sched/token_routing/peer_transport.py` | 传输层封装 |
| `tokenrouter_vllm/configs/glimprouter_qwen3.yaml` | 示例配置（双引擎、路由阈值、传输地址） |
| `tokenrouter_vllm/scripts/verify_a100.py` | 运行环境自检 |
| `tokenrouter_vllm/CODEX_EVAL_GUIDE.md` | 评测流程说明 |

## 运行

```bash
pip install -e .

# 只打印将要执行的命令，不真正拉起引擎
python tokenrouter_vllm/scripts/launch_dual_engine.py \
  --config tokenrouter_vllm/configs/glimprouter_qwen3.yaml \
  --dry-run

# 正式启动两个引擎
python tokenrouter_vllm/scripts/launch_dual_engine.py \
  --config tokenrouter_vllm/configs/glimprouter_qwen3.yaml
```

- `--vllm-bin`：vllm 可执行文件路径，默认取 PATH 中的 `vllm`。
- `--startup-timeout`：等待引擎就绪的秒数。
- 配置中的 `model`、`cuda_visible_devices`、`tensor_parallel_size`、`gpu_memory_utilization` 需按自己机器调整。

## 状态

- 实验性代码，接口与行为随时可能变化，未做完整回归。
- 仓库刻意不含 `.github/`（CI 与依赖机器人配置），避免在个人仓库里触发自动化。
- 除上表列出的文件外，其余目录是 vLLM 0.33.0 的源码快照，仅作为运行底座。
