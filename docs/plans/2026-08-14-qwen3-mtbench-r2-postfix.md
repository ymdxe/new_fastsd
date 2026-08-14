# Qwen3 MT-Bench 两 A5000 四方法实验记录（post-fix）

## 固定方案

- 仓库：`ymdxe/new_fastsd`，分支 `agent/token-budget-continuous-batching`，提交 `69f5eca`。
- node1：Qwen3-0.6B Draft，两个进程，物理 GPU0/1（RTX A5000）。
- node2：Qwen3-8B Target，FastSD/SpecEdge 使用物理 GPU0（RTX A6000）。
- 数据：MT-Bench 第一轮 80 条问题，Qwen3 chat template，`enable_thinking=false`。
- `workload_hash=f100fe17c5b30626e80da573c307f0340d5005eb12c5fbd00d15f921f6a04455`。
- 生成：`max_new_tokens=256`，temperature 0，`gamma=4`，seed 42，EOS stop。
- FastSD：`token_budget=64`、`batch_size=2`、`kv_batch_mode=varlen`、pipeline/proactive 开启、gamma step=2；每个 Draft 显式 `max_tasks_per_draft=40`。
- SpecEdge：官方固定 revision 与适配层，`max_budget=32`、两个 client。
- Draft-only：两个 A5000 Draft worker，不访问 Target。

配置文件：`configs/evaluation/qwen3_8b_0.6b_mt_bench_2gpu_seed42_r2_20260814.json`。

## 当前结果（80/80）

| 方法 | 总 tokens | wallclock (s) | 吞吐 (tok/s) | TTFT avg / P95 / P99 (ms) | TPOT avg / P95 / P99 (ms) | E2E avg / P95 / P99 (ms) | 接受率 | accepted/verify |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FastSD | 16,646 | 1,317.805 | 12.6316 | 1,012.01 / 1,362.51 / 1,771.25 | 147.82 / 261.91 / 296.51 | 32,612.73 / 58,710.28 / 72,760.27 | 0.27090 | 3.30364 |
| SpecEdge* | 16,805 | 1,070.625 | 15.6964 | 820.85 / 1,688.83 / 1,780.42 | 110.13 / 215.35 / 235.67 | 25,203.92 / 56,485.81 / 61,141.73 | N/A | 3.72163 |
| Draft-only | 14,410 | 169.667 | 84.9313 | 47.26 / 33.90 / 853.70 | 22.80 / 28.16 / 32.71 | 4,143.36 / 6,410.58 / 7,531.26 | N/A | N/A |
| standard_sd | — | — | — | — | — | — | — | — |

`*` SpecEdge 在运行中途与另一用户的 node2 GPU0 任务重叠，故该行是跑通后的观测值，不能作为干净公平性能结论。
Draft-only 不具有 8B Target 的质量含义，不能直接与三种 Target 方法比较回答质量。

FastSD scheduler 落盘指标：`iterations=59403`、`plans=6035`、`used_tokens=73192`、`verify_slices=6148`、`prefill_slices=165`、`partial_verify=1`、`partial_prefill=65`。本轮显式关闭 target cache offload，因此 prefetch 命中/异步搬运计数为 0。

## standard_sd 阻塞

standard_sd 首次尝试使用 node2 GPU1；该卡只有 Xorg 显示进程，模型加载时 PyTorch 报 `CUDA-capable device(s) is/are busy or unavailable`，没有产生有效请求。此时 GPU0 仍被另一用户 `gaojq/task_2_RL.py` 占用约 17.3GiB，GPU2 也有另一用户计算任务。为避免杀掉他人进程或污染结果，standard_sd 未伪造结果；需要下一次在 node2 GPU0/其他空闲 A6000 真正空闲时重跑。

## 证据路径

- FastSD Edge：`exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42_r2_20260814/fastsd_full80/`（node1）。
- FastSD Cloud scheduler：`exp/exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42_r2_20260814/fastsd_cloud/scheduler_metrics.json`（node2；Cloud 的 `exp_name` 传入了带 `exp/` 的路径，因此出现双 `exp` 前缀）。
- SpecEdge raw：`exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42_r2_20260814/specedge/raw/qwen3_8b_0.6b_mt_bench_2gpu_seed42_r2_20260814/`（node1）。
- Draft-only：`exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42_r2_20260814/draft_only/`（node1）。
- 统一归一化结果：`exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42_r2_20260814/normalized/`（node1）。

本轮没有运行统一 MT-Bench LLM judge，也没有把文本质量或 target-output parity 宣称为通过；下一步应先在空闲 A6000 上补 standard_sd，再按同一 manifest 运行 judge/parity 检验。
