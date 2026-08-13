# Qwen3-0.6B / Qwen3-8B 跨服务器完整实验方案

## 1. 目标与证据边界

本方案评估 `new_fastsd` 的统一 token-budget continuous batching 改造，固定使用：

- node1：Qwen3-0.6B Draft，RTX A5000；
- node2：Qwen3-8B Target，RTX A6000；
- Edge/Cloud 协议：一轮 speculative verify 对应一次 HTTP request/response；
- Target：单个模型实例，不使用 tensor parallel；
- Draft：先 1 张 A5000，再扩展到 2/4 张 A5000。

本文分清两类证据：

1. **预实验**只证明模型、网络、协议、多进程和 batching 能跑通；
2. **正式实验**才用于比较性能、扩展性、尾延迟和质量。

预实验的 1/4 个短请求不能用于论文级性能结论。

## 2. 固定环境

### 2.1 服务器与模型

| 角色 | 服务器 | GPU | 模型 | 路径 |
|---|---|---|---|---|
| Cloud Target | node2 | 1 × RTX A6000 48 GiB，固定物理 GPU 0 | Qwen3-8B | `/home/hdd/zhangh/models/Qwen3-8B` |
| Edge Draft（单卡） | node1 | 1 × RTX A5000 24 GiB，物理 GPU 0 | Qwen3-0.6B | `/home/hdd/zhangh/models/Qwen3-0.6B` |
| Edge Draft（多卡） | node1 | 2/4 × RTX A5000，每卡一个 Draft 进程 | Qwen3-0.6B | 同上 |

两个服务器的仓库路径均为：

```text
/home/hdd/zhangh/workspace/new_fastsd
```

Python 环境均为：

```text
/home/hdd/zhangh/envs/new_fastsd/bin/python
Python 3.10
PyTorch 2.2.1+cu121
Transformers 4.57.6
```

### 2.2 网络

Cloud 必须只监听 node2 loopback：

```bash
export CLOUD_SERVICE_HOST=127.0.0.1
export CLOUD_SERVICE_PORT=8001
```

node1 通过 SSH 隧道访问 `127.0.0.1:18001`。不要将未认证的 8001 暴露到公网。

Windows 中继隧道示例：

```powershell
$sshExe = (Get-Command ssh).Source

$forward = Start-Process -FilePath $sshExe `
  -ArgumentList @('-o','BatchMode=yes','-o','ExitOnForwardFailure=yes',
                  '-N','-L','127.0.0.1:28001:127.0.0.1:8001','node2') `
  -WindowStyle Hidden -PassThru

$reverse = Start-Process -FilePath $sshExe `
  -ArgumentList @('-o','BatchMode=yes','-o','ExitOnForwardFailure=yes',
                  '-N','-R','127.0.0.1:18001:127.0.0.1:28001','node1') `
  -WindowStyle Hidden -PassThru
```

完成实验后按本次返回的明确 PID 停止两个隧道进程。

## 3. 运行前门禁

每次正式运行前必须记录以下信息：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
/home/hdd/zhangh/envs/new_fastsd/bin/python -m unittest discover -s tests -v
```

门禁标准：

- node1/node2 SHA 完全一致；
- 仅允许已知的未跟踪历史日志，不允许已跟踪源码有修改；
- Target/Draft 指定 GPU 无其他计算进程；
- Qwen3-0.6B 与 Qwen3-8B tokenizer 校验通过；
- Cloud `/health` 为 `ok`，且 node1 可通过隧道访问；
- scheduler/KV/协议测试全部通过。

正式实验应使用 `scripts/eval_suite.py prepare` 生成唯一 canonical manifest，并固定
`workload_hash`。不同方法不得各自重新采样到达时间。

## 4. 已完成的跨服务器预实验

### 4.1 代码基线

```text
0b8ec17e78a6fc25c5369b8b0c1dc50987928c0e
agent/token-budget-continuous-batching
```

该 SHA 包含 `CLOUD_SERVICE_HOST`，使 node2 Cloud 可以仅绑定 loopback。

两个服务器均通过：

- 15 个 scheduler 测试；
- 1 个真实 Torch/Transformers KV mixed-cached-length 测试。

### 4.2 预实验固定参数

```text
dataset                  = HumanEval
max_tasks_per_draft      = 1
max_tokens               = 16
temperature              = 0
gamma                    = 4
batch_size/max_num_seqs  = 4
token_budget             = 64
min_prefill_chunk_tokens = 16
prefill_chunk_quantum    = 32
pipeline                 = on
proactive draft          = off
```

关闭 proactive draft 是为了让本轮只验证服务器、网络、调度和 batching，不混入重叠优化。

### 4.3 单 A5000 结果

拓扑：node1 GPU0 Draft + node2 GPU0 Target。

结果：通过，Edge exit code 为 0，1 个请求生成 16 tokens。

| 指标 | 数值 |
|---|---:|
| system output tok/s | 1.3434 |
| active-window output tok/s | 2.7596 |
| request E2E | 5798.02 ms |
| TTFT | 1925.54 ms |
| decode TTFT | 853.33 ms |
| TPOT | 258.17 ms |
| acceptance rate | 0.1111 |
| scheduler plans | 18 |
| verify slices | 14 |
| prefill slices | 4 |
| partial Prefill | 3 |

证据路径：

```text
node1: exp/cross_preflight_single_a5000/edge_metrics_summary.json
node1: /tmp/new_fastsd_cross_single_20260813_edge.log
node2: exp/cross_preflight_single_a5000/scheduler_metrics.json
node2: /tmp/new_fastsd_cross_single_20260813_cloud.log
```

### 4.4 四 A5000 同步并发结果

拓扑：node1 GPU0-3 各一个 Draft + node2 GPU0 Target。

为避免四个模型加载完成时间不同导致请求错峰，本轮启用：

```text
arrival_distribution = poisson
arrival_rate_rps      = 1000
arrival_seed          = 42
```

Edge 的进程 barrier 保证四个 Draft 模型完成加载后再开始到达时间线。

结果：通过，Edge exit code 为 0，4 个请求各生成 16 tokens，共 64 tokens。

| 指标 | 数值 |
|---|---:|
| actual arrival span | 0.603 ms |
| system output tok/s | 2.3458 |
| active-window output tok/s | 8.4268 |
| request E2E avg | 6699.91 ms |
| TTFT avg | 2865.98 ms |
| decode TTFT avg | 1012.63 ms |
| TPOT avg | 255.60 ms |
| acceptance rate | 0.1515 |
| scheduler plans | 49 |
| verify slices | 54 |
| prefill slices | 16 |
| partial Prefill | 12 |

Cloud 日志确认真实 continuous batch 已形成：Prefill batch 中出现过 2 个和 3 个不同
session，而不是四个请求完全串行执行。

证据路径：

```text
node1: exp/cross_preflight_multi_a5000_sync/edge_metrics_summary.json
node1: /tmp/new_fastsd_cross_multi_sync_20260813_edge.log
node2: exp/cross_preflight_multi_a5000_sync/scheduler_metrics.json
node2: /tmp/new_fastsd_cross_multi_sync_20260813_cloud.log
```

## 5. 正式完整实验矩阵

### 5.1 核心方法

沿用 `docs/FOUR_METHOD_EVALUATION.md` 的统一入口：

1. `fastsd`：本次 token-budget continuous batching；
2. `specedge`：固定官方 revision + 本仓库指标适配器；
3. `standard_sd`：关闭 FastSD 调度、pipeline 和 proactive draft；
4. `draft_only`：相同数量 A5000，不访问 Target。

所有方法必须使用同一 canonical manifest、模型、tokenizer、停止条件和输出长度。

### 5.2 数据集与生成参数

| 数据集 | 请求数 | max new tokens | 主要质量指标 |
|---|---:|---:|---|
| HumanEval | 164 | 256 | pass@1 |
| GSM8K | 全集或固定子集 | 128 | numeric exact match |
| MT-Bench | 固定第一轮问题 | 256 | 输出完整率；质量评分另行执行 |

共同参数：

```text
temperature = 0
gamma       = 4
stop_policy = eos
```

### 5.3 拓扑与负载

主扩展性变量：

```text
Draft GPU count = 1, 2, 4
Target GPU count = 1（固定 node2 A6000 GPU0）
每张 A5000 = 1 个完整 Qwen3-0.6B Draft 进程
```

先用每种拓扑 20 个请求做负载校准。Poisson rate 从：

```text
0.05, 0.1, 0.2, 0.5 requests/s
```

选出低负载、接近饱和、过载三个点，再冻结到正式配置。另保留 1 RPS 作为压力测试，
但不得把过载排队结果与稳定负载结果混在同一均值中。

到达种子至少使用：

```text
1234, 2026, 3407
```

### 5.4 Scheduler 参数与消融

核心配置先固定：

```text
batch_size               = Draft GPU count
token_budget             = 512
min_prefill_chunk_tokens = 16
prefill_chunk_quantum    = 128
```

完成核心四方法比较后，再执行下列消融；每次只改变一个变量：

- `token_budget = 64, 128, 256, 512`；
- `prefill_chunk_quantum = 32, 64, 128`；
- `Draft GPU count = 1, 2, 4`；
- immediate closed-loop 与冻结的 Poisson manifest。

若 512 在校准阶段发生 OOM 或尾延迟失控，应保留失败证据，并将 256 设为主配置，而不是
静默丢弃失败点。

### 5.5 重复、顺序与热身

- 每个正式点运行 3 个到达种子；
- 每次冷启动后先运行 5 个 warmup 请求，warmup 不进入正式汇总；
- 四种方法按 seed 交错运行，避免全部 FastSD 都在同一时间段；
- 任意时刻只运行一个 Target 方法，避免 A6000 竞争；
- 每轮记录 GPU 型号、显存、驱动、SHA、manifest hash 和完整命令。

## 6. 指标与验收

### 6.1 公共指标

- TTFT、scheduled TTFT；
- TPOT；
- request E2E P50/P90/P95/P99；
- system output tok/s 与 active-window output tok/s；
- acceptance rate、mean accepted tokens per verify；
- HumanEval pass@1 / GSM8K exact match；
- 峰值显存、OOM/timeout/HTTP error 数；
- 可选的 Target GPU energy/token。

### 6.2 Scheduler 指标

- `used_tokens / (plans * token_budget)`；
- Verify/Prefill slice 数；
- partial Verify/Prefill 数；
- mixed Verify+Prefill plan 数；
- 实际 Verify/Prefill batch size；
- Prefill 两周期 overdue 触发次数；
- prefetch hit/miss/eviction。

若当前 JSON 尚未直接输出某项，应在正式实验前先补计数器和测试，不能从日志文本手工推断
整套正式结果。

### 6.3 正式开跑条件

以下条件全部满足才进入 164-request 正式运行：

- 单卡与四卡预实验均 exit code 0；
- 同步四卡日志确认真实多 session batch；
- 两服务器单测通过且 SHA 一致；
- 所有 HTTP 请求无 5xx、timeout 或 Future 悬挂；
- 预实验结束后 Cloud、Edge、隧道进程均能按明确 PID 释放；
- 正式配置、manifest hash、运行顺序已经冻结。

当前前五项已在 2026-08-13 预实验中通过；正式配置和 manifest 仍需在完整实验启动前冻结。

## 7. MT-Bench 两 A5000 完整实验（2026-08-13）

### 7.1 最终有效配置

本轮使用 MT-Bench 80 条第一轮问题，node1 两张 A5000 各运行一个 Qwen3-0.6B
Draft worker，node2 GPU0 的一张 A6000 运行 Qwen3-8B Target。最终有效代码为：

```text
03bd03d50a9a99ba17610606999790a0ca64f49e
agent/token-budget-continuous-batching
workload_hash=f100fe17c5b30626e80da573c307f0340d5005eb12c5fbd00d15f921f6a04455
```

固定参数：

```text
num_requests             = 80（每个 Draft worker 40）
max_new_tokens           = 256
temperature              = 0
gamma                    = 4
batch_size               = 2
token_budget             = 512
min_prefill_chunk_tokens = 16
prefill_chunk_quantum    = 128
arrival_distribution     = immediate（两 worker 闭环）
pipeline                 = off
proactive draft          = on
```

MT-Bench 使用 Qwen3 chat template、`add_generation_prompt=True` 和
`enable_thinking=False`。本实验只执行每条样本的第一轮问题；第二轮改写/自评问题保存在
reference 中但未生成回答。

### 7.2 正式运行前发现并修复的问题

完整 80 请求运行暴露了短 smoke 未覆盖的四个边界：

1. 非 pipeline 的 full-prefix verify 需要追加上一轮 correction token；此前 planner 未将其作为
   bridge forward token 计费，曾触发 logits 越界。修复提交 `a02c4ed`。
2. Cloud HTTP 父进程曾把 `torch.Tensor` 通过 `multiprocessing.Queue` 传给 Target worker，
   长运行后出现 `received 0 items of ancdata`。现在 IPC 只传 Python list/int，worker 内部再
   tensorize。修复提交 `2219c06`。
3. generic Edge 原来直接编码 MT-Bench raw question，未应用 Qwen3 chat template；同时统一
   normalizer 错把闭环 worker 的相对 completion time 当作全局 wallclock。修复提交
   `977a8bc`。
4. bridge token 已物理追加时，verify 又把逻辑 target logit 位置额外右移一位，导致错误接受
   Draft token并形成退化重复。修复后所有 draft token i 均由逻辑位置
   `prefix_len + i - 1` 的 logits 验证。修复提交 `03bd03d`。

所有失败目录和日志均保留，没有覆盖为成功结果。本地最终 50 个单元/静态测试通过，其中
1 个真实 Torch KV 测试因 Windows 本地缺 Torch 跳过；node1/node2 的 scheduler 与真实 KV
门禁均通过。

### 7.3 最终公共指标

结果：80/80 请求完成，两个 Draft worker 各 40 条；Edge/Cloud 错误均为 0，退化重复终止
为 0。共生成 16,643 tokens，80 条输出均非空，其中 46 条达到 256-token 上限。

| 指标 | 最终结果 |
|---|---:|
| 真实运行 wallclock | 1143.409 s |
| system output throughput | 14.5556 tok/s |
| TTFT avg / P50 / P90 / P95 / P99 | 940.92 / 905.16 / 1190.54 / 1273.78 / 1464.31 ms |
| decode TTFT avg | 458.19 ms |
| TPOT avg / P50 / P90 / P95 / P99 | 128.82 / 126.45 / 179.49 / 183.56 / 209.14 ms |
| request E2E avg / P50 / P90 / P95 / P99 | 28058.61 / 29237.39 / 43315.88 / 45877.29 / 47555.05 ms |
| weighted acceptance rate | 0.48848 |
| mean accepted tokens / verify | 2.13555 |
| 平均生成长度 | 208.04 tokens |
| 非空输出 | 80 / 80 |
| 达到 256-token 上限 | 46 / 80 |
| HTTP/worker errors | 0 |
| degenerate-repeat early stops | 0 |

这里不报告 MT-Bench judge 分数：尚未运行统一的 GPT-4/LLM judge。上述数据是系统性能、
协议稳定性和输出完整性证据，不等于 MT-Bench 质量胜率。

### 7.4 分类别指标

| 类别 | 请求数 | 平均 tokens | TTFT avg (ms) | TPOT avg (ms) | E2E avg (ms) | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| coding | 10 | 255.0 | 943.22 | 108.86 | 28577.85 | 0.6066 |
| extraction | 10 | 80.0 | 1190.49 | 96.39 | 8579.67 | 0.7246 |
| humanities | 10 | 256.0 | 926.88 | 159.66 | 41641.46 | 0.4060 |
| math | 10 | 225.2 | 888.03 | 82.04 | 19326.84 | 0.7872 |
| reasoning | 10 | 195.0 | 861.18 | 123.19 | 24912.81 | 0.5124 |
| roleplay | 10 | 208.3 | 833.68 | 171.66 | 35310.51 | 0.3248 |
| stem | 10 | 253.8 | 929.19 | 136.64 | 35426.01 | 0.4956 |
| writing | 10 | 191.0 | 954.69 | 152.12 | 30693.72 | 0.3700 |

### 7.5 Scheduler 指标与解释

```text
iterations       = 23456
plans            = 5511
used_tokens      = 34633
verify_slices    = 5663
prefill_slices   = 97
partial_verify   = 0
partial_prefill  = 17
prefetch_plans   = 5511
prefetch hit/miss/eviction = 0/0/0
```

平均每个 plan 使用 6.284 tokens；按 `plans * token_budget` 计算的预算利用率为 1.2274%。
本轮只有两个闭环会话，Verify 每轮通常只消费约 `gamma` 个 token，因此 `token_budget=512`
远高于可形成的实际工作量。这个结果说明 512 不是两 worker MT-Bench 的有效调优点；正式
消融应补 `token_budget=16/32/64/128`，或使用开放环更多并发请求后再讨论预算利用率。

### 7.6 最终证据路径

本地：

```text
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/fastsd_03bd03d/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/fastsd_03bd03d_cloud/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/normalized/fastsd/
```

服务器：

```text
node1: exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/fastsd_03bd03d/
node1: /tmp/new_fastsd_mtbench_2gpu_03bd03d_edge.log
node2: exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/fastsd_03bd03d_cloud/
node2: /tmp/new_fastsd_mtbench_2gpu_03bd03d_cloud.log
```

## 8. 结果目录约定

正式结果统一放在：

```text
exp/comparison/<run_id>/
  manifests/
  fastsd/
  specedge/
  standard_sd/
  draft_only/
  normalized/
  comparison.csv
  environment/
```

`environment/` 至少保存：

```text
node1_git.txt
node2_git.txt
node1_nvidia_smi.txt
node2_nvidia_smi.txt
commands.txt
config.json
workload_hash.txt
```

任何失败运行也应保留独立目录和错误日志，不覆盖成功运行。
