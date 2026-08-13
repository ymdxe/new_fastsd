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

## 8. SpecEdge、普通投机解码与 Draft-only 对照（2026-08-13）

### 8.1 公平性约束与有效版本

三个对照继续使用第 7 节完全相同的 canonical manifest：

```text
dataset       = MT-Bench 80 条第一轮问题
workload_hash = f100fe17c5b30626e80da573c307f0340d5005eb12c5fbd00d15f921f6a04455
max_new_tokens = 256
temperature = 0, top_k = 0, top_p = 1, seed = 42
```

普通投机解码使用 node1 GPU0/1 上两个 Qwen3-0.6B Draft worker，以及 node2 GPU0
上的 Qwen3-8B Target；`profile=vanilla`、`server_sched_mode=vanilla`、pipeline 和
proactive draft 均关闭，`gamma=4`。最终有效实现提交为 `23db00d95dfb`。

Draft-only 只使用 node1 GPU0/1，每卡一个 Qwen3-0.6B 自回归 worker；它不访问 node2，
也不执行 Target verify。最终有效实现提交为 `c234a605ac26`。两个入口均使用 Qwen3 chat
template、`add_generation_prompt=True` 和 `enable_thinking=False`。

SpecEdge 使用固定官方子模块 `1edcaf02ffc41a7b57726450c5357ed216a3b9bc` 的核心
`SpecExecBatchServer`、`SpecExecClient`、tree drafting 和 proactive drafting；本仓库
`baselines/specedge/integration/` 只负责 canonical dataset、Qwen3 chat template、精确请求指标、
BF16 位保真 gRPC 序列化和可配置回环端口。最终有效适配提交为 `ebf85d210eb3`。

### 8.2 对照运行暴露并修复的问题

1. Draft-only 原入口直接编码 MT-Bench 裸问题，与 FastSD 的 Qwen3 chat template 不一致；
   提交 `98117e1` 统一了 prompt 格式。
2. tensor-free Cloud IPC 已在 FastSD ingress 恢复 tensor，但 strict-FCFS vanilla 分支在恢复
   之前直接调用 `.to()`，首个 `/prefill` 因 list 无 `.to()` 崩溃。提交 `23db00d` 将
   tensorize 提升为两个调度分支共享的 ingress 步骤。
3. Draft-only immediate 模式曾把 closed-loop worker 本地排队时间记入 `arrival_lag`，而 Edge
   immediate 模式固定为 0，导致不可比的 scheduled TTFT。提交 `c234a60` 对齐了辅助指标。
4. wrapper 初次由系统 Python 启动，缺少 `auto_gptq`；正式运行显式使用
   `/home/hdd/zhangh/envs/new_fastsd/bin/python`。所有失败日志均保留，未计入结果。
5. 两台服务器最初均无官方 `.venv`，node2 已有 `/home/hdd/zhangh/envs/specedge`；node1
   使用官方 `uv.lock` 和 `uv 0.12.3` 安装相同 Python 3.14.7、Torch 2.9.0+cu128 环境。
6. node2 的公网 8000 已由 root Docker proxy 使用，不能终止；提交 `203a6b0` 增加只绑定
   `127.0.0.1:18000` 的 server launcher，仍直接实例化官方 `SpecExecBatchServer`。
7. 官方 `util.encode()` 经 NumPy 序列化 BF16 会报 `unsupported ScalarType BFloat16`；
   提交 `ebf85d2` 在适配层把 BF16 view 为 `uint16` 后发送，服务端仍用官方
   `torch.frombuffer(..., dtype=bfloat16)` 还原，传输位模式不变。

最终本地回归为 52 项：51 通过，1 个真实 Torch KV 测试因 Windows 本地无 Torch 跳过。

### 8.3 四方法公共指标

| 方法 | 请求 | 生成 tokens | wallclock (s) | throughput (tok/s) | TTFT avg / P95 / P99 (ms) | TPOT avg / P95 / P99 (ms) | E2E avg / P95 / P99 (ms) | acceptance | accepted / verify |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FastSD | 80 | 16,643 | 1143.409 | 14.5556 | 940.92 / 1273.78 / 1464.31 | 128.82 / 183.56 / 209.14 | 28058.61 / 45877.29 / 47555.05 | 0.48848 | 2.13555 |
| SpecEdge | 80 | 16,805 | 605.461 | 27.7557 | 533.59 / 642.10 / 1230.80 | 67.49 / 83.56 / 92.04 | 14762.80 / 20487.51 / 21880.43 | N/A | 3.72163 |
| standard_sd | 80 | 16,699 | 1532.675 | 10.8953 | 1255.34 / 1983.03 / 2125.40 | 176.47 / 288.79 / 308.96 | 37855.40 / 58732.78 / 66527.31 | 0.49503 | 2.17240 |
| draft_only | 80 | 14,410 | 162.370 | 88.7480 | 39.40 / 32.76 / 598.65 | 21.84 / 25.01 / 27.56 | 3976.98 / 6365.65 / 7052.08 | N/A | N/A |

Draft-only 的平均 TTFT 高于 P95，是因为两个 worker 各自首次模型执行形成少数约 0.6 秒
冷启动离群点；P50/P90/P95 分别为 24.25/26.74/32.76 ms。所有方法都报告真实整段
wallclock throughput，不使用各 worker active-window 吞吐替代。

相对 standard_sd，FastSD 的系统吞吐提高 33.59%，wallclock 降低 25.40%，TTFT 平均值
降低 25.05%、P95 降低 35.77%，TPOT 平均值降低 27.00%、P95 降低 36.44%，请求 E2E
平均值降低 25.88%。这组结果只说明当前两 closed-loop session、两 A5000 Draft 的系统性能；
`token_budget=512` 仍然严重过配，不能据此外推高并发收益。

相对 FastSD，SpecEdge 系统吞吐提高 90.69%，TTFT 平均值降低 43.29%，TPOT 平均值
降低 47.61%，请求 E2E 平均值降低 47.39%；相对 standard_sd 吞吐提高 154.75%。SpecEdge
使用 `max_budget=32` 的树形候选和 proactive drafting，FastSD/standard_sd 是线性 `gamma=4`，
这是各方法算法配置而非相同 draft-token 工作量。SpecEdge cycle 日志包含 4,525 个非 Prefill
verify cycle，按 cycle 加权平均接受 3.657 tokens；表中 3.72163 是先逐请求求均值再对 80 请求
等权平均。树中验证的分支 token 没有与线性 draft 数相同定义的分母，因此不报告伪造的
acceptance rate。

Draft-only 的速度不能解释为与 8B Target 方法同质量：它生成的是 0.6B 模型输出，生成 token
总数和停止位置也不同。本轮未运行统一 MT-Bench LLM judge，因此不报告质量胜率。

### 8.4 完整性与正确性边界

- standard_sd：80/80 非空，两个 worker 各 40 条，47 条达到 256-token 上限，Edge/Cloud
  错误为 0；
- draft_only：80/80 非空，两个 worker 各 40 条，35 条达到 256-token 上限，worker 错误为 0；
- SpecEdge：80/80 非空，两个 client 各 40 条，50 条达到 256-token 上限，4,605 个 cycle，
  client/server 错误均为 0；
- 三者 workload hash 均与 FastSD 完全一致；
- 运行结束后 node1 GPU0/1 和 node2 GPU0 均回到约 15 MiB，8001 服务和两段 SSH 隧道已释放。

必须保留一个尚未关闭的正确性风险：逐样本比较 FastSD 与 standard_sd，只有 26/80 输出文本
逐字一致，60/80 的生成 token 数相同。动态 batch 与单请求 GEMM 可能因 BF16 数值路径不同在
接近的 logits 上选择不同 token，但现有证据也不能排除 FastSD KV/bridge 状态语义仍有问题。
所以当前数据可作为性能和稳定性结果，不能作为 target-output parity 已通过的证据。正式论文实验
前必须补同一 Target 的 monolithic greedy oracle，并比较 token-level 首个分叉位置和最终 KV/logits。

SpecEdge 与 standard_sd、FastSD 都只有 14/80 输出文本逐字一致，生成长度分别有 59/80
一致。因此 SpecEdge 性能数据同样不能作为 target-output parity 已通过的证据。当前四方法均未
运行统一 MT-Bench LLM judge，不能从系统时延推导回答质量。

### 8.5 MT-Bench 分类别指标

standard_sd：

| 类别 | 请求 | 平均 tokens | TTFT avg (ms) | TPOT avg (ms) | E2E avg (ms) | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| coding | 10 | 254.9 | 1226.88 | 157.10 | 41061.59 | 0.6300 |
| extraction | 10 | 80.3 | 1471.04 | 152.16 | 13488.12 | 0.7400 |
| humanities | 10 | 256.0 | 1107.24 | 167.16 | 43732.77 | 0.4200 |
| math | 10 | 225.1 | 1026.52 | 116.18 | 26518.18 | 0.7900 |
| reasoning | 10 | 194.6 | 1254.72 | 173.26 | 34767.96 | 0.5200 |
| roleplay | 10 | 208.9 | 1439.70 | 259.78 | 53862.75 | 0.3200 |
| stem | 10 | 255.1 | 1163.74 | 176.96 | 46106.57 | 0.5000 |
| writing | 10 | 195.0 | 1352.89 | 209.14 | 43305.31 | 0.3700 |

draft_only：

| 类别 | 请求 | 平均 tokens | TTFT avg (ms) | TPOT avg (ms) | E2E avg (ms) |
|---|---:|---:|---:|---:|---:|
| coding | 10 | 251.5 | 23.83 | 21.44 | 5396.38 |
| extraction | 10 | 79.2 | 25.38 | 21.55 | 1709.35 |
| humanities | 10 | 244.0 | 23.86 | 21.40 | 5222.84 |
| math | 10 | 216.3 | 25.72 | 22.95 | 5016.21 |
| reasoning | 10 | 109.4 | 24.18 | 21.30 | 2337.78 |
| roleplay | 10 | 146.5 | 24.55 | 21.38 | 3136.74 |
| stem | 10 | 246.2 | 25.52 | 22.90 | 5639.38 |
| writing | 10 | 147.9 | 142.13 | 21.83 | 3357.13 |

SpecEdge：

| 类别 | 请求 | 平均 tokens | TTFT avg (ms) | TPOT avg (ms) | E2E avg (ms) |
|---|---:|---:|---:|---:|---:|
| coding | 10 | 256.0 | 498.80 | 62.40 | 16410.95 |
| extraction | 10 | 81.2 | 558.87 | 61.72 | 5781.59 |
| humanities | 10 | 256.0 | 480.95 | 70.69 | 18507.41 |
| math | 10 | 225.4 | 479.83 | 59.74 | 13864.47 |
| reasoning | 10 | 194.5 | 497.62 | 64.29 | 13170.25 |
| roleplay | 10 | 212.2 | 521.43 | 78.65 | 16828.28 |
| stem | 10 | 256.0 | 532.10 | 67.66 | 17784.44 |
| writing | 10 | 199.2 | 699.12 | 74.74 | 15754.98 |

### 8.6 证据路径

```text
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/standard_sd_23db00d/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/standard_sd_23db00d_cloud/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/draft_only/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/normalized/standard_sd/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/normalized/draft_only/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/specedge/raw/qwen3_8b_0.6b_mt_bench_2gpu_seed42/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/specedge/requests/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/normalized/specedge/
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/normalized/comparison_all_four.csv
```

### 8.7 SpecEdge 可复现运行命令

首次在每台服务器准备环境：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd/baselines/specedge/official
UV_PROJECT_ENVIRONMENT=/home/hdd/zhangh/envs/specedge \
UV_CACHE_DIR=/home/hdd/zhangh/cache/uv \
/home/hdd/zhangh/tools/uv/bin/uv sync --python 3.14 --frozen
```

node2 启动 Target server：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd
FASTSD_EVAL_ROLE=server \
FASTSD_EVAL_DATASET_FILE=exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/inputs/canonical.jsonl \
PYTHONPATH=baselines/specedge/integration:baselines/specedge/official/src \
/home/hdd/zhangh/envs/specedge/bin/python -O baselines/specedge/integration/server.py \
  --config exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/specedge/specedge.yaml \
  --host 127.0.0.1 --port 18000
```

Windows 上用两个终端建立仅回环隧道：

```powershell
ssh -N -L 18000:127.0.0.1:18000 node2
ssh -N -R 18000:127.0.0.1:18000 node1
```

node1 启动两个 A5000 client：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd
/home/hdd/zhangh/envs/specedge/bin/python \
  baselines/specedge/integration/client_host.py \
  --config exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/specedge/specedge.yaml
```

归一化：

```bash
/home/hdd/zhangh/envs/new_fastsd/bin/python scripts/eval_suite.py normalize \
  --config configs/evaluation/qwen3_8b_0.6b_mt_bench_2gpu.json \
  --method specedge \
  --input exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/specedge/raw/qwen3_8b_0.6b_mt_bench_2gpu_seed42
```

## 9. 结果目录约定

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
