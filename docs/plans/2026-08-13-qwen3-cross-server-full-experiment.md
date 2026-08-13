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

## 7. 结果目录约定

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
