# FastSD / SpecEdge CPU adapter / 标准投机解码 / Draft-only 统一评测

本评测入口让四种方法读取同一份请求清单，并统一模型、生成长度、到达过程和指标定义：

| 方法名 | 实际执行路径 | 拓扑 |
|---|---|---|
| `fastsd` | FastSD 调度与 stateful EdgeClient 协议 | node3 单进程 CPU32 draft + 可配置网络 + node2 单张逻辑 `cuda:0` target |
| `specedge_cpu_adapted` | 固定版本的官方 Tree/SpecExec/proactive 核心 + 本仓库显式 CPU engine/wire adapter | node3 CPU32 draft + 可配置网络 + node2 单张逻辑 `cuda:0` server |
| `standard_sd` | FastSD 的 `vanilla` profile，关闭主动 draft、pipeline 和 FastSD 调度 | 与 FastSD 相同的 CPU32/target 拓扑 |
| `draft_only` | 独立 CPU draft worker，不访问 target | 与前三组相同的 node3 CPU32 资源口径 |

官方源码仍固定在 `baselines/specedge/official/`，适配代码只放在
`baselines/specedge/integration/`，没有改写官方子模块的算法实现。CPU 运行必须标为
`specedge_cpu_adapted`，不能写成 untouched official SpecEdge。

## 本次 CPU draft / GPU target 实验边界

- FastSD、standard SD、`specedge_cpu_adapted` 的 draft 都使用 CPU fp32；node2 target 使用
  bf16。FastSD 保持 `EdgeClient` 的 `/session/init`、`/prefill`、`/verify` 和 rollback/KV
  生命周期，不使用一次性 `/generate` worker。
- latency track 是单个 stateful 进程、32 个 PyTorch CPU threads；throughput track 才配置
  4 个进程、每个 8 threads。`scripts/prepare_32worker_experiment.py` 生成的是历史一次性
  worker/多模型副本方案，现已阻止作为本实验入口。
- node3 interpreter 由 `eval_suite.py plan --python <absolute-interpreter>` 显式传入并原样
  写入 `commands.txt`；target URL/端口和两端 bind host 也由配置提供，默认只绑定
  `127.0.0.1`。受限网络时，在 node3 使用本地端口转发，把 `127.0.0.1:18001` 转发到
  node2 的 `127.0.0.1:18001`；不要假设 node3 到 node2 的应用端口直连可用。网络路径
  开销必须计入 TTFT/E2E。
- shared-load CPU 口径固定为物理 CPU `56-71,80-95`（NUMA node2/node3 各 16 核），
  统一前缀为 `nice -n 5 numactl --physcpubind=56-71,80-95 --interleave=2,3`，并设置
  OMP/MKL/OpenBLAS/Torch threads=32。`cpu_preflight.json` 与 `cpu_postflight.json` 保存
  affinity、loadavg 和 `mpstat` 快照；manifest 记录 `shared_load`、cpuset、method order
  与 `repeat_index`。不停止或重绑其他用户进程。
- `target_only` 是 greedy oracle，只用于 token parity/质量参照，不计入四方法性能表。
- `max_new_tokens=16` 的结果只称为 `communication_smoke`；HumanEval、MGSM、GSM8K 和
  MT-Bench 的质量结论应使用更长的生成预算。当前 MT-Bench canonical 路径只取第一轮，
  因此必须标为 `first_turn_only`，不能宣称完整 MT-Bench judge。

## 支持的数据集

`src/evaluation.py` 当前支持仓库中的：

- HumanEval：`task_id`、`prompt`、测试与参考实现
- GSM8K：`question`、`answer`
- MGSM：`question_id`、`question`、`answer`
- MT-Bench：`question_id`、`category`、`turns`

`prepare` 会生成 `canonical.jsonl`。每条记录包含全局序号、统一 prompt、reference 和
`scheduled_arrival_s`。四种方法都按全局序号 round-robin 分配给相同数量的边缘 worker；
泊松到达时间只生成一次，不由各方法各自重新采样。`workload_hash` 不一致时，汇总脚本
拒绝比较。

默认实验配置是：

```text
configs/evaluation/qwen3_8b_1.7b_four_method_cpu.json
draft  = /path/to/models/Qwen3-1.7B
target = /path/to/models/Qwen3-8B
HumanEval, 164 requests, immediate arrival, seed 42
max_new_tokens=256, temperature=0, gamma=4, stop_policy=eos
1 x 32-thread CPU draft, 1 x logical cuda:0 target
```

四个正式数据集配置由同一模板生成，保证 generation/topology/seed 一致，且 run ID 唯一：

```bash
python scripts/make_evaluation_matrix.py \
  --base-config configs/evaluation/qwen3_8b_1.7b_four_method_cpu.json \
  --data-root data \
  --output-dir configs/evaluation/matrix/qwen3_8b_1.7b_four_method_cpu
```

入口生成 HumanEval-164、MGSM-110、GSM8K-1319、MT-Bench-80 四个 config 和
`matrix_manifest.json`；MT-Bench 明确写入 `first_turn_only`。不要手工修改四份 JSON。

## 运行前检查

两台服务器都应先同步到同一提交并初始化子模块。官方 SpecEdge 独立环境必须由其
`uv.lock` 创建，不能直接复用 FastSD 的 Python 3.10 环境：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd
git submodule update --init --recursive
python scripts/run_tests.py

cd baselines/specedge/official
UV_PROJECT_ENVIRONMENT=/home/hdd/zhangh/envs/specedge \
UV_CACHE_DIR=/home/hdd/zhangh/cache/uv \
/home/hdd/zhangh/tools/uv/bin/uv sync --python 3.14 --frozen
```

模型必须先通过架构、词表和 tokenizer 指纹检查；“能下载”不等于 draft/target 可互换
token id：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd
/home/hdd/zhangh/envs/new_fastsd/bin/python scripts/eval_suite.py validate-models \
  --config configs/evaluation/qwen3_8b_0.6b_humaneval.json
```

## 生成清单与完整命令

在 node3、node2 的相同仓库版本上各执行一次 `prepare`。输出 hash 必须相同：

```bash
cd <node3-repo>
<node3-edge-python> scripts/eval_suite.py prepare \
  --config <resolved-config.json>

cd <node2-repo>
<node2-target-python> scripts/eval_suite.py prepare \
  --config <resolved-config.json>

<node3-edge-python> scripts/eval_suite.py workload-hash \
  --node3-manifest <node3-repo>/exp/comparison/<run_id>/run_manifest.json \
  --node2-manifest <node2-repo>/exp/comparison/<run_id>/run_manifest.json
```

配置中的 `models.node3_draft` 与 `models.node2_draft` 必须分别指向两台主机的
Qwen3-1.7B 副本；Cloud target 命令使用 node2 副本，SpecEdge client/server YAML
分别使用 node3/node2 值。缺少 host-specific 值时才回退到 legacy `models.draft`。

然后生成按窗口和服务器区分的可复制命令。配置中的
`node3_repo/node2_repo` 以及四个 host-specific interpreter 键会被解析后原样写入
manifest、plan 和 `commands.txt`；示例中的 `python` 可替换为 node3 的显式 Python 3.14
interpreter。若使用 plan CLI 覆盖解释器或 repo，必须先用同样的 resolved config
重新 `prepare`；plan 会拒绝与既有 manifest 不一致的 override：

```bash
python scripts/eval_suite.py plan \
  --config configs/evaluation/qwen3_8b_1.7b_four_method_cpu.json \
  --python /absolute/path/to/python314_glibc \
  --target-host 127.0.0.1 --target-port 18001
```

若 node3 到 node2 的应用端口被策略丢弃，在执行上述计划前由现场网络方案建立双段
SSH 本地转发（占位示意，不包含凭据或固定主机名）：

```bash
ssh -J <jump-host> -N -L 18001:127.0.0.1:18001 <node2-hop>
```

SpecEdge 端口同样按现场拓扑决定是否转发到 node3 的 `127.0.0.1:18000`。计划中的
`target_host/specedge_host` 可以覆盖，不能写死不可达的 `node2`。

`prepare` 和 `plan` 会把带 UTC 时间戳的命令块追加保存到
`exp/comparison/<run_id>/commands.txt`，并记录 `run_status.jsonl`。FastSD 和 standard SD 的 Edge 启动脚本还会在
实际启动前，把最终展开后的 `python edge/edge.py ...` 命令追加到各自实验目录的
`commands.txt`，因此该文件同时保留计划命令和实际执行命令。

正式现场还应把资源 sidecar 的原始输出/路径写入同一个 `commands.txt`：`mpstat`、
`pidstat`、`nvidia-smi` 以及 run 前后 load 快照。仓库不新增监控脚本；这些命令由正式
运行记录执行并保留原始日志，且不假设固定 CPU 集合被独占。

GPU sidecar 必须记录实际 CSV 列布局。node2/node3 当前的无 GPU name 六列格式使用
`compact-no-name6`，不要依靠数值范围猜测列含义：

```bash
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader,nounits > "$RUN/<method>/nvidia-smi.csv"
python scripts/eval_suite.py resources \
  --method-dir "$RUN/<method>" \
  --gpu-nvidia-csv "$RUN/<method>/nvidia-smi.csv" \
  --gpu-index 1 \
  --gpu-csv-schema compact-no-name6
```

`full8`（含 `name`、GPU 与显存利用率、已用/空闲显存和功耗）可以省略 schema；
两种六列格式都必须显式传入 `--gpu-csv-schema compact-name6` 或
`--gpu-csv-schema compact-no-name6`。生成的 `resource_metrics.json` 会保留 schema
名称和列定义，便于审计原始 sidecar 与解析结果是否匹配。

四种方法应顺序运行，避免互相抢 GPU。FastSD 与 standard SD 共用 8001 端口，但需要
分别以 `fastsd` 和 `vanilla` 调度模式启动 target。若 node2 的 8000 已被占用，SpecEdge
使用 `baselines/specedge/integration/server.py --host 127.0.0.1 --port 18000`；它只替换
绑定端口，仍直接运行官方 `SpecExecBatchServer`。本次 MT-Bench 的完整命令见实验记录
第 8.7 节。

## 统一指标

每种方法都写 request-level JSONL，再归一化为相同 schema：

- `ttft_ms`：worker 实际开始处理请求到第一个可见输出 token；
- `scheduled_ttft_ms`：清单计划到达时刻到首 token，包含 worker 排队/到达滞后；
- `tpot_ms`：首 token 后的耗时除以剩余输出 token 数；
- `e2e_ms`、系统 output token/s、P50/P90/P95/P99；
- `request_bytes`、`response_bytes`、`rpc_count`：FastSD 记录实际 HTTP JSON
  应用层序列化字节，SpecEdge 记录 protobuf `ByteSize`；两者均不含协议封装、TCP/IP
  或 SSH；
- 投机方法的接受率或每次验证平均接受 token 数；
- GSM8K/MGSM 数值 exact match；
- HumanEval 会额外导出 `humaneval_samples.jsonl`，供 pass@1 工具执行。

归一化示例（在 node3）：

```bash
RUN=/home/hdd/zhangh/workspace/new_fastsd/exp/comparison/qwen3_8b_0.6b_humaneval
CFG=configs/evaluation/qwen3_8b_0.6b_humaneval.json

/home/hdd/zhangh/envs/new_fastsd/bin/python scripts/eval_suite.py normalize --config "$CFG" --method fastsd --input "$RUN/fastsd"
python scripts/eval_suite.py normalize --config "$CFG" --method specedge_cpu_adapted --input "$RUN/specedge/raw/qwen3_8b_1.7b_four_method_cpu_humaneval"
/home/hdd/zhangh/envs/new_fastsd/bin/python scripts/eval_suite.py normalize --config "$CFG" --method standard_sd --input "$RUN/standard_sd"
/home/hdd/zhangh/envs/new_fastsd/bin/python scripts/eval_suite.py normalize --config "$CFG" --method draft_only --input "$RUN/draft_only"
```

生成最终 CSV：

```bash
/home/hdd/zhangh/envs/new_fastsd/bin/python scripts/eval_suite.py compare \
  "$RUN/normalized/fastsd/summary.json" \
  "$RUN/normalized/specedge/summary.json" \
  "$RUN/normalized/standard_sd/summary.json" \
  "$RUN/normalized/draft_only/summary.json" \
  --output "$RUN/comparison.csv"
```

HumanEval pass@1 会执行模型生成的 Python 代码，应只在隔离实验环境中运行：

```bash
evaluate_functional_correctness \
  "$RUN/normalized/specedge/humaneval_samples.jsonl" \
  --problem_file=data/humaneval.jsonl
```

直接 FastSD/SpecEdge 配对分析使用同一 `sample_id`，固定 seed=42、10,000 次 bootstrap，
并输出 TTFT/E2E/TPOT/吞吐的提升百分比及 95% CI：

```bash
python scripts/eval_suite.py paired --config "$CFG" \
  --left fastsd="$RUN/normalized/fastsd/requests.jsonl" \
  --right specedge_cpu_adapted="$RUN/normalized/specedge_cpu_adapted/requests.jsonl"
```

## 当前验证边界

2026-08-13 的历史记录曾在同一 MT-Bench 80 请求 manifest 上完成旧 GPU 拓扑的四方法运行，
不能作为本次 CPU56-71,80-95 shared-load 结果。当前 target 只使用逻辑 `cuda:0`；物理
GPU 选择由外部 `CUDA_VISIBLE_DEVICES` 决定，不在配置中硬编码物理编号。完整拓扑、命令、
指标和证据边界见
`docs/plans/2026-08-13-qwen3-cross-server-full-experiment.md`。

历史目录中的 16-token 四方法结果只证明通信路径覆盖，不是完整质量评测；旧结果中的
`specedge` GPU client 也不能作为本次 `specedge_cpu_adapted` 结论。新运行应在 target-only
oracle 完成后执行：

```bash
python scripts/eval_suite.py parity --config "$CFG" \
  --input fastsd="$RUN/normalized/fastsd/requests.jsonl" \
  --input specedge_cpu_adapted="$RUN/normalized/specedge_cpu_adapted/requests.jsonl" \
  --input standard_sd="$RUN/normalized/standard_sd/requests.jsonl" \
  --input draft_only="$RUN/normalized/draft_only/requests.jsonl" \
  --input target_only="$RUN/target_only/requests.jsonl"
```

正式 parity 的 `fastsd`、`specedge_cpu_adapted`、`standard_sd` 相对
`target_only` 是 exact gate：任一 token 或样本缺失/不一致都会以非零状态结束；
`draft_only` 只出现在报告中，不影响该 gate。communication smoke 默认只比较前 20 个
reference samples，也可用 `--max-samples` 显式限制。
