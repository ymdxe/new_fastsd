# FastSD / SpecEdge / 标准投机解码 / Draft-only 统一评测

本评测入口让四种方法读取同一份请求清单，并统一模型、生成长度、到达过程和指标定义：

| 方法名 | 实际执行路径 | 拓扑 |
|---|---|---|
| `fastsd` | FastSD 调度与协议 | 多个 A5000 draft + 网络 + 一个 A6000 target |
| `specedge` | 固定版本的官方 SpecEdge 核心 + 本仓库数据/指标适配器 | 多个 A5000 client + 网络 + 一个 A6000 server |
| `standard_sd` | FastSD 的 `vanilla` profile，关闭主动 draft、pipeline 和 FastSD 调度 | 与 FastSD 相同 |
| `draft_only` | 独立 draft worker 池，不访问 target | 与前三组相同数量的 A5000 |

官方源码仍固定在 `baselines/specedge/official/`，适配代码只放在
`baselines/specedge/integration/`，没有改写官方子模块的算法实现。

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
configs/evaluation/qwen3_8b_0.6b_humaneval.json
draft  = /home/hdd/zhangh/models/Qwen3-0.6B
target = /home/hdd/zhangh/models/Qwen3-8B
HumanEval, 164 requests, Poisson 1 RPS, seed 1234
max_new_tokens=256, temperature=0, gamma=4, stop_policy=eos
2 x A5000 draft, 1 x A6000 target
```

换数据集时只需复制 JSON 配置并修改 `dataset.name`、`data_path`、请求数和到达率。

## 运行前检查

两台服务器都应先同步到同一提交并初始化子模块。官方 SpecEdge 独立环境必须由其
`uv.lock` 创建，不能直接复用 FastSD 的 Python 3.10 环境：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd
git submodule update --init --recursive
python scripts/run_tests.py

cd baselines/specedge/official
uv sync
```

模型必须先通过架构、词表和 tokenizer 指纹检查；“能下载”不等于 draft/target 可互换
token id：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd
/home/hdd/zhangh/envs/fastsd/bin/python scripts/eval_suite.py validate-models \
  --config configs/evaluation/qwen3_8b_0.6b_humaneval.json
```

## 生成清单与完整命令

在 node1、node2 的相同仓库版本上各执行一次 `prepare`。输出 hash 必须相同：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd
/home/hdd/zhangh/envs/fastsd/bin/python scripts/eval_suite.py prepare \
  --config configs/evaluation/qwen3_8b_0.6b_humaneval.json
```

然后生成按窗口和服务器区分的可复制命令：

```bash
/home/hdd/zhangh/envs/fastsd/bin/python scripts/eval_suite.py plan \
  --config configs/evaluation/qwen3_8b_0.6b_humaneval.json
```

四种方法应顺序运行，避免互相抢 GPU。FastSD 与 standard SD 共用 8001 端口，但需要
分别以 `fastsd` 和 `vanilla` 调度模式启动 target。SpecEdge 使用 8000 端口。node1 到
node2 的隧道把它们映射为 18001 和 18000。

## 统一指标

每种方法都写 request-level JSONL，再归一化为相同 schema：

- `ttft_ms`：worker 实际开始处理请求到第一个可见输出 token；
- `scheduled_ttft_ms`：清单计划到达时刻到首 token，包含 worker 排队/到达滞后；
- `tpot_ms`：首 token 后的耗时除以剩余输出 token 数；
- `e2e_ms`、系统 output token/s、P50/P90/P95/P99；
- 投机方法的接受率或每次验证平均接受 token 数；
- GSM8K/MGSM 数值 exact match；
- HumanEval 会额外导出 `humaneval_samples.jsonl`，供 pass@1 工具执行。

归一化示例（在 node1）：

```bash
RUN=/home/hdd/zhangh/workspace/new_fastsd/exp/comparison/qwen3_8b_0.6b_humaneval
CFG=configs/evaluation/qwen3_8b_0.6b_humaneval.json

/home/hdd/zhangh/envs/fastsd/bin/python scripts/eval_suite.py normalize --config "$CFG" --method fastsd --input "$RUN/fastsd"
/home/hdd/zhangh/envs/fastsd/bin/python scripts/eval_suite.py normalize --config "$CFG" --method specedge --input "$RUN/specedge/raw/qwen3_8b_0.6b_humaneval"
/home/hdd/zhangh/envs/fastsd/bin/python scripts/eval_suite.py normalize --config "$CFG" --method standard_sd --input "$RUN/standard_sd"
/home/hdd/zhangh/envs/fastsd/bin/python scripts/eval_suite.py normalize --config "$CFG" --method draft_only --input "$RUN/draft_only"
```

生成最终 CSV：

```bash
/home/hdd/zhangh/envs/fastsd/bin/python scripts/eval_suite.py compare \
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

## 当前验证边界

Windows 本地已经验证数据适配、泊松清单、YAML 生成、日志归一化、公共指标和静态编译。
2026-08-13 还完成了真实跨服务器预实验：node1 的 1/4 张 A5000 运行 Qwen3-0.6B
Draft，node2 的一张 A6000 运行 Qwen3-8B Target；单卡和同步四卡 smoke 均通过，且 Cloud
日志确认形成了多 session Prefill batch。完整拓扑、命令、指标和正式实验矩阵见
`docs/plans/2026-08-13-qwen3-cross-server-full-experiment.md`。

这些短 smoke 只证明端到端可运行，不等同于四方法正式结果。SpecEdge 的正式运行仍要求
其独立 Python 3.14 环境通过门禁；论文级指标必须使用冻结的 canonical manifest、多 seed
重复和统一归一化流程。
