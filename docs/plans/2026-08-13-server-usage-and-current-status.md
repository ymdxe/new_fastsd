# new_fastsd 服务器使用与当前状态（2026-08-13）

本文是当前两台服务器的操作基准，所有路径和结果均以 `new_fastsd` 为准，不引用旧
`fastsd` 工作空间。

## 1. 已核对的服务器状态

| 项目 | node1 | node2 |
|---|---|---|
| 工作空间 | `/home/hdd/zhangh/workspace/new_fastsd` | `/home/hdd/zhangh/workspace/new_fastsd` |
| Git | `b2ed81792871a58e23aa4aed6ae7ef2eae28bb29` | 同左 |
| 官方 SpecEdge 子模块 | `1edcaf02ffc41a7b57726450c5357ed216a3b9bc` | 同左 |
| GPU 用途 | A5000：GPU 0/1 各运行一个 0.6B Draft/SpecEdge client | A6000：固定 GPU 0 运行 8B Target/SpecEdge server |
| Python 环境 | `envs/new_fastsd`；SpecEdge 使用 `envs/specedge` | 同左 |
| 当前显存 | A5000 GPU 0–3 各约 15 MiB | A6000 GPU 0/1 各约 18 MiB；GPU 2 有其他进程约 2.8 GiB |
| 监听端口 | 当前无 8001/18000/18001 | 8000 已被 root Docker 占用；当前无 8001/18000 |

两台机器工作树都只有此前保留的未跟踪实验日志，没有已跟踪源码修改。不要删除这些日志，
它们是失败/预实验的审计证据。

## 2. 角色和端口规则

### FastSD / standard_sd

- node2 只运行 Cloud Target，固定 `CUDA_VISIBLE_DEVICES=0`，监听 `127.0.0.1:8001`。
- node1 运行一个或多个 A5000 Draft worker；通过 SSH 隧道访问 Cloud，不能把未认证的
  8001 暴露到公网。
- `fastsd` 与 `standard_sd` 必须分开运行；后者用 `profile=vanilla`、关闭 proactive
  draft 和 pipeline。两组之间释放进程、确认 GPU 空闲后再开始下一组。

### 官方 SpecEdge

- node2 GPU 0 启动官方 `SpecExecBatchServer`，只绑定 `127.0.0.1:18000`。不要停止
  node2 上已有的 root Docker `0.0.0.0:8000`。
- node1 GPU 0/1 各运行一个 client。Windows 端建立两段仅回环 SSH 转发：
  `-L 18000:node2:18000` 和 `-R 18000:node1:18000`；实验结束按明确 PID 关闭。
- SpecEdge 必须使用 `/home/hdd/zhangh/envs/specedge/bin/python` 和官方 `uv.lock` 环境，
  不能把官方子模块算法改写进 FastSD。

### Draft-only

- 仅使用 node1 GPU 0/1 的 0.6B worker，不访问 node2，不代表 8B Target 的质量。

## 3. 每次实验的门禁

在两台服务器分别执行：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
/home/hdd/zhangh/envs/new_fastsd/bin/python -m unittest discover -s tests -v
```

只有当两端 SHA、子模块 SHA、模型路径、tokenizer 指纹和 workload hash 一致，且目标 GPU
无其他实验进程时，才可开始正式组。方法必须串行运行，避免 GPU 争用。

## 4. 当前已完成工作记录

在同一份 MT-Bench 80 条第一轮请求清单（`workload_hash=f100fe17c5b30626e80da573c307f0340d5005eb12c5fbd00d15f921f6a04455`）上，已完成：

- FastSD：两张 A5000 Draft + 一张 A6000 Target；
- standard_sd：普通线性投机解码，同一拓扑；
- draft-only：两张 A5000 自回归 Draft；
- 官方 SpecEdge：两张 A5000 client + 一张 A6000 server。

结果和原始证据统一保存在：

```text
exp/comparison/qwen3_8b_0.6b_mt_bench_2gpu_seed42/
```

四方法汇总在 `normalized/comparison_all_four.csv`，详细指标、修复记录和可复制命令见
`docs/plans/2026-08-13-qwen3-cross-server-full-experiment.md` 第 7–8 节。

当前结果是单 seed、单轮 MT-Bench 性能结果；尚未完成统一 LLM judge、monolithic greedy
oracle 的逐 token parity、多 seed 重复和更高并发。因此不能据此宣称回答质量或 target
正确性已经通过。

## 5. 下一次运行顺序

1. 按第 3 节门禁记录两端状态和新实验 SHA。
2. 复制配置并运行 `prepare`，确认两端 `workload_hash` 相同。
3. 按 FastSD → standard_sd → SpecEdge → draft-only（或预先注册的实验顺序）逐组运行，
   每组保存 raw、normalized 和运行日志。
4. 释放服务/隧道，复查 GPU 显存和端口，最后再生成汇总 CSV。

