# Deprecated: legacy one-shot worker experiment

本文档保留为历史记录。它描述的一次性 `/generate` worker、多模型副本和手工 SpecEdge
KV patch 不符合当前 stateful 四方法实验；`scripts/prepare_32worker_experiment.py` 已
主动拒绝执行。请改用 `scripts/eval_suite.py prepare/plan`，主 latency track 为单进程
32 threads，throughput track 才使用 4x8，并使用仓库内显式 CPU/线缆 adapter。

# 32-Worker CPU+GPU 实验指南

本文档说明如何在 node2（GPU target）+ node3（32 CPU draft workers）上运行 FastSD 和 SpecEdge 对比实验。

## 准备的代码文件

### 1. `scripts/prepare_32worker_experiment.py`
**功能：** 创建实验目录结构和配置文件

**用途：**
- 生成隔离的实验目录
- 记录 Git SHA、模型指纹、拓扑信息
- 生成 FastSD 和 SpecEdge 的执行命令
- 创建 SpecEdge KV offload 补丁参考代码

**使用示例：**
```bash
python scripts/prepare_32worker_experiment.py \
  --run_id qwen3_8b_1.7b_32cpu_workers_20260818 \
  --exp_root /home/hdd/zhangh/workspace/new_fastsd/exp \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --target_model /home/hdd/zhangh/models/Qwen3-8B \
  --num_workers 32 \
  --dataset humaneval \
  --node2_gpu cuda:0 \
  --node3_cores "0-31" \
  --gamma 4 \
  --max_tokens 256 \
  --temperature 0.0 \
  --arrival_rate 1.0 \
  --arrival_seed 1234
```

**输出：**
- `exp/<run_id>/experiment_manifest.json` - 完整元数据
- `exp/<run_id>/fastsd/commands.txt` - FastSD 执行命令
- `exp/<run_id>/specedge/commands.txt` - SpecEdge 执行命令
- `exp/<run_id>/specedge/specedge_kv_offload_patch.py` - KV offload 补丁

---

### 2. `scripts/launch_cpu_draft_workers.py`
**功能：** 在 node3 上启动 32 个 CPU-bound draft workers

**特性：**
- 每个 worker 绑定到一个物理 CPU 核心（使用 `taskset -c`）
- 每个 worker 加载独立的 Qwen3-1.7B 模型副本
- 设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 避免线程过量
- 禁用 GPU（`CUDA_VISIBLE_DEVICES=""`）
- 每个 worker 监听唯一端口（base_port + worker_id）

**使用示例（32 workers）：**
```bash
# SSH to node3
ssh node3
cd /home/hdd/zhangh/workspace/new_fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate

python scripts/launch_cpu_draft_workers.py \
  --num_workers 32 \
  --start_core 0 \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --base_port 19000 \
  --log_dir exp/<run_id>/fastsd/draft_logs \
  --worker_script scripts/cpu_draft_worker.py \
  --max_tokens 256 \
  --temperature 0.0 \
  --manifest_out exp/<run_id>/fastsd/draft_workers_manifest.json
```

**使用示例（2 workers，冒烟测试）：**
```bash
python scripts/launch_cpu_draft_workers.py \
  --num_workers 2 \
  --start_core 0 \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --base_port 19000 \
  --log_dir exp/<run_id>/smoke_test/draft_logs \
  --worker_script scripts/cpu_draft_worker.py \
  --manifest_out exp/<run_id>/smoke_test/draft_workers_manifest.json
```

**输出：**
- 32 个后台进程，PID 记录在 manifest JSON
- 每个 worker 的 stdout/stderr 分别保存到 `draft_logs/draft_worker_N.{out,err}`
- Workers 监听端口 19000-19031

**终止 workers：**
- Ctrl+C（脚本会捕获信号并终止所有子进程）
- 或手动：`pkill -f cpu_draft_worker.py`

---

### 3. `scripts/cpu_draft_worker.py`
**功能：** 单个 CPU draft worker 的 HTTP 服务

**特性：**
- 使用 FastAPI 提供 `/health` 和 `/generate` 端点
- 加载一个 Qwen3-1.7B 模型实例到 CPU
- 接收请求并生成 draft tokens
- 返回 draft 序列、生成时间和 worker ID

**API 端点：**

**GET /health**
```json
{
  "status": "healthy",
  "worker_id": 0,
  "generation_count": 42,
  "device": "cpu"
}
```

**POST /generate**

Request:
```json
{
  "prompt": "Write a Python function to...",
  "max_new_tokens": 10,
  "temperature": 0.0,
  "top_k": 0,
  "top_p": 1.0,
  "request_id": "req_123"
}
```

Response:
```json
{
  "draft_tokens": [123, 456, 789, ...],
  "worker_id": 0,
  "generation_time_ms": 45.2,
  "request_id": "req_123"
}
```

---

## 实验执行流程

### 阶段 1：准备环境

#### 在本地 Windows 机器：
```bash
cd C:\code\workspace\fastsd
git status  # 确认在 pre-kvcache 分支
git log --oneline -3

# 生成实验配置
python scripts/prepare_32worker_experiment.py \
  --run_id qwen3_8b_1.7b_32cpu_workers_20260818 \
  --exp_root C:\code\workspace\fastsd\exp \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --target_model /home/hdd/zhangh/models/Qwen3-8B \
  --num_workers 2  # 先生成 2-worker 冒烟测试配置

# 将生成的脚本同步到服务器
# scp exp/qwen3_8b_1.7b_32cpu_workers_20260818/* node2:/home/hdd/zhangh/workspace/new_fastsd/exp/...
```

#### 在 node2 和 node3：
```bash
# 创建隔离的工作目录
ssh node2
mkdir -p /home/hdd/zhangh/workspace/new_fastsd_exp_32worker
cd /home/hdd/zhangh/workspace/new_fastsd_exp_32worker

# 克隆仓库并切换分支
git clone --recurse-submodules https://github.com/<your-repo>/fastsd.git .
git checkout pre-kvcache
git submodule update --init --recursive

# 记录 commit SHA
git log --oneline -1 > exp_git_sha.txt
cd baselines/specedge/official
git log --oneline -1 >> ../../../exp_git_sha.txt

# 在 node3 重复相同操作
ssh node3
# ... 相同的克隆和 checkout 步骤
```

---

### 阶段 2：冒烟测试（2 workers）

#### Node3: 启动 2 个 CPU draft workers
```bash
ssh node3
cd /home/hdd/zhangh/workspace/new_fastsd_exp_32worker
source /home/hdd/zhangh/envs/fastsd/bin/activate

# 检查 CPU 拓扑
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|NUMA"
numactl --hardware

# 启动 2 个 workers
python scripts/launch_cpu_draft_workers.py \
  --num_workers 2 \
  --start_core 0 \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --base_port 19000 \
  --log_dir exp/smoke_test/draft_logs \
  --worker_script scripts/cpu_draft_worker.py \
  --manifest_out exp/smoke_test/draft_workers_manifest.json

# 等待模型加载（约 1-2 分钟）
# 查看日志确认启动成功
tail -f exp/smoke_test/draft_logs/draft_worker_0.out
tail -f exp/smoke_test/draft_logs/draft_worker_1.out
```

#### 验证 workers 健康状态：
```bash
# 从 node1 或 node2 测试
curl http://node3:19000/health
curl http://node3:19001/health

# 预期输出：
# {"status":"healthy","worker_id":0,"generation_count":0,"device":"cpu"}
```

#### Node2: 启动 target service
```bash
ssh node2
cd /home/hdd/zhangh/workspace/new_fastsd_exp_32worker
source /home/hdd/zhangh/envs/fastsd/bin/activate

# 检查 GPU
nvidia-smi

FASTSD_TARGET_DEVICE=cuda:0 \
python cloud/cloud_service.py \
  --target_model /home/hdd/zhangh/models/Qwen3-8B \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --dataset humaneval \
  --server_sched_mode fastsd \
  --batch_size 2 \
  --token_budget 512 \
  > exp/smoke_test/target.out 2> exp/smoke_test/target.err &

# 等待 target 加载
tail -f exp/smoke_test/target.err

# 验证 target 健康
curl http://127.0.0.1:8001/health
```

#### 运行冒烟测试（手动验证）：
```bash
# TODO: 需要实现 coordinator 脚本
# 该脚本应该：
# 1. 向 worker_0 发送一个测试请求
# 2. 收集 draft tokens
# 3. 向 target 发送 verify 请求
# 4. 检查 KV cache 是否正确 offload
# 5. 记录 TTFT, TPOT, E2E

# 临时手动测试：
curl -X POST http://node3:19000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "def hello():", "max_new_tokens": 10, "request_id": "test_1"}'
```

---

### 阶段 3：FastSD 完整实验（32 workers）

#### Node3: 启动 32 workers
```bash
ssh node3
# 终止之前的 2-worker 进程
pkill -f cpu_draft_worker.py

# 启动 32 workers
python scripts/launch_cpu_draft_workers.py \
  --num_workers 32 \
  --start_core 0 \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --base_port 19000 \
  --log_dir exp/fastsd_32w/draft_logs \
  --worker_script scripts/cpu_draft_worker.py \
  --manifest_out exp/fastsd_32w/draft_workers_manifest.json

# 验证所有 workers 启动
for i in {0..31}; do
  curl -s http://127.0.0.1:$((19000+i))/health | jq -r '.status'
done | sort | uniq -c
# 预期：32 healthy
```

#### Node2: 重启 target（32 batch size）
```bash
ssh node2
# 终止之前的 target
pkill -f cloud_service.py

FASTSD_TARGET_DEVICE=cuda:0 \
python cloud/cloud_service.py \
  --target_model /home/hdd/zhangh/models/Qwen3-8B \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --dataset humaneval \
  --server_sched_mode fastsd \
  --batch_size 32 \
  --token_budget 1024 \
  > exp/fastsd_32w/target.out 2> exp/fastsd_32w/target.err &
```

#### 运行完整实验：
```bash
# TODO: 实现 coordinator
# python scripts/run_fastsd_32worker_experiment.py \
#   --workload exp/canonical_humaneval_164.jsonl \
#   --draft_workers exp/fastsd_32w/draft_workers_manifest.json \
#   --target_url http://node2:8001 \
#   --output exp/fastsd_32w/results
```

---

### 阶段 4：SpecEdge 实验（32 clients + KV offload）

**重要：** SpecEdge KV offload 补丁需要手动应用到官方代码。

#### 应用 KV offload 补丁：
```bash
ssh node2
cd /home/hdd/zhangh/workspace/new_fastsd_exp_32worker/baselines/specedge/official

# 1. 阅读补丁参考
cat ../../../exp/qwen3_8b_1.7b_32cpu_workers_20260818/specedge/specedge_kv_offload_patch.py

# 2. 手动修改 src/script/batch_server.py
# - 在 __init__ 中添加 idle_kv_device 配置
# - 添加 prepare_batch_kv_caches() 方法
# - 添加 offload_batch_kv_caches() 方法
# - 在 execute_batch() 中调用这两个方法

# 3. 验证修改
git diff src/script/batch_server.py > exp/specedge_kv_offload.patch

# 4. 测试修改（dry run）
SPECEDGE_IDLE_KV_DEVICE=cpu python -c "from src.script import batch_server; print('Import OK')"
```

#### 启动 SpecEdge server（带 KV offload）：
```bash
ssh node2
cd /home/hdd/zhangh/workspace/new_fastsd_exp_32worker/baselines/specedge/official
source /home/hdd/zhangh/envs/specedge/bin/activate

SPECEDGE_IDLE_KV_DEVICE=cpu \
SPECEDGE_TARGET_DEVICE=cuda:0 \
python src/script/batch_server.py \
  --config ../../../exp/qwen3_8b_1.7b_32cpu_workers_20260818/specedge/specedge_config.yaml \
  > exp/specedge_32c/server.out 2> exp/specedge_32c/server.err &

# 验证 KV offload 日志
grep "KV-OFFLOAD" exp/specedge_32c/server.err
```

#### Node3: 启动 32 SpecEdge clients：
```bash
ssh node3
cd /home/hdd/zhangh/workspace/new_fastsd_exp_32worker/baselines/specedge/official
source /home/hdd/zhangh/envs/specedge/bin/activate

python ../../integration/client_host.py \
  --config ../../../exp/qwen3_8b_1.7b_32cpu_workers_20260818/specedge/specedge_config.yaml
```

---

## 指标收集

### FastSD 指标
- **位置：** `exp/fastsd_32w/target.out`、`exp/fastsd_32w/draft_logs/draft_worker_*.out`
- **内容：**
  - Per-request JSONL（包含 TTFT, TPOT, E2E, accepted tokens）
  - Target 侧：`scheduler_metrics.json`（prefetch_async_submitted, completed, failed）
  - Draft 侧：每个 worker 的 generation_count、平均 generation_time_ms

### SpecEdge 指标
- **位置：** `exp/specedge_32c/server.err`、SpecEdge raw results
- **内容：**
  - Per-batch KV offload 时间：`copy_to_gpu_ms`, `copy_to_cpu_ms`
  - Per-request TTFT, TPOT, E2E（通过 integration adapter 记录）
  - KV offload summary（在 server 关闭时打印）

---

## 未完成部分（需要实现）

### 1. FastSD coordinator 脚本
**功能：** 协调 32 个 draft workers 和 1 个 target

**待实现：**
```python
# scripts/run_fastsd_32worker_coordinator.py
# - 读取 canonical workload
# - 按 Poisson 到达分配请求到 32 个 draft workers
# - 收集 draft outputs
# - 发送 verify 到 target
# - 记录每个请求的完整时间线
```

### 2. SpecEdge KV offload 实际代码
**当前状态：** 只有参考补丁（`specedge_kv_offload_patch.py`）

**待实现：**
- 手动应用补丁到 `baselines/specedge/official/src/script/batch_server.py`
- 测试 CPU↔GPU 拷贝正确性
- 验证拷贝时间计入 TTFT/E2E

### 3. SpecEdge integration adapter 增强
**当前状态：** `baselines/specedge/integration/client.py` 已有基础框架

**待实现：**
- 记录 draft 侧生成时间
- 记录 network RTT
- 与 server 侧 KV offload 时间对齐

---

## 检查清单

### 实验前：
- [ ] node2 和 node3 都 clone 代码到隔离目录
- [ ] node2 和 node3 的 Git SHA 一致
- [ ] SpecEdge 子模块已初始化（`git submodule update --init --recursive`）
- [ ] 模型路径存在且可访问
- [ ] 计算模型 SHA256（`sha256sum config.json`）
- [ ] node3 CPU 拓扑确认（32 physical cores）
- [ ] node2 GPU 可用（`nvidia-smi`）
- [ ] 端口未被占用（node3:19000-19031, node2:8001）

### 冒烟测试后：
- [ ] 2 个 draft workers 启动成功
- [ ] workers 健康检查通过
- [ ] target 启动成功且可连接
- [ ] 手动发送一个请求，验证端到端流程
- [ ] KV cache 正确 offload（检查 target logs）
- [ ] 指标正确记录（TTFT, TPOT, generation_time_ms）

### 完整实验后：
- [ ] 32 个 workers 全部启动
- [ ] 实验完成，无中断
- [ ] FastSD 结果 JSONL 包含所有请求
- [ ] SpecEdge 结果 JSONL 包含所有请求
- [ ] KV offload summary 已记录
- [ ] 原始日志已保存
- [ ] 汇总指标已生成

---

## 故障排查

### Workers 启动失败
```bash
# 检查日志
tail -100 exp/fastsd_32w/draft_logs/draft_worker_0.err

# 常见问题：
# - 模型路径错误
# - CPU 核心绑定失败（权限问题）
# - 端口已被占用
# - 内存不足（32 x 1.7B 模型需要大量 RAM）
```

### Target 连接失败
```bash
# 检查 target 是否运行
curl http://127.0.0.1:8001/health

# 检查防火墙
ss -tlnp | grep 8001

# 检查 target 日志
tail -100 exp/fastsd_32w/target.err
```

### KV offload 未生效
```bash
# 检查环境变量
grep FASTSD_DISABLE_TARGET_CACHE_OFFLOAD exp/fastsd_32w/target.err

# 检查 prefetch metrics
jq '.fastsd_scheduler.prefetch_async_submitted' exp/fastsd_32w/scheduler_metrics.json

# 如果 submitted=0，参考第 5 节审阅报告的诊断方法
```

---

## 相关文档
- [FOUR_METHOD_EVALUATION.md](FOUR_METHOD_EVALUATION.md) - 四方法统一评测
- [REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md) - 仓库结构指南
- `baselines/specedge/README.md` - SpecEdge 基线说明
