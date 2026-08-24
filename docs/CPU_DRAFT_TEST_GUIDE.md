# 说明：旧版一次性 worker 仅作通信 smoke harness

本文件中的 `scripts/cpu_draft_worker.py` `/generate` 路径不是 FastSD 的正式实现，不能
替代 `EdgeClient` 的 `/session/init`、`/prefill`、`/verify` 和 rollback/KV 生命周期。
正式四方法实验请使用 `scripts/eval_suite.py prepare/plan`；max_new_tokens=16 的历史结果
只能标为 `communication_smoke`。

# CPU Draft + GPU Target 通信测试指南

## 概述

本测试验证以下架构：
- **node3**: 1个CPU进程运行Qwen3-1.7B draft模型
- **node2**: 1个GPU (A6000) 运行Qwen3-8B target模型
- **通信**: node3的CPU draft进程通过HTTP与node2的GPU target通信

## 已完成的代码修改

### 1. `edge/edge.py` - 支持CPU设备
**修改位置**: `run_draft_process_http` 函数 (line 527-539)

**修改内容**:
```python
# 原代码（只支持GPU）:
gpu_id = (proc_id % max(1, self.args.edge_gpus)) + self.args.edge_gpu_start
device = f"cuda:{gpu_id}"

# 新代码（支持CPU和GPU）:
use_cpu = getattr(self.args, "edge_use_cpu", False)
if use_cpu:
    device = "cpu"
    self.color_print(f"[Edge {proc_id}] loading draft model on CPU", 3)
else:
    gpu_id = (proc_id % max(1, self.args.edge_gpus)) + self.args.edge_gpu_start
    device = f"cuda:{gpu_id}"
    self.color_print(f"[Edge {proc_id}] loading draft model on {device}", 3)
```

**新增参数**: `--edge_use_cpu`
- 当设置此参数时，draft模型加载到CPU而非GPU
- 默认值: `False` (向后兼容，保持原有GPU行为)

### 2. `scripts/test_cpu_draft_communication.py` - 测试脚本
**功能**:
- 创建简单的1请求测试workload
- 启动CPU draft worker
- 连接到GPU target
- 发送请求并记录结果
- 解析metrics并显示关键指标

### 3. `scripts/test_cpu_draft_communication.sh` - Bash测试包装脚本
**功能**:
- 检查环境配置（模型路径、target连接性）
- 显示CPU拓扑和内存信息
- 设置CPU运行环境变量
- 调用Python测试脚本

## 测试步骤

### 前置准备

#### 在node2上启动target服务：
```bash
ssh node2
cd /home/hdd/zhangh/workspace/new_fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate

# 确认GPU可用
nvidia-smi

# 启动target
FASTSD_TARGET_DEVICE=cuda:0 \
python cloud/cloud_service.py \
  --target_model /home/hdd/zhangh/models/Qwen3-8B \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --dataset humaneval \
  --server_sched_mode fastsd \
  --batch_size 1 \
  > /tmp/fastsd_target_test.out 2> /tmp/fastsd_target_test.err &

# 等待target启动（约30-60秒）
tail -f /tmp/fastsd_target_test.err

# 验证target健康
curl http://127.0.0.1:8001/health
```

预期输出:
```json
{"status":"ok"}
```

#### 在node3上准备环境：
```bash
ssh node3
cd /home/hdd/zhangh/workspace/new_fastsd

# 确认在pre-kvcache分支
git branch
git log --oneline -1

# 拉取最新修改（如果需要）
git pull origin pre-kvcache

# 激活虚拟环境
source /home/hdd/zhangh/envs/fastsd/bin/activate

# 检查Python环境
python --version
which python

# 确认模型存在
ls -lh /home/hdd/zhangh/models/Qwen3-1.7B/config.json

# 检查CPU拓扑
lscpu | grep -E "CPU\(s\):|Thread|Core|Socket|NUMA"
```

---

### 测试方法1: 使用Bash脚本（推荐）

```bash
# 在node3上执行
cd /home/hdd/zhangh/workspace/new_fastsd

# 设置target URL（如果node2不能直接访问，使用隧道）
export TARGET_URL="http://node2:8001"
# 或者使用127.0.0.1（如果已设置SSH隧道）
# export TARGET_URL="http://127.0.0.1:8001"

# 运行测试
bash scripts/test_cpu_draft_communication.sh
```

**预期输出**:
```
==========================================
CPU Draft + GPU Target Communication Test
==========================================

Configuration:
  Draft model: /home/hdd/zhangh/models/Qwen3-1.7B
  Target URL: http://node2:8001
  Experiment dir: /home/hdd/zhangh/workspace/new_fastsd/exp/test_cpu_draft_20260818_143025

Checking target connectivity...
✓ Target is reachable at http://node2:8001

CPU Topology:
CPU(s):              32
Thread(s) per core:  1
Core(s) per socket:  16
Socket(s):           2
NUMA node(s):        2

Available Memory:
              total        used        free      shared  buff/cache   available
Mem:           125G         15G        100G        128M         10G        108G

Environment:
  CUDA_VISIBLE_DEVICES: 
  OMP_NUM_THREADS: 1
  MKL_NUM_THREADS: 1

==========================================
Starting test...
==========================================

[Edge 0] loading draft model on CPU
[TEST] Created test workload: ...
...
✓ SUCCESS: Communication test passed

--- METRICS ---
{
  "task_id": "test_cpu_draft_1",
  "ttft_ms": 1234.56,
  "e2e_ms": 5678.90,
  "tpot_ms": 123.45,
  "accepted_tokens": 15,
  "output_tokens": 20
}

==========================================
SUMMARY
==========================================

✓ Test PASSED:
  - CPU draft worker started successfully
  - Connected to GPU target
  - Completed request processing
  - Generated metrics

Key Metrics:
  TTFT: 1234.56 ms
  E2E: 5678.90 ms
  TPOT: 123.45 ms
  Accepted tokens: 15
  Output tokens: 20
```

---

### 测试方法2: 直接调用Python脚本

```bash
# 在node3上执行
cd /home/hdd/zhangh/workspace/new_fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate

# 禁用GPU
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/test_cpu_draft_communication.py \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --target_url http://node2:8001 \
  --exp_dir exp/test_cpu_draft_$(date +%Y%m%d_%H%M%S) \
  --dataset humaneval \
  --gamma 4 \
  --max_tokens 64 \
  --temperature 0.0
```

---

### 测试方法3: 手动运行edge.py（调试用）

```bash
# 在node3上执行
cd /home/hdd/zhangh/workspace/new_fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 创建测试workload
cat > /tmp/test_workload.jsonl << 'EOF'
{"task_id": "test_1", "prompt": "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n", "sample_id": 0}
EOF

# 运行edge.py
python edge/edge.py \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --server_url http://node2:8001 \
  --profile vanilla \
  --exp_name exp/test_manual_$(date +%Y%m%d_%H%M%S) \
  --dataset humaneval \
  --data_path /tmp/test_workload.jsonl \
  --num_drafts 1 \
  --edge_use_cpu \
  --gamma 4 \
  --max_tokens 64 \
  --temp 0.0 \
  --top_k 0 \
  --top_p 1.0

# 检查结果
ls -lh exp/test_manual_*/edge_metrics_proc0.jsonl
cat exp/test_manual_*/edge_metrics_proc0.jsonl | jq
```

---

## 故障排查

### 问题1: Target连接失败
**症状**:
```
✗ ERROR: Cannot reach target at http://node2:8001
```

**解决方案**:
```bash
# 1. 确认target在node2上运行
ssh node2 "ps aux | grep cloud_service.py"

# 2. 确认target端口监听
ssh node2 "ss -tlnp | grep 8001"

# 3. 测试网络连通性
ping node2
curl http://node2:8001/health

# 4. 如果node3不能直接访问node2:8001，使用SSH隧道
# 在node3上执行：
ssh -N -L 8001:127.0.0.1:8001 node2 &
# 然后使用 http://127.0.0.1:8001 作为target_url
```

### 问题2: 模型加载失败
**症状**:
```
FileNotFoundError: /home/hdd/zhangh/models/Qwen3-1.7B/config.json
```

**解决方案**:
```bash
# 确认模型路径
ls -lh /home/hdd/zhangh/models/Qwen3-1.7B/

# 检查config.json存在
cat /home/hdd/zhangh/models/Qwen3-1.7B/config.json | head -20

# 确认模型文件完整
ls -lh /home/hdd/zhangh/models/Qwen3-1.7B/*.safetensors
```

### 问题3: 内存不足
**症状**:
```
RuntimeError: [enforce fail at alloc_cpu.cpp:114] . DefaultCPUAllocator: not enough memory
```

**解决方案**:
```bash
# 检查可用内存
free -h

# Qwen3-1.7B约需6-8GB RAM (bf16)
# 如果内存不足，尝试：
# 1. 关闭其他进程
# 2. 使用量化模型
# 3. 使用更小的模型（如Qwen3-0.6B）
```

### 问题4: CPU性能过慢
**症状**:
- Draft生成时间 > 5秒

**解决方案**:
```bash
# 1. 确认CPU线程数设置
echo $OMP_NUM_THREADS  # 应为1

# 2. 检查CPU频率
lscpu | grep MHz

# 3. 使用更小的max_tokens进行测试
python scripts/test_cpu_draft_communication.py \
  --max_tokens 32  # 减少生成token数
```

### 问题5: torch.bfloat16不支持
**症状**:
```
RuntimeError: "bfloat16" is not supported on CPU
```

**解决方案**:
修改 `edge/edge.py` 中的 `_load_draft_model` 函数:
```python
# 在edge.py line 129
return AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": device},
    torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,  # CPU使用fp32
    trust_remote_code=True,
).eval()
```

---

## 验证成功的标准

测试成功应满足以下条件：

1. ✓ **进程启动**: edge进程正常启动，无崩溃
2. ✓ **模型加载**: Draft模型成功加载到CPU
3. ✓ **连接建立**: 成功连接到node2的target服务
4. ✓ **请求完成**: 至少完成1个请求的完整流程（prefill + verify rounds）
5. ✓ **指标生成**: 生成 `edge_metrics_proc0.jsonl` 文件
6. ✓ **指标合理**: 
   - TTFT < 10秒
   - E2E < 30秒
   - accepted_tokens > 0
   - output_tokens > 0

---

## 下一步：扩展到32个CPU workers

测试成功后，可以扩展到32个CPU workers：

```bash
# 在node3上执行
cd /home/hdd/zhangh/workspace/new_fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate

# 创建32个worker的workload（每个worker处理一部分请求）
python scripts/eval_suite.py prepare \
  --config configs/evaluation/qwen3_8b_1.7b_humaneval_32cpu.json

# 启动32个worker（使用multiprocessing）
python edge/edge.py \
  --draft_model /home/hdd/zhangh/models/Qwen3-1.7B \
  --server_url http://node2:8001 \
  --profile fastsd \
  --exp_name exp/fastsd_32cpu_$(date +%Y%m%d_%H%M%S) \
  --dataset humaneval \
  --data_path data/humaneval.jsonl \
  --num_drafts 32 \
  --edge_use_cpu \
  --gamma 4 \
  --max_tokens 256 \
  --arrival_distribution poisson \
  --arrival_rate 1.0
```

**注意事项**:
- 32个进程共享一个模型实例（PyTorch的模型对象是线程安全的）
- 每个进程会独立处理其分配的请求
- 确保node3有足够内存（至少64GB空闲）
- 监控CPU使用率：`htop` 或 `top`

---

## 预期结果报告

测试成功后，请记录以下信息：

### 环境信息
- node3 CPU型号: _______
- node3 核心数: _______
- node3 内存: _______
- node2 GPU型号: _______
- Draft模型: Qwen3-1.7B
- Target模型: Qwen3-8B

### 性能指标（1 CPU worker）
- TTFT: _______ ms
- E2E: _______ ms
- TPOT: _______ ms
- Draft generation time (avg): _______ ms
- Accepted tokens (avg): _______
- Output tokens: _______

### 通信延迟
- Network RTT (node3 ↔ node2): _______ ms
- HTTP request overhead: _______ ms

### 可行性结论
- [ ] CPU draft可行（TTFT < 10s）
- [ ] CPU draft不可行（需要GPU）
- [ ] 需要优化（瓶颈：_______）

---

请在node3上运行测试，并告诉我结果！
