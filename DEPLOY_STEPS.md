# 快速部署指南

## 当前状态
- ✓ 本地代码已修改完成
- ✓ 已提交到本地 git (commit: b3a1508)
- ⏳ 待推送到远程仓库
- ⏳ 待同步到 node2 和 node3

## 第一步：推送到远程仓库

在本地 Windows 机器上执行：

```bash
cd C:\code\workspace\fastsd
git push origin pre-kvcache
```

## 第二步：在 node2 上更新代码

```bash
# SSH 到 node2
ssh node2

# 进入仓库目录
cd /home/hdd/zhangh/workspace/new_fastsd

# 拉取最新代码
git fetch origin
git checkout pre-kvcache
git pull origin pre-kvcache

# 验证更新成功
git log --oneline -3
# 应该看到: b3a1508 Add CPU draft worker support and test scripts

# 查看修改的文件
ls -lh edge/edge.py
ls -lh scripts/test_cpu_draft_communication.py
ls -lh scripts/diagnose_environment.sh
```

## 第三步：在 node3 上更新代码

```bash
# SSH 到 node3
ssh node3

# 进入仓库目录
cd /home/hdd/zhangh/workspace/new_fastsd

# 拉取最新代码
git fetch origin
git checkout pre-kvcache
git pull origin pre-kvcache

# 验证更新成功
git log --oneline -3
# 应该看到: b3a1508 Add CPU draft worker support and test scripts

# 查看修改的文件
ls -lh edge/edge.py
ls -lh scripts/test_cpu_draft_communication.sh
```

## 第四步：诊断环境（在 node2 上）

```bash
ssh node2
cd /home/hdd/zhangh/workspace/new_fastsd
bash scripts/diagnose_environment.sh
```

把输出结果发给我，我会根据实际环境调整测试脚本。

## 第五步：诊断环境（在 node3 上）

```bash
ssh node3
cd /home/hdd/zhangh/workspace/new_fastsd
bash scripts/diagnose_environment.sh
```

同样把输出发给我。

## 关键修改说明

### 1. edge/edge.py
- 在 `run_draft_process_http` 函数中添加了 CPU 设备支持
- 新增 `--edge_use_cpu` 参数
- 当设置此参数时，draft 模型加载到 CPU 而非 GPU

### 2. 新增测试脚本
- `scripts/test_cpu_draft_communication.py`: Python 测试脚本
- `scripts/test_cpu_draft_communication.sh`: Bash 包装脚本
- `scripts/diagnose_environment.sh`: 环境诊断脚本

### 3. 文档
- `docs/CPU_DRAFT_TEST_GUIDE.md`: 详细测试指南
- `docs/32_WORKER_EXPERIMENT_GUIDE.md`: 32-worker 实验指南

## 下一步

完成上述步骤后，告诉我：
1. git push 是否成功
2. node2 和 node3 的环境诊断输出
3. 是否遇到任何错误

然后我们就可以开始实际测试了！
