#!/bin/bash
# 环境诊断脚本 - 在node2和node3上运行

echo "=========================================="
echo "FastSD 环境诊断"
echo "=========================================="
echo ""

# 基本信息
echo "主机名: $(hostname)"
echo "用户: $(whoami)"
echo "当前目录: $(pwd)"
echo ""

# 检查仓库
echo "========== Git 仓库 =========="
if [ -d ".git" ]; then
    echo "✓ 当前在 Git 仓库中"
    echo "  分支: $(git branch --show-current)"
    echo "  最新提交: $(git log --oneline -1)"
else
    echo "✗ 当前不在 Git 仓库中"
fi
echo ""

# 检查Python环境
echo "========== Python 环境 =========="
echo "系统 Python: $(which python3) ($(python3 --version 2>&1))"
echo "当前 Python: $(which python 2>/dev/null || echo 'not found')"

# 查找可能的虚拟环境
echo ""
echo "查找虚拟环境:"
for venv_path in \
    "/home/hdd/zhangh/envs/fastsd" \
    "/home/hdd/zhangh/envs/new_fastsd" \
    "$HOME/.venv" \
    "$HOME/venv" \
    "$(pwd)/.venv" \
    "$(pwd)/venv"; do
    if [ -f "$venv_path/bin/activate" ]; then
        echo "  ✓ 找到: $venv_path"
        echo "    Python: $($venv_path/bin/python --version 2>&1)"
    fi
done

# 检查conda
if command -v conda &> /dev/null; then
    echo ""
    echo "Conda 环境:"
    conda env list | grep -E "fastsd|new_fastsd" || echo "  未找到相关conda环境"
fi

echo ""

# 检查模型
echo "========== 模型路径 =========="
for model_path in \
    "/home/hdd/zhangh/models/Qwen3-1.7B" \
    "/home/hdd/zhangh/models/Qwen3-8B" \
    "/home/hdd/zhangh/models/Qwen3-0.6B"; do
    if [ -d "$model_path" ]; then
        echo "  ✓ $model_path"
        if [ -f "$model_path/config.json" ]; then
            vocab_size=$(grep -o '"vocab_size": [0-9]*' "$model_path/config.json" | cut -d' ' -f2)
            echo "    vocab_size: $vocab_size"
        fi
    else
        echo "  ✗ $model_path (不存在)"
    fi
done
echo ""

# 检查GPU
echo "========== GPU 信息 =========="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "  未检测到GPU"
fi
echo ""

# 检查CPU
echo "========== CPU 信息 =========="
echo "  CPU核心数: $(nproc)"
echo "  物理核心: $(lscpu | grep '^Core(s) per socket:' | awk '{print $4}')"
echo "  Socket数: $(lscpu | grep '^Socket(s):' | awk '{print $2}')"
echo "  NUMA节点: $(lscpu | grep '^NUMA node(s):' | awk '{print $3}')"
echo ""

# 检查内存
echo "========== 内存信息 =========="
free -h | grep -E "^Mem:|^Swap:"
echo ""

# 检查网络
echo "========== 网络连通性 =========="
if [ "$(hostname)" = "gpunode2-R8424-G12" ]; then
    echo "  当前在 node2"
    echo "  检查端口 8001: $(ss -tlnp 2>/dev/null | grep :8001 || echo '未监听')"
else
    echo "  当前在 node3"
    echo "  ping node2: $(ping -c 1 -W 1 gpunode2-R8424-G12 &>/dev/null && echo '✓ 可达' || echo '✗ 不可达')"
fi
echo ""

echo "=========================================="
echo "诊断完成"
echo "=========================================="
