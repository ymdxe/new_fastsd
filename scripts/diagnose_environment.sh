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
if git rev-parse --git-dir >/dev/null 2>&1; then
    echo "✓ 当前在 Git 仓库中"
    # The newer branch-display option is unavailable on the Git 1.8.3.1
    # installed on the CentOS 7 servers.  symbolic-ref works on both old and
    # new Git; report the commit explicitly when HEAD is detached.
    branch_ref="$(git symbolic-ref HEAD 2>/dev/null)"
    if [ -n "$branch_ref" ]; then
        case "$branch_ref" in
            refs/heads/*) branch_name="${branch_ref#refs/heads/}" ;;
            *) branch_name="$branch_ref" ;;
        esac
        echo "  分支: $branch_name"
    else
        detached_commit="$(git rev-parse --short HEAD 2>/dev/null)"
        if [ -n "$detached_commit" ]; then
            echo "  分支: detached (commit $detached_commit)"
        else
            echo "  分支: detached (commit unavailable)"
        fi
    fi
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
model_roots=()
add_model_root() {
    local candidate_root existing_root
    candidate_root="$1"
    [ -n "$candidate_root" ] || return 0
    for existing_root in "${model_roots[@]}"; do
        [ "$existing_root" = "$candidate_root" ] && return 0
    done
    model_roots[${#model_roots[@]}]="$candidate_root"
}

# FASTSD_MODEL_ROOTS is a colon-separated override for site-specific layouts.
# Keep both known server roots as defaults so the same script reports the
# node3 draft copy and the node2 target copy without assuming one filesystem.
if [ -n "${FASTSD_MODEL_ROOTS:-}" ]; then
    saved_ifs="$IFS"
    IFS=:
    for configured_root in $FASTSD_MODEL_ROOTS; do
        add_model_root "$configured_root"
    done
    IFS="$saved_ifs"
fi
add_model_root "${FASTSD_MODEL_ROOT:-}"
add_model_root "/home/zhangh/models"
add_model_root "/home/hdd/zhangh/models"
[ -n "${HOME:-}" ] && add_model_root "$HOME/models"
add_model_root "$(pwd)/models"

echo "  搜索根目录: ${model_roots[*]}"
for model_name in Qwen3-1.7B Qwen3-8B Qwen3-0.6B; do
    found_model=0
    for model_root in "${model_roots[@]}"; do
        model_path="${model_root%/}/$model_name"
        if [ -d "$model_path" ]; then
            found_model=1
            echo "  ✓ $model_path"
            if [ -f "$model_path/config.json" ]; then
                vocab_size=$(grep -o '"vocab_size": [0-9]*' "$model_path/config.json" | cut -d' ' -f2)
                echo "    vocab_size: $vocab_size"
            fi
        fi
    done
    if [ "$found_model" -eq 0 ]; then
        echo "  ✗ $model_name (搜索根目录中不存在)"
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
echo "  CPU核心数: $(nproc 2>/dev/null || echo 'unknown')"

# lscpu's English keys are stable across the two servers when forced through
# the C locale.  The old script printed only Core(s) per socket (24); report
# the total physical-core count and retain the inputs used to derive it.
lscpu_value() {
    LC_ALL=C lscpu 2>/dev/null | awk -F: -v wanted="$1" '
        {
            key = $1
            value = $2
            sub(/^[[:space:]]+/, "", key)
            sub(/[[:space:]]+$/, "", key)
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            if (key == wanted) {
                print value
                exit
            }
        }'
}
cores_per_socket="$(lscpu_value 'Core(s) per socket')"
sockets="$(lscpu_value 'Socket(s)')"
physical_core_inputs_valid=1
case "$cores_per_socket" in
    ''|*[!0-9]*) physical_core_inputs_valid=0 ;;
esac
case "$sockets" in
    ''|*[!0-9]*) physical_core_inputs_valid=0 ;;
esac
if [ "$physical_core_inputs_valid" -eq 1 ]; then
    physical_cores=$((cores_per_socket * sockets))
    echo "  物理核心: $physical_cores (每Socket $cores_per_socket × $sockets Socket)"
else
    echo "  物理核心: unknown (lscpu 缺少可验证的 Core(s) per socket/Socket(s))"
fi
echo "  Socket数: ${sockets:-unknown}"
numa_nodes="$(lscpu_value 'NUMA node(s)')"
echo "  NUMA节点: ${numa_nodes:-unknown}"
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
