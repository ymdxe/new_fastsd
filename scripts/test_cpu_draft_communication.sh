#!/bin/bash
# Test CPU draft worker communication with GPU target
# Run this script on node3

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
DRAFT_MODEL="${DRAFT_MODEL:-/home/hdd/zhangh/models/Qwen3-1.7B}"
TARGET_URL="${TARGET_URL:-http://node2:8001}"
EXP_DIR="${EXP_DIR:-$REPO_ROOT/exp/test_cpu_draft_$(date +%Y%m%d_%H%M%S)}"

echo "=========================================="
echo "CPU Draft + GPU Target Communication Test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Draft model: $DRAFT_MODEL"
echo "  Target URL: $TARGET_URL"
echo "  Experiment dir: $EXP_DIR"
echo ""

# Check if draft model exists
if [ ! -d "$DRAFT_MODEL" ]; then
    echo "ERROR: Draft model not found: $DRAFT_MODEL"
    echo "Set DRAFT_MODEL environment variable to the correct path"
    exit 1
fi

# Check if target is reachable
echo "Checking target connectivity..."
if curl -s --connect-timeout 5 "$TARGET_URL/health" > /dev/null 2>&1; then
    echo "✓ Target is reachable at $TARGET_URL"
else
    echo "✗ ERROR: Cannot reach target at $TARGET_URL"
    echo ""
    echo "Make sure the target service is running on node2:"
    echo "  ssh node2"
    echo "  cd /home/hdd/zhangh/workspace/new_fastsd"
    echo "  FASTSD_TARGET_DEVICE=cuda:0 python cloud/cloud_service.py \\"
    echo "    --target_model /home/hdd/zhangh/models/Qwen3-8B \\"
    echo "    --draft_model $DRAFT_MODEL \\"
    echo "    --dataset humaneval \\"
    echo "    --server_sched_mode fastsd"
    exit 1
fi

# Check CPU topology
echo ""
echo "CPU Topology:"
lscpu | grep -E "^CPU\(s\):|^Thread|^Core|^Socket|^NUMA"

# Check available memory
echo ""
echo "Available Memory:"
free -h | grep -E "^Mem:|^Swap:"

# Disable GPU for this test
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo ""
echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"

# Run test
echo ""
echo "=========================================="
echo "Starting test..."
echo "=========================================="
echo ""

cd "$REPO_ROOT"
python scripts/test_cpu_draft_communication.py \
    --draft_model "$DRAFT_MODEL" \
    --target_url "$TARGET_URL" \
    --exp_dir "$EXP_DIR" \
    --dataset humaneval \
    --gamma 4 \
    --max_tokens 64 \
    --temperature 0.0

exit_code=$?

echo ""
echo "=========================================="
echo "Test completed with exit code: $exit_code"
echo "=========================================="

exit $exit_code
