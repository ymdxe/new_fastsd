#!/bin/bash
# 部署脚本：将本地修改同步到node2和node3

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "FastSD 代码部署到远程服务器"
echo "=========================================="
echo ""

# 检查是否在git仓库中
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "错误: 不在git仓库中"
    exit 1
fi

cd "$REPO_ROOT"

# 显示当前状态
echo "当前分支: $(git branch --show-current)"
echo "最新提交: $(git log --oneline -1)"
echo ""

# 检查是否有未提交的修改
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "检测到未提交的修改:"
    git status --short
    echo ""

    read -p "是否要提交这些修改? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "输入提交信息: " commit_msg
        git add -A
        git commit -m "$commit_msg"
        echo "✓ 已提交"
    else
        echo "跳过提交，将使用stash"
        git stash push -m "auto-stash before deploy $(date +%Y%m%d_%H%M%S)"
    fi
    echo ""
fi

# 推送到远程
BRANCH=$(git branch --show-current)
echo "推送分支 $BRANCH 到远程..."
git push origin "$BRANCH" || {
    echo "警告: 推送失败，但继续部署"
}
echo ""

# 部署到node2
echo "=========================================="
echo "部署到 node2"
echo "=========================================="
echo ""

read -p "node2 SSH地址 (默认: node2): " NODE2_HOST
NODE2_HOST=${NODE2_HOST:-node2}

read -p "node2 仓库路径 (默认: /home/hdd/zhangh/workspace/new_fastsd): " NODE2_PATH
NODE2_PATH=${NODE2_PATH:-/home/hdd/zhangh/workspace/new_fastsd}

ssh "$NODE2_HOST" "cd $NODE2_PATH && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH && git submodule update --init --recursive" && {
    echo "✓ node2 更新成功"
    ssh "$NODE2_HOST" "cd $NODE2_PATH && git log --oneline -1"
} || {
    echo "✗ node2 更新失败"
}
echo ""

# 部署到node3
echo "=========================================="
echo "部署到 node3"
echo "=========================================="
echo ""

read -p "node3 SSH地址 (默认: node3): " NODE3_HOST
NODE3_HOST=${NODE3_HOST:-node3}

read -p "node3 仓库路径 (默认: /home/hdd/zhangh/workspace/new_fastsd): " NODE3_PATH
NODE3_PATH=${NODE3_PATH:-/home/hdd/zhangh/workspace/new_fastsd}

ssh "$NODE3_HOST" "cd $NODE3_PATH && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH && git submodule update --init --recursive" && {
    echo "✓ node3 更新成功"
    ssh "$NODE3_HOST" "cd $NODE3_PATH && git log --oneline -1"
} || {
    echo "✗ node3 更新失败"
}
echo ""

echo "=========================================="
echo "部署完成"
echo "=========================================="
echo ""
echo "验证命令:"
echo "  ssh $NODE2_HOST 'cd $NODE2_PATH && git log --oneline -3'"
echo "  ssh $NODE3_HOST 'cd $NODE3_PATH && git log --oneline -3'"
