#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PROFILE="${1:-both}"
EXP_NAME="${2:-profile_${PROFILE}}"
shift 2 || true

case "$PROFILE" in
  vanilla|proactive_only|pipeline_only|both)
    ;;
  *)
    echo "Unsupported profile: $PROFILE"
    echo "Usage: bash scripts/run_vanilla_profile.sh {vanilla|proactive_only|pipeline_only|both} [exp_name] [extra args...]"
    exit 1
    ;;
esac

COMMON_ARGS=(--profile "$PROFILE" --exp_name "$EXP_NAME" --server_sched_mode "vanilla" "$@")

SERVER_URL_VALUE="${SERVER_URL:-http://127.0.0.1:8001}"
PYTHON_BIN_VALUE="${PYTHON_BIN:-python}"
EDGE_COMMAND=("$PYTHON_BIN_VALUE" edge/edge.py --server_url "$SERVER_URL_VALUE" "${COMMON_ARGS[@]}")
COMMAND_LOG="${REPO_ROOT}/exp/${EXP_NAME}/commands.txt"
mkdir -p "$(dirname "$COMMAND_LOG")"
{
  printf '\n# command_record_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'command='
  printf '%q ' "${EDGE_COMMAND[@]}"
  printf '\n'
} >> "$COMMAND_LOG"

set +e
"${EDGE_COMMAND[@]}"
EXIT_STATUS=$?
set -e
{
  printf 'exit_status=%s\n' "$EXIT_STATUS"
} >> "$COMMAND_LOG"
echo "Done. Metrics summary: $(pwd)/exp/${EXP_NAME}/edge_metrics_summary.json"
exit "$EXIT_STATUS"
