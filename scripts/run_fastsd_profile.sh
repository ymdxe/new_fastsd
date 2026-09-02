#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXP_NAME="${1:-fastsd_run}"
shift 1 || true

COMMON_ARGS=(
  --profile custom
  --server_sched_mode fastsd
  --enable_latency_priority
  --overlap_prefill_first_draft
  --enable_proactive_draft
  --no-enable_pipeline
  --exp_name "$EXP_NAME"
  "$@"
)

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
