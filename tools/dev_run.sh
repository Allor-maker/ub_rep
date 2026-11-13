#!/usr/bin/env bash
# Run the agent on a COPY of a tasks dir inside the runtime image, with the official
# container restrictions (read-only rootfs, /tmp tmpfs, no capabilities, workdir /app).
# Usage: tools/dev_run.sh <tasks_dir> <run_name> [image] [extra env args...]
# LLM credentials: LLM_BASE_URL/LLM_API_KEY from the environment, or NEURALDEEP_* from ../.env.
set -euo pipefail
[[ $# -ge 2 && -d "$1" && "$2" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "usage: $0 <existing_tasks_dir> <run_name> [image] [docker -e args]" >&2; exit 2; }
SRC_TASKS="$(realpath -e "$1")"; NAME="$2"; IMAGE="${3:-hack:dev}"; shift $(( $# >= 3 ? 3 : 2 ))
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$REPO/experiments/runs/$NAME"
ENV_FILE="${ENV_FILE:-$REPO/../.env}"
if [[ -z "${LLM_API_KEY:-}" && -f "$ENV_FILE" ]]; then
  LLM_API_KEY="$(grep -E '^(NEURALDEEP_API_KEY|LLM_API_KEY)=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
  LLM_BASE_URL="$(grep -E '^(NEURALDEEP_BASE_URL|LLM_BASE_URL)=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
fi
export LLM_API_KEY LLM_BASE_URL
rm -rf "$RUN_DIR" && mkdir -p "$RUN_DIR/logs" && cp -r "$SRC_TASKS" "$RUN_DIR/tasks"
find "$RUN_DIR/tasks" -name .DS_Store -delete; chmod -R a+rwX "$RUN_DIR"
docker run --rm --name "agent-$NAME" \
  -e LLM_BASE_URL -e LLM_API_KEY -e AGENT_LOG_DIR=/data/logs "$@" \
  -v "$REPO:/app:ro" -v "$RUN_DIR/tasks:/data/tasks:rw" -v "$RUN_DIR/logs:/data/logs:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cap-drop ALL --security-opt no-new-privileges --cpus 4 --memory 16g --pids-limit 512 \
  --workdir /app --entrypoint python "$IMAGE" /app/run_agent.py --tasks-dir /data/tasks \
  2> "$RUN_DIR/logs/stderr.log" | tee "$RUN_DIR/logs/summary_stdout.json"
grep "^\[agent" "$RUN_DIR/logs/stderr.log" | tail -n 200 > "$RUN_DIR/logs/progress.log" || true
