#!/usr/bin/env bash
# End-to-end scenario: the built image runs with the organizers' container flags against a
# scripted fake LLM (tests/e2e/fake_llm.py) on tests/e2e/tasks, then tools/e2e_check.py checks it.
# Usage: tools/e2e.sh <full|budget|ratelimit|dead|all> [image]      (no API key, no internet needed)
set -uo pipefail
SCENARIO="${1:?usage: $0 <full|budget|ratelimit|dead|all> [image]}"; IMAGE="${2:-astra:ci}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$SCENARIO" == all ]]; then
  failed=0
  for s in full budget ratelimit dead; do "$0" "$s" "$IMAGE" || failed=1; done
  exit $failed
fi

case "$SCENARIO" in
  full)      FAKE=();                        AGENT=() ;;
  budget)    FAKE=();                        AGENT=(-e BUDGET=250 -e AGENT_STOP_AT_BUDGET=1) ;;
  ratelimit) FAKE=(-e FAKE_MAX_PARALLEL=1);  AGENT=() ;;
  dead)      FAKE=(-e FAKE_STATUS=402);      AGENT=() ;;
  *) echo "unknown scenario $SCENARIO" >&2; exit 2 ;;
esac

RUN="$(mktemp -d "${TMPDIR:-/tmp}/astra-e2e-$SCENARIO-XXXXXX")"
cp -r "$REPO/tests/e2e/tasks" "$RUN/tasks"
mkdir -p "$RUN/stats" && chmod -R a+rwX "$RUN"
NET="astra-e2e-$$"; FAKE_NAME="astra-fake-llm-$$"
cleanup() { docker rm -f "$FAKE_NAME" >/dev/null 2>&1; docker network rm "$NET" >/dev/null 2>&1; }
trap cleanup EXIT
docker network create "$NET" >/dev/null

docker run -d --rm --name "$FAKE_NAME" --network "$NET" --network-alias fake-llm \
  -v "$REPO/tests/e2e:/e2e:ro" -v "$RUN/stats:/stats" -e FAKE_STATS=/stats/stats.json "${FAKE[@]}" \
  --entrypoint python "$IMAGE" /e2e/fake_llm.py >/dev/null
for _ in $(seq 1 50); do
  docker exec "$FAKE_NAME" python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    >/dev/null 2>&1 && break
  sleep 0.2
done

# The organizers' command (plus the fake proxy's address); 10 minutes is far more than needed.
# CPUs: the regulation allows 4; CI runners may have fewer (GitHub private repos: 2).
CPUS=$(nproc); (( CPUS > 4 )) && CPUS=4
timeout -k 10s 600s docker run --rm --network "$NET" \
  -e LLM_BASE_URL=http://fake-llm:8000/v1 -e LLM_API_KEY=e2e-fake-key "${AGENT[@]}" \
  -v "$RUN/tasks:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus "$CPUS" --memory 16g --pids-limit 512 \
  --workdir /app --entrypoint python \
  "$IMAGE" /app/run_agent.py --tasks-dir /data/tasks > "$RUN/summary.json" 2> "$RUN/stderr.log"
code=$?
cp "$RUN/stats/stats.json" "$RUN/stats.json" 2>/dev/null
if [[ $code -ne 0 ]]; then
  echo "FAIL: scenario $SCENARIO: the agent exited with code $code"; tail -n 30 "$RUN/stderr.log"; exit 1
fi
if ! python3 "$REPO/tools/e2e_check.py" "$SCENARIO" "$RUN"; then
  echo "--- agent log (tail) ---"; grep '^\[astra' "$RUN/stderr.log" | tail -n 40; exit 1
fi
