#!/usr/bin/env bash
# Grade a finished run with the grader of one of our task sets (grade.py + private_answers.json).
# Usage: tools/grade_set.sh <run_name> <grader_dir>
#
# Their convention for infeasible tasks is answer "UNSOLVABLE"; the organizers' rule is
# "no answer.txt". The agent follows the organizers, so for grading we work on a COPY and
# write UNSOLVABLE where the agent itself decided the task was infeasible.
# The grader executes model-written mylib.py, so it runs in a container without network.
set -euo pipefail
[[ $# -eq 2 && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && -f "$2/grade.py" ]] || {
  echo "usage: $0 <run_name> <grader_dir>" >&2; exit 2; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$REPO/experiments/runs/$1"
GRADER="$(realpath -e "$2")"
[[ -d "$RUN/tasks" && -f "$RUN/logs/summary.json" ]] || { echo "no finished run $RUN" >&2; exit 2; }
GRADED="$RUN/graded"
rm -rf "$GRADED" && command cp -r "$RUN/tasks" "$GRADED"
python3 - "$RUN/logs/summary.json" "$GRADED" <<'EOF'
import json, sys, pathlib
summary, graded = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2])
marked = 0
for task in summary["tasks"]:
    answer = graded / task["id"] / "answer.txt"
    if task["status"] == "infeasible" and not answer.exists():
        answer.write_text("UNSOLVABLE\n"); marked += 1
print(f"marked UNSOLVABLE: {marked}")
EOF
chmod -R a+rwX "$GRADED"
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=512m --cap-drop ALL \
  --security-opt no-new-privileges --memory 2g --pids-limit 256 --user "$(id -u):$(id -g)" \
  -v "$GRADER:/grader:ro" -v "$GRADED:/tasks:ro" -v "$RUN/logs:/out:rw" -w /tmp \
  python:3.12-slim python /grader/grade.py --tasks /tasks --verbose --json-report /out/set_report.json \
  > "$RUN/logs/set_grade.txt"
grep -vE '^[a-z_]+_[0-9]+ ' "$RUN/logs/set_grade.txt"  # summary lines only, not per-task
