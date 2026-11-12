"""Make a copy of a finished run's tasks as if the run had been stopped at a token budget.

Usage: python tools/cut_at_budget.py RUN_NAME [BUDGET=100000]
Writes experiments/runs/RUN_NAME/cut<BUDGET>/ (task dirs); grade it with the set's own grader.

A task keeps its answer.txt only if the progress line that finished it shows a running total
within the budget. Up to the budget a run without AGENT_STOP_AT_BUDGET behaves like a stopped one
(verification is skipped once the budget is spent), so this replays the stop; the only error is
a task that was in flight at the cut.
"""

import re
import shutil
import sys
from pathlib import Path

LINE = re.compile(r"^\[astra\s+[\d.]+s\]\s+(\S+)\s+(answered|infeasible|failed)\s+.*?total=\s*([\d.]+)")


def main() -> None:
    name = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 100000.0
    run = Path(__file__).resolve().parent.parent / "experiments" / "runs" / name
    finished: dict[str, float] = {}
    for line in (run / "logs" / "stderr.log").read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if m and m.group(1) not in finished:
            finished[m.group(1)] = float(m.group(3))
    out = run / f"cut{int(budget)}"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(run / "tasks", out)
    removed = kept = 0
    for task in sorted(p for p in out.iterdir() if p.is_dir()):
        answer = task / "answer.txt"
        if not answer.exists():
            continue
        if finished.get(task.name, float("inf")) <= budget:
            kept += 1
        else:
            answer.unlink()
            removed += 1
    print(f"{name}: kept {kept} answers within {budget:.0f}, removed {removed}; copy: {out}")


if __name__ == "__main__":
    main()
