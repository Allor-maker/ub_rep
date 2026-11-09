"""Summarize finished runs: python tools/compare_runs.py RUN_NAME... (dirs under experiments/runs/)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "experiments" / "runs"

print(f"{'run':<44}{'answered':>9}{'infeas':>8}{'failed':>8}{'tokens':>10}{'calls':>7}{'tok/LLM-task':>14}{'sec':>7}")
for name in sys.argv[1:]:
    path = ROOT / name / "logs" / "summary.json"
    if not path.is_file():
        print(f"{name:<44} (missing)")
        continue
    data = json.loads(path.read_text())
    s, tasks = data["summary"], data["tasks"]
    llm_tasks = sum(1 for t in tasks if t["calls"] > 0)
    print(f"{name:<44}{s['answered']:>9}{s['infeasible']:>8}{s['failed']:>8}{s['tokens_spent']:>10.0f}"
          f"{s['llm_calls']:>7}{s['tokens_spent'] / max(llm_tasks, 1):>14.0f}{s['elapsed_s']:>7.0f}")
