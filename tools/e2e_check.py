"""Check an end-to-end run against tests/e2e/expected.json (stdlib only).

Usage: python3 tools/e2e_check.py <scenario> <run_dir>
run_dir holds tasks/ (after the run), summary.json (agent stdout), stats.json (fake LLM counters).
Exit code 0 = all checks passed; every failed check is printed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ORIGINAL = REPO / "tests" / "e2e" / "tasks"
EXPECTED = json.loads((REPO / "tests" / "e2e" / "expected.json").read_text())


def main() -> int:
    scenario, run = sys.argv[1], Path(sys.argv[2])
    tasks = run / "tasks"
    problems: list[str] = []
    try:
        summary = json.loads((run / "summary.json").read_text())
    except (OSError, ValueError) as exc:
        print(f"FAIL: agent printed no JSON summary on stdout ({exc})")
        return 1
    stats = json.loads((run / "stats.json").read_text()) if (run / "stats.json").exists() else {}
    answers = {}
    for tid, exp in EXPECTED.items():
        path = tasks / tid / "answer.txt"
        answer = path.read_text().strip() if path.exists() else None
        answers[tid] = answer
        # Safety checks that hold in every scenario.
        if exp.get("absent") and answer is not None:
            problems.append(f"{tid}: answer.txt must not be written (got {answer!r})")
        if answer is not None and "answer" in exp and answer != exp["answer"]:
            problems.append(f"{tid}: wrong answer {answer!r}, expected {exp['answer']!r}")
        for rel in exp.get("unchanged", []):
            if (tasks / tid / "workspace" / rel).read_bytes() != (ORIGINAL / tid / "workspace" / rel).read_bytes():
                problems.append(f"{tid}: {rel} was modified")

    def require_all_answers() -> None:
        for tid, exp in EXPECTED.items():
            if "answer" in exp and answers[tid] is None:
                problems.append(f"{tid}: no answer.txt (expected {exp['answer']!r})")
        if answers.get("coding_fix") is not None:
            result = subprocess.run([sys.executable, "test_mylib.py"], cwd=tasks / "coding_fix" / "workspace",
                                    capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                problems.append("coding_fix: tests still fail after the agent's edit")

    peak = stats.get("peak_in_flight", 0)
    if peak > 2:
        problems.append(f"{peak} parallel LLM requests (default limit is 2)")
    if scenario == "full":
        require_all_answers()
        per_rule = stats.get("per_rule", {})
        rules = {r["match"]: r for r in json.loads((REPO / "tests" / "e2e" / "replies.json").read_text())}
        for tid, exp in EXPECTED.items():
            if "max_requests" not in exp:
                continue
            instruction = json.loads((ORIGINAL / tid / "task.json").read_text())["instruction"]
            match = next((m for m in rules if m in instruction), None)
            if match and per_rule.get(match, 0) > exp["max_requests"]:
                problems.append(f"{tid}: {per_rule[match]} LLM requests, expected at most {exp['max_requests']}")
        if summary.get("llm_failed_calls"):
            problems.append(f"{summary['llm_failed_calls']} failed LLM calls on a healthy proxy")
    elif scenario == "budget":
        if "emulated external stop" not in str(summary.get("llm_disabled_reason")):
            problems.append(f"the budget did not run out: {summary.get('llm_disabled_reason')!r}")
        if answers["math_free"] != "4":
            problems.append("the free (fast path) answer is missing")
        if all(answers[t] is not None for t, e in EXPECTED.items() if "answer" in e):
            problems.append("every task was answered although the budget was tiny")
    elif scenario == "ratelimit":
        require_all_answers()
        rejected = stats.get("statuses", {}).get("429", 0)
        if rejected < 1 or summary.get("llm_rate_limited", 0) < 1:
            problems.append("the fake proxy never rate limited: the scenario did not test anything")
        if rejected > 3:
            problems.append(f"{rejected} requests rejected with 429: parallelism did not adapt")
    elif scenario == "dead":
        if answers["math_free"] != "4":
            problems.append("the free (fast path) answer is missing")
        if any(answers[t] is not None for t in EXPECTED if t != "math_free"):
            problems.append("answers appeared although the proxy rejected everything")
        if stats.get("requests", 0) > 2:
            problems.append(f"{stats['requests']} requests to a dead proxy (should stop after the first)")
    else:
        problems.append(f"unknown scenario {scenario!r}")

    print(f"scenario={scenario} answered={summary.get('answered')} tokens={summary.get('tokens_spent')} "
          f"requests={stats.get('requests')} statuses={stats.get('statuses')} peak_parallel={peak}")
    for problem in problems:
        print("FAIL:", problem)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
