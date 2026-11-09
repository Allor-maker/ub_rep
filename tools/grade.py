"""Grade a finished run: python tools/grade.py RUN_TASKS_DIR ANSWERS.json [--summary summary.json]

ANSWERS.json: {task_id: {"expected": str|null, "reward": n, "type": str, "check": str, "test_file": str|null}}
or the organizers' sample layout (pass --from-task-json to read "expected" from each task.json).
expected null = infeasible task: "skipped" (no answer.txt) is the desired outcome.
Coding tasks are graded by re-running the original test file in the workspace.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def run_tests(workspace: Path, test_file: str | None) -> bool:
    if not workspace.is_dir():
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for item in workspace.iterdir():
            if item.is_file():
                (tmp_path / item.name).write_bytes(item.read_bytes())
        if test_file is not None:
            (tmp_path / "test_mylib.py").write_text(test_file)  # the ORIGINAL tests
        tests = sorted(p.name for p in tmp_path.glob("test*.py"))
        if not tests:
            return False
        result = subprocess.run([sys.executable, tests[0]], cwd=tmp_path, capture_output=True, timeout=60)
        return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path)
    parser.add_argument("answers", type=Path, nargs="?")
    parser.add_argument("--from-task-json", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.from_task_json:
        key = {}
        for tj in sorted(args.tasks.glob("*/task.json")):
            t = json.loads(tj.read_text())
            key[t["task_id"]] = {"expected": t.get("expected"), "reward": t.get("reward", 1),
                                 "type": t.get("task_type"), "check": t.get("check"), "test_file": None}
    else:
        key = json.loads(args.answers.read_text())
    stats = defaultdict(lambda: defaultdict(int))
    reward = total_reward = 0
    rows = []
    for task_id, info in sorted(key.items()):
        task_dir = args.tasks / task_id
        answer_path = task_dir / "answer.txt"
        answer = answer_path.read_text(encoding="utf-8-sig").strip() if answer_path.is_file() else None
        expected = info["expected"]
        kind = info["type"]
        if expected is None:
            outcome = "infeasible_skipped" if answer is None else "infeasible_answered"
        else:
            total_reward += info["reward"]
            if answer is None:
                outcome = "missing"
            elif expected == "tests_pass" or info.get("check") == "unittest":
                outcome = "correct" if answer == "DONE" and run_tests(task_dir / "workspace", info.get("test_file")) \
                    else "incorrect"
            else:
                outcome = "correct" if answer == str(expected).strip() else "incorrect"
            if outcome == "correct":
                reward += info["reward"]
        stats[kind][outcome] += 1
        stats["ALL"][outcome] += 1
        rows.append((task_id, outcome, answer, expected))
    for task_id, outcome, answer, expected in rows:
        if outcome not in ("correct", "infeasible_skipped"):
            print(f"  {task_id:<22} {outcome:<20} got={answer!r} expected={expected!r}")
    for kind in sorted(stats):
        print(f"{kind:<14} " + "  ".join(f"{k}={v}" for k, v in sorted(stats[kind].items())))
    print(f"REWARD {reward}/{total_reward}")
    if args.summary and args.summary.is_file():
        s = json.loads(args.summary.read_text())
        s = s.get("summary", s)
        print(f"TOKENS {s.get('tokens_spent')} calls={s.get('llm_calls')} elapsed={s.get('elapsed_s')}s "
              f"estimated={s.get('estimated_tokens')}")


if __name__ == "__main__":
    main()
