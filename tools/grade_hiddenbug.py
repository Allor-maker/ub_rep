"""Grade a run of a set built by tools/make_hiddenbug_set.py: the agent's mylib.py is run
against the full original tests (feasible tasks only; answer must be DONE).

Usage: tools/grade_hiddenbug.py <run_tasks_dir> <hidden_dir>
  e.g. tools/grade_hiddenbug.py experiments/runs/hb-x/tasks ../datasets/hiddenbug_coding_hidden
Runs model-written code: use on local test sets only.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    run, hidden = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads((hidden / "meta.json").read_text())
    feasible = {tid: m for tid, m in sorted(meta.items()) if m.get("feasible", True)}
    ok = got = total = 0
    for tid, m in feasible.items():
        total += m["reward"]
        answer_file = run / tid / "answer.txt"
        answer = answer_file.read_text().strip() if answer_file.exists() else ""
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(run / tid / "workspace" / "mylib.py", tmp)
            shutil.copy(hidden / tid / "test_mylib.py", tmp)
            r = subprocess.run([sys.executable, "-m", "unittest", "test_mylib"], cwd=tmp,
                               capture_output=True, text=True, timeout=60)
        fails = [line.split()[1] for line in r.stderr.splitlines() if line.startswith(("FAIL:", "ERROR:"))]
        good = r.returncode == 0 and answer == "DONE"
        ok += good
        got += m["reward"] * good
        print(f"{tid} answer={answer or '-':6} full_tests={'OK ' if r.returncode == 0 else 'FAIL'} "
              f"hidden_bug_fixed={m['dropped'] not in fails} {'' if good else fails}")
    print(f"correct {ok}/{len(feasible)}  reward {got}/{total}")


if __name__ == "__main__":
    main()
