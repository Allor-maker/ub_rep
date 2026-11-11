"""Build a "hidden bug" coding set from organizer-style tasks ("some tests fail, fix mylib.py").

For every task with >= 2 failing tests one failing test method is removed from the visible
test_mylib.py, so one bug is covered only by the hidden (full original) tests. Checks that the
agent fixes all bugs, not only the ones the visible tests catch.

Usage: tools/make_hiddenbug_set.py <out_dir> <source>...
  source: a set dir with tasks/ (+ optional answers.json, expected=None marks infeasible tasks)
          or a single task dir.
Writes <out_dir>/<task>/ (tasks for the agent) and <out_dir>_hidden/ (full tests + meta.json),
graded by tools/grade_hiddenbug.py. Task names get the set name as prefix (ids repeat across sets).
"""
import ast
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path


def failing_tests(ws: Path) -> list[str]:
    r = subprocess.run([sys.executable, "-m", "unittest", "test_mylib"], cwd=ws, capture_output=True, text=True)
    return sorted({line.split()[1] for line in r.stderr.splitlines() if line.startswith(("FAIL:", "ERROR:"))})


def sources(args: list[str]):
    for arg in map(Path, args):
        if (arg / "task.json").exists():
            yield arg.name, arg, True
            continue
        answers_file = arg / "answers.json"
        answers = json.loads(answers_file.read_text()) if answers_file.exists() else {}
        for task in sorted((arg / "tasks").glob("coding_*")):
            feasible = answers.get(task.name, {}).get("expected", "DONE") is not None
            yield f"{arg.name}_{task.name}", task, feasible


def main() -> None:
    out = Path(sys.argv[1])
    hidden = out.parent / (out.name + "_hidden")
    out.mkdir(parents=True, exist_ok=True)
    hidden.mkdir(exist_ok=True)
    rng = random.Random(1111)
    meta = {}
    for name, src, feasible in sorted(sources(sys.argv[2:])):  # name order fixes the random picks
        ws = src / "workspace"
        failing = failing_tests(ws)
        if len(failing) < 2:
            continue
        drop = rng.choice(failing)
        tree = ast.parse((ws / "test_mylib.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                node.body = [b for b in node.body if not (isinstance(b, ast.FunctionDef) and b.name == drop)]
        dst = out / name
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "answer.txt", ".DS_Store"))
        (dst / "workspace" / "test_mylib.py").write_text(ast.unparse(tree) + "\n")
        task = json.loads((dst / "task.json").read_text())
        task["task_id"] = name
        (dst / "task.json").write_text(json.dumps(task, indent=1, ensure_ascii=False))
        (hidden / name).mkdir(exist_ok=True)
        shutil.copy(ws / "test_mylib.py", hidden / name / "test_mylib.py")
        meta[name] = {"dropped": drop, "failing": failing, "reward": task["reward"], "feasible": feasible}
    (hidden / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(meta)} tasks ({sum(m['feasible'] for m in meta.values())} feasible) -> {out}")


if __name__ == "__main__":
    main()
