"""Coding tasks: make the provided tests pass by editing source files (never the tests).

Verification is free and objective: the tests run locally in a sandbox copy before and
after every model edit. Only a passing set of edits is copied back into the real
workspace, so a failed attempt never damages the task.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path, PurePosixPath

from ..config import is_reasoning
from ..context import list_files, read_text
from ..llm import LLMClient
from ..sandbox import make_sandbox, remove_sandbox, run_process
from ..tasks import Task
from ..trace import Trace
from .common import SolveResult, says_unsolvable

SYSTEM = """You fix bugs in Python source files so that the given tests pass.
Never edit test files. Keep the public API; make the minimal correct fix (general code, no special-casing of test inputs).
Reply ONLY with the complete corrected content of every file you change, each as:
### FILE: <relative path>
```python
<full file content>
```
If the tests cannot pass by editing the non-test files (e.g. contradictory tests), reply exactly UNSOLVABLE."""

_FILE_BLOCK = re.compile(r"###\s*FILE:\s*`?([^\s`]+)`?\s*\n```[a-zA-Z0-9]*[ \t]*\n(.*?)```", re.S)


def is_test_file(rel: str) -> bool:
    p = PurePosixPath(rel)
    return (
        p.name.startswith("test")
        or p.name.endswith("_test.py")
        or "tests" in p.parts[:-1]
        or p.name == "conftest.py"
    )


def failing_tests(output: str) -> frozenset[str]:
    """Names of failing tests (unittest/pytest style); used to detect a repair without progress."""
    names = re.findall(r"^(?:FAIL|ERROR): (\S+)", output, re.M)
    names += re.findall(r"^FAILED (\S+)", output, re.M)
    return frozenset(names)


def answer_word(instruction: str) -> str:
    m = re.search(r"write ONLY the word ['\"`]?([A-Za-z0-9_]+)", instruction)
    return m.group(1) if m else "DONE"


def test_command(task: Task, root: Path) -> list[str]:
    for snippet in re.findall(r"`([^`]+)`", task.instruction):
        try:
            argv = shlex.split(snippet)
        except ValueError:
            continue
        safe = not any(ch in snippet for ch in ";|&><$")
        if argv and argv[0] in ("python", "python3") and len(argv) >= 2 and safe:
            return [sys.executable, *argv[1:]]
    return [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test*.py"]


def _apply(blocks: list[tuple[str, str]], sandbox: Path) -> dict[str, str]:
    changed: dict[str, str] = {}
    root = sandbox.resolve()
    for rel, content in blocks:
        rel = rel.strip().lstrip("./") if not rel.startswith("../") else rel
        if rel.startswith("workspace/"):
            rel = rel[len("workspace/"):]
        target = (sandbox / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            continue
        if is_test_file(rel) or not rel.endswith(".py"):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        changed[rel] = content
    return changed


async def solve(
    task: Task,
    llm: LLMClient | None,
    model: str | None,
    trace: Trace,
    *,
    repairs: int = 2,
    timeout: float = 60.0,
    cost_cap: float = 2500.0,
) -> SolveResult:
    sandbox = make_sandbox(task.dir, task.workspace)
    try:
        argv = test_command(task, sandbox)
        result = await run_process(argv, sandbox, timeout=timeout)
        trace.exec("tests:initial", result)
        word = answer_word(task.instruction)
        if result.ok:
            return SolveResult("answer", word, reason="tests already pass", files={})
        if llm is None or model is None:
            return SolveResult("failed", reason="tests fail and no LLM")
        sources = []
        for path in list_files(sandbox, limit=200):
            rel = path.relative_to(sandbox).as_posix()
            if path.suffix == ".py" and path.stat().st_size <= 40_000:
                sources.append(f"### {rel}\n```python\n{read_text(path)}\n```")
        user = (
            f"TASK:\n{task.instruction}\n\nFILES:\n" + "\n".join(sources)
            + f"\n\nTEST OUTPUT ({' '.join(argv[1:])}):\n{(result.stdout + result.stderr)[-3000:]}"
        )
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        changed: dict[str, str] = {}
        start_cost = trace.cost
        previous_failures = failing_tests(result.stdout + result.stderr)
        for attempt in range(repairs + 1):
            if trace.cost - start_cost >= cost_cap:
                return SolveResult("failed", reason="per-task cost cap reached")
            reply = await llm.chat(task.id, model, messages, max_tokens=6000 if is_reasoning(model) else 2000)
            trace.llm("solve" if attempt == 0 else "repair", messages, reply)
            if says_unsolvable(reply.text):
                return SolveResult("infeasible", reason="model: UNSOLVABLE")
            blocks = _FILE_BLOCK.findall(reply.text)
            changed.update(_apply(blocks, sandbox))
            if not blocks:
                feedback = "No file blocks found. Use the exact '### FILE: <path>' + code block format."
            else:
                result = await run_process(argv, sandbox, timeout=timeout)
                trace.exec(f"tests:attempt{attempt + 1}", result)
                if result.ok:
                    return SolveResult("answer", word, files=changed)
                failures = failing_tests(result.stdout + result.stderr)
                if attempt > 0 and failures and failures == previous_failures:
                    # Same tests fail after a repair: likely contradictory tests; stop spending.
                    return SolveResult("failed", reason="no progress: same tests still fail")
                previous_failures = failures
                feedback = f"Tests still fail:\n{(result.stdout + result.stderr)[-2500:]}"
            messages = messages + [
                {"role": "assistant", "content": reply.text[-6000:]},
                {"role": "user", "content": feedback + "\nReply with the full corrected files in the same format."},
            ]
        return SolveResult("failed", reason="tests still fail after repairs")
    finally:
        remove_sandbox(sandbox)
