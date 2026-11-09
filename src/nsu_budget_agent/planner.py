"""A-priori token cost estimate and task ordering.

Estimate (competition tokens) = expected_calls * (prompt_tokens / 10 + expected_output).
prompt_tokens come from the actual prompt the solver would send (system + instruction +
file preview / sources), counted at ~3.2 chars per token. Output and call-count priors
per solver kind are calibrated on our trajectories (see README) and can be refit.

Order: zero-cost (fast path) first, then by reward / estimate, so if the hidden budget
runs out, the most valuable tokens were spent first.
"""

from __future__ import annotations

from .context import build_preview, list_files
from .solvers import coding, program
from .tasks import Task

CHARS_PER_TOKEN = 3.7
# kind -> (expected output tokens per call, expected calls per task)
# Fitted on 89 single-call successes (fresh50 + practice80): mean completion 69 / 63 / 126.
PRIORS = {
    "program": (75.0, 1.1),
    "program_kb": (70.0, 1.15),
    "coding": (130.0, 1.2),
}


def kind_of(task: Task) -> str:
    if task.is_coding:
        return "coding"
    md = [p for p in list_files(task.workspace) if p.suffix.lower() == ".md"]
    return "program_kb" if len(md) > 6 else "program"


def estimate(task: Task) -> float:
    kind = kind_of(task)
    if kind == "coding":
        chars = len(coding.SYSTEM) + len(task.instruction) + 3000
        chars += sum(min(p.stat().st_size, 40_000) for p in list_files(task.workspace) if p.suffix == ".py")
    else:
        root = task.workspace if task.has_workspace else task.dir
        chars = len(program.SYSTEM) + len(task.instruction) + len(build_preview(root)) + 80
    out, calls = PRIORS[kind]
    return round(calls * (chars / CHARS_PER_TOKEN / 10 + out), 1)


def order(tasks: list[Task], estimates: dict[str, float]) -> list[Task]:
    return sorted(tasks, key=lambda t: (-(t.reward / max(estimates.get(t.id, 1.0), 1.0)), -t.reward, t.id))
