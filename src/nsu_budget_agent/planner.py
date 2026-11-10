"""A-priori token cost estimate and task ordering.

Estimate (competition tokens) = expected_calls * (prompt_tokens / 10 + expected_output).
prompt_tokens come from the actual prompt the solver would send (system + instruction +
file preview / sources), counted at ~3.2 chars per token. Output and call-count priors
per solver kind are calibrated on our trajectories (see README) and can be refit.

Order: zero-cost (fast path) first, then by reward / estimate, so if the hidden budget
runs out, the most valuable tokens were spent first.

The priors come from OUR task sets; the final set is new. ``Calibrator`` therefore corrects
the estimate online, per task type, from the tasks already solved in this very run (a
type that turns out 2x more expensive than its prior sinks in the queue). Offline
simulation on 10 graded sets: no change where the priors are right, +5..9 points of the
reachable reward where a set's cost profile differs (README, "Оценка расхода").
"""

from __future__ import annotations

import math

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


class Calibrator:
    """Per-type factor actual/estimate: geometric mean over finished tasks of the type, shrunk
    towards 1 by PRIOR_WEIGHT pseudo-observations (a few tasks cannot swing the order)."""

    PRIOR_WEIGHT = 3.0

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._log_sum: dict[str, float] = {}
        self._count: dict[str, int] = {}

    def factor(self, task_type: str) -> float:
        if not self.enabled:
            return 1.0
        n = self._count.get(task_type, 0)
        return math.exp(self._log_sum.get(task_type, 0.0) / (n + self.PRIOR_WEIGHT))

    def observe(self, task_type: str, estimate: float, actual: float) -> None:
        if not self.enabled or estimate <= 0 or actual <= 0:
            return
        self._log_sum[task_type] = self._log_sum.get(task_type, 0.0) + math.log(actual / estimate)
        self._count[task_type] = self._count.get(task_type, 0) + 1

    def snapshot(self) -> dict[str, dict]:
        return {t: {"factor": round(self.factor(t), 2), "tasks": n} for t, n in sorted(self._count.items())}
