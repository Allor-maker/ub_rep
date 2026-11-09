"""Task discovery. Tolerant: a broken task is skipped, never fatal for the run."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Fields that may carry the reference answer in public examples. They never reach a prompt.
_HIDDEN_FIELDS = ("expected", "answer", "solution", "gold")


@dataclass
class Task:
    id: str
    type: str
    instruction: str
    reward: float
    check: str
    dir: Path
    meta: dict = field(default_factory=dict)

    @property
    def workspace(self) -> Path:
        """Directory the instruction calls "current": workspace/ if present, else the task dir."""
        ws = self.dir / "workspace"
        return ws if ws.is_dir() else self.dir

    @property
    def has_workspace(self) -> bool:
        return (self.dir / "workspace").is_dir()

    @property
    def is_coding(self) -> bool:
        return self.type == "coding" or self.check in ("unittest", "tests_pass", "pytest")


def _reward(value) -> float:
    try:
        r = float(value)
        return r if math.isfinite(r) and r >= 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def load_task(task_dir: Path) -> Task | None:
    try:
        raw = json.loads((task_dir / "task.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    instruction = raw.get("instruction") or raw.get("prompt") or raw.get("question")
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    meta = {k: v for k, v in raw.items() if k not in _HIDDEN_FIELDS}
    return Task(
        id=str(raw.get("task_id") or task_dir.name),
        type=str(raw.get("task_type") or raw.get("type") or "unknown").strip().lower(),
        instruction=instruction.strip(),
        reward=_reward(raw.get("reward", 1)),
        check=str(raw.get("check") or "").strip().lower(),
        dir=task_dir,
        meta=meta,
    )


def load_tasks(root: Path) -> list[Task]:
    tasks: list[Task] = []
    try:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        print(f"[astra] cannot list tasks dir {root}: {exc}", file=sys.stderr)
        return tasks
    if (root / "task.json").is_file():  # a single task passed directly
        candidates = [root]
    seen: set[str] = set()
    for task_dir in candidates:
        task = load_task(task_dir)
        if task is None:
            continue
        if task.id in seen:
            task.id = f"{task.id}@{task_dir.name}"
        seen.add(task.id)
        tasks.append(task)
    return tasks
