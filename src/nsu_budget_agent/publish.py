"""Write results immediately after each solved task.

* ``answer.txt`` goes next to ``task.json`` (organizers' answer, 2026-11-07 Q&A).
  An infeasible task gets NO file (also their answer).
* Files are 0644 and replaced atomically (temp file + rename), so a kill never leaves
  a half-written answer and a non-root checker can read them.
* If ``OUT`` / ``--out`` is given, an aggregate JSON {task_id: answer} is rewritten after
  every answer (``OUT/answers.json`` when OUT is a directory).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from .solvers.common import to_json_answer
from .tasks import Task


def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".astra-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Publisher:
    def __init__(self, out: Path | None = None) -> None:
        self.out = out
        self.answers: dict[str, str] = {}

    def _out_path(self) -> Path | None:
        if self.out is None:
            return None
        if self.out.is_dir() or str(self.out).endswith(("/", os.sep)):
            return self.out / "answers.json"
        return self.out

    def apply_files(self, task: Task, files: dict[str, str]) -> None:
        """Copy test-verified source edits into the real workspace."""
        root = task.workspace.resolve()
        for rel, content in files.items():
            target = (task.workspace / rel).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                continue
            atomic_write(target, content)

    def publish(self, task: Task, answer: str) -> bool:
        if not task.is_coding:
            answer = to_json_answer(answer, task.instruction)
        text = answer.strip() + "\n"
        ok = False
        for target in (task.dir / "answer.txt",):
            try:
                atomic_write(target, text)
                ok = True
            except OSError as exc:
                print(f"[astra] cannot write {target}: {exc}", file=sys.stderr)
        self.answers[task.id] = answer.strip()
        out = self._out_path()
        if out is not None:
            try:
                atomic_write(out, json.dumps(self.answers, ensure_ascii=False, indent=1))
            except OSError as exc:
                print(f"[astra] cannot write OUT {out}: {exc}", file=sys.stderr)
        return ok
