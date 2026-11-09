"""Isolated execution of model-written code.

* every attempt works on a temporary COPY of the task workspace, so inputs never change
  (coding fixes are copied back explicitly, only after the tests pass);
* the child gets a scrubbed environment: no LLM key, no proxy address;
* wall-clock timeout kills the whole process group; memory/CPU/file-size rlimits apply.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX development hosts
    resource = None  # type: ignore[assignment]

MAX_OUTPUT = 64_000
LIB_DIR = Path(__file__).resolve().parent / "sandbox_lib"  # agentlib helpers for scripts
MAX_COPY_BYTES = 512 * 1024 * 1024
_SKIP_NAMES = {"task.json", "answer.txt", "__pycache__", ".agent_solve.py"}


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def clean_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(home),
        "TMPDIR": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }


def _limits(cpu_s: int) -> None:  # runs in the child before exec
    if resource is None:
        return
    for name, value in (
        ("RLIMIT_AS", 4 * 1024**3),
        ("RLIMIT_CPU", cpu_s),
        ("RLIMIT_FSIZE", 256 * 1024**2),
        ("RLIMIT_CORE", 0),
    ):
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def make_sandbox(task_dir: Path, workspace: Path, *, parent: Path | None = None) -> Path:
    """Copy the task's working files into a fresh temp dir.

    Inside the copy, ``workspace`` is a symlink to ``.`` so both "sales.csv" and
    "workspace/sales.csv" (two styles seen in instructions) resolve to the same file.
    """
    root = Path(tempfile.mkdtemp(prefix="astra-", dir=str(parent) if parent else None))
    if workspace.is_dir() and _dir_size(workspace) <= MAX_COPY_BYTES:
        for item in workspace.iterdir():
            if item.name in _SKIP_NAMES:
                continue
            target = root / item.name
            if item.is_dir() and not item.is_symlink():
                shutil.copytree(item, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
            elif item.is_file():
                shutil.copy2(item, target)
    if workspace != task_dir and not (root / "workspace").exists():
        try:
            (root / "workspace").symlink_to(".", target_is_directory=True)
        except OSError:
            pass
    return root


def remove_sandbox(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


async def run_process(
    argv: list[str],
    cwd: Path,
    *,
    timeout: float,
    stdin: bytes | None = None,
    extra_env: dict[str, str] | None = None,
) -> ExecResult:
    env = clean_env(cwd)
    env.update(extra_env or {})
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=env,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        preexec_fn=(lambda: _limits(int(timeout) + 5)) if resource is not None else None,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
        timed_out = False
    except (asyncio.TimeoutError, TimeoutError):
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=5)
        except (asyncio.TimeoutError, TimeoutError):
            out, err = b"", b""
    finally:
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return ExecResult(
        returncode=proc.returncode if proc.returncode is not None else -9,
        stdout=out[-MAX_OUTPUT:].decode("utf-8", "replace"),
        stderr=(err[-MAX_OUTPUT:].decode("utf-8", "replace") + ("\n[timeout]" if timed_out else "")),
        timed_out=timed_out,
    )


async def run_python(code: str, cwd: Path, *, timeout: float = 20.0) -> ExecResult:
    """Run a script stored OUTSIDE the sandbox dir, with the sandbox as the cwd."""
    holder = Path(tempfile.mkdtemp(prefix="astra-code-"))
    script = holder / "solve.py"
    script.write_text(code, encoding="utf-8")
    try:
        # cwd on PYTHONPATH: a task may ask to use a module from its workspace.
        return await run_process(
            [sys.executable, "-B", str(script)], cwd, timeout=timeout, extra_env={"PYTHONPATH": f"{cwd}{os.pathsep}{LIB_DIR}"}
        )
    finally:
        shutil.rmtree(holder, ignore_errors=True)
