"""Create isolated smoke data, then exercise the real CLI locally or in Docker.

--source exports only public artifacts from an existing task set; no expected/gold
or old answers are copied. Without --source, six portable synthetic tasks are used.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.public_reference import public_reference  # noqa: E402

from nsu_budget_agent.task_io import DirectoryTaskLoader  # noqa: E402
from nsu_budget_agent.task_io.workspace import artifact_file, work_directory  # noqa: E402


def add_task(
    target: Path,
    task_id: str,
    instruction: str,
    kind: str | None,
    check: str | None,
    files: dict[str, bytes],
) -> Path:
    target.mkdir()
    (target / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "instruction": instruction,
                "reward": 1,
                "type": kind,
                "check": check,
            }
        ),
        encoding="utf-8",
    )
    workspace = target / "workspace"
    workspace.mkdir()
    for name, content in files.items():
        path = workspace.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return target


def portable_tasks(target: Path) -> dict[str, tuple[Path, str | None]]:
    cases = [
        ("math", "Compute (17-5)*3.", "math", "exact_int", {}, "36"),
        (
            "csv",
            "In account.csv, what is the sum of amount where label equals 'x'?",
            "data_analysis",
            "exact_int",
            {"account.csv": b"amount,label\n11,x\n99,y\n5,x\n"},
            "16",
        ),
        (
            "search",
            "In which city is the headquarters of Anne Example's employer?",
            "search",
            "exact_str",
            {
                "a.md": b"# Anne Example\nEmployer: Wider Company\n",
                "b.md": b"# Wider Company\nHeadquarters: Novel Town\n",
            },
            "Novel Town",
        ),
        (
            "missing",
            "In which year was Anne Example born?",
            "search",
            "exact_str",
            {"a.md": b"# Anne Example\nEmployer: Wider Company\n"},
            None,
        ),
        (
            "unknown",
            "A warehouse starts with 10 boxes. It loses some boxes. How many boxes are left?",
            "math",
            "exact_int",
            {},
            None,
        ),
        (
            "coding",
            "Fix the available program.",
            "coding",
            "unittest",
            {"main.py": b"raise ValueError()\n"},
            None,
        ),
    ]
    return {
        task_id: (add_task(target / task_id, task_id, instruction, kind, check, files), answer)
        for task_id, instruction, kind, check, files, answer in cases
    }


async def export_public(source: Path, target: Path) -> dict[str, tuple[Path, str | None]]:
    report = await DirectoryTaskLoader().load(source)
    if report.issues or not report.tasks:
        raise RuntimeError("source task directory is empty or has load issues")
    cases = {}
    for index, task in enumerate(report.tasks):
        raw_files = {}
        with work_directory(task) as directory:
            for artifact in task.artifacts:
                with artifact_file(directory, artifact.path) as descriptor:
                    chunks = []
                    while chunk := os.read(descriptor, 65_536):
                        chunks.append(chunk)
                    raw_files[artifact.path.as_posix()] = b"".join(chunks)
        # Only public fields and loader-allowed files are exported, not task.json/gold/expected.
        destination = add_task(
            target / f"task-{index:04}",
            task.task_id,
            task.instruction,
            task.type_hint,
            task.check_hint,
            raw_files,
        )
        public_text = {name: data.decode("utf-8-sig") for name, data in raw_files.items()}
        cases[task.task_id] = (destination, public_reference(task.instruction, public_text))
    return cases


def check_app_payload(image: str) -> None:
    code = (
        "from pathlib import Path; "
        "r=Path('/app'); "
        "assert (r/'run_agent.py').is_file(); "
        "assert (r/'src/nsu_budget_agent/solvers/composition.py').is_file(); "
        "assert not (r/'tests').exists(); "
        "assert not (r/'src/nsu_budget_agent/orchestration/sdk_probe.py').exists(); "
        "bad=[p for p in r.rglob('*') if p.name.lower() in ('tests','gold','expected','.env') "
        "or p.name.lower().startswith(('gold-','gold_','gold.','expected-','expected_','expected.')) "
        "or p.name.endswith(('.pyc','.md','.key','.pem'))]; "
        "assert not bad, bad; print('runtime application payload: clean')"
        "; s=Path('/opt/venv/lib/python3.11/site-packages'); "
        "badtests=[p for p in s.rglob('*') if (p.is_dir() and p.name in ('test','tests','__pycache__')) "
        "or (p.is_file() and (p.name.startswith('test_') and p.suffix=='.py' "
        "or p.name.endswith(('_test.py','.pyc'))))]; "
        "assert not badtests, badtests[:10]; print('runtime dependencies: no test modules or bytecode')"
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            "/tmp:rw,size=2g,mode=1777",
            "--entrypoint",
            "python",
            image,
            "-c",
            code,
        ],
        check=True,
        timeout=120,
    )


def execute(mode: str, image: str, tasks: Path, scratch: Path) -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("LLM_API_KEY", "LLM_BASE_URL")
    }
    env.update(
        BUDGET="0", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )
    container = "nsu-a6-smoke-" + uuid4().hex
    if mode == "docker":
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container,
            "--read-only",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--pids-limit",
            "512",
            "--memory",
            "4g",
            "--cpus",
            "4",
            "--tmpfs",
            "/tmp:rw,size=2g,mode=1777",
            "--mount",
            f"type=bind,src={tasks},dst=/data/tasks",
            "--env",
            "BUDGET=0",
        ]
        if os.name == "posix":
            command.extend(["--user", f"{getattr(os, 'getuid')()}:{getattr(os, 'getgid')()}"])
        command.extend([image, "python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"])
    else:
        env.update(
            HOME=str(scratch),
            TMPDIR=str(scratch),
            TEMP=str(scratch),
            TMP=str(scratch),
            PYTHONPATH=str(ROOT / "src"),
        )
        command = [sys.executable, str(ROOT / "run_agent.py"), "--tasks-dir", str(tasks)]
    try:
        completed = subprocess.run(
            command,
            cwd=scratch,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
    finally:
        if mode == "docker":
            # Only the unique container created by this invocation can be removed.
            subprocess.run(
                ["docker", "rm", "--force", container], capture_output=True, check=False, timeout=15
            )
    if completed.returncode:
        raise RuntimeError(f"CLI failed ({completed.returncode}): {completed.stderr[-3000:]}")
    summaries = [
        json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{"tasks":')
    ]
    if len(summaries) != 1:
        raise RuntimeError("missing or duplicate CLI summary")
    return summaries[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("host", "docker"), required=True)
    parser.add_argument("--image", default="nsu-budget-agent:a6-smoke")
    parser.add_argument(
        "--source", type=Path, help="existing public smoke task directory; never modified"
    )
    parser.add_argument("--work-parent", type=Path, default=ROOT if os.name == "nt" else None)
    args = parser.parse_args()
    if args.mode == "docker":
        check_app_payload(args.image)
    with tempfile.TemporaryDirectory(prefix="nsu-a6-smoke-", dir=args.work_parent) as temp:
        scratch = Path(temp).absolute()
        tasks = scratch / "tasks"
        tasks.mkdir()
        # A5 never puts staging inside the task tree; SDK-relative logs stay in scratch.
        cases = (
            asyncio.run(export_public(args.source.absolute(), tasks))
            if args.source
            else portable_tasks(tasks)
        )
        summary = execute(args.mode, args.image, tasks, scratch)
        solved = 0
        for task_id, (root, answer) in cases.items():
            path = root / "answer.txt"
            if answer is None:
                if path.exists():
                    raise AssertionError(f"unexpected answer for unsolved task {task_id}")
            else:
                if not path.is_file() or path.read_bytes() != answer.encode("utf-8"):
                    raise AssertionError(f"wrong or missing answer for task {task_id}")
                solved += 1
        if (
            summary["saved"] != solved
            or summary["limit_units"] != 0
            or summary["spent_units"] != 0
            or summary["held_units"] != 0
            or summary["accounting_mode"] != "strict"
            or summary["workflow_invocations"] < solved
        ):
            raise AssertionError("unexpected budget, publication or workflow summary")
        print(
            f"{args.mode} smoke: {solved}/{len(cases)} public-reference answers verified; "
            f"unsolved files absent; SDK workflows={summary['workflow_invocations']}; budget units=0"
        )


if __name__ == "__main__":
    main()
