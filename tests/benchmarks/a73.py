"""Isolated public-input replay/live harness; external grader runs after agent exit.

This is test orchestration, never a production solver or an answer-key reader.
"""

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

from nsu_budget_agent.solvers import composition
from nsu_budget_agent.task_io import DirectoryTaskLoader
from nsu_budget_agent.task_io.workspace import artifact_file, work_directory
from nsu_budget_agent.validation.data_plan import data_question
from nsu_budget_agent.validation.llm import question

ROOT = Path(__file__).resolve().parents[2]


def independent_math(spec):
    # Deliberately different computations from the verifier (sieve, brute LCM,
    # row enumeration and explicit permutations), using public parameters only.
    if spec.kind == "prime_sum":
        count = spec.values[0]
        flags = bytearray(b"\x01") * 100000
        flags[0:2] = b"\x00\x00"
        for n in range(2, math.isqrt(len(flags)) + 1):
            if flags[n]:
                for multiple in range(n * n, len(flags), n):
                    flags[multiple] = 0
        return str(sum([i for i, valid in enumerate(flags) if valid][:count]))
    if spec.kind == "lcm":
        candidate = max(spec.values)
        while any(candidate % value for value in spec.values):
            candidate += max(spec.values)
            if candidate > 1000000:
                raise ValueError("replay oracle bound")
        return str(candidate)
    if spec.kind == "series":
        count, first, delta = spec.values
        return str(sum(first + i * delta for i in range(count)))
    if spec.kind == "arrangements":
        import itertools

        size, count = spec.values
        return str(sum(1 for _ in itertools.permutations(range(size), count)))
    raise ValueError("no independent replay oracle")


def build_offline(ledger, config):
    async def deny(*args, **kwargs):
        raise AssertionError("live transport forbidden in offline replay")

    httpx.AsyncHTTPTransport.handle_async_request = deny
    tasks = None

    async def handler(request):
        nonlocal tasks
        if tasks is None:
            loaded = await DirectoryTaskLoader().load(Path(os.environ["ASTRA_REPLAY_TASKS"]))
            tasks = {t.instruction: t for t in loaded.tasks}
        payload = json.loads(json.loads(request.content)["messages"][-1]["content"])
        task = tasks[payload["instruction"]]
        if payload["role"] != "solve":
            result = {
                "verdict": "reject",
                "reason": "replay has no review oracle",
                "next_action": "skip",
                "missing_conditions": ["independent verification"],
            }
        elif (spec := data_question(task)) is not None:
            result = {
                "kind": "tool_plan",
                "tool_name": spec.call.tool_name,
                "arguments": spec.call.arguments,
            }
        elif (spec := question(task)) is not None:
            result = {
                "kind": "direct_answer",
                "answer": independent_math(spec),
                "evidence_paths": [],
            }
        else:
            result = {"kind": "decline", "reason": "no independent replay oracle"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(result)}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 48,
                    "total_tokens": 168,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    return composition.build_competition(ledger, config, transport=httpx.MockTransport(handler))


async def export_public(source, destination, select):
    loaded = await DirectoryTaskLoader().load(source)
    if loaded.issues or not loaded.tasks:
        raise ValueError("invalid public benchmark")
    chosen = [t for t in loaded.tasks if not select or t.task_id in select]
    if select and {t.task_id for t in chosen} != set(select):
        raise ValueError("requested public task missing")
    for task in chosen:
        root = destination / task.task_id
        root.mkdir()
        (root / "task.json").write_text(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "task_type": task.type_hint,
                    "instruction": task.instruction,
                    "reward": task.reward,
                    "check": task.check_hint,
                }
            ),
            encoding="utf-8",
        )
        workspace = root / "workspace"
        workspace.mkdir()
        with work_directory(task) as source_directory:
            for artifact in task.artifacts:
                target = workspace.joinpath(*artifact.path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with artifact_file(source_directory, artifact.path) as descriptor:
                    with target.open("wb") as stream:
                        while block := os.read(descriptor, 65536):
                            stream.write(block)
    return len(chosen)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("offline", "smoke", "full"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--grader", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--select", default="")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--offline-budget", type=int, default=5000)
    args = parser.parse_args()
    select = args.select.split(",") if args.select else []
    if args.mode == "smoke" and not 2 <= len(select) <= 3:
        raise ValueError("smoke requires two or three explicit public task IDs")
    if args.mode != "offline" and (not args.allow_live or args.credentials is None):
        raise ValueError("live mode requires explicit permission and configured credentials")
    record = Path(tempfile.gettempdir()) / "astra-a73-live-record.json"
    history = json.loads(record.read_text()) if record.exists() else {}
    if args.mode != "offline" and args.mode in history:
        raise ValueError("this authorized live stage was already attempted")
    if args.mode == "full" and not history.get("smoke", {}).get("successful"):
        raise ValueError("full stage requires a successful smoke")
    # Windows sandbox may hide ancestor identities under the user profile.
    # Keep anchored traversal checks intact and use the supported workspace.
    runs = ROOT / "experiments" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    runroot = Path(tempfile.mkdtemp(prefix="astra-a73-" + args.mode + "-", dir=runs))
    tasks = runroot / "tasks"
    tasks.mkdir()
    count = asyncio.run(export_public(args.source.absolute(), tasks, select))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_", "NEURALDEEP_"))}
    env.update(
        TEMP=str(runroot),
        TMP=str(runroot),
        TMPDIR=str(runroot),
        BUDGET=(
            str(args.offline_budget)
            if args.mode == "offline"
            else "1500"
            if args.mode == "smoke"
            else "5000"
        ),
        PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        LLM_MODE="competition",
        LLM_MODEL="qwen3.6-35b-a3b-noreason",
        LLM_OUTPUT_CAP="512",
        LLM_HTTP_TIMEOUT_S="45",
    )
    if args.mode == "offline":
        env.update(
            LLM_API_KEY="offline-never-live",
            LLM_BASE_URL="https://offline.invalid/v1",
            ASTRA_REPLAY_TASKS=str(tasks),
        )
    else:
        from dotenv import dotenv_values

        configured = dotenv_values(args.credentials)
        for target, alias in (
            ("LLM_API_KEY", "NEURALDEEP_API_KEY"),
            ("LLM_BASE_URL", "NEURALDEEP_BASE_URL"),
        ):
            value = configured.get(target) or configured.get(alias)
            if not value:
                raise ValueError("configured live credential is missing")
            env[target] = value
        history[args.mode] = {"runroot": str(runroot), "successful": False, "started": True}
        record.write_text(json.dumps(history), encoding="utf-8")
    timeout = 900 if args.mode == "full" else 240
    command = [
        sys.executable,
        str(ROOT / "run_agent.py"),
        "--tasks-dir",
        str(tasks),
        "--llm-mode",
        "competition",
        "--timeout-s",
        str(timeout),
        "--attempt-timeout-s",
        "60",
    ]
    if args.mode == "offline":
        command += ["--components", "tests.benchmarks.a73:build_offline"]
    with (runroot / "agent.log").open("w", encoding="utf-8") as stream:
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout + 60,
            )
        except subprocess.TimeoutExpired:
            print(
                json.dumps({"runroot": str(runroot), "outcome": "externally_stopped_unknown_usage"})
            )
            return 2
    lines = (runroot / "agent.log").read_text(encoding="utf-8", errors="replace").splitlines()
    summaries = [json.loads(line) for line in lines if line.startswith('{"tasks":')]
    if not summaries:
        print(
            json.dumps(
                {
                    "runroot": str(runroot),
                    "outcome": "missing_agent_summary",
                    "exit": result.returncode,
                }
            )
        )
        return 2
    summary = summaries[-1]
    (runroot / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # The external grader is a separate process after the agent has exited.
    graded = subprocess.run(
        [sys.executable, str(args.grader), "--tasks", str(tasks)],
        cwd=runroot,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (runroot / "grade.log").write_text(graded.stdout + graded.stderr, encoding="utf-8")
    import re

    match = re.search(r"CORRECT=(\d+)/\d+ INCORRECT=(\d+) MISSING=(\d+)", graded.stdout)
    correct, incorrect, missing = map(int, match.groups()) if match else (0, 0, count)
    # Grader denominator may be the full set even for a smoke subset.
    successful = (
        result.returncode == 0
        and summary["held_units"] == 0
        and correct == count
        and incorrect == 0
    )
    if args.mode != "offline":
        history[args.mode]["successful"] = successful
        record.write_text(json.dumps(history), encoding="utf-8")
    print(
        json.dumps(
            {
                "runroot": str(runroot),
                "mode": args.mode,
                "task_count": count,
                "correct": correct,
                "incorrect": incorrect,
                "missing_in_selected": count - correct - incorrect,
                "spent_units": summary["spent_units"],
                "held_units": summary["held_units"],
                "http_dispatched": summary.get("http_dispatched"),
                "successful": successful,
            }
        )
    )
    print(
        "\n".join(
            line
            for line in graded.stdout.splitlines()
            if line.startswith(("CORRECT=", "REWARD=", "math ", "search ", "data_analysis "))
        )
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(run())
