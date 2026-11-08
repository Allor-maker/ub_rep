"""Opt-in sequential live diagnosis with one durable, fail-closed session journal.

Only public TaskSpecs/artifacts enter the child. Raw model JSON is local/ignored.
Reported counters are not a provider billing attestation. Never auto-retry.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import httpx
from tests.benchmarks.a73 import ROOT, export_public

from nsu_budget_agent.llm.competition import normalize_competition_usage
from nsu_budget_agent.solvers import composition

SESSION = ROOT / "experiments/runs/a74"
JOURNAL = SESSION / "journal.json"


def save(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_journal():
    return json.loads(JOURNAL.read_text(encoding="utf-8"))


class GuardedTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.inner = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request):
        journal = load_journal()
        wire = json.loads(request.content)
        input_estimate = len(request.content) + 1024 + 256 * len(wire["messages"])
        reserve = input_estimate + 10 * wire["max_tokens"]
        if (
            journal["held_units"]
            or journal["http"] >= 25
            or journal["spent_units"] + reserve > 100_000
            or not journal["quota_ok"]
        ):
            raise RuntimeError("cumulative live guard refused dispatch")
        journal["held_units"] = reserve
        journal["http"] += 1
        entry = {"http": journal["http"], "reserve": reserve, "run": os.environ["A74_RUN"]}
        journal["calls"].append(entry)
        save(JOURNAL, journal)  # Persist unknown outcome BEFORE physical dispatch.
        response = await self.inner.handle_async_request(request)
        await response.aread()
        try:
            body = response.json()
        except ValueError:
            body = None
        usage = normalize_competition_usage(
            body,
            request_id="diagnostic",
            attempt_id="diagnostic",
            model=wire["model"],
            success=response.status_code == 200,
        )
        entry.update(status=response.status_code, usage=asdict(usage))
        if usage.status == "exact":
            cost = usage.input_tokens + 10 * usage.output_tokens
            journal["spent_units"] += cost
            journal["held_units"] = 0
            entry["cost_units"] = cost
            if (
                cost > reserve
                or usage.input_tokens > input_estimate
                or usage.output_tokens > wire["max_tokens"]
            ):
                journal["quota_ok"] = False
        else:
            journal["quota_ok"] = False
        save(JOURNAL, journal)
        # No headers, URL, request environment or arbitrary HTTP body is logged.
        choices = body.get("choices", []) if isinstance(body, dict) else []
        content = choices[0].get("message", {}).get("content") if choices else None
        if isinstance(content, str):
            content = content.replace(os.environ["LLM_API_KEY"], "[REDACTED]")
        with (Path(os.environ["A74_RUN"]) / "model.jsonl").open("a", encoding="utf-8") as out:
            payload = json.loads(wire["messages"][-1]["content"])
            out.write(
                json.dumps(
                    {
                        "role": payload["role"],
                        "model": wire["model"],
                        "artifact_provenance": payload["artifact_provenance"],
                        "tool_schemas": payload.get("tool_schemas"),
                        "reply": content,
                    }
                )
                + "\n"
            )
        return response

    async def aclose(self):
        await self.inner.aclose()


class DiagnosticValidator:
    def __init__(self, delegate):
        self.delegate = delegate

    def for_run(self, store):
        return DiagnosticValidator(self.delegate.for_run(store))

    async def validate(self, task, result, ctx):
        checked = await self.delegate.validate(task, result, ctx)
        with (Path(os.environ["A74_RUN"]) / "validation.jsonl").open("a", encoding="utf-8") as out:
            out.write(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "attempt_id": result.attempt_id,
                        "validation": asdict(checked),
                    },
                    default=str,
                )
                + "\n"
            )
        return checked


def build(ledger, config):
    from dataclasses import replace

    components = composition.build_competition(ledger, config, transport=GuardedTransport())
    tools = components.tools
    original = tools.execute

    async def execute(call_id, call, ctx):
        result = await original(call_id, call, ctx)
        if call.tool_name != "read_file":
            with (Path(os.environ["A74_RUN"]) / "tools.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps({"call": asdict(call), "result": asdict(result)}) + "\n")
        return result

    tools.execute = execute
    if os.environ.get("A74_REPAIR") == "numeric_expression_only":
        from nsu_budget_agent.contracts.models import ChatMessage

        controller = components.decision_controller
        original_request = controller.make_request

        def make_request(task, route, role):
            request = original_request(task, route, role)
            if role == "solve":
                payload = json.loads(request.messages[-1].content)
                payload["feedback"] = {
                    "error_code": "invalid_expression_syntax",
                    "reason": "Previous local_math ToolCall failed: function calls are unavailable. "
                    "Correct only the ToolCall to numeric literals/operators matching the schema. "
                    "If you computed the final number, direct_answer is also independently checked. "
                    "Do not invent tools, functions or sources.",
                }
                return replace(
                    request,
                    messages=(
                        *request.messages[:-1],
                        ChatMessage(role="user", content=json.dumps(payload, sort_keys=True)),
                    ),
                )
            return request

        controller.make_request = make_request
    return replace(components, validator=DiagnosticValidator(components.validator))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repair-code", choices=("numeric_expression_only",))
    args = parser.parse_args()
    journal = load_journal()
    if journal["held_units"] or not journal["quota_ok"] or journal["http"] >= 25:
        raise RuntimeError("live session halted")
    runroot = SESSION / args.label
    runroot.mkdir()  # Refuse overwriting/replaying an earlier experiment.
    tasks = runroot / "tasks"
    tasks.mkdir()
    asyncio.run(export_public(ROOT.parent / "live-test/tasks", tasks, args.select.split(",")))
    from dotenv import dotenv_values

    configured = dotenv_values(ROOT.parent / "token-predictor/.env")
    env = dict(os.environ)
    for target, alias in (
        ("LLM_API_KEY", "NEURALDEEP_API_KEY"),
        ("LLM_BASE_URL", "NEURALDEEP_BASE_URL"),
    ):
        env[target] = env.get(target) or configured.get(target) or configured[alias]
    env.update(
        PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
        PYTHONIOENCODING="utf-8",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TEMP=str(runroot),
        TMP=str(runroot),
        TMPDIR=str(runroot),
        A74_RUN=str(runroot),
        LLM_MODE="competition",
        LLM_MODEL="qwen3.6-35b-a3b-noreason",
        LLM_OUTPUT_CAP="512",
        BUDGET=str((100_000 - journal["spent_units"]) // 10),
    )
    env.pop("A74_REPAIR", None)
    if args.repair_code:
        if journal.get("numeric_expression_repair_used"):
            raise RuntimeError("the single syntax repair was already used")
        journal["numeric_expression_repair_used"] = True
        save(JOURNAL, journal)
        env["A74_REPAIR"] = args.repair_code
    command = [
        sys.executable,
        str(ROOT / "run_agent.py"),
        "--tasks-dir",
        str(tasks),
        "--components",
        "tests.benchmarks.a74:build",
        "--llm-mode",
        "competition",
        "--timeout-s",
        "240",
        "--attempt-timeout-s",
        "60",
    ]
    with (runroot / "agent.log").open("w", encoding="utf-8") as out:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=300
        )
    summaries = [
        json.loads(line)
        for line in (runroot / "agent.log").read_text(encoding="utf-8").splitlines()
        if line.startswith('{"tasks":')
    ]
    if not summaries:
        raise RuntimeError("missing summary; inspect local log before continuing")
    save(runroot / "summary.json", summaries[-1])
    if summaries[-1]["held_units"]:
        journal = load_journal()
        journal["held_units"] = max(journal["held_units"], summaries[-1]["held_units"])
        journal["quota_ok"] = False
        save(JOURNAL, journal)
    grade = subprocess.run(
        [sys.executable, str(ROOT.parent / "live-test/grade.py"), "--tasks", str(tasks)],
        cwd=ROOT.parent / "live-test",
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    (runroot / "grade.log").write_text(grade.stdout + grade.stderr, encoding="utf-8")
    print(
        json.dumps(
            {
                "run": args.label,
                "exit": result.returncode,
                "summary": summaries[-1],
                "journal": load_journal(),
            },
            ensure_ascii=True,
        )
    )
    print(grade.stdout)


if __name__ == "__main__":
    lock = SESSION / "session.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        run()
    finally:
        os.close(descriptor)
        lock.unlink()
