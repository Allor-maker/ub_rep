"""Explicit A7.5 experiments; real SDK/Gateway, public exports and durable budget.

Never opens a grader or answer key. Evaluation is a separate post-development
command. The default offline mode has zero budget and denies physical transport.
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import httpx
from tests.benchmarks.a73 import ROOT, export_public
from tests.benchmarks.a74 import DiagnosticValidator, save

from nsu_budget_agent.llm.competition import normalize_competition_usage
from nsu_budget_agent.solvers import composition

SESSION = ROOT / "experiments/runs/a75"
JOURNAL = SESSION / "journal.json"


def journal():
    return json.loads(JOURNAL.read_text(encoding="utf-8"))


def check_quota(environment):
    state = journal()
    state["http"] += 1
    entry = {"http": state["http"], "method": "GET /limits", "cost_units": 0}
    state["calls"].append(entry)
    state["quota_can_request"] = False
    save(JOURNAL, state)
    try:
        response = httpx.get(
            environment["LLM_BASE_URL"].rstrip("/") + "/limits",
            headers={"Authorization": "Bearer " + environment["LLM_API_KEY"]},
            timeout=15,
        )
        entry["status"] = response.status_code
        data = response.json()
        available = (
            response.status_code == 200
            and data["decision"]["can_request"] is True
            and data["chat"]["session"]["remaining"] > 0
            and data["chat"]["week"]["remaining"] > 0
            and environment["LLM_MODEL"] not in data.get("blocked_models", [])
        )
        entry["can_request"] = available
        state["quota_can_request"] = available
        if not available:
            state["closed"] = True
            state["stop_reason"] = "provider_quota_unavailable"
        save(JOURNAL, state)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        state["closed"] = True
        state["stop_reason"] = "provider_quota_unknown"
        save(JOURNAL, state)
        raise RuntimeError("provider quota unavailable before inference") from None
    if not available:
        raise RuntimeError("provider quota unavailable before inference")


class SessionTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.inner = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request):
        if os.environ.get("A75_MODE") == "offline":
            raise AssertionError("real HTTP forbidden in offline diagnosis")
        state = journal()
        wire = json.loads(request.content)
        estimate = len(request.content) + 1024 + 256 * len(wire["messages"])
        reserve = estimate + 10 * wire["max_tokens"]
        if (
            state["held_units"]
            or state["closed"]
            or not state.get("quota_can_request", False)
            or state["spent_units"] + reserve > 500000
        ):
            raise RuntimeError("cumulative A7.5 budget is unavailable")
        state["http"] += 1
        state["held_units"] = reserve
        payload = json.loads(wire["messages"][-1]["content"])
        entry = {
            "http": state["http"],
            "run": os.environ["A75_RUN"],
            "role": payload["role"],
            "method": "POST /chat/completions",
            "reserve_units": reserve,
            "input_estimate": estimate,
            "output_cap": wire["max_tokens"],
        }
        state["calls"].append(entry)
        save(JOURNAL, state)  # Durable unknown outcome before physical dispatch.
        response = await self.inner.handle_async_request(request)
        raw = bytearray()
        async for chunk in response.aiter_bytes():
            if len(raw) + len(chunk) > 1048576:
                raise RuntimeError("oversized provider response; durable usage remains unknown")
            raw.extend(chunk)
        await response.aclose()
        response = httpx.Response(
            response.status_code, headers=response.headers, content=bytes(raw)
        )
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
            entry["cost_units"] = cost
            state["spent_units"] += cost
            state["held_units"] = 0
            if usage.input_tokens > estimate or usage.output_tokens > wire["max_tokens"]:
                state["closed"] = True
                state["stop_reason"] = "reported_bound_breach"
        else:
            state["closed"] = True
            state["stop_reason"] = "unknown_usage"
        save(JOURNAL, state)
        choices = body.get("choices", []) if isinstance(body, dict) else []
        content = choices[0].get("message", {}).get("content") if choices else None
        if isinstance(content, str):
            content = content.replace(os.environ["LLM_API_KEY"], "[REDACTED]")
        with (Path(os.environ["A75_RUN"]) / "model.jsonl").open("a", encoding="utf-8") as out:
            out.write(
                json.dumps(
                    {
                        "role": payload["role"],
                        "model": wire["model"],
                        "instruction": payload["instruction"],
                        "reply": content,
                    }
                )
                + "\n"
            )
        return response

    async def aclose(self):
        await self.inner.aclose()


def build(ledger, config):
    from dataclasses import replace

    components = composition.build_competition(ledger, config, transport=SessionTransport())
    execute = components.tools.execute

    async def record(call_id, call, ctx):
        result = await execute(call_id, call, ctx)
        if call.tool_name != "read_file":
            with (Path(os.environ["A75_RUN"]) / "tools.jsonl").open("a", encoding="utf-8") as out:
                out.write(
                    json.dumps({"call": asdict(call), "result": asdict(result)}, default=str) + "\n"
                )
        return result

    components.tools.execute = record
    return replace(components, validator=DiagnosticValidator(components.validator))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--label", required=True)
    parser.add_argument("--select", default="")
    parser.add_argument("--budget-tokens", type=int, default=20000)
    parser.add_argument("--output-cap", type=int, default=512)
    args = parser.parse_args()
    SESSION.mkdir(parents=True, exist_ok=True)
    lock = SESSION / "session.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        if not JOURNAL.exists():
            save(
                JOURNAL,
                {
                    "limit_units": 500000,
                    "spent_units": 0,
                    "held_units": 0,
                    "http": 0,
                    "closed": False,
                    "calls": [],
                },
            )
        state = journal()
        if args.mode == "live" and (state["closed"] or state["held_units"]):
            raise RuntimeError("A7.5 live session halted")
        runroot = SESSION / args.label
        runroot.mkdir()  # Existing outputs are never deleted or overwritten.
        tasks = runroot / "tasks"
        tasks.mkdir()
        count = asyncio.run(
            export_public(
                ROOT.parent / "live-test/tasks2",
                tasks,
                args.select.split(",") if args.select else [],
            )
        )
        # The private evaluator hashes task.json bytes as well as artifacts.
        # Preserve its public bytes, after proving that no hidden fields were
        # introduced and that the loader's public export has identical JSON.
        for task_root in tasks.iterdir():
            original = ROOT.parent / "live-test/tasks2" / task_root.name / "task.json"
            public = original.read_bytes()
            document = json.loads(public.decode("utf-8-sig"))
            if set(document) != {
                "task_id",
                "task_type",
                "instruction",
                "reward",
                "check",
            } or document != json.loads((task_root / "task.json").read_text(encoding="utf-8")):
                raise RuntimeError("public manifest cannot be exported byte-exactly")
            (task_root / "task.json").write_bytes(public)
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("LLM_", "NEURALDEEP_", "A74_"))
        }
        env.update(
            PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
            PYTHONIOENCODING="utf-8",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TEMP=str(runroot),
            TMP=str(runroot),
            TMPDIR=str(runroot),
            A75_RUN=str(runroot),
            A74_RUN=str(runroot),
            A75_MODE=args.mode,
            LLM_MODE="competition",
            LLM_MODEL="qwen3.6-35b-a3b-noreason",
            LLM_OUTPUT_CAP=str(args.output_cap),
            LLM_REQUIRE_INDEPENDENT="1",
            LLM_LENGTH_RETRY="1",
        )
        if args.mode == "offline":
            env.update(
                LLM_API_KEY="offline-never-dispatched",
                LLM_BASE_URL="https://offline.invalid/v1",
                BUDGET="0",
            )
        else:
            from dotenv import dotenv_values

            configured = dotenv_values(ROOT.parent / "token-predictor/.env")
            for target, alias in (
                ("LLM_API_KEY", "NEURALDEEP_API_KEY"),
                ("LLM_BASE_URL", "NEURALDEEP_BASE_URL"),
            ):
                env[target] = os.environ.get(target) or configured.get(target) or configured[alias]
            remaining = (500000 - state["spent_units"]) // 10
            env["BUDGET"] = str(min(args.budget_tokens, remaining))
            check_quota(env)
        command = [
            sys.executable,
            str(ROOT / "run_agent.py"),
            "--tasks-dir",
            str(tasks),
            "--components",
            "tests.benchmarks.a75:build",
            "--llm-mode",
            "competition",
            "--mode",
            "portfolio",
            "--timeout-s",
            "900",
            "--attempt-timeout-s",
            "60",
        ]
        fingerprint = hashlib.sha256()
        for path in sorted((ROOT / "src/nsu_budget_agent").rglob("*.py")):
            fingerprint.update(path.relative_to(ROOT).as_posix().encode())
            fingerprint.update(path.read_bytes())
        save(
            runroot / "experiment.json",
            {
                "mode": args.mode,
                "task_count": count,
                "model": env["LLM_MODEL"],
                "budget_competition_tokens": env["BUDGET"],
                "output_cap": args.output_cap,
                "require_independent": True,
                "length_retry": True,
                "spent_before_units": state["spent_units"],
                "source_code_sha256": fingerprint.hexdigest(),
            },
        )
        with (runroot / "agent.log").open("w", encoding="utf-8") as out:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=960
            )
        summaries = [
            json.loads(line)
            for line in (runroot / "agent.log").read_text(encoding="utf-8").splitlines()
            if line.startswith('{"tasks":')
        ]
        if not summaries:
            raise RuntimeError("missing agent summary; inspect logs before continuing")
        summary = summaries[-1]
        save(runroot / "summary.json", summary)
        if args.mode == "live" and summary["held_units"]:
            state = journal()
            state["closed"] = True
            state["stop_reason"] = "gateway_unknown_usage"
            state["held_units"] = max(state["held_units"], summary["held_units"])
            save(JOURNAL, state)
        print(
            json.dumps(
                {
                    "run": args.label,
                    "mode": args.mode,
                    "task_count": count,
                    "exit": result.returncode,
                    "summary": summary["recovery_summary"],
                    "spent_units": summary["spent_units"],
                    "held_units": summary["held_units"],
                    "http": summary["http_dispatched"],
                    "cumulative_units": journal()["spent_units"],
                }
            )
        )
    finally:
        os.close(descriptor)
        lock.unlink()


if __name__ == "__main__":
    run()
