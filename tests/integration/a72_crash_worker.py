"""Test-only real SDK child: parent kills it during a mocked subsequent HTTP."""

import json
import os
import sys
import time
from pathlib import Path

import httpx
from tests.integration.test_a71_router import direct, review

from nsu_budget_agent.orchestration.cli import main
from nsu_budget_agent.solvers import composition


def run():
    tasks, marker, staging = map(Path, sys.argv[1:])
    original = composition.build_competition

    async def deny_live(*args, **kwargs):
        raise AssertionError("real HTTP forbidden")

    httpx.AsyncHTTPTransport.handle_async_request = deny_live

    def configured(ledger, config, **kwargs):
        def deliver(request):
            assert ledger.snapshot().held_units > 0
            payload = json.loads(json.loads(request.content)["messages"][-1]["content"])
            if "crash task" in payload["instruction"]:
                assert (tasks / "accepted" / "answer.txt").read_bytes() == b"15"
                assert (tasks / "free" / "answer.txt").read_bytes() == b"5"
                with marker.open("w") as stream:
                    stream.write("second HTTP entered")
                    stream.flush()
                    os.fsync(stream.fileno())
                time.sleep(120)  # Parent uses SIGKILL/TerminateProcess, never normal shutdown.
                raise AssertionError("parent failed to kill worker")
            value = direct("15") if payload["role"] == "solve" else review()
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": json.dumps(value)}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 48,
                        "total_tokens": 168,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )

        kwargs["transport"] = httpx.MockTransport(deliver)
        return original(ledger, config, **kwargs)

    composition.build_competition = configured
    try:
        raise SystemExit(
            main(
                [
                    "--tasks-dir",
                    str(tasks),
                    "--staging-dir",
                    str(staging),
                    "--llm-mode",
                    "competition",
                ]
            )
        )
    finally:
        marker.with_suffix(".finally").write_text("normal cleanup ran")


if __name__ == "__main__":
    run()
