"""Test helpers: temp task dirs and a scripted fake OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx


def make_task(root: Path, task_id: str, instruction: str, *, task_type: str = "math", reward: float = 1,
              check: str = "exact_int", files: dict[str, str] | None = None) -> Path:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(json.dumps({
        "task_id": task_id, "task_type": task_type, "instruction": instruction,
        "reward": reward, "expected": "SECRET-EXPECTED", "check": check,
    }))
    for rel, content in (files or {}).items():
        path = task_dir / "workspace" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return task_dir


def temp_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="agent-test-"))


class FakeLLM:
    """Answers chat completions with a handler(messages, model) -> str; records requests."""

    def __init__(self, handler, *, status: int = 200, usage: tuple[int, int] = (1000, 100)):
        self.handler, self.status, self.usage = handler, status, usage
        self.requests: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        def respond(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.requests.append(body)
            if self.status != 200:
                return httpx.Response(self.status, json={"error": {"message": "budget exhausted"}})
            text = self.handler(body["messages"], body["model"])
            return httpx.Response(200, json={
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": self.usage[0], "completion_tokens": self.usage[1]},
            })

        return httpx.MockTransport(respond)
