"""Per-task trajectory: every LLM call, sandbox execution and decision, for logs and README."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

_CLIP = 6000


def _clip(text: str, limit: int = _CLIP) -> str:
    return text if len(text) <= limit else text[: limit // 2] + "\n...[clipped]...\n" + text[-limit // 2 :]


@dataclass
class Trace:
    task_id: str
    events: list[dict] = field(default_factory=list)
    cost: float = 0.0
    calls: int = 0

    def llm(self, phase: str, messages: list[dict], reply) -> None:
        self.cost += reply.cost
        self.calls += 1
        self.events.append(
            {
                "t": round(time.time(), 2),
                "kind": "llm",
                "phase": phase,
                "model": reply.model,
                "prompt": [{"role": m["role"], "content": _clip(m["content"])} for m in messages[-1:]],
                "reply": _clip(reply.text),
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "cost": round(reply.cost, 1),
                "finish": reply.finish,
                "seconds": reply.seconds,
            }
        )

    def exec(self, what: str, result) -> None:
        self.events.append(
            {
                "t": round(time.time(), 2),
                "kind": "exec",
                "what": what,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "stdout": _clip(result.stdout, 1500),
                "stderr": _clip(result.stderr, 1500),
            }
        )

    def note(self, kind: str, **data) -> None:
        self.events.append({"t": round(time.time(), 2), "kind": kind, **data})
