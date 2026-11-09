"""OpenAI-compatible chat client with retries and competition-token accounting.

Cost of a call = prompt_tokens / 10 + completion_tokens (reasoning tokens are part of
completion_tokens in OpenAI/vLLM semantics; if a proxy reports them separately and they
exceed completion_tokens, the larger value is charged).
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field

import httpx


class LLMUnavailable(RuntimeError):
    """Budget exhausted, key rejected or proxy dead: stop all further LLM work."""


class LLMError(RuntimeError):
    """This call failed; other calls may still succeed."""


@dataclass
class Reply:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish: str
    seconds: float
    estimated: bool = False

    @property
    def cost(self) -> float:
        return self.prompt_tokens / 10 + self.completion_tokens


@dataclass
class Ledger:
    budget: float | None = None
    spent: float = 0.0
    calls: int = 0
    per_task: dict[str, float] = field(default_factory=dict)

    def add(self, task_id: str, cost: float) -> None:
        self.spent += cost
        self.calls += 1
        self.per_task[task_id] = self.per_task.get(task_id, 0.0) + cost

    @property
    def remaining(self) -> float | None:
        return None if self.budget is None else self.budget - self.spent

    def can_afford(self, estimate: float) -> bool:
        return self.budget is None or self.spent + estimate <= self.budget


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 3)


_EXHAUSTED_WORDS = ("budget", "quota", "exhaust", "insufficient", "limit exceeded", "balance")


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        ledger: Ledger,
        timeout_s: float = 120.0,
        max_parallel: int = 4,
        deadline: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.ledger = ledger
        self.deadline = deadline  # time.monotonic() value; no request starts after it
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=15.0), transport=transport
        )
        self._sem = asyncio.Semaphore(max_parallel)
        self._consecutive_failures = 0
        self.dead_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.dead_reason is None

    def _die(self, reason: str) -> LLMUnavailable:
        if self.dead_reason is None:
            self.dead_reason = reason
            print(f"[astra] LLM disabled: {reason}", file=sys.stderr)
        return LLMUnavailable(reason)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        task_id: str,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> Reply:
        if self.dead_reason:
            raise LLMUnavailable(self.dead_reason)
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        last_error = "unknown"
        for attempt in range(6):
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise LLMError("deadline")
            async with self._sem:
                if self.dead_reason:
                    raise LLMUnavailable(self.dead_reason)
                started = time.monotonic()
                try:
                    response = await self._client.post(self.url, headers=self._headers, json=body)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = f"transport: {type(exc).__name__}"
                    response = None
            if response is not None:
                status = response.status_code
                if status == 200:
                    try:
                        reply = self._parse(response.json(), model, messages, time.monotonic() - started)
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        last_error = f"bad response: {exc}"
                    else:
                        self._consecutive_failures = 0
                        self.ledger.add(task_id, reply.cost)
                        return reply
                else:
                    text = response.text[:400].lower()
                    last_error = f"http {status}: {text[:160]}"
                    if status in (401, 402, 403) or (
                        status in (400, 429) and any(w in text for w in _EXHAUSTED_WORDS)
                        and "rate" not in text
                    ):
                        raise self._die(last_error)
                    if status not in (408, 409, 425, 429, 500, 502, 503, 504):
                        self._fail()
                        raise LLMError(last_error)
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            await asyncio.sleep(min(float(retry_after), 30.0))
                            continue
                        except ValueError:
                            pass
            await asyncio.sleep(min(2.0 * (attempt + 1), 15.0) + random.random())
        self._fail()
        raise LLMError(last_error)

    def _fail(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 8:
            raise self._die("too many consecutive failures")

    @staticmethod
    def _parse(data: dict, model: str, messages: list[dict], seconds: float) -> Reply:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usage = data.get("usage") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        estimated = prompt is None or completion is None
        if prompt is None:
            prompt = sum(_approx_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
        if completion is None:
            completion = _approx_tokens(text) + (_approx_tokens(reasoning) if reasoning else 0)
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens") or 0
        completion = max(int(completion), int(reasoning_tokens))
        return Reply(
            text=text,
            model=model,
            prompt_tokens=int(prompt),
            completion_tokens=int(completion),
            finish=str(choice.get("finish_reason") or ""),
            seconds=round(seconds, 2),
            estimated=estimated,
        )
