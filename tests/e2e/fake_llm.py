"""Deterministic OpenAI-compatible fake LLM for end-to-end runs (stdlib only, no network needed).

The reply is picked by the first rule of replies.json whose "match" string occurs in the user
message; the n-th reply of the rule is used on the n-th turn of a conversation (repairs).
Like the real proxy, it reports usage, and it can misbehave on purpose:

  FAKE_STATUS=402       every request fails with this status ("budget exhausted")
  FAKE_429_EVERY=N      every N-th request gets 429 key-parallel with Retry-After: 1
  FAKE_MAX_PARALLEL=N   a request arriving while N are already in flight gets 429 (like a parallel limit)
  FAKE_DELAY=0.2        seconds each request takes

Counters (requests, statuses, peak parallel requests) are written to $FAKE_STATS after every request.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "replies.json"), encoding="utf-8") as fh:
    RULES = json.load(fh)
STATUS = int(os.environ.get("FAKE_STATUS", "200"))
EVERY_429 = int(os.environ.get("FAKE_429_EVERY", "0"))
MAX_PARALLEL = int(os.environ.get("FAKE_MAX_PARALLEL", "0"))
DELAY = float(os.environ.get("FAKE_DELAY", "0.2"))
STATS = os.environ.get("FAKE_STATS", "/tmp/fake_llm_stats.json")

_lock = threading.Lock()
_stats = {"requests": 0, "statuses": {}, "in_flight": 0, "peak_in_flight": 0, "models": {}, "per_rule": {}}


def _save() -> None:
    with open(STATS + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(_stats, fh)
    os.replace(STATS + ".tmp", STATS)


def reply_for(messages: list[dict]) -> tuple[str, str]:
    """(rule match string or "", reply text)."""
    user = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
    turn = sum(1 for m in messages if m.get("role") == "assistant")
    for rule in RULES:
        if rule["match"] in user:
            replies = rule["replies"]
            return rule["match"], replies[min(turn, len(replies) - 1)]
    return "", "UNSOLVABLE"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, status: int, body: dict, headers: dict | None = None) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("health"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        with _lock:
            _stats["requests"] += 1
            number = _stats["requests"]
            _stats["in_flight"] += 1
            busy = _stats["in_flight"]
            _stats["peak_in_flight"] = max(_stats["peak_in_flight"], busy)
            model = body.get("model", "?")
            _stats["models"][model] = _stats["models"].get(model, 0) + 1
        status, headers = 200, {}
        try:
            time.sleep(DELAY)
            if STATUS != 200:
                status = STATUS
                self._send(status, {"error": {"message": "budget exhausted", "code": str(status)}})
                return
            if (EVERY_429 and number % EVERY_429 == 0) or (MAX_PARALLEL and busy > MAX_PARALLEL):
                status, headers = 429, {"Retry-After": "1", "X-Window": "key-parallel"}
                self._send(status, {"error": {"message": "too many parallel requests", "code": "429"}}, headers)
                return
            messages = body.get("messages") or []
            rule, text = reply_for(messages)
            with _lock:
                _stats["per_rule"][rule] = _stats["per_rule"].get(rule, 0) + 1
            prompt = sum(len(str(m.get("content") or "")) for m in messages) // 4 + 1
            self._send(200, {
                "id": f"fake-{number}", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": len(text) // 4 + 1},
            })
        finally:
            with _lock:
                _stats["in_flight"] -= 1
                key = str(status)
                _stats["statuses"][key] = _stats["statuses"].get(key, 0) + 1
                _save()


if __name__ == "__main__":
    _save()
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("FAKE_PORT", "8000"))), Handler).serve_forever()
