"""Run configuration from CLI arguments and environment.

Everything is optional except the tasks directory: the official launch passes only
``--tasks-dir`` plus ``LLM_BASE_URL``/``LLM_API_KEY``, so no other setting may be required.
Unknown CLI arguments are ignored on purpose (the organizers may add some).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

MODELS = (
    "qwen3.8-27b",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-noreason",
)

# Chosen on 130 graded tasks (see README "Модели"): cheapest model with ~98-100% accuracy.
DEFAULT_MODEL = "qwen3.8-27b-noreason"
DEFAULT_SECOND_MODEL = "qwen3.8-27b"  # reasoning: strongest, best at spotting infeasible tasks


def is_reasoning(model: str) -> bool:
    """Reasoning variants spend part of max_tokens on (billed) reasoning before the answer."""
    return not model.endswith("-noreason")


def _env_float(env: dict, name: str, default: float | None) -> float | None:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(env: dict, name: str, default: int) -> int:
    value = _env_float(env, name, None)
    return int(value) if value is not None else default


def _model(env: dict, name: str, default: str | None) -> str | None:
    raw = env.get(name, "").strip()
    if raw.lower() in ("none", "off", "0"):
        return None
    return raw if raw in MODELS else default


@dataclass
class Config:
    tasks_dir: Path
    out: Path | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    # Second opinion for tasks the main model failed (second pass, only if budget remains).
    second_model: str | None = DEFAULT_SECOND_MODEL
    budget: float | None = None  # competition tokens; None = unknown (the proxy enforces it)
    concurrency: int = 4
    deadline_s: float = 3300.0
    llm_timeout_s: float = 120.0
    max_repairs: int = 1
    coding_repairs: int = 2
    sandbox_timeout_s: float = 20.0
    log_dir: Path | None = None
    second_pass: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.base_url and self.api_key)


def parse_config(argv: list[str] | None = None, environ: dict | None = None) -> Config:
    env = dict(os.environ if environ is None else environ)
    parser = argparse.ArgumentParser(description="Astra v2 harness")
    parser.add_argument("--tasks-dir", "--tasks", dest="tasks_dir")
    parser.add_argument("--out", "--output", dest="out")
    parser.add_argument("--log-dir", dest="log_dir")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--deadline-s", type=float)
    parser.add_argument("--model")
    parser.add_argument("positional", nargs="*")
    args, _unknown = parser.parse_known_args(argv)

    tasks = args.tasks_dir or env.get("TASKS") or (args.positional[0] if args.positional else None)
    out = args.out or env.get("OUT") or None
    log_dir = args.log_dir or env.get("AGENT_LOG_DIR") or None
    model = args.model if args.model in MODELS else _model(env, "AGENT_MODEL", DEFAULT_MODEL)
    concurrency = args.concurrency or _env_int(env, "AGENT_CONCURRENCY", 4)
    deadline = args.deadline_s or _env_float(env, "AGENT_DEADLINE_S", 3300.0)
    return Config(
        tasks_dir=Path(tasks or "/data/tasks"),
        out=Path(out) if out else None,
        base_url=(env.get("LLM_BASE_URL") or "").strip().rstrip("/") or None,
        api_key=(env.get("LLM_API_KEY") or "").strip() or None,
        model=model or DEFAULT_MODEL,
        second_model=_model(env, "AGENT_SECOND_MODEL", DEFAULT_SECOND_MODEL),
        budget=_env_float(env, "BUDGET", None),
        concurrency=max(1, min(int(concurrency), 16)),
        deadline_s=max(30.0, min(float(deadline or 3300.0), 3550.0)),
        llm_timeout_s=_env_float(env, "AGENT_LLM_TIMEOUT_S", 120.0) or 120.0,
        log_dir=Path(log_dir) if log_dir else None,
        second_pass=env.get("AGENT_SECOND_PASS", "1") != "0",
    )
