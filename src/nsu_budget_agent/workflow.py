"""openJiuwen integration: every task phase runs as an openJiuwen Workflow.

A phase is a linear chain of components (e.g. ``solve -> publish``). Each component wraps
one async step of ours; step exceptions are captured inside the component, so the
framework never sees them. If openJiuwen itself is unavailable or fails, the remaining
steps run directly: a framework problem must never cost a solved task.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

Step = tuple[str, Callable[[], Awaitable[None]]]

_SDK: dict | None = None
_SDK_ERROR: str | None = None


def _sdk() -> dict | None:
    global _SDK, _SDK_ERROR
    if _SDK is not None or _SDK_ERROR is not None:
        return _SDK
    # openJiuwen cancels a whole workflow after 60 s by default (WORKFLOW_EXECUTE_TIMEOUT), even
    # with per-node timeout=-1: a slow LLM reply then lost the task. Our steps carry their own
    # timeouts (HTTP, sandbox, cost caps), so the framework-wide limit is switched off.
    os.environ.setdefault("WORKFLOW_EXECUTE_TIMEOUT", "-1")
    try:
        # openJiuwen binds console log handlers to sys.stdout at import: keep stdout clean.
        with contextlib.redirect_stdout(sys.stderr):
            from openjiuwen.core.workflow import (  # type: ignore[import-untyped]
                End,
                Start,
                Workflow,
                WorkflowCard,
                WorkflowComponent,
                WorkflowExecutionState,
                create_workflow_session,
            )
        quiet_framework_logs()
    except Exception as exc:  # noqa: BLE001 - any import problem disables the SDK path
        _SDK_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[astra] openJiuwen unavailable, running steps directly: {_SDK_ERROR}", file=sys.stderr)
        return None
    _SDK = {
        "End": End,
        "Start": Start,
        "Workflow": Workflow,
        "WorkflowCard": WorkflowCard,
        "WorkflowComponent": WorkflowComponent,
        "COMPLETED": WorkflowExecutionState.COMPLETED,
        "session": create_workflow_session,
    }
    return _SDK


def quiet_framework_logs() -> None:
    """Framework INFO chatter -> WARNING, console handlers -> stderr."""
    for logger in list(logging.root.manager.loggerDict.values()):
        if not isinstance(logger, logging.Logger):
            continue
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
                handler.setStream(sys.stderr)
        if logger.handlers and logger.level < logging.WARNING:
            logger.setLevel(logging.WARNING)


def sdk_status() -> str:
    return "openjiuwen" if _sdk() is not None else f"direct ({_SDK_ERROR})"


@dataclass
class PhaseRun:
    done: set[str] = field(default_factory=set)
    errors: dict[str, BaseException] = field(default_factory=dict)
    engine: str = "direct"


async def _call(run: PhaseRun, name: str, fn: Callable[[], Awaitable[None]]) -> None:
    if name in run.done:
        return
    run.done.add(name)
    try:
        await fn()
    except asyncio.CancelledError:
        # Cancelled by the framework mid-step (e.g. its timeout): let the direct fallback rerun it.
        run.done.discard(name)
        raise
    except Exception as exc:  # noqa: BLE001 - reported to the caller via run.errors
        run.errors[name] = exc


async def run_phase(name: str, steps: list[Step]) -> PhaseRun:
    run = PhaseRun()
    sdk = _sdk()
    if sdk is not None:
        try:
            base = sdk["WorkflowComponent"]

            class StepComponent(base):  # type: ignore[misc, valid-type]
                def __init__(self, step_name: str, fn: Callable[[], Awaitable[None]]):
                    super().__init__()
                    self.step_name, self.fn = step_name, fn

                async def invoke(self, inputs, session, context):
                    await _call(run, self.step_name, self.fn)
                    return {"tick": int((inputs or {}).get("tick", 0)) + 1}

            flow = sdk["Workflow"](
                card=sdk["WorkflowCard"](id=uuid.uuid4().hex, name=name, version="1")
            )
            flow.set_start_comp("start", sdk["Start"](), inputs_schema={"tick": "${tick}"})
            previous = "start"
            for step_name, fn in steps:
                flow.add_workflow_comp(
                    step_name,
                    StepComponent(step_name, fn),
                    max_retries=0,
                    timeout=-1.0,
                    inputs_schema={"tick": "${" + previous + ".tick}"},
                )
                flow.add_connection(previous, step_name)
                previous = step_name
            flow.set_end_comp("end", sdk["End"](), inputs_schema={"tick": "${" + previous + ".tick}"})
            flow.add_connection(previous, "end")
            output = await flow.invoke({"tick": 0}, sdk["session"]())
            if output.state == sdk["COMPLETED"]:
                run.engine = "openjiuwen"
        except Exception as exc:  # noqa: BLE001
            print(f"[astra] workflow '{name}' fell back to direct steps: {type(exc).__name__}", file=sys.stderr)
    for step_name, fn in steps:  # whatever the framework did not run
        await _call(run, step_name, fn)
    if sdk is not None:
        quiet_framework_logs()  # loggers created lazily during the first runs
    return run


async def shutdown() -> None:
    if _SDK is None:
        return
    try:
        from openjiuwen.core.runner import Runner  # type: ignore[import-untyped]

        await Runner.stop()
    except Exception:  # noqa: BLE001
        pass
