"""Synthetic B doubles. Public fixture instructions select test scenarios only."""

import asyncio
import os
import sys
from pathlib import PurePosixPath
from uuid import uuid4

from nsu_budget_agent.contracts.models import (
    CandidateAnswer,
    ChatMessage,
    CheckResult,
    ErrorInfo,
    LLMReply,
    LLMRequest,
    OperationFailed,
    SolverOffer,
    StopAttempt,
    ToolCall,
    ToolResult,
    ToolSpec,
    UsageRecord,
    ValidationResult,
)
from nsu_budget_agent.orchestration.runtime import Components
from nsu_budget_agent.planning import StaticSolverRegistry
from nsu_budget_agent.task_io._files import open_directory
from nsu_budget_agent.task_io.workspace import artifact_file, manifest, snapshot

MODEL = "qwen3.8-27b"


class FakeGateway:
    def __init__(self, ledger, *, unknown=False):
        self.ledger = ledger
        self.calls = 0
        self.unknown = unknown
        self.dispatched = asyncio.Event()

    def bind(self, attempt_id, step_index, budget):
        gateway = self

        class Session:
            async def complete(self, request):
                request_id = uuid4().hex
                gateway.ledger.reserve(request_id, attempt_id, request.model, 2, 2)
                gateway.ledger.mark_dispatched(request_id)
                gateway.calls += 1
                gateway.dispatched.set()
                if gateway.unknown:
                    try:
                        await asyncio.sleep(3600)
                    finally:
                        gateway.ledger.record_usage(
                            UsageRecord(
                                request_id=request_id,
                                attempt_id=attempt_id,
                                model=request.model,
                                status="unknown",
                                input_tokens=None,
                                output_tokens=None,
                            )
                        )
                usage = UsageRecord(
                    request_id=request_id,
                    attempt_id=attempt_id,
                    model=request.model,
                    status="exact",
                    input_tokens=2,
                    output_tokens=1,
                    reasoning_tokens=1,
                )
                gateway.ledger.record_usage(usage)
                return LLMReply(request_id=request_id, text="fixture", finish="stop", usage=usage)

        return Session()


class FakeSolver:
    solver_id = "generic"

    def __init__(self):
        self.steps = []

    def assess(self, task, tools):
        return SolverOffer(
            task_id=task.task_id,
            solver_id=self.solver_id,
            applicability="yes",
            required_tools=(),
            expected_cost_units=22 if task.instruction == "llm" else 0,
        )

    async def step(self, ctx, history):
        self.steps.append((ctx.task.task_id, ctx.step_index))
        mode = ctx.task.instruction
        if mode == "error":
            raise RuntimeError("fixture error")
        if mode == "refuse":
            return StopAttempt(status="no_progress", reason="fixture refusal")
        if mode == "unsupported":
            return StopAttempt(status="unsupported", reason="fixture unsupported")
        if mode == "no_answer":
            return ToolCall(tool_name="read", arguments={"path": "input.txt"})
        if not history:
            if mode == "llm":
                await ctx.llm.complete(
                    LLMRequest(
                        model=MODEL,
                        messages=(ChatMessage(role="user", content="fixture"),),
                        max_output_tokens=2,
                    )
                )
            return ToolCall(tool_name="read", arguments={"path": "input.txt"})
        data = history[-1].result.data
        return CandidateAnswer(
            text="" if mode == "empty" else str(data), evidence_paths=(PurePosixPath("input.txt"),)
        )


class FakeTools:
    def catalog(self):
        return (
            ToolSpec(
                name="read",
                description="Fixture staging reader",
                input_schema={"type": "object"},
                output_schema={},
                mutates_files=False,
            ),
        )

    async def execute(self, call_id, call, ctx):
        path = PurePosixPath(call.arguments["path"])
        if path.is_absolute() or ".." in path.parts:
            raise OperationFailed(ErrorInfo(code="denied", message="unsafe fixture path"))
        with open_directory(ctx.workspace_root) as root, artifact_file(root, path) as descriptor:
            data = os.read(descriptor, ctx.limits.max_read_bytes + 1)
        if len(data) > ctx.limits.max_read_bytes:
            raise OperationFailed(ErrorInfo(code="resource_limit", message="fixture read limit"))
        return ToolResult(
            call_id=call_id, tool_name=call.tool_name, status="ok", data=data.decode("utf-8")
        )


class FakeValidator:
    def __init__(self, *, blocking=False, tamper=False):
        self.blocking, self.tamper = blocking, tamper
        self.roots = []

    async def validate(self, task, result, ctx):
        self.roots.append(ctx.workspace_root)
        artifacts = manifest(
            ctx.workspace_root, frozenset(ctx.protected_paths), max_bytes=1_048_576
        )
        valid = snapshot(result.candidate, artifacts) == result.snapshot_sha256
        original = {a.path: a.sha256 for a in task.artifacts}
        preserved = all(
            a.sha256 == original[a.path] for a in artifacts if a.path in ctx.protected_paths
        )
        checks = tuple(
            CheckResult(
                check_id=name,
                status="passed" if passed else "failed",
                blocking=True,
                message="fixture check",
            )
            for name, passed in (
                ("format", bool(result.candidate.text) and task.instruction != "reject"),
                ("paths", not result.changes),
                ("protected_files", preserved),
                ("snapshot", valid),
            )
        )
        if self.blocking:
            checks += (
                CheckResult(
                    check_id="visible_tests",
                    status="inconclusive",
                    blocking=True,
                    message="fixture unavailable",
                ),
            )
        # Validation gets a disposable copy: temporary files cannot affect the snapshot.
        (ctx.workspace_root / "validator-scratch.tmp").write_text("temporary", encoding="utf-8")
        if self.tamper:
            (ctx.workspace_root / "input.txt").write_text("modified copy", encoding="utf-8")
        checks += (
            CheckResult(
                check_id="semantics",
                status="inconclusive",
                blocking=False,
                message="fixture validation is not semantic verification",
            ),
        )
        status = "failed" if any(c.status == "failed" for c in checks) else "inconclusive"
        return ValidationResult(
            attempt_id=result.attempt_id,
            snapshot_sha256=result.snapshot_sha256,
            checks=checks,
            status=status,
        )


def build(ledger, config):
    from nsu_budget_agent.orchestration.sdk_probe import _offline_audit

    # Applies before SDK import in the CLI subprocess. Limited socket guard, not a sandbox.
    sys.addaudithook(_offline_audit)
    registry = StaticSolverRegistry()
    registry.register(FakeSolver())
    return Components(
        registry=registry,
        tools=FakeTools(),
        validator=FakeValidator(),
        gateway=FakeGateway(ledger),
        deterministic_solvers=("generic",),
    )
