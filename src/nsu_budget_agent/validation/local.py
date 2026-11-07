"""Fail-closed answer-only validation with recomputation from public evidence."""

import re
from typing import Literal, cast
from uuid import uuid4

from nsu_budget_agent.contracts.models import (
    AttemptResult,
    CheckResult,
    ContractError,
    TaskSpec,
    ToolCall,
    ToolContext,
    ValidationResult,
    parse_rel_path,
)
from nsu_budget_agent.contracts.ports import ToolExecutor
from nsu_budget_agent.solvers.plans import plan
from nsu_budget_agent.task_io.workspace import joined_io, manifest, snapshot


class LocalValidator:
    def __init__(self, tools: ToolExecutor) -> None:
        self.tools = tools

    async def validate(
        self, task: TaskSpec, result: AttemptResult, ctx: ToolContext
    ) -> ValidationResult:
        if (
            result.status != "candidate"
            or result.candidate is None
            or result.snapshot_sha256 is None
        ):
            raise ContractError("validator requires a candidate snapshot")
        candidate = result.candidate
        checks = []

        def add(
            name: str,
            status: Literal["passed", "failed", "inconclusive"],
            message: str,
            *,
            blocking: bool = True,
        ) -> None:
            checks.append(
                CheckResult(check_id=name, status=status, blocking=blocking, message=message)
            )

        valid_format = (
            bool(candidate.text)
            and len(candidate.text.encode("utf-8")) <= ctx.limits.max_output_bytes
        )
        if task.check_hint == "exact_int":
            valid_format &= re.fullmatch(r"-?(?:0|[1-9]\d*)", candidate.text) is not None
        known_format = task.check_hint in (None, "exact_int", "exact_str")
        add(
            "format",
            ("passed" if valid_format else "failed") if known_format else "inconclusive",
            "explicit format checked" if known_format else "unsupported check hint",
        )
        try:
            protected = frozenset(ctx.protected_paths) | frozenset(
                a.path for a in task.artifacts if a.protected
            )
            artifacts = await joined_io(
                manifest,
                ctx.workspace_root,
                protected,
                max_bytes=min(ctx.limits.max_read_bytes, 8 * 1024 * 1024),
            )
            current, original = {a.path: a for a in artifacts}, {a.path: a for a in task.artifacts}
            paths_ok = (
                result.task_id == task.task_id
                and result.attempt_id == ctx.attempt_id
                and result.workspace_root == ctx.workspace_root
                and not result.changes
                and set(current) == set(original)
                and all(p in current for p in candidate.evidence_paths)
                and all(
                    (a.size_bytes, a.sha256) == (original[p].size_bytes, original[p].sha256)
                    for p, a in current.items()
                )
            )
            protection_ok = all(
                p in current and p in original and current[p].sha256 == original[p].sha256
                for p in protected
            )
            add(
                "paths",
                "passed" if paths_ok else "failed",
                "answer-only manifest and evidence checked",
            )
            add("protected_files", "passed" if protection_ok else "failed", "source hashes checked")
            add(
                "snapshot",
                "passed" if snapshot(candidate, artifacts) == result.snapshot_sha256 else "failed",
                "snapshot recomputed",
            )
        except Exception:
            for name in ("paths", "protected_files", "snapshot"):
                add(name, "inconclusive", "local integrity could not be established")
        try:
            parsed = plan(task)
        except ValueError:
            add("supported_task", "failed", "task has no supported deterministic interpretation")
            add("semantics", "inconclusive", "semantic verification unavailable", blocking=False)
        else:
            call_id = uuid4().hex
            verified = await self.tools.execute(call_id, parsed.call, ctx)
            data = verified.data
            if (
                verified.call_id != call_id
                or verified.tool_name != parsed.call.tool_name
                or verified.status != "ok"
                or verified.truncated
                or not isinstance(data, dict)
            ):
                add("semantics", "inconclusive", "deterministic verification could not complete")
            else:
                evidence = data.get("evidence_paths")
                required = (
                    {parse_rel_path(cast(str, p)) for p in evidence}
                    if isinstance(evidence, list) and all(isinstance(p, str) for p in evidence)
                    else None
                )
                correct = (
                    data.get("answer") == candidate.text
                    and required is not None
                    and required == set(candidate.evidence_paths)
                )
                add(
                    "semantics",
                    "passed" if correct else "failed",
                    "answer recomputed from the public instruction and evidence",
                )
            if parsed.call.tool_name == "query_csv":
                arguments = parsed.call.arguments
                call_id = uuid4().hex
                checked = await self.tools.execute(
                    call_id,
                    ToolCall(tool_name="audit_csv", arguments=arguments),
                    ctx,
                )
                data = checked.data
                if (
                    checked.call_id != call_id
                    or checked.tool_name != "audit_csv"
                    or checked.status != "ok"
                    or checked.truncated
                    or not isinstance(data, dict)
                    or type(data.get("valid")) is not bool
                    or (data.get("answer") is not None and not isinstance(data.get("answer"), str))
                    or (
                        data.get("valid") is True
                        and arguments.get("aggregate") in ("count", "count_rows", "count_distinct")
                        and not isinstance(data.get("answer"), str)
                    )
                ):
                    add("csv_independent", "inconclusive", "independent CSV audit unavailable")
                else:
                    expected_answer = data.get("answer")
                    correct = data["valid"] is True and (
                        expected_answer is None or expected_answer == candidate.text
                    )
                    add(
                        "csv_independent",
                        "passed" if correct else "failed",
                        "independent CSV completeness and order/row count checked",
                    )
        if task.check_hint == "unittest":
            add("visible_tests", "inconclusive", "isolated test execution is unavailable")
        status: Literal["passed", "failed", "inconclusive"] = (
            "failed"
            if any(c.status == "failed" for c in checks)
            else ("inconclusive" if any(c.status == "inconclusive" for c in checks) else "passed")
        )
        return ValidationResult(
            attempt_id=result.attempt_id,
            snapshot_sha256=result.snapshot_sha256,
            checks=tuple(checks),
            status=status,
        )
