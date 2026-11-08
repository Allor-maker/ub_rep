"""A7 checks: intact A5 mandatory checks, computation and explicit review risk."""

import csv
import io
import re
import time
from dataclasses import replace
from decimal import Context, Decimal, Inexact, localcontext
from pathlib import PurePosixPath
from typing import Literal, cast
from uuid import uuid4

from nsu_budget_agent.contracts.models import (
    AttemptResult,
    CheckResult,
    TaskSpec,
    ToolCall,
    ToolContext,
    ValidationResult,
)
from nsu_budget_agent.contracts.ports import ToolExecutor
from nsu_budget_agent.orchestration.decision import DecisionStore, Proposal, State
from nsu_budget_agent.solvers.plans import plan
from nsu_budget_agent.solvers.sources import selected_sources

from .certification import independent_math
from .data_plan import _markdown_answer, _matches_original_text, data_question
from .instruction import instruction_matches
from .llm import FallbackValidator, question
from .local import LocalValidator


def _csv_compute(text: str, args: dict, deadline: float) -> str:
    """Independent Decimal/DictReader recomputation of the proposed operation.

    This verifies computation, not natural-language instruction binding.
    Generic plans deliberately exclude distinct orders and declared row counts.
    """
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    header = reader.fieldnames
    columns, predicates, aggregate = args["columns"], args["filters"], args["aggregate"]
    arity = {"count": 0, "count_rows": 0, "count_distinct": 1, "sum": 1, "product_sum": 2}
    if (
        not header
        or any(not c for c in header)
        or len(header) != len(set(header))
        or not isinstance(aggregate, str)
        or aggregate not in arity
        or not isinstance(columns, list)
        or len(columns) != arity[aggregate]
        or any(c not in header for c in columns)
        or not isinstance(predicates, list)
        or len(predicates) > 16
    ):
        raise ValueError("independent CSV schema")
    for p in predicates:
        if (
            not isinstance(p, dict)
            or set(p) != {"column", "op", "value"}
            or p["column"] not in header
            or p["op"] not in ("eq", "ge", "le")
            or not isinstance(p["value"], str)
            or not p["value"].strip()
        ):
            raise ValueError("independent CSV filter")

    def number(raw: str) -> Decimal:
        if re.fullmatch(r"[+-]?\d{1,40}(?:\.\d{1,20})?", raw) is None:
            raise ValueError("independent CSV number")
        return Decimal(raw)

    if args.get("expected_columns") is not None and set(args["expected_columns"]) != set(header):
        raise ValueError("independent declared columns")
    decimal_context = Context(prec=256, Emin=-1024, Emax=1024)
    decimal_context.traps[Inexact] = True
    with localcontext(decimal_context):
        total = Decimal(0)
        identities = set()
        count = 0
        for count, row in enumerate(reader, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("independent computation deadline")
            if count > 100_000 or None in row or any(v is None for v in row.values()):
                raise ValueError("independent CSV shape")
            accepted = True
            for p in predicates:
                left, right = row[p["column"]], p["value"]
                if not left.strip():
                    raise ValueError("missing filter data")
                if p["op"] == "eq":
                    accepted &= left == right
                elif p["op"] == "ge":
                    accepted &= number(left) >= number(right)
                else:
                    accepted &= number(left) <= number(right)
            value = Decimal(1)
            if aggregate != "count_distinct":
                for column in columns:
                    value *= number(row[column])
            if accepted:
                if aggregate == "count_distinct":
                    identities.add(row[columns[0]])
                else:
                    total += value
        if args.get("expected_rows") is not None and args["expected_rows"] != count:
            raise ValueError("independent declared row count")
        if aggregate == "count_distinct":
            return str(len(identities))
        rendered = format(total, "f")
        return (rendered.rstrip("0").rstrip(".") if "." in rendered else rendered) if total else "0"


class DecisionValidator:
    def __init__(self, tools: ToolExecutor, store: DecisionStore) -> None:
        self.tools, self.store = tools, store
        self.local, self.fallback = LocalValidator(tools), FallbackValidator(tools)

    def for_run(self, store: DecisionStore) -> "DecisionValidator":
        return DecisionValidator(self.tools, store)

    async def _compute(self, task: TaskSpec, proposal: Proposal, ctx: ToolContext) -> bool:
        call, candidate = proposal.call, proposal.candidate
        if call is None:
            # A direct answer has no independently verified computation claim.
            return True
        if call.tool_name == "local_math":
            return not candidate.evidence_paths and candidate.text == independent_math(
                call.arguments, ctx.deadline_monotonic
            )
        documents: dict[PurePosixPath, str] = {}
        for artifact in selected_sources(task):
            request_id = uuid4().hex
            checked = await self.tools.execute(
                request_id,
                ToolCall(tool_name="read_file", arguments={"path": artifact.path.as_posix()}),
                ctx,
            )
            data = checked.data
            if (
                checked.call_id != request_id
                or checked.tool_name != "read_file"
                or checked.status != "ok"
                or checked.truncated
                or not isinstance(data, dict)
                or data.get("path") != artifact.path.as_posix()
                or not isinstance(data.get("text"), str)
                or not _matches_original_text(cast(str, data["text"]), artifact)
            ):
                raise ValueError("independent source unavailable or changed")
            documents[artifact.path] = cast(str, data["text"])
        if call.tool_name == "query_csv":
            paths = tuple(p for p in documents if p.as_posix() == call.arguments["path"])
            if len(paths) != 1:
                return False
            answer = _csv_compute(documents[paths[0]], call.arguments, ctx.deadline_monotonic)
            evidence: tuple[PurePosixPath, ...] = paths
        else:
            selected = {
                p: text
                for p, text in documents.items()
                if p.as_posix() in cast(list, call.arguments["paths"])
            }
            answer, evidence = _markdown_answer(selected, call.arguments, ctx.deadline_monotonic)
        return candidate.text == answer and set(candidate.evidence_paths) == set(evidence)

    async def validate(
        self, task: TaskSpec, result: AttemptResult, ctx: ToolContext
    ) -> ValidationResult:
        route = self.store.get(task)
        baseline = await self.local.validate(task, result, ctx)
        if (
            route.state != State.VALIDATION
            or route.proposal is None
            or route.proposal.candidate != result.candidate
        ):
            return replace(
                baseline,
                checks=(
                    *baseline.checks,
                    CheckResult(
                        check_id="router_provenance",
                        status="failed",
                        blocking=True,
                        message="candidate does not match active router proposal",
                    ),
                ),
                status="failed",
            )
        try:
            plan(task)
            known_local = True
        except ValueError:
            known_local = False
        if known_local:
            try:
                computed = (
                    route.proposal.call is not None
                    and instruction_matches(task, route.proposal.call)
                    and await self._compute(task, route.proposal, ctx)
                )
            except (
                ValueError,
                KeyError,
                TypeError,
                ArithmeticError,
                csv.Error,
                TimeoutError,
                SyntaxError,
            ):
                computed = False
            independent = CheckResult(
                check_id="independent_computation",
                status="passed" if computed else "failed",
                blocking=True,
                message="independent Decimal/AST or Markdown calculation against source hashes",
            )
            certified_checks = (*baseline.checks, independent)
            route.independent_semantics = all(c.status == "passed" for c in certified_checks)
            return replace(
                baseline,
                checks=certified_checks,
                status="passed" if route.independent_semantics else "failed",
            )
        if question(task) is not None or data_question(task) is not None:
            verified = await self.fallback.validate(task, result, ctx)
            route.independent_semantics = all(c.status == "passed" for c in verified.checks)
            return verified
        # Only this explicit policy can replace unsupported semantics; integrity
        # checks, visible tests, format, source/snapshot checks remain unchanged.
        checks = [c for c in baseline.checks if c.check_id not in ("supported_task", "semantics")]
        allowed = {a.path for a in selected_sources(task)}
        proposal = route.proposal
        approved = (
            route.approved_attempt == result.attempt_id
            and 0 < route.calls <= 3
            and not route.complete
            and set(proposal.candidate.evidence_paths).issubset(allowed)
            and (not allowed or bool(proposal.candidate.evidence_paths))
            and route.feedback is not None
            and route.feedback.verdict == "accept"
            and not route.feedback.missing_conditions
        )
        checks.append(
            CheckResult(
                check_id="review_policy",
                status="passed" if approved else "failed",
                blocking=True,
                message="bounded reviewed candidate; semantic correctness is not certified",
            )
        )
        if proposal.call is not None:
            try:
                computed = await self._compute(task, proposal, ctx)
            except (ValueError, KeyError, TypeError, ArithmeticError, csv.Error, TimeoutError):
                computed = False
            checks.append(
                CheckResult(
                    check_id="independent_computation",
                    status="passed" if computed else "failed",
                    blocking=True,
                    message="proposed computation independently checked against input hashes",
                )
            )
        checks.append(
            CheckResult(
                check_id="semantics",
                status="inconclusive",
                blocking=False,
                message="instruction binding relies on LLM review; false publication remains possible",
            )
        )
        status: Literal["passed", "failed", "inconclusive"] = (
            "failed" if any(c.status == "failed" for c in checks) else "inconclusive"
        )
        return replace(baseline, checks=tuple(checks), status=status)
