"""Offline without a key; competition or explicitly reviewed strict A7.1 with a key."""

import argparse
import asyncio
import importlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from nsu_budget_agent.budget import BudgetLedger, budget_from_env
from nsu_budget_agent.contracts.models import (
    AttemptLimits,
    ContractError,
    ModelId,
    OperationFailed,
    RunConfig,
    ToolLimits,
)

from .runtime import AgentRunner, ComponentFactory, Components


def _reason_code(reason: str) -> str | None:
    known = {
        "review rejected or unresolved after bounded strategy": "review_rejected_or_unresolved",
        "insufficient calls or steps for mandatory route checks": "insufficient_calls_or_steps",
        "mandatory role or reservation unavailable": "mandatory_route_unavailable",
        "minimum route reservation unavailable": "minimum_reservation_unavailable",
        "generic evidence exceeds bounded text policy": "source_limit_exceeded",
        "rejected answer and plan have no new verifiable information": "rejected_candidate_repeated",
        "rejected answer cannot return without a new verifiable plan": "rejected_answer_repeated",
        "paid route cannot repeat without new information": "paid_route_repeated",
        "coding pipeline is unavailable": "coding_unavailable",
        "sensitive task source is forbidden": "sensitive_source_forbidden",
    }
    if reason in known:
        return known[reason]
    return reason if re.fullmatch(r"[a-z]+(?:_[a-z]+)+", reason) else None


async def _run(args, factory: ComponentFactory) -> int:
    from typing import get_args

    loop = asyncio.get_running_loop()
    units = budget_from_env(unit=args.budget_unit)
    llm_mode = (
        args.llm_mode
        or os.environ.get("LLM_MODE")
        or (
            "competition"
            if args.components == "nsu_budget_agent.solvers.composition:build_production"
            else "strict"
        )
    )
    if llm_mode not in ("strict", "competition"):
        raise ContractError("invalid LLM_MODE")
    competition = llm_mode == "competition" and bool(os.environ.get("LLM_API_KEY", "").strip())
    if args.timeout_s <= 0 or args.timeout_s > 3600:
        raise ContractError("timeout must be in (0, 3600]")
    if not 1 <= args.attempt_timeout_s <= 300:
        raise ContractError("attempt timeout must be in [1, 300]")
    config = RunConfig(
        total_budget_units=units,
        deadline_monotonic=loop.time() + args.timeout_s,
        attempt_limits=AttemptLimits(
            max_cost_units=units,
            max_llm_calls=4,
            max_steps=8,
            max_duration_s=min(args.attempt_timeout_s, args.timeout_s),
            max_attempts_per_task=2,
            allowed_models=get_args(ModelId),
            tool_limits=ToolLimits(
                timeout_s=5,
                max_read_bytes=1_048_576,
                max_output_bytes=1_048_576,
                max_write_bytes=1_048_576,
                max_process_memory_bytes=134_217_728,
                max_process_count=1,
            ),
        ),
        mode=args.mode,
        max_parallel_tasks=1,
        accounting_mode="best_effort" if competition else "strict",
    )
    ledger = BudgetLedger(limit_units=units, deadline_monotonic=config.deadline_monotonic)
    components = factory(ledger, config)
    if not isinstance(components, Components):
        raise ContractError("composition factory must return Components")
    runner = AgentRunner(
        config=config, ledger=ledger, components=components, staging_parent=args.staging_dir
    )
    try:
        report = await runner.run(args.tasks_dir)
        events = getattr(runner.components.gateway, "events", ())
        diagnostics = {}
        for progress in report.progress:
            attempts = [a for a in report.attempts if a.task_id == progress.task_id]
            details = []
            for attempt in attempts:
                snapshot = ledger.snapshot(attempt.attempt_id)
                details.append(
                    {
                        "attempt_id": attempt.attempt_id,
                        "solver": attempt.solver_id,
                        "status": attempt.status,
                        "error_code": attempt.error.code if attempt.error else None,
                        "spent_units": snapshot.spent_units,
                        "held_units": snapshot.held_units,
                        "reserved_calls": snapshot.calls_used,
                        "http_dispatched": sum(
                            e.phase == "dispatched" and e.attempt_id == attempt.attempt_id
                            for e in events
                        ),
                    }
                )
            reason = (
                "paid_blocked_unknown_usage"
                if progress.status == "deferred" and report.budget.held_units
                else None
            )
            if reason is None and progress.status == "deferred":
                reason = "insufficient_budget_or_deadline"
            diagnostics[progress.task_id] = {"reason_code": reason, "attempts": details}
        task_by_id = {t.task_id: t for t in report.loaded.tasks}
        attempted = {a.task_id for a in report.attempts}
        llm_tasks = {
            a.task_id
            for a in report.attempts
            if any(e.phase == "dispatched" and e.attempt_id == a.attempt_id for e in events)
        }
        published = {r.task_id for r in report.receipts if r.status in ("saved", "already_saved")}
        categories = {}
        final_reasons = {}
        for trace in report.decision_traces:
            codes = [
                code for _, reason in trace.events if (code := _reason_code(reason)) is not None
            ]
            if codes:
                final_reasons[trace.task_id] = codes[-1]
        skipped = Counter(
            final_reasons.get(tid) or diagnostics[tid]["reason_code"] or "unsupported_or_unselected"
            for tid in task_by_id
            if tid not in published
        )
        for category in sorted({t.type_hint or "unknown" for t in report.loaded.tasks}):
            ids = {tid for tid, t in task_by_id.items() if (t.type_hint or "unknown") == category}
            snapshots = [ledger.snapshot(a.attempt_id) for a in report.attempts if a.task_id in ids]
            categories[category] = {
                "published": len(published & ids),
                "llm_tasks": len(llm_tasks & ids),
                "spent_units": sum(s.spent_units for s in snapshots),
                "held_units": sum(s.held_units for s in snapshots),
                "competition_tokens": str(sum(s.spent_units for s in snapshots) // 10)
                + "."
                + str(sum(s.spent_units for s in snapshots) % 10),
            }
        print(
            json.dumps(
                {
                    "tasks": {p.task_id: p.status for p in report.progress},
                    "load_issues": len(report.loaded.issues),
                    "assessment_errors": len(report.assessment_errors),
                    "saved": sum(r.status == "saved" for r in report.receipts),
                    "recovery_summary": {
                        "discovered": len(task_by_id),
                        "processed": len(attempted),
                        "solved_without_llm": len(published - llm_tasks),
                        "llm_tasks": len(llm_tasks),
                        "published": len(published),
                        "categories": categories,
                        "truncated_replies": sum(
                            t.truncated_replies for t in report.decision_traces
                        ),
                        "remaining_units": report.budget.available_units,
                        "skip_reasons": dict(skipped),
                    },
                    "validation_diagnostics": [
                        {
                            "attempt_id": v.attempt_id,
                            "status": v.status,
                            "checks": {c.check_id: c.status for c in v.checks},
                        }
                        for v in report.validations
                    ],
                    "limit_units": report.budget.limit_units,
                    "spent_units": report.budget.spent_units,
                    "held_units": report.budget.held_units,
                    "workflow_invocations": report.workflow_invocations,
                    "component_steps": report.component_steps,
                    "decision_routes": {
                        trace.task_id: {
                            "state": trace.state.value,
                            "transitions": [state.value for state, _ in trace.events],
                            "llm_requests_attempted": trace.llm_requests_attempted,
                            "independent_semantics": trace.independent_semantics,
                            "truncated_replies": trace.truncated_replies,
                            "reviewer_verdict": trace.reviewer_verdict,
                            "source_count": trace.source_count,
                            "context_source_count": trace.context_source_count,
                            "tool_normalizations": list(trace.normalizations),
                            "reason_codes": [
                                code
                                for _, reason in trace.events
                                if (code := _reason_code(reason)) is not None
                            ],
                        }
                        for trace in report.decision_traces
                    },
                    "sdk": "openjiuwen 0.1.19",
                    "accounting_mode": config.accounting_mode,
                    "task_diagnostics": diagnostics,
                    "llm_events": [asdict(event) for event in events],
                    "http_dispatched": sum(e.phase == "dispatched" for e in events),
                    "attempt_timeout_s": config.attempt_limits.max_duration_s,
                    "llm_mode": "offline"
                    if all(
                        s.solver_id in components.deterministic_solvers
                        for s in components.registry.all_solvers()
                    )
                    else (
                        "competition"
                        if getattr(components.gateway, "competition_mode", False)
                        else "strict"
                    ),
                },
                ensure_ascii=True,
            )
        )
        return 0 if not report.loaded.issues else 1
    finally:
        await runner.aclose()


def main(argv: list[str] | None = None, *, factory: ComponentFactory | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A5-lite workflow harness; requires trusted solver composition"
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument(
        "--budget-unit",
        choices=("competition_tokens", "units_x10"),
        default="competition_tokens",
        help="BUDGET unit (default: competition_tokens); override if organizer specifies otherwise",
    )
    parser.add_argument(
        "--components",
        default="nsu_budget_agent.solvers.composition:build_production",
        help="trusted module:factory; defaults to offline without a key or competition A7.1",
    )
    parser.add_argument(
        "--llm-mode",
        choices=("strict", "competition"),
        help="overrides LLM_MODE; standard keyed startup defaults to competition",
    )
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--timeout-s", type=float, default=3300)
    parser.add_argument(
        "--attempt-timeout-s",
        type=float,
        default=60,
        help="bounded per-attempt workflow budget, including model calls and validation (default 60s)",
    )
    parser.add_argument("--mode", choices=("auto", "coverage", "portfolio"), default="auto")
    args = parser.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        if factory is None:
            if args.components is None or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", args.components
            ):
                raise ContractError(
                    "B components are not installed; supply a trusted --components module:factory"
                )
            module, name = args.components.split(":")
            factory = getattr(importlib.import_module(module), name)
        return asyncio.run(_run(args, factory))
    except (ContractError, ImportError, AttributeError):
        print(
            "Configuration error: BUDGET, budget units, SDK and trusted components must be configured"
        )
        return 2
    except OperationFailed:
        print("Run failed: task directory could not be loaded safely")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
