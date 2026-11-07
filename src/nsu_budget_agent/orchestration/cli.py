"""Trusted composition entry point; defaults to offline production B-minimal."""

import argparse
import asyncio
import importlib
import json
import os
import re
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


async def _run(args, factory: ComponentFactory) -> int:
    from typing import get_args

    loop = asyncio.get_running_loop()
    units = budget_from_env(unit=args.budget_unit)
    if args.timeout_s <= 0 or args.timeout_s > 3600:
        raise ContractError("timeout must be in (0, 3600]")
    config = RunConfig(
        total_budget_units=units,
        deadline_monotonic=loop.time() + args.timeout_s,
        attempt_limits=AttemptLimits(
            max_cost_units=units,
            max_llm_calls=4,
            max_steps=8,
            max_duration_s=min(30, args.timeout_s),
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
        accounting_mode="strict",
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
        print(
            json.dumps(
                {
                    "tasks": {p.task_id: p.status for p in report.progress},
                    "load_issues": len(report.loaded.issues),
                    "assessment_errors": len(report.assessment_errors),
                    "saved": sum(r.status == "saved" for r in report.receipts),
                    "limit_units": report.budget.limit_units,
                    "spent_units": report.budget.spent_units,
                    "held_units": report.budget.held_units,
                    "workflow_invocations": report.workflow_invocations,
                    "component_steps": report.component_steps,
                    "sdk": "openjiuwen 0.1.19",
                    "accounting_mode": config.accounting_mode,
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
        default="nsu_budget_agent.solvers.composition:build",
        help="trusted module:factory; defaults to offline B-minimal production components",
    )
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--timeout-s", type=float, default=3300)
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
