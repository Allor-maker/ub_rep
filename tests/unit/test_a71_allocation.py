"""100-task A4/A2 stress: reservation floors must preserve descending reward."""

import asyncio
import unittest
from dataclasses import replace
from pathlib import Path

import httpx
from tests.unit.test_a35_provider import ENV, MODELS, mock_compatibility

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.models import (
    AttemptLimits,
    RunConfig,
    TaskProgress,
    TaskSpec,
    ToolLimits,
    UsageRecord,
)
from nsu_budget_agent.planning.decision import DecisionPortfolioPlanner
from nsu_budget_agent.solvers.composition import build_with_router
from nsu_budget_agent.solvers.decision import State


class AllocationStressTests(unittest.IsolatedAsyncioTestCase):
    async def test_100_tasks_reservation_13312_keeps_top_25_and_all_rewards(self):
        deadline = asyncio.get_running_loop().time() + 300
        limits = ToolLimits(
            timeout_s=1,
            max_read_bytes=10000,
            max_output_bytes=10000,
            max_write_bytes=1000,
            max_process_memory_bytes=10000,
            max_process_count=1,
        )
        config = RunConfig(
            total_budget_units=1_000_000,
            deadline_monotonic=deadline,
            mode="portfolio",
            attempt_limits=AttemptLimits(
                max_cost_units=1_000_000,
                max_llm_calls=3,
                max_steps=8,
                max_duration_s=30,
                max_attempts_per_task=2,
                allowed_models=MODELS,
                tool_limits=limits,
            ),
        )
        ledger = BudgetLedger(limit_units=1_000_000, deadline_monotonic=deadline)
        components = build_with_router(
            ledger,
            config,
            compatibility=replace(mock_compatibility(), input_bound=lambda request, body: 8192),
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    AssertionError("no HTTP in planner/A2 stress")
                )
            ),
            environ=ENV,
        )
        self.addAsyncCleanup(components.gateway.aclose)
        tasks = tuple(
            TaskSpec(
                task_id=str(i),
                instruction="Give the result of adding eleven and four.",
                reward=100 - i,
                task_root=Path.cwd(),
                work_root=Path.cwd(),
                artifacts=(),
            )
            for i in range(100)
        )
        for task in tasks:
            components.decision_store.get(task).move(State.UNSOLVED, "completed free pass")
        offers = tuple(
            components.registry.get("generic").assess(t, components.tools.catalog()) for t in tasks
        )
        progress = tuple(
            TaskProgress(task_id=t.task_id, status="pending", attempted_solvers=("local_router",))
            for t in tasks
        )
        fallback = DecisionPortfolioPlanner(deterministic_solvers=("local_router",))
        initial = fallback.choose(
            tasks, offers, progress, ledger.snapshot(), config, asyncio.get_running_loop().time()
        )
        self.assertGreaterEqual(initial.budget.max_cost_units, 13312)
        order = []
        for index in range(100):
            decision = components.planner.choose(
                tasks,
                offers,
                progress,
                ledger.snapshot(),
                config,
                asyncio.get_running_loop().time(),
            )
            self.assertIsNotNone(decision)
            self.assertGreaterEqual(decision.budget.max_cost_units, 13312)
            order.append(int(decision.task_id))
            attempt_id = f"attempt{index}"
            ledger.open_attempt(attempt_id, decision.budget)
            for call in range(2):
                request_id = f"request{index}-{call}"
                ledger.reserve(request_id, attempt_id, MODELS[0], 8192, 512)
                ledger.mark_dispatched(request_id)
                ledger.record_usage(
                    UsageRecord(
                        request_id=request_id,
                        attempt_id=attempt_id,
                        model=MODELS[0],
                        status="exact",
                        input_tokens=120,
                        output_tokens=48,
                    )
                )
            ledger.close_attempt(attempt_id)
            progress = tuple(
                replace(p, status="submitted", attempted_solvers=("local_router", "generic"))
                if p.task_id == decision.task_id
                else p
                for p in progress
            )
        self.assertEqual(order, list(range(100)))
        self.assertEqual(ledger.snapshot().spent_units, 120_000)
        self.assertEqual(ledger.snapshot().held_units, 0)
        self.assertEqual(ledger.snapshot().calls_used, 200)
