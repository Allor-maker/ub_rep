"""A4 policy and registry with fake B components, no attempts executed."""

import asyncio
import dataclasses
import itertools
import unittest
from pathlib import Path

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.models import (
    AttemptLimits,
    AttemptResult,
    BudgetSnapshot,
    ContractError,
    ErrorInfo,
    RunConfig,
    SolverOffer,
    TaskProgress,
    TaskSpec,
    ToolLimits,
    ToolSpec,
)
from nsu_budget_agent.planning import BaselinePlanner, StaticSolverRegistry, next_not_before_wakeup

MODEL = "qwen3.8-27b"
TOOLS = (
    ToolSpec(
        name="read",
        description="Fake file reader",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        mutates_files=False,
    ),
)
LIMITS = AttemptLimits(
    max_cost_units=600,
    max_llm_calls=3,
    max_steps=5,
    max_duration_s=20,
    allowed_models=(MODEL,),
    tool_limits=ToolLimits(
        timeout_s=1,
        max_read_bytes=100,
        max_output_bytes=100,
        max_write_bytes=100,
        max_process_memory_bytes=1000,
        max_process_count=1,
    ),
)
CONFIG = RunConfig(total_budget_units=1000, deadline_monotonic=100, attempt_limits=LIMITS)
SNAPSHOT = BudgetSnapshot(
    limit_units=1000, spent_units=100, held_units=100, calls_used=1, blocked=False
)


def task(task_id="task", reward=2, type_hint="math"):
    root = Path.cwd() / "nonexistent-input" / task_id
    return TaskSpec(
        task_id=task_id,
        instruction="Public task",
        reward=reward,
        task_root=root,
        work_root=root,
        artifacts=(),
        type_hint=type_hint,
    )


def offer(task_id="task", solver_id="generic", **changes):
    return SolverOffer(
        **{
            "task_id": task_id,
            "solver_id": solver_id,
            "applicability": "unknown",
            "required_tools": (),
            **changes,
        }
    )


class FakeSolver:
    def __init__(self, solver_id="generic", **assessment):
        self.solver_id = solver_id
        self.assessment = assessment

    def assess(self, task, tools):
        return offer(task.task_id, self.solver_id, **self.assessment)

    async def step(self, ctx, history):
        raise AssertionError("A4 must not execute solvers")


class PlannerTests(unittest.TestCase):
    def choose(
        self,
        *,
        tasks=None,
        offers=None,
        progress=(),
        budget=SNAPSHOT,
        config=CONFIG,
        now=10,
        planner=None,
    ):
        return (planner or BaselinePlanner()).choose(
            (task(),) if tasks is None else tasks,
            (offer(),) if offers is None else offers,
            progress,
            budget,
            config,
            now,
        )

    def test_unknown_estimates_get_profile_caps_not_free_work(self):
        item = offer()
        before = dataclasses.asdict(item)
        decision = self.choose(offers=(item,))
        self.assertEqual(decision.budget.max_cost_units, 600)
        self.assertEqual(decision.budget.max_llm_calls, 3)
        self.assertEqual(decision.budget.deadline_monotonic, 30)
        self.assertEqual(dataclasses.asdict(item), before)
        self.assertIsNone(item.expected_cost_units)
        self.assertIsNone(item.success_probability)

    def test_all_input_permutations_have_id_tie_break(self):
        tasks = (task("b"), task("a"))
        offers = tuple(offer(t.task_id, solver_id) for t in tasks for solver_id in ("z", "generic"))
        for task_order in itertools.permutations(tasks):
            for offer_order in itertools.permutations(offers):
                decision = self.choose(tasks=task_order, offers=offer_order)
                self.assertEqual((decision.task_id, decision.solver_id), ("a", "generic"))

    def test_known_reward_probability_cost_and_time_rank(self):
        tasks = (task("cheap", 5), task("expensive", 20))
        offers = (
            offer("cheap", expected_cost_units=10, success_probability=0.9),
            offer("expensive", expected_cost_units=500, success_probability=1),
        )
        self.assertEqual(self.choose(tasks=tasks, offers=offers).task_id, "cheap")
        offers = (
            offer("task", "slow", expected_duration_s=10),
            offer("task", "fast", expected_duration_s=1),
        )
        self.assertEqual(self.choose(offers=offers).solver_id, "fast")
        offers = (
            offer("task", "low", expected_cost_units=10, success_probability=0.1),
            offer("task", "high", expected_cost_units=10, success_probability=0.9),
        )
        self.assertEqual(self.choose(offers=offers).solver_id, "high")

    def test_partial_estimates_do_not_become_zero_cost_or_certainty(self):
        unknown = offer("task", "unknown")
        known = offer("task", "known", expected_cost_units=10, success_probability=0.8)
        self.assertEqual(self.choose(offers=(unknown, known)).solver_id, "known")
        self.assertEqual(self.choose(offers=(unknown,)).solver_id, "unknown")
        self.assertEqual(
            self.choose(offers=(offer(expected_cost_units=0),)).budget.max_llm_calls, 3
        )

    def test_trusted_deterministic_routes_need_no_llm_budget(self):
        planner = BaselinePlanner(deterministic_solvers=("det",))
        offers = (
            offer(
                "task", "generic", applicability="yes", expected_cost_units=1, success_probability=1
            ),
            offer("task", "det", applicability="yes"),
        )
        decision = self.choose(offers=offers, planner=planner)
        self.assertEqual(decision.solver_id, "det")
        self.assertEqual(decision.budget.max_llm_calls, 0)
        self.assertEqual(decision.budget.max_cost_units, 0)
        self.assertEqual(decision.budget.allowed_models, ())
        empty = dataclasses.replace(SNAPSHOT, spent_units=1000, held_units=0, blocked=True)
        self.assertEqual(self.choose(offers=offers, planner=planner, budget=empty).solver_id, "det")
        self.assertIsNone(self.choose(budget=empty))

    def test_global_no_llm_profile_allows_zero_cost_task(self):
        limits = dataclasses.replace(LIMITS, max_llm_calls=0, allowed_models=())
        decision = self.choose(config=dataclasses.replace(CONFIG, attempt_limits=limits))
        self.assertEqual(decision.budget.max_cost_units, 0)
        self.assertEqual(decision.budget.max_llm_calls, 0)

    def test_filters_no_applicability_and_missing_tools_per_offer(self):
        offers = (offer("task", "special", applicability="no", reason="not supported"), offer())
        self.assertEqual(self.choose(offers=offers).solver_id, "generic")
        required = offer(required_tools=("read",))
        self.assertIsNone(self.choose(offers=(required,)))
        self.assertIsNotNone(self.choose(offers=(required,), planner=BaselinePlanner(tools=TOOLS)))
        unavailable = dataclasses.replace(
            TOOLS[0], available=False, unavailable_reason="no backend"
        )
        self.assertIsNone(
            self.choose(offers=(required,), planner=BaselinePlanner(tools=(unavailable,)))
        )
        offers = (offer("task", "unknown"), offer("task", "yes", applicability="yes"))
        self.assertEqual(self.choose(offers=offers).solver_id, "yes")

    def test_budget_known_cost_fit_and_minimum_reservation(self):
        budget = dataclasses.replace(SNAPSHOT, spent_units=900, held_units=90)
        decision = self.choose(offers=(offer(expected_cost_units=10),), budget=budget)
        self.assertEqual(decision.budget.max_cost_units, 10)
        self.assertIsNone(self.choose(offers=(offer(expected_cost_units=11),), budget=budget))
        self.assertIsNone(self.choose(budget=dataclasses.replace(budget, held_units=91)))
        self.assertIsNone(self.choose(offers=(offer(expected_cost_units=601),)))
        self.assertIsNone(self.choose(budget=dataclasses.replace(SNAPSHOT, blocked=True)))

    def test_time_and_step_caps(self):
        limits = dataclasses.replace(LIMITS, max_llm_calls=10)
        config = dataclasses.replace(CONFIG, attempt_limits=limits)
        decision = self.choose(config=config, now=99)
        self.assertEqual(decision.budget.deadline_monotonic, 100)
        self.assertEqual(decision.budget.max_llm_calls, 5)
        self.assertEqual(decision.budget.tool_limits, LIMITS.tool_limits)
        self.assertIsNone(self.choose(now=100))
        self.assertIsNone(self.choose(now=99, offers=(offer(expected_duration_s=2),)))
        self.assertIsNone(self.choose(offers=(offer(expected_duration_s=21),)))

    def test_terminal_running_and_exhausted_tasks_never_relaunch(self):
        for status in ("running", "submitted", "exhausted"):
            with self.subTest(status=status):
                self.assertIsNone(
                    self.choose(progress=(TaskProgress(task_id="task", status=status),))
                )
        state = TaskProgress(
            task_id="task", status="pending", attempted_solvers=("generic", "generic")
        )
        self.assertIsNone(self.choose(progress=(state,)))

    def test_global_concurrency_and_deferred_time(self):
        tasks = (task("a"), task("b"), task("task"))
        progress = tuple(TaskProgress(task_id=name, status="running") for name in ("a", "b"))
        self.assertIsNone(self.choose(tasks=tasks, progress=progress))
        state = TaskProgress(task_id="task", status="deferred", not_before_monotonic=20)
        self.assertIsNone(self.choose(progress=(state,), now=19))
        self.assertIsNotNone(self.choose(progress=(state,), now=20))

    def test_auto_coverage_first_attempts_before_retries_and_low_reward(self):
        tasks = (task("rich", 10**300), task("poor", 0))
        offers = (offer("rich"), offer("poor"))
        progress = (TaskProgress(task_id="rich", status="pending", attempted_solvers=("generic",)),)
        self.assertEqual(self.choose(tasks=tasks, offers=offers, progress=progress).task_id, "poor")
        progress = (TaskProgress(task_id="rich", status="submitted"),)
        self.assertEqual(self.choose(tasks=tasks, offers=offers, progress=progress).task_id, "poor")

    def test_coverage_waits_for_deferred_first_attempt_and_reports_wakeup(self):
        tasks = (task("rich", 100), task("poor", 0))
        offers = (offer("rich"), offer("poor"))
        progress = (
            TaskProgress(task_id="rich", status="pending", attempted_solvers=("generic",)),
            TaskProgress(task_id="poor", status="deferred", not_before_monotonic=20),
        )
        planner = BaselinePlanner()
        self.assertIsNone(
            self.choose(tasks=tasks, offers=offers, progress=progress, planner=planner)
        )
        self.assertEqual(
            next_not_before_wakeup(planner, tasks, offers, progress, SNAPSHOT, CONFIG, 10), 20
        )
        self.assertEqual(
            self.choose(tasks=tasks, offers=offers, progress=progress, now=20).task_id, "poor"
        )
        # A submitted task never returns to the candidate queue.
        finished = (progress[0], dataclasses.replace(progress[1], status="submitted"))
        self.assertEqual(self.choose(tasks=tasks, offers=offers, progress=finished).task_id, "rich")

    def test_coverage_does_not_wait_for_permanently_inapplicable_first_offer(self):
        tasks = (task("rich"), task("poor"))
        progress = (
            TaskProgress(task_id="rich", status="pending", attempted_solvers=("generic",)),
            TaskProgress(task_id="poor", status="deferred", not_before_monotonic=20),
        )
        for bad in (
            offer("poor", applicability="no", reason="not applicable"),
            offer("poor", required_tools=("unavailable",)),
            offer("poor", expected_cost_units=601),
            offer("poor", expected_duration_s=81),
        ):
            with self.subTest(offer=bad):
                offers = (offer("rich"), bad)
                self.assertEqual(
                    self.choose(tasks=tasks, offers=offers, progress=progress).task_id, "rich"
                )
        past_deadline = (progress[0], dataclasses.replace(progress[1], not_before_monotonic=100))
        self.assertEqual(
            self.choose(
                tasks=tasks, offers=(offer("rich"), offer("poor")), progress=past_deadline
            ).task_id,
            "rich",
        )

    def test_coverage_waits_for_held_funds_and_rechecks_after_release(self):
        tasks = (task("rich"), task("poor"))
        offers = (offer("rich", expected_cost_units=10), offer("poor", expected_cost_units=100))
        progress = (TaskProgress(task_id="rich", status="pending", attempted_solvers=("generic",)),)
        held = dataclasses.replace(SNAPSHOT, held_units=850)
        self.assertIsNone(self.choose(tasks=tasks, offers=offers, progress=progress, budget=held))
        released = dataclasses.replace(held, held_units=0)
        self.assertEqual(
            self.choose(tasks=tasks, offers=offers, progress=progress, budget=released).task_id,
            "poor",
        )

    def test_next_wakeup_none_does_not_claim_global_completion(self):
        tasks = (task("rich"), task("poor"))
        progress = (TaskProgress(task_id="poor", status="deferred", not_before_monotonic=20),)
        planner = BaselinePlanner()
        self.assertIsNone(
            next_not_before_wakeup(planner, tasks, (), progress, SNAPSHOT, CONFIG, 10)
        )
        # A different budget/offer snapshot can become viable later without changing progress.
        offers = (offer("poor"),)
        self.assertEqual(
            next_not_before_wakeup(planner, tasks, offers, progress, SNAPSHOT, CONFIG, 10), 20
        )

    def test_portfolio_prefers_positive_reward_to_zero_reward_llm_spend(self):
        tasks = (task("zero", 0), task("valuable", 10000))
        offers = (
            offer("zero", applicability="yes", expected_cost_units=20, success_probability=1),
            offer("valuable", applicability="yes"),
        )
        budget = dataclasses.replace(SNAPSHOT, spent_units=980, held_units=0)
        config = dataclasses.replace(CONFIG, mode="portfolio")
        chosen = self.choose(tasks=tasks, offers=offers, budget=budget, config=config)
        self.assertEqual(chosen.task_id, "valuable")
        self.assertIsNone(offers[1].expected_cost_units)
        self.assertIsNone(offers[1].success_probability)
        # No-LLM work is still allowed, and zero reward remains eligible on its own.
        free = BaselinePlanner(deterministic_solvers=("free",))
        with_free = (offer("zero", "free"), offers[1])
        self.assertEqual(
            self.choose(
                tasks=tasks, offers=with_free, budget=budget, config=config, planner=free
            ).task_id,
            "zero",
        )
        self.assertEqual(
            self.choose(
                tasks=(tasks[0],), offers=(offers[0],), budget=budget, config=config
            ).task_id,
            "zero",
        )

    def test_explicit_portfolio_and_coverage(self):
        tasks = (task("rich", 100), task("poor", 1))
        offers = (offer("rich"), offer("poor"))
        progress = (TaskProgress(task_id="rich", status="pending", attempted_solvers=("generic",)),)
        self.assertEqual(
            self.choose(
                tasks=tasks,
                offers=offers,
                progress=progress,
                config=dataclasses.replace(CONFIG, mode="portfolio"),
            ).task_id,
            "rich",
        )
        self.assertEqual(
            self.choose(
                tasks=tasks,
                offers=offers,
                progress=progress,
                config=dataclasses.replace(CONFIG, mode="coverage"),
            ).task_id,
            "poor",
        )

    def test_extreme_and_zero_reward_exact_ordering(self):
        tasks = (task("large", 10**300 + 1), task("small", 10**300))
        offers = tuple(
            offer(t.task_id, expected_cost_units=10, success_probability=0.9) for t in tasks
        )
        self.assertEqual(self.choose(tasks=tasks, offers=offers).task_id, "large")
        self.assertIsNotNone(self.choose(tasks=(task(reward=0),)))

    def test_unsupported_solver_allows_another_route(self):
        result = AttemptResult(
            attempt_id="a",
            task_id="task",
            solver_id="special",
            status="unsupported",
            workspace_root=Path.cwd(),
            reason="unsupported by this solver",
        )
        state = TaskProgress(
            task_id="task", status="pending", attempted_solvers=("special",), last_result=result
        )
        offers = (offer("task", "special", applicability="yes"), offer())
        self.assertEqual(self.choose(offers=offers, progress=(state,)).solver_id, "generic")
        self.assertIsNone(self.choose(offers=(offers[0],), progress=(state,)))

    def test_pure_repeated_choose_has_no_state_or_execution(self):
        planner = BaselinePlanner()
        state = TaskProgress(task_id="task", status="pending")
        one = self.choose(planner=planner, progress=(state,))
        self.assertEqual(one, self.choose(planner=planner, progress=(state,)))
        self.assertEqual(state.status, "pending")
        self.assertFalse((Path.cwd() / "nonexistent-input").exists())
        self.assertIsNone(self.choose(offers=()))
        self.assertIsNone(self.choose(tasks=(), offers=()))

    def test_malformed_snapshot_relationships_are_contract_errors(self):
        for changes in (
            {"tasks": (task(), task())},
            {"offers": (offer(), offer())},
            {"offers": (offer("missing"),)},
            {"progress": (TaskProgress(task_id="missing", status="pending"),)},
            {"budget": dataclasses.replace(SNAPSHOT, limit_units=2000)},
            {"now": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.choose(**changes)


class RegistryTests(unittest.TestCase):
    def test_explicit_registration_freeze_unique_generic_and_get(self):
        registry = StaticSolverRegistry()
        with self.assertRaises(ContractError):
            registry.all_solvers()
        registry.register(FakeSolver("special"))
        registry.register(FakeSolver())
        with self.assertRaises(ContractError):
            registry.register(FakeSolver())
        self.assertEqual(
            [solver.solver_id for solver in registry.all_solvers()], ["generic", "special"]
        )
        self.assertEqual(registry.get("special").solver_id, "special")
        with self.assertRaises(ContractError):
            registry.register(FakeSolver("late"))
        with self.assertRaises(ContractError):
            registry.get("missing")

    def test_new_solver_and_unknown_type_require_no_queue_changes(self):
        registry = StaticSolverRegistry()
        registry.register(FakeSolver())
        registry.register(FakeSolver("new", applicability="yes", required_tools=("read",)))
        item = task(type_hint="new-unfamiliar-label")
        offers = tuple(solver.assess(item, TOOLS) for solver in registry.all_solvers())
        decision = BaselinePlanner(tools=TOOLS).choose((item,), offers, (), SNAPSHOT, CONFIG, 10)
        self.assertEqual(decision.solver_id, "new")


class LedgerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_planner_observes_actual_a2_reservations_and_halt(self):
        now = asyncio.get_running_loop().time()
        config = dataclasses.replace(CONFIG, deadline_monotonic=now + 60)
        ledger = BudgetLedger(limit_units=1000, deadline_monotonic=now + 60)
        planner = BaselinePlanner()
        first = planner.choose((task(),), (offer(),), (), ledger.snapshot(), config, now)
        ledger.open_attempt("a", first.budget)
        ledger.reserve("request", "a", MODEL, 0, 60)
        second = planner.choose((task(),), (offer(),), (), ledger.snapshot(), config, now)
        self.assertEqual(second.budget.max_cost_units, 400)
        ledger.halt(ErrorInfo(code="quota_exhausted", message="test"))
        self.assertIsNone(planner.choose((task(),), (offer(),), (), ledger.snapshot(), config, now))


if __name__ == "__main__":
    unittest.main()
