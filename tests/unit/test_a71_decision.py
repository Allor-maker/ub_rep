"""State ownership, strict role contracts and no duplicate physical proposal requests."""

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    CandidateAnswer,
    ContractError,
    TaskSpec,
    ToolLimits,
)
from nsu_budget_agent.contracts.ports import SolverContext
from nsu_budget_agent.orchestration.decision import DecisionController
from nsu_budget_agent.solvers.composition import build_with_router
from nsu_budget_agent.solvers.decision import (
    TRANSITIONS,
    DecisionStore,
    Review,
    State,
    free_interpretation,
)


class NoNetworkSession:
    async def complete(self, request):
        raise AssertionError("duplicate must be rejected before reaching gateway")


class DecisionModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.task = TaskSpec(
            task_id="test",
            instruction="Unfamiliar instruction.",
            reward=7,
            task_root=Path(self.temp.name),
            work_root=Path(self.temp.name),
            artifacts=(),
        )

    async def test_transition_table_rejects_every_undeclared_edge(self):
        for source in State:
            for destination in State:
                store = DecisionStore()
                route = store.get(self.task)
                route.state = source
                if destination in TRANSITIONS[source]:
                    route.move(destination, "test declared edge")
                    self.assertEqual(route.state, destination)
                else:
                    with self.assertRaises(ContractError):
                        route.move(destination, "test undeclared edge")

    async def test_unknown_solver_has_no_invented_uncertain_candidate(self):
        call, complete, reason, consumed = free_interpretation(self.task)
        self.assertIsNone(call)
        self.assertFalse(complete)
        self.assertTrue(reason)
        self.assertEqual(consumed, "")
        call, complete, reason, _ = free_interpretation(
            replace(self.task, instruction="Compute 2+3. Then multiply by 9.")
        )
        self.assertIsNone(call)
        self.assertFalse(complete)

    async def test_store_rejects_same_task_identity_with_different_instruction(self):
        store = DecisionStore()
        self.assertIs(store.get(self.task), store.get(self.task))
        with self.assertRaises(ContractError):
            store.get(replace(self.task, instruction="Different condition"))

    async def test_duplicate_fingerprint_denied_before_session(self):
        store = DecisionStore()
        route = store.get(self.task)
        route.move(State.UNSOLVED, "no interpretation")
        route.move(State.LLM_SOLVE, "new strategy")
        solver = DecisionController(store, "qwen3.6-35b-a3b-noreason")
        budget = AttemptBudget(
            max_cost_units=10000,
            max_llm_calls=3,
            max_steps=8,
            deadline_monotonic=asyncio.get_running_loop().time() + 10,
            allowed_models=(solver.model,),
            tool_limits=ToolLimits(
                timeout_s=1,
                max_read_bytes=1000,
                max_output_bytes=1000,
                max_write_bytes=1000,
                max_process_memory_bytes=1000,
                max_process_count=1,
            ),
        )
        ctx = SolverContext(
            task=self.task,
            attempt_id="attempt",
            step_index=0,
            budget=budget,
            tools=(),
            llm=NoNetworkSession(),
        )
        # Simulate a gateway exception at the first dispatch boundary; the
        # fingerprint must already be consumed even though no reply was returned.
        with self.assertRaises(AssertionError):
            await solver._request(ctx, route, "solve")
        self.assertEqual(len(route.fingerprints), 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            await solver._request(ctx, route, "solve")
        self.assertEqual(route.calls, 1)

    async def test_review_contract_strict_fields_types_and_combinations(self):
        valid = {
            "verdict": "accept",
            "reason": "covers conditions",
            "next_action": "publish",
            "missing_conditions": [],
        }
        self.assertEqual(Review.parse(valid).verdict, "accept")
        for change in (
            {"verdict": True},
            {"next_action": "shell"},
            {"reason": ""},
            {"missing_conditions": "none"},
            {"missing_conditions": [1]},
            {"missing_conditions": ["unverified"]},
            {"extra": "command"},
        ):
            with self.assertRaises(ValueError):
                Review.parse({**valid, **change})

    async def test_review_accept_never_certifies_and_terminal_state_cannot_restart(self):
        store = DecisionStore()
        route = store.get(self.task)
        route.move(State.UNSOLVED, "unknown")
        route.move(State.LLM_SOLVE, "solver")
        route.move(State.LLM_REVIEW, "review")
        route.move(State.VALIDATION, "accept")
        store.finish(self.task, validated=True, published=True)
        self.assertNotIn(State.CERTIFIED, [s for s, _ in route.events])
        self.assertEqual(route.state, State.PUBLISH)
        with self.assertRaises(ContractError):
            route.move(State.LLM_SOLVE, "try again")
        with self.assertRaises(ContractError):
            route.move(State.CERTIFIED, "pretend proof")

    async def test_strict_router_factory_cannot_enable_untrusted_provider(self):
        from nsu_budget_agent.budget import BudgetLedger
        from nsu_budget_agent.contracts.models import AttemptLimits, RunConfig

        tool_limits = ToolLimits(
            timeout_s=1,
            max_read_bytes=1000,
            max_output_bytes=1000,
            max_write_bytes=1000,
            max_process_memory_bytes=1000,
            max_process_count=1,
        )
        config = RunConfig(
            total_budget_units=10000,
            deadline_monotonic=asyncio.get_running_loop().time() + 10,
            attempt_limits=AttemptLimits(
                max_cost_units=10000,
                max_llm_calls=3,
                max_steps=8,
                max_duration_s=10,
                max_attempts_per_task=2,
                allowed_models=("qwen3.6-35b-a3b-noreason",),
                tool_limits=tool_limits,
            ),
        )
        ledger = BudgetLedger(limit_units=10000, deadline_monotonic=config.deadline_monotonic)
        with self.assertRaises(ContractError):
            build_with_router(ledger, config, environ={})
        self.assertEqual(ledger.snapshot().calls_used, 0)

    async def test_partial_candidate_dto_preserves_conditions_and_evidence(self):
        from nsu_budget_agent.solvers.decision import Proposal

        proposal = Proposal(
            candidate=CandidateAnswer(text="30"),
            call=None,
            complete=False,
            uncertainty="unconsumed status condition",
            used_conditions="region west",
        )
        with self.assertRaises((AttributeError, TypeError)):
            proposal.complete = True
