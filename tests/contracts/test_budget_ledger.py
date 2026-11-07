"""A2: C04/C05/C06 with concurrent asyncio callers and no transport/SDK."""

import asyncio
import dataclasses
import unittest
from unittest.mock import patch

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    ContractError,
    ErrorInfo,
    OperationFailed,
    ToolLimits,
    UsageRecord,
)
from nsu_budget_agent.contracts.ports import BudgetController

MODEL = "qwen3.8-27b"
OTHER_MODEL = "qwen3.8-27b-noreason"
LIMITS = ToolLimits(
    timeout_s=1,
    max_read_bytes=100,
    max_output_bytes=100,
    max_write_bytes=100,
    max_process_memory_bytes=1000,
    max_process_count=1,
)


class BudgetLedgerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.loop = asyncio.get_running_loop()
        self.deadline = self.loop.time() + 60
        self.ledger = BudgetLedger(limit_units=1000, deadline_monotonic=self.deadline)
        self.budget = AttemptBudget(
            max_cost_units=1000,
            max_llm_calls=3,
            max_steps=5,
            deadline_monotonic=self.deadline,
            allowed_models=(MODEL,),
            tool_limits=LIMITS,
        )
        self.ledger.open_attempt("a", self.budget)

    def reserve(self, request_id="r", attempt_id="a", input_tokens=120, output_tokens=60):
        return self.ledger.reserve(request_id, attempt_id, MODEL, input_tokens, output_tokens)

    def usage(self, request_id="r", attempt_id="a", **changes):
        return UsageRecord(
            **{
                "request_id": request_id,
                "attempt_id": attempt_id,
                "model": MODEL,
                "status": "exact",
                "input_tokens": 120,
                "output_tokens": 40,
                "reasoning_tokens": 20,
                **changes,
            }
        )

    def fail_with(self, code, operation, *args):
        with self.assertRaises(OperationFailed) as failure:
            operation(*args)
        self.assertEqual(failure.exception.error.code, code)

    async def test_reservation_settlement_and_reasoning_counted_once(self):
        # Also statically checked: the concrete class satisfies the unchanged port.
        controller: BudgetController = self.ledger
        lease = self.reserve()
        self.assertEqual(lease.reserved_units, 720)
        self.assertEqual(controller.snapshot().held_units, 720)
        dispatched = controller.mark_dispatched("r")
        settled = controller.record_usage(self.usage())
        self.assertEqual(settled.state, "settled")
        self.assertEqual(settled.usage.cost_units, 520)
        self.assertEqual(controller.snapshot().available_units, 480)
        self.assertEqual(controller.snapshot().held_units, 0)
        self.assertEqual(controller.snapshot().calls_used, 1)
        self.assertEqual(controller.snapshot("a"), controller.snapshot())
        # Previously returned snapshots never mutate with ledger state.
        self.assertEqual(lease.state, "reserved")
        self.assertEqual(dispatched.state, "dispatched")

    async def test_zero_exact_is_not_unknown_or_a_free_call_slot(self):
        self.reserve()
        self.ledger.mark_dispatched("r")
        zero = self.usage(input_tokens=0, output_tokens=0, reasoning_tokens=0)
        self.ledger.record_usage(zero)
        snapshot = self.ledger.snapshot()
        self.assertEqual(
            (snapshot.spent_units, snapshot.held_units, snapshot.calls_used), (0, 0, 1)
        )

    async def test_exact_boundary_and_one_input_token_keep_integer_units(self):
        self.reserve()
        self.ledger.mark_dispatched("r")
        self.ledger.record_usage(self.usage(output_tokens=60))
        self.assertEqual(self.ledger.snapshot().spent_units, 720)
        self.assertFalse(self.ledger.snapshot().blocked)
        self.reserve("tiny", input_tokens=1, output_tokens=1)
        self.ledger.mark_dispatched("tiny")
        self.ledger.record_usage(
            self.usage("tiny", input_tokens=1, output_tokens=0, reasoning_tokens=0)
        )
        self.assertEqual(self.ledger.snapshot().spent_units, 721)
        self.assertEqual(self.ledger.snapshot().available_units, 279)

    async def test_settlement_releases_only_difference_for_next_request(self):
        self.reserve()
        self.ledger.mark_dispatched("r")
        self.ledger.record_usage(self.usage())
        self.reserve("next", input_tokens=0, output_tokens=48)
        self.assertEqual(self.ledger.snapshot().available_units, 0)
        self.assertEqual(self.ledger.snapshot().held_units, 480)
        self.ledger.record_usage(self.usage())
        self.assertEqual(self.ledger.snapshot().spent_units, 520)
        self.fail_with("budget_exhausted", self.reserve, "extra", "a", 0, 1)

    async def test_replays_preserve_balances_and_do_not_reopen_closed_attempts(self):
        lease = self.reserve()
        self.assertIs(self.reserve(), lease)
        self.ledger.open_attempt("a", dataclasses.replace(self.budget))
        with self.assertRaises(ContractError):
            self.ledger.open_attempt("a", dataclasses.replace(self.budget, max_llm_calls=4))
        for params in (
            ("r", "a", MODEL, 121, 60),
            ("r", "a", OTHER_MODEL, 120, 60),
            ("r", "other", MODEL, 120, 60),
        ):
            with self.subTest(params=params), self.assertRaises(ContractError):
                self.ledger.reserve(*params)
        self.ledger.close_attempt("a")
        self.ledger.open_attempt("a", self.budget)
        self.assertEqual(self.reserve().state, "released")
        with self.assertRaises(ContractError):
            self.reserve("new")
        self.assertEqual(self.ledger.snapshot().calls_used, 0)

    async def test_release_unsent_refunds_both_units_and_slot(self):
        self.reserve()
        released = self.ledger.release_unsent("r")
        self.assertIs(self.ledger.release_unsent("r"), released)
        self.assertEqual(self.ledger.snapshot().available_units, 1000)
        self.assertEqual(self.ledger.snapshot().calls_used, 0)
        self.reserve("fresh")
        self.assertEqual(self.reserve().state, "released")  # IDs are never recycled.

    async def test_exact_and_unknown_replays_and_late_exact(self):
        self.reserve()
        dispatched = self.ledger.mark_dispatched("r")
        self.assertIs(self.ledger.mark_dispatched("r"), dispatched)
        unknown = self.usage(status="unknown", input_tokens=120, output_tokens=None)
        uncertain = self.ledger.record_usage(unknown)
        self.assertIs(self.ledger.record_usage(unknown), uncertain)
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.assertEqual(self.ledger.snapshot().spent_units, 0)
        # Different partial counts are not authoritative and never reduce the hold.
        self.ledger.record_usage(dataclasses.replace(unknown, input_tokens=0, output_tokens=0))
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        settled = self.ledger.record_usage(self.usage())
        self.assertIs(self.ledger.record_usage(self.usage()), settled)
        self.assertIs(self.ledger.record_usage(unknown), settled)
        self.assertEqual(self.reserve().state, "settled")
        self.assertEqual(self.ledger.snapshot().spent_units, 520)

    async def test_conflicting_exact_halts_without_overwriting_first_record(self):
        self.reserve()
        self.ledger.mark_dispatched("r")
        first = self.ledger.record_usage(self.usage())
        self.fail_with(
            "accounting_violation", self.ledger.record_usage, self.usage(output_tokens=30)
        )
        self.assertTrue(self.ledger.snapshot().blocked)
        self.assertEqual(self.ledger.snapshot().spent_units, 520)
        self.assertIs(self.ledger.record_usage(self.usage()), first)
        self.fail_with("accounting_violation", self.reserve, "new")

    async def test_over_reservation_records_full_expense_then_halts(self):
        self.reserve()
        self.reserve("pending", input_tokens=0, output_tokens=10)
        self.ledger.mark_dispatched("r")
        excess = self.usage(input_tokens=500, output_tokens=100, reasoning_tokens=20)
        self.fail_with("accounting_violation", self.ledger.record_usage, excess)
        snapshot = self.ledger.snapshot()
        self.assertEqual(
            (snapshot.spent_units, snapshot.held_units, snapshot.available_units), (1500, 100, 0)
        )
        self.assertTrue(snapshot.blocked)
        self.assertEqual(self.ledger.record_usage(excess).state, "settled")
        self.fail_with("accounting_violation", self.ledger.mark_dispatched, "pending")
        self.ledger.release_unsent("pending")
        self.assertEqual(self.ledger.snapshot().spent_units, 1500)

    async def test_usage_cannot_precede_dispatch_or_mismatch_identity(self):
        self.reserve()
        for status in ("unknown", "exact"):
            with self.subTest(status=status), self.assertRaises(ContractError):
                self.ledger.record_usage(self.usage(status=status))
        self.ledger.mark_dispatched("r")
        for record in (
            self.usage(attempt_id="other"),
            self.usage(model=OTHER_MODEL),
            self.usage(request_id="unknown"),
        ):
            with self.assertRaises(ContractError):
                self.ledger.record_usage(record)
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.assertFalse(self.ledger.snapshot().blocked)

    async def test_invalid_state_transitions_do_not_change_balances(self):
        for state in ("reserved", "released", "dispatched", "uncertain", "settled"):
            ledger = BudgetLedger(limit_units=1000, deadline_monotonic=self.deadline)
            ledger.open_attempt("a", self.budget)
            ledger.reserve("r", "a", MODEL, 120, 60)
            if state == "released":
                ledger.release_unsent("r")
            elif state != "reserved":
                ledger.mark_dispatched("r")
                if state in ("uncertain", "settled"):
                    ledger.record_usage(
                        self.usage(status="unknown" if state == "uncertain" else "exact")
                    )
            before = ledger.snapshot()
            if state in ("released", "uncertain", "settled"):
                with self.subTest(state=state), self.assertRaises(ContractError):
                    ledger.mark_dispatched("r")
            if state in ("dispatched", "uncertain", "settled"):
                with self.subTest(state=state), self.assertRaises(ContractError):
                    ledger.release_unsent("r")
            if state == "released":
                with self.assertRaises(ContractError):
                    ledger.record_usage(self.usage())
            self.assertEqual(ledger.snapshot(), before)

    async def test_global_attempt_and_call_limits_are_independent(self):
        self.ledger.open_attempt(
            "small", dataclasses.replace(self.budget, max_cost_units=20, max_llm_calls=1)
        )
        self.reserve("s", "small", 0, 2)
        self.fail_with("budget_exhausted", self.reserve, "s2", "small", 0, 1)
        self.assertTrue(self.ledger.snapshot("small").blocked)
        self.ledger.release_unsent("s")
        self.reserve("s3", "small", 0, 2)
        self.ledger.mark_dispatched("s3")
        self.ledger.record_usage(
            self.usage("s3", "small", input_tokens=0, output_tokens=0, reasoning_tokens=0)
        )
        self.assertEqual(self.ledger.snapshot("small").available_units, 20)
        self.fail_with("budget_exhausted", self.reserve, "s4", "small", 0, 1)
        self.reserve()  # A different attempt still has slots and global funds.
        self.fail_with("budget_exhausted", self.reserve, "too-big", "a", 0, 29)
        with self.assertRaises(ContractError):
            self.ledger.mark_dispatched("too-big")  # No lease was inserted on failure.

    async def test_minimum_cost_boundaries_and_deterministic_attempt(self):
        for limit in (0, 9, 10):
            ledger = BudgetLedger(limit_units=limit, deadline_monotonic=self.deadline)
            ledger.open_attempt("a", self.budget)
            self.assertEqual(ledger.snapshot().blocked, limit < 10)
            if limit < 10:
                self.fail_with("budget_exhausted", ledger.reserve, "r", "a", MODEL, 0, 1)
            else:
                ledger.reserve("r", "a", MODEL, 0, 1)
                self.assertEqual(ledger.snapshot().available_units, 0)
        self.ledger.open_attempt(
            "local",
            dataclasses.replace(self.budget, max_cost_units=0, max_llm_calls=0, allowed_models=()),
        )
        self.assertTrue(self.ledger.snapshot("local").blocked)
        self.ledger.close_attempt("local")

    async def test_deadlines_checked_at_open_reserve_and_dispatch(self):
        earlier = self.deadline - 20
        self.ledger.open_attempt(
            "early", dataclasses.replace(self.budget, deadline_monotonic=earlier)
        )
        self.reserve("early-r", "early", 0, 1)
        with patch.object(self.loop, "time", return_value=earlier):
            self.fail_with("timeout", self.reserve, "new", "early", 0, 1)
            self.fail_with("timeout", self.ledger.mark_dispatched, "early-r")
            self.fail_with(
                "timeout",
                self.ledger.open_attempt,
                "late",
                dataclasses.replace(self.budget, deadline_monotonic=earlier),
            )
            self.assertTrue(self.ledger.snapshot("early").blocked)
            self.assertFalse(self.ledger.snapshot().blocked)
            self.ledger.release_unsent("early-r")
        with self.assertRaises(ContractError):
            self.ledger.open_attempt(
                "bad", dataclasses.replace(self.budget, deadline_monotonic=self.deadline + 1)
            )
        with patch.object(self.loop, "time", return_value=self.deadline):
            self.assertTrue(self.ledger.snapshot().blocked)

    async def test_close_and_halt_keep_inflight_liabilities_and_allow_reconciliation(self):
        self.reserve("unsent", input_tokens=0, output_tokens=1)
        self.reserve()
        self.ledger.mark_dispatched("r")
        error = ErrorInfo(code="quota_exhausted", message="quota ended")
        self.ledger.halt(error)
        self.ledger.halt(ErrorInfo(code="provider_error", message="later reason"))
        self.fail_with("quota_exhausted", self.ledger.mark_dispatched, "unsent")
        self.fail_with("quota_exhausted", self.reserve, "new", "a", 0, 1)
        self.assertEqual(self.ledger.mark_dispatched("r").state, "dispatched")
        self.ledger.close_attempt("a")
        self.ledger.close_attempt("a")
        self.assertEqual(self.reserve("unsent", input_tokens=0, output_tokens=1).state, "released")
        self.assertEqual(self.reserve().state, "uncertain")
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.ledger.record_usage(self.usage())
        self.assertEqual(self.ledger.snapshot().spent_units, 520)
        self.assertEqual(self.ledger.snapshot().held_units, 0)
        self.assertTrue(self.ledger.snapshot("a").blocked)

    async def test_unknown_and_settled_records_survive_close(self):
        self.reserve()
        self.ledger.mark_dispatched("r")
        unknown = self.ledger.record_usage(self.usage(status="unknown"))
        self.ledger.close_attempt("a")
        self.assertIs(self.ledger.record_usage(self.usage(status="unknown")), unknown)
        self.ledger.record_usage(self.usage())
        self.ledger.close_attempt("a")
        self.assertEqual(self.ledger.snapshot().spent_units, 520)

    async def test_concurrent_reserves_use_one_global_balance(self):
        self.ledger.open_attempt("b", self.budget)
        barrier = asyncio.Event()

        async def caller(attempt_id):
            await barrier.wait()
            try:
                return self.reserve(attempt_id, attempt_id)
            except OperationFailed as error:
                return error.error.code

        tasks = [asyncio.create_task(caller(key)) for key in ("a", "b")]
        barrier.set()
        outcomes = await asyncio.gather(*tasks)
        self.assertEqual(sum(result == "budget_exhausted" for result in outcomes), 1)
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.assertEqual(self.ledger.snapshot().calls_used, 1)

    async def test_concurrent_replayed_usage_is_charged_once(self):
        self.reserve()
        self.ledger.mark_dispatched("r")

        async def deliver():
            await asyncio.sleep(0)
            return self.ledger.record_usage(self.usage())

        replies = await asyncio.gather(*(deliver() for _ in range(20)))
        self.assertTrue(all(reply is replies[0] for reply in replies))
        self.assertEqual(self.ledger.snapshot().spent_units, 520)

    async def test_concurrent_attempt_funds_slots_and_identical_reserves(self):
        for local_limit, call_limit, output_cap in ((1000, 3, 60), (5000, 1, 1)):
            ledger = BudgetLedger(limit_units=5000, deadline_monotonic=self.deadline)
            ledger.open_attempt(
                "a",
                dataclasses.replace(
                    self.budget, max_cost_units=local_limit, max_llm_calls=call_limit
                ),
            )
            barrier = asyncio.Event()

            async def reserve(request_id):
                await barrier.wait()
                try:
                    return ledger.reserve(request_id, "a", MODEL, 0, output_cap)
                except OperationFailed as error:
                    return error.error.code

            tasks = [asyncio.create_task(reserve(key)) for key in ("first", "second")]
            barrier.set()
            outcomes = await asyncio.gather(*tasks)
            with self.subTest(limit=local_limit, calls=call_limit):
                self.assertEqual(sum(result == "budget_exhausted" for result in outcomes), 1)
                self.assertEqual(ledger.snapshot().calls_used, 1)
                self.assertEqual(ledger.snapshot().held_units, 10 * output_cap)

        async def same_id():
            await asyncio.sleep(0)
            return self.reserve()

        leases = await asyncio.gather(*(same_id() for _ in range(10)))
        self.assertTrue(all(lease is leases[0] for lease in leases))
        self.assertEqual(self.ledger.snapshot().held_units, 720)

    async def test_usage_can_exceed_local_reservation_without_global_overdraw(self):
        self.reserve("small", input_tokens=0, output_tokens=1)
        self.ledger.mark_dispatched("small")
        usage = self.usage("small", input_tokens=0, output_tokens=2, reasoning_tokens=0)
        self.fail_with("accounting_violation", self.ledger.record_usage, usage)
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot.spent_units, 20)
        self.assertEqual(snapshot.available_units, 980)
        self.assertTrue(snapshot.blocked)

    async def test_cancellation_before_and_after_dispatch(self):
        for dispatched in (False, True):
            started = asyncio.Event()
            request_id = f"cancel-{dispatched}"

            async def call():
                self.reserve(request_id, input_tokens=0, output_tokens=10)
                if dispatched:
                    self.ledger.mark_dispatched(request_id)
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    if dispatched:
                        self.ledger.record_usage(
                            self.usage(
                                request_id,
                                status="unknown",
                                input_tokens=None,
                                output_tokens=None,
                                reasoning_tokens=None,
                            )
                        )
                    else:
                        self.ledger.release_unsent(request_id)

            task = asyncio.create_task(call())
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            snapshot = self.ledger.snapshot()
            self.assertEqual(snapshot.held_units, 100 if dispatched else 0)
            self.assertEqual(snapshot.calls_used, 1 if dispatched else 0)

    async def test_timeout_after_dispatch_preserves_hold_until_late_exact(self):
        self.reserve()
        self.ledger.mark_dispatched("r")
        with self.assertRaises(TimeoutError):
            try:
                async with asyncio.timeout(0):
                    await asyncio.Event().wait()
            finally:
                self.ledger.record_usage(
                    self.usage(
                        status="unknown",
                        input_tokens=None,
                        output_tokens=None,
                        reasoning_tokens=None,
                    )
                )
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.ledger.close_attempt("a")
        self.ledger.record_usage(self.usage())
        self.assertEqual(self.ledger.snapshot().available_units, 480)

    async def test_threads_and_other_loops_are_rejected_before_mutation(self):
        def from_thread():
            with self.assertRaises(ContractError):
                self.ledger.snapshot()

            async def from_other_loop():
                with self.assertRaises(ContractError):
                    self.reserve()

            asyncio.run(from_other_loop())

        await asyncio.to_thread(from_thread)
        self.assertEqual(self.ledger.snapshot().held_units, 0)

    async def test_invalid_inputs_and_unknown_ids_leave_no_state(self):
        for limit in (-1, True, 1.5):
            with self.assertRaises(ContractError):
                BudgetLedger(limit_units=limit, deadline_monotonic=self.deadline)
        for operation in (
            self.ledger.mark_dispatched,
            self.ledger.release_unsent,
            self.ledger.close_attempt,
            self.ledger.snapshot,
        ):
            with self.assertRaises(ContractError):
                operation("unknown")
        for args in (
            ("r", "a", MODEL, True, 1),
            ("r", "a", MODEL, 0, 0),
            ("r", "a", "other-model", 0, 1),
            ("r", "a", OTHER_MODEL, 0, 1),
            ("r", "unknown", MODEL, 0, 1),
        ):
            with self.assertRaises(ContractError):
                self.ledger.reserve(*args)
        self.assertEqual(self.ledger.snapshot().calls_used, 0)
