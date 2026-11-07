"""Single-event-loop BudgetController implementation; no transport or SDK."""

import asyncio
from dataclasses import dataclass, replace
from typing import NoReturn

from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    BudgetLease,
    BudgetSnapshot,
    ContractError,
    ErrorInfo,
    Id,
    ModelId,
    OperationFailed,
    Units,
    UsageRecord,
    _id,
    _instance,
    _int,
    _number,
)


@dataclass(slots=True)
class _Attempt:
    budget: AttemptBudget
    closed: bool = False


class BudgetLedger:
    """Construct and use inside one running asyncio loop, never from workers.

    Synchronous methods contain no await: validation and insertion are one event
    loop transaction. Immutable leases are the source of truth, not cached totals.
    """

    def __init__(self, *, limit_units: Units, deadline_monotonic: float) -> None:
        _int(limit_units, "limit_units")
        _number(deadline_monotonic, "deadline_monotonic")
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise ContractError("BudgetLedger requires a running event loop") from exc
        self._limit = limit_units
        self._deadline = deadline_monotonic
        self._attempts: dict[Id, _Attempt] = {}
        self._leases: dict[Id, BudgetLease] = {}
        self._halt_error: ErrorInfo | None = None

    def _check_loop(self) -> None:
        try:
            current = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise ContractError("BudgetLedger cannot be called outside its event loop") from exc
        if current is not self._loop:
            raise ContractError("BudgetLedger cannot be shared between event loops")

    def _attempt(self, attempt_id: Id) -> _Attempt:
        _id(attempt_id, "attempt_id")
        if attempt_id not in self._attempts:
            raise ContractError("unknown attempt_id")
        return self._attempts[attempt_id]

    def _lease(self, request_id: Id) -> BudgetLease:
        _id(request_id, "request_id")
        if request_id not in self._leases:
            raise ContractError("unknown request_id")
        return self._leases[request_id]

    def _allow_new_call(self, attempt: _Attempt) -> None:
        if self._halt_error is not None:
            raise OperationFailed(self._halt_error)
        if attempt.closed:
            raise ContractError("attempt is closed")
        if self._loop.time() >= min(self._deadline, attempt.budget.deadline_monotonic):
            raise OperationFailed(ErrorInfo(code="timeout", message="budget deadline expired"))

    def open_attempt(self, attempt_id: Id, budget: AttemptBudget) -> None:
        self._check_loop()
        _id(attempt_id, "attempt_id")
        _instance(budget, AttemptBudget, "budget")
        if attempt_id in self._attempts:
            if self._attempts[attempt_id].budget != budget:
                raise ContractError("attempt_id already registered with another budget")
            return  # Does not reopen a closed attempt or reset spent slots.
        if budget.deadline_monotonic > self._deadline:
            raise ContractError("attempt deadline exceeds run deadline")
        if self._loop.time() >= budget.deadline_monotonic:
            raise OperationFailed(ErrorInfo(code="timeout", message="attempt deadline expired"))
        self._attempts[attempt_id] = _Attempt(budget=budget)

    def reserve(
        self,
        request_id: Id,
        attempt_id: Id,
        model: ModelId,
        input_bound_tokens: int,
        output_cap_tokens: int,
    ) -> BudgetLease:
        self._check_loop()
        proposed = BudgetLease(
            request_id=request_id,
            attempt_id=attempt_id,
            model=model,
            input_bound_tokens=input_bound_tokens,
            output_cap_tokens=output_cap_tokens,
            state="reserved",
        )
        if request_id in self._leases:
            previous = self._leases[request_id]
            if replace(previous, state="reserved", usage=None) != proposed:
                raise ContractError("request_id already reserved with other parameters")
            return previous  # Identity lookup; never allocates another slot or reuses an ID.
        attempt = self._attempt(attempt_id)
        self._allow_new_call(attempt)
        if model not in attempt.budget.allowed_models:
            raise ContractError("model is not allowed for the attempt")
        global_budget = self.snapshot()
        local_budget = self.snapshot(attempt_id)
        if local_budget.calls_used >= attempt.budget.max_llm_calls or proposed.reserved_units > min(
            global_budget.available_units, local_budget.available_units
        ):
            raise OperationFailed(
                ErrorInfo(code="budget_exhausted", message="budget or call limit")
            )
        self._leases[request_id] = proposed
        return proposed

    def mark_dispatched(self, request_id: Id) -> BudgetLease:
        self._check_loop()
        lease = self._lease(request_id)
        if lease.state == "dispatched":
            return lease
        if lease.state != "reserved":
            raise ContractError("only reserved requests can be dispatched")
        self._allow_new_call(self._attempt(lease.attempt_id))
        updated = replace(lease, state="dispatched")
        self._leases[request_id] = updated
        return updated

    def release_unsent(self, request_id: Id) -> BudgetLease:
        self._check_loop()
        lease = self._lease(request_id)
        if lease.state == "released":
            return lease
        if lease.state != "reserved":
            raise ContractError("cannot release a request after dispatch")
        updated = replace(lease, state="released")
        self._leases[request_id] = updated
        return updated

    def _accounting_violation(self, message: str) -> NoReturn:
        error = ErrorInfo(code="accounting_violation", message=message)
        self.halt(error)
        raise OperationFailed(error)

    def record_usage(self, usage: UsageRecord) -> BudgetLease:
        self._check_loop()
        _instance(usage, UsageRecord, "usage")
        lease = self._lease(usage.request_id)
        if (lease.attempt_id, lease.model) != (usage.attempt_id, usage.model):
            raise ContractError("usage does not match the lease identity")
        if lease.state not in ("dispatched", "uncertain", "settled"):
            raise ContractError("usage requires a dispatched request")
        if lease.state == "settled":
            if usage.status == "unknown" or lease.usage == usage:
                return lease
            self._accounting_violation("conflicting exact usage for request_id")
        if usage.status == "unknown" and lease.state == "uncertain":
            return lease  # Keep the first uncertain diagnostic; no counters are authoritative.
        updated = replace(
            lease, state="settled" if usage.status == "exact" else "uncertain", usage=usage
        )
        self._leases[usage.request_id] = updated
        cost = usage.cost_units
        if cost is not None and cost > lease.reserved_units:
            # Commit the full authoritative expense BEFORE reporting the violation.
            self._accounting_violation("exact usage exceeds the reservation")
        return updated

    def _totals(self, attempt_id: Id | None) -> tuple[Units, Units, int]:
        spent = held = calls = 0
        for lease in self._leases.values():
            if attempt_id is not None and lease.attempt_id != attempt_id:
                continue
            if lease.state == "settled":
                assert lease.usage is not None and lease.usage.cost_units is not None
                spent += lease.usage.cost_units
            elif lease.state in ("reserved", "dispatched", "uncertain"):
                held += lease.reserved_units
            if lease.state != "released":
                calls += 1
        return spent, held, calls

    def snapshot(self, attempt_id: Id | None = None) -> BudgetSnapshot:
        self._check_loop()
        attempt = self._attempt(attempt_id) if attempt_id is not None else None
        spent, held, calls = self._totals(attempt_id)
        limit = self._limit if attempt is None else attempt.budget.max_cost_units
        blocked = self._halt_error is not None or self._loop.time() >= self._deadline
        global_spent, global_held, _ = self._totals(None) if attempt else (spent, held, calls)
        # A legal request costs at least 10 Units: zero input plus one output token.
        blocked |= min(limit - spent - held, self._limit - global_spent - global_held) < 10
        if attempt is not None:
            blocked |= (
                attempt.closed
                or self._loop.time() >= attempt.budget.deadline_monotonic
                or calls >= attempt.budget.max_llm_calls
            )
        return BudgetSnapshot(
            limit_units=limit, spent_units=spent, held_units=held, calls_used=calls, blocked=blocked
        )

    def close_attempt(self, attempt_id: Id) -> None:
        self._check_loop()
        attempt = self._attempt(attempt_id)
        if attempt.closed:
            return
        for request_id, lease in self._leases.items():
            if lease.attempt_id != attempt_id:
                continue
            if lease.state == "reserved":
                self._leases[request_id] = replace(lease, state="released")
            elif lease.state == "dispatched":
                self._leases[request_id] = replace(
                    lease,
                    state="uncertain",
                    usage=UsageRecord(
                        request_id=request_id,
                        attempt_id=attempt_id,
                        model=lease.model,
                        status="unknown",
                        input_tokens=None,
                        output_tokens=None,
                    ),
                )
        attempt.closed = True

    def halt(self, error: ErrorInfo) -> None:
        self._check_loop()
        _instance(error, ErrorInfo, "error")
        if self._halt_error is None:
            self._halt_error = error
