"""A4 ranking with a conservative equal-share cap for first paid attempts."""

from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nsu_budget_agent.orchestration.decision import DecisionController

from nsu_budget_agent.contracts.models import (
    BudgetSnapshot,
    PlanDecision,
    RunConfig,
    SolverOffer,
    TaskProgress,
    TaskSpec,
)

from .baseline import BaselinePlanner


class DecisionPortfolioPlanner(BaselinePlanner):
    def __init__(
        self, *, tools=(), deterministic_solvers=(), controller: "DecisionController | None" = None
    ):
        super().__init__(tools=tools, deterministic_solvers=deterministic_solvers)
        self.controller = controller
        self.deferred_task_ids: set[str] = set()

    def _rank(self, task, offer, state, no_llm):
        base = super()._rank(task, offer, state, no_llm)
        if no_llm or self.controller is None or not self.controller.require_independent:
            return base
        from nsu_budget_agent.validation.data_plan import data_question
        from nsu_budget_agent.validation.llm import question

        known = question(task) is not None or data_question(task) is not None
        floor = max(1, self.controller.minimum_reservation(task))
        # Reservation is a conservative cost proxy, not an observed probability.
        return (True, not known, -Fraction(task.reward, floor), *base)

    def choose(
        self,
        tasks: tuple[TaskSpec, ...],
        offers: tuple[SolverOffer, ...],
        progress: tuple[TaskProgress, ...],
        budget: BudgetSnapshot,
        config: RunConfig,
        now: float,
    ) -> PlanDecision | None:
        floors = {}
        negotiated = {}
        self.deferred_task_ids.clear()
        eligible = []
        task_by_id = {t.task_id: t for t in tasks}
        for offer in offers:
            if offer.solver_id == "generic" and self.controller is not None:
                from nsu_budget_agent.orchestration.decision import State

                route = self.controller.store.get(task_by_id[offer.task_id])
                if route.state not in (State.UNSOLVED, State.UNCERTAIN):
                    continue
                calls = self.controller.minimum_calls(route.task)
                steps = self.controller.minimum_steps(route.task)
                if (
                    config.attempt_limits.max_llm_calls < calls
                    or config.attempt_limits.max_steps < steps
                ):
                    continue
                try:
                    reserve = self.controller.minimum_reservation(route.task)
                except ValueError:
                    continue
                available = min(budget.available_units, config.attempt_limits.max_cost_units)
                if self.controller.reserve_mandatory_calls:
                    proposal = self.controller.negotiate_caps(
                        route.task, available, config.attempt_limits.allowed_models
                    )
                    if proposal is None:
                        self.deferred_task_ids.add(offer.task_id)
                        continue
                    negotiated[offer.task_id], floors[offer.task_id] = proposal
                else:
                    floors[offer.task_id] = reserve * min(
                        self.controller.route_call_cap(route.task),
                        config.attempt_limits.max_llm_calls,
                    )
                if not self.controller.reserve_mandatory_calls and reserve > available:
                    self.deferred_task_ids.add(offer.task_id)
                    continue
            eligible.append(offer)
        offers = tuple(eligible)
        decision = super().choose(tasks, offers, progress, budget, config, now)
        if decision is None or decision.solver_id != "generic" or self.controller is None:
            return decision
        if decision.task_id in negotiated:
            route = self.controller.store.get(task_by_id[decision.task_id])
            route.negotiated_caps = negotiated[decision.task_id]
            route.preflight_floor = floors[decision.task_id]
        states = {s.task_id: s for s in progress}
        waiting = sum(
            offer.solver_id == "generic"
            and offer.applicability != "no"
            and (
                offer.task_id not in states
                or (
                    states[offer.task_id].status in ("pending", "deferred")
                    and "generic" not in states[offer.task_id].attempted_solvers
                )
            )
            for offer in offers
        )
        share = budget.available_units // max(1, waiting)
        return replace(
            decision,
            budget=replace(
                decision.budget,
                max_cost_units=min(
                    decision.budget.max_cost_units, max(share, floors.get(decision.task_id, 0))
                ),
            ),
        )
