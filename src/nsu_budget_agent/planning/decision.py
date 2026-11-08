"""A4 ranking with a conservative equal-share cap for first paid attempts."""

from dataclasses import replace
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
                floors[offer.task_id] = reserve * min(
                    self.controller.route_call_cap(route.task), config.attempt_limits.max_llm_calls
                )
                if reserve > min(budget.available_units, config.attempt_limits.max_cost_units):
                    self.deferred_task_ids.add(offer.task_id)
                    continue
            eligible.append(offer)
        offers = tuple(eligible)
        decision = super().choose(tasks, offers, progress, budget, config, now)
        if decision is None or decision.solver_id != "generic" or self.controller is None:
            return decision
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
