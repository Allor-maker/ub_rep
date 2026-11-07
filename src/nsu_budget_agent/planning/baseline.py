"""Pure baseline policy over Contracts v1 snapshots; never executes a decision."""

from fractions import Fraction

from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    BudgetSnapshot,
    ContractError,
    PlanDecision,
    RunConfig,
    SolverOffer,
    TaskProgress,
    TaskSpec,
    ToolSpec,
    _ids,
    _items,
    _number,
    _unique,
)
from nsu_budget_agent.contracts.ports import Planner


class BaselinePlanner:
    """tools and deterministic_solvers are trusted startup policy, never task data.

    A deterministic solver is explicitly forbidden LLM calls. A zero *estimate*
    alone cannot confer that guarantee. Global reservations remain A2's job.
    """

    def __init__(
        self,
        *,
        tools: tuple[ToolSpec, ...] = (),
        deterministic_solvers: tuple[str, ...] = (),
    ) -> None:
        _items(tools, ToolSpec, "tools")
        _unique(tuple(tool.name for tool in tools), "tool name")
        _ids(deterministic_solvers, "deterministic solver")
        self._tools = frozenset(tool.name for tool in tools if tool.available)
        self._deterministic = frozenset(deterministic_solvers)

    @staticmethod
    def _rank(task: TaskSpec, offer: SolverOffer, state: TaskProgress, no_llm: bool) -> tuple:
        complete = offer.expected_cost_units is not None and offer.success_probability is not None
        # The +1 is a scoring regularizer in Units, not an invented cost estimate.
        score = (
            Fraction(task.reward)
            * Fraction(str(offer.success_probability))
            / (1 + offer.expected_cost_units)
            if complete and offer.expected_cost_units is not None
            else Fraction(0)
        )
        return (
            not no_llm,
            offer.applicability != "yes",
            not complete,
            -score,
            -task.reward,
            offer.expected_cost_units is None,
            offer.expected_cost_units if offer.expected_cost_units is not None else 0,
            offer.success_probability is None,
            -offer.success_probability if offer.success_probability is not None else 0,
            offer.expected_duration_s is None,
            offer.expected_duration_s if offer.expected_duration_s is not None else 0,
            offer.solver_id in state.attempted_solvers,
            task.task_id,
            offer.solver_id,
        )

    def choose(
        self,
        tasks: tuple[TaskSpec, ...],
        offers: tuple[SolverOffer, ...],
        progress: tuple[TaskProgress, ...],
        budget: BudgetSnapshot,
        config: RunConfig,
        now: float,
    ) -> PlanDecision | None:
        _number(now, "now", minimum=0)
        _items(tasks, TaskSpec, "tasks")
        _items(offers, SolverOffer, "offers")
        _items(progress, TaskProgress, "progress")
        _unique(tuple(task.task_id for task in tasks), "task_id")
        _unique(tuple(state.task_id for state in progress), "progress task_id")
        _unique(tuple((offer.task_id, offer.solver_id) for offer in offers), "solver offer")
        task_by_id = {task.task_id: task for task in tasks}
        states = {state.task_id: state for state in progress}
        if not set(states).issubset(task_by_id) or any(
            offer.task_id not in task_by_id for offer in offers
        ):
            raise ContractError("progress/offers must refer to loaded tasks")
        if budget.limit_units != config.total_budget_units:
            raise ContractError("planner requires the matching global budget snapshot")
        if now >= config.deadline_monotonic:
            return None
        if sum(state.status == "running" for state in progress) >= config.max_parallel_tasks:
            return None
        limits = config.attempt_limits
        duration = min(limits.max_duration_s, config.deadline_monotonic - now)
        deadline = min(config.deadline_monotonic, now + duration)
        if deadline <= now:
            return None
        cap = min(limits.max_cost_units, budget.available_units)
        coverage = config.mode == "coverage" or (config.mode == "auto" and len(tasks) == 2)
        candidates: list[tuple[tuple, PlanDecision]] = []
        waiting_first = False
        for offer in offers:
            task = task_by_id[offer.task_id]
            state = states.get(task.task_id, TaskProgress(task_id=task.task_id, status="pending"))
            if state.status not in ("pending", "deferred"):
                continue
            if len(state.attempted_solvers) >= limits.max_attempts_per_task:
                continue
            if offer.applicability == "no" or not self._tools.issuperset(offer.required_tools):
                continue
            if (
                state.last_result is not None
                and state.last_result.status == "unsupported"
                and state.last_result.solver_id == offer.solver_id
            ):
                continue
            no_llm = offer.solver_id in self._deterministic or limits.max_llm_calls == 0
            allocation = 0 if no_llm else cap
            if coverage and not state.attempted_solvers:
                wake = max(now, state.not_before_monotonic)
                future_cap = (
                    0
                    if no_llm
                    else min(limits.max_cost_units, max(0, budget.limit_units - budget.spent_units))
                )
                could_use_budget = no_llm or (
                    future_cap >= 10 and (not budget.blocked or budget.held_units > 0)
                )
                future_cost_fits = (
                    offer.expected_cost_units is None or offer.expected_cost_units <= future_cap
                )
                future_duration = min(limits.max_duration_s, config.deadline_monotonic - wake)
                future_time_fits = wake < config.deadline_monotonic and (
                    offer.expected_duration_s is None
                    or offer.expected_duration_s <= future_duration
                )
                unavailable_now = wake > now or (
                    not no_llm
                    and budget.held_units > 0
                    and (
                        budget.blocked
                        or allocation < 10
                        or (
                            offer.expected_cost_units is not None
                            and offer.expected_cost_units > allocation
                        )
                    )
                )
                if could_use_budget and future_cost_fits and future_time_fits and unavailable_now:
                    waiting_first = True
            if now < state.not_before_monotonic:
                continue
            if not no_llm and (budget.blocked or allocation < 10):
                continue
            if offer.expected_cost_units is not None and offer.expected_cost_units > allocation:
                continue
            if offer.expected_duration_s is not None and offer.expected_duration_s > duration:
                continue
            attempt = AttemptBudget(
                max_cost_units=allocation,
                max_llm_calls=0 if no_llm else min(limits.max_llm_calls, limits.max_steps),
                max_steps=limits.max_steps,
                deadline_monotonic=deadline,
                allowed_models=() if no_llm else limits.allowed_models,
                tool_limits=limits.tool_limits,
            )
            rank = (
                bool(state.attempted_solvers) if coverage else False,
                *self._rank(task, offer, state, no_llm),
            )
            candidates.append(
                (
                    rank,
                    PlanDecision(task_id=task.task_id, solver_id=offer.solver_id, budget=attempt),
                )
            )
        if not candidates:
            return None
        if not coverage and any(
            task_by_id[decision.task_id].reward > 0 for _, decision in candidates
        ):
            candidates = [
                (rank, decision)
                for rank, decision in candidates
                if task_by_id[decision.task_id].reward > 0 or decision.budget.max_llm_calls == 0
            ]
        best_rank, best = min(candidates, key=lambda candidate: candidate[0])
        if coverage and waiting_first and best_rank[0]:
            return None
        return best


def next_not_before_wakeup(
    planner: Planner,
    tasks: tuple[TaskSpec, ...],
    offers: tuple[SolverOffer, ...],
    progress: tuple[TaskProgress, ...],
    budget: BudgetSnapshot,
    config: RunConfig,
    now: float,
) -> float | None:
    """Earliest time gate yielding a decision under the current immutable snapshot.

    Call only after choose returned None. Resource or running-task changes can make
    a decision possible earlier, so the caller must also await those events.
    """
    _number(now, "now", minimum=0)
    times = sorted(
        {
            state.not_before_monotonic
            for state in progress
            if state.status in ("pending", "deferred")
            and now < state.not_before_monotonic < config.deadline_monotonic
        }
    )
    for wake in times:
        if planner.choose(tasks, offers, progress, budget, config, wake) is not None:
            return wake
    return None
