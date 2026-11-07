"""Explicit startup-only registry; no task-controlled imports."""

from nsu_budget_agent.contracts.models import ContractError, _id
from nsu_budget_agent.contracts.ports import Solver


class StaticSolverRegistry:
    def __init__(self) -> None:
        self._solvers: dict[str, Solver] = {}
        self._frozen = False

    def register(self, solver: Solver) -> None:
        if self._frozen:
            raise ContractError("solver registration is closed")
        _id(solver.solver_id, "solver_id")
        if solver.solver_id in self._solvers:
            raise ContractError("duplicate solver ID")
        if not callable(getattr(solver, "assess", None)) or not callable(
            getattr(solver, "step", None)
        ):
            raise ContractError("solver must provide assess and step")
        self._solvers[solver.solver_id] = solver

    def _freeze(self) -> None:
        if "generic" not in self._solvers:
            raise ContractError("exactly one generic solver is required")
        self._frozen = True

    def all_solvers(self) -> tuple[Solver, ...]:
        self._freeze()
        return tuple(self._solvers[key] for key in sorted(self._solvers))

    def get(self, solver_id: str) -> Solver:
        self._freeze()
        try:
            return self._solvers[solver_id]
        except KeyError:
            raise ContractError("unknown solver ID") from None
