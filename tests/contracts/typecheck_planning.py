"""A4 implementations satisfy unchanged Contracts v1 ports."""

from nsu_budget_agent.contracts.ports import Planner, SolverRegistry, TaskAdapter, TaskLoader
from nsu_budget_agent.planning import BaselinePlanner, StaticSolverRegistry
from nsu_budget_agent.task_io import DirectoryTaskLoader, OrganizerTaskAdapter

adapter: TaskAdapter = OrganizerTaskAdapter()
loader: TaskLoader = DirectoryTaskLoader(adapter)
planner: Planner = BaselinePlanner()
registry: SolverRegistry = StaticSolverRegistry()
