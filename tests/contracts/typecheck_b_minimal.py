"""Structural compatibility of production B with Contracts v1."""

from nsu_budget_agent.contracts.ports import LLMGateway, Solver, ToolExecutor, Validator
from nsu_budget_agent.solvers.composition import DisabledLLMGateway
from nsu_budget_agent.solvers.deterministic import DeterministicSolver, UnsupportedGenericSolver
from nsu_budget_agent.tools import LocalToolExecutor
from nsu_budget_agent.validation import LocalValidator

tools: ToolExecutor = LocalToolExecutor()
validator: Validator = LocalValidator(tools)
math: Solver = DeterministicSolver("math")
search: Solver = DeterministicSolver("search")
data_analysis: Solver = DeterministicSolver("data_analysis")
generic: Solver = UnsupportedGenericSolver()
gateway: LLMGateway = DisabledLLMGateway()
