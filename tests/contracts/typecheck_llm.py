"""The A3 class and its bound capability structurally satisfy unchanged v1 ports."""

from nsu_budget_agent.contracts.models import AttemptBudget
from nsu_budget_agent.contracts.ports import LLMGateway, LLMSession
from nsu_budget_agent.llm import NeuralDeepGateway


def as_gateway(gateway: NeuralDeepGateway) -> LLMGateway:
    return gateway


def session(gateway: NeuralDeepGateway, budget: AttemptBudget) -> LLMSession:
    return gateway.bind("attempt", 0, budget)
