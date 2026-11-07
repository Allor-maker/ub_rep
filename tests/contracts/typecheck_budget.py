"""Mypy checks the A2 implementation against the existing A1 port."""

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.ports import BudgetController


def as_controller(ledger: BudgetLedger) -> BudgetController:
    return ledger
