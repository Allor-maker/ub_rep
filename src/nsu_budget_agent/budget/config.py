"""Opt-in BUDGET reader; the external unit must be chosen by the caller."""

import os
from collections.abc import Mapping
from decimal import Context, Decimal, DecimalException, InvalidOperation
from typing import Literal, cast

from nsu_budget_agent.contracts.models import ContractError, Units

BudgetUnit = Literal["competition_tokens", "units_x10"]


def parse_budget(value: str, *, unit: BudgetUnit) -> Units:
    """Convert competition units by floor(10 * value), or accept integral Units.

    There is deliberately no default unit and no conversion through binary float.
    """
    if not isinstance(value, str) or not value.strip():
        raise ContractError("BUDGET must be a nonempty decimal string")
    if unit not in ("competition_tokens", "units_x10"):
        raise ContractError("BUDGET unit must be explicitly supported")
    try:
        # Parsing with a private context also keeps invalid input from changing
        # the caller's flags or depending on its InvalidOperation trap.
        context = Context(traps=[InvalidOperation])
        amount = Decimal(value.strip(), context=context)
        if not amount.is_finite() or amount < 0:
            raise ContractError("BUDGET must be finite and nonnegative")
        if amount.is_zero():
            return 0
        if unit == "competition_tokens":
            # Constructing a Decimal from its tuple is exact and does not use
            # the caller's precision, rounding, Emax, Emin, traps or flags.
            sign, digits, exponent = amount.as_tuple()
            scaled = Decimal((sign, digits, cast(int, exponent) + 1), context=context)
            # Amount is nonnegative, so truncating to int is floor.
            return int(scaled)
        integral = int(amount)
        if amount != integral:
            raise ContractError("BUDGET in units_x10 must be integral")
        return integral
    except DecimalException as exc:
        raise ContractError("BUDGET must be a representable decimal number") from exc


def budget_from_env(*, unit: BudgetUnit, environ: Mapping[str, str] | None = None) -> Units:
    """Read BUDGET explicitly, without reading configuration at module import."""
    environment = os.environ if environ is None else environ
    if "BUDGET" not in environment:
        raise ContractError("BUDGET is required; no implicit budget is configured")
    return parse_budget(environment["BUDGET"], unit=unit)
