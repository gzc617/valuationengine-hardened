"""Reverse DCF solver: backs out market-implied assumptions"""

from __future__ import annotations

from copy import deepcopy

from scipy.optimize import brentq

from valuationengine.core import dcf
from valuationengine.core.models import Assumptions, Company, assumption_field_names


def solve(
    company: Company,
    base_assumptions: Assumptions,
    field: str,
    target: str = "market_cap",
    bracket: tuple[float, float] = (-0.10, 0.50),
) -> dict:
    """
    Solve for the assumption that makes DCF output match a market target.

    Finds x such that f(x) = 0 where f compares DCF output to market_cap or current_price.

    Args:
        company: Company to value.
        base_assumptions: Base assumptions with all fields except `field` held fixed.
        field: Assumptions attribute to solve for (e.g. revenue_growth).
        target: 'market_cap' or 'current_price'.
        bracket: (low, high) search interval for brentq.

    Returns:
        Dict with field, implied_value, target, target_value, and interpretation.
    """
    if field not in assumption_field_names():
        raise ValueError(f"Unknown Assumptions field '{field}'.")
    if target not in {"market_cap", "current_price"}:
        raise ValueError("target must be 'market_cap' or 'current_price'.")

    target_value = company.market_cap if target == "market_cap" else company.current_price

    def objective(x: float) -> float:
        assumptions = deepcopy(base_assumptions)
        setattr(assumptions, field, x)
        result = dcf.run(company, assumptions)
        if target == "market_cap":
            return result.equity_value - company.market_cap
        return result.value_per_share - company.current_price

    try:
        implied_value = brentq(objective, bracket[0], bracket[1])
    except ValueError as exc:
        raise ValueError(
            f"Could not solve for '{field}' over bracket {bracket}. "
            f"Ensure the target is bracketed by opposite-sign endpoints. Original error: {exc}"
        ) from exc

    return {
        "field": field,
        "implied_value": implied_value,
        "target": target,
        "target_value": target_value,
        "interpretation": _interpretation(field, implied_value, target, company, base_assumptions),
    }


def _interpretation(
    field: str,
    value: float,
    target: str,
    company: Company,
    assumptions: Assumptions,
) -> str:
    pct = value * 100
    years = assumptions.projection_years
    target_label = "market cap" if target == "market_cap" else "current price"

    if field == "revenue_growth":
        return (
            f"At the {target_label}, the market is pricing in {pct:.1f}% annual revenue growth "
            f"for {years} years."
        )
    if field == "operating_margin":
        return (
            f"At the {target_label}, the market is pricing in a {pct:.1f}% EBIT margin "
            f"over the forecast."
        )
    if field == "terminal_growth":
        return (
            f"At the {target_label}, the market is pricing in {pct:.1f}% perpetual terminal growth."
        )
    return (
        f"At the {target_label}, the implied {field.replace('_', ' ')} is {pct:.1f}% "
        f"to reconcile the DCF with the market."
    )
