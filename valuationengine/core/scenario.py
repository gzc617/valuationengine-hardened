"""Bull, base, and bear scenario runner"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from valuationengine.core import dcf
from valuationengine.core.models import Assumptions, Company


def run(
    company: Company,
    scenarios: dict[str, Assumptions],
    valuation_fn: Callable | None = None,
) -> dict[str, object]:
    """
    Run a valuation for each named scenario.

    Args:
        company: Company to value.
        scenarios: Mapping of scenario name to Assumptions.
        valuation_fn: Valuation runner; defaults to dcf.run.
        Statement-model assumptions stay on the statement DCF.

    Returns:
        Dict mapping scenario name to the valuation result object.
    """
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    fn = valuation_fn or dcf.run
    return {name: fn(company, assumptions) for name, assumptions in scenarios.items()}


def build_bull_base_bear(
    base: Assumptions,
    growth_delta: float = 0.03,
    margin_delta: float = 0.03,
) -> dict[str, Assumptions]:
    """
    Build bull, base, and bear scenarios from a base Assumptions object.

    Bull: revenue_growth + delta, operating_margin + delta
    Bear: revenue_growth - delta, operating_margin - delta

    Args:
        base: Base-case assumptions to adjust.
        growth_delta: Adjustment to revenue growth for bull/bear.
        margin_delta: Adjustment to operating margin for bull/bear.

    Returns:
        Dict with keys 'bull', 'base', 'bear' mapping to Assumptions copies.
    """
    bull = deepcopy(base)
    bear = deepcopy(base)
    base_copy = deepcopy(base)

    bull.revenue_growth = _adjust_scalar_or_list(base.revenue_growth, growth_delta)
    bull.operating_margin = _adjust_scalar_or_list(base.operating_margin, margin_delta)
    bear.revenue_growth = _adjust_scalar_or_list(base.revenue_growth, -growth_delta)
    bear.operating_margin = _adjust_scalar_or_list(base.operating_margin, -margin_delta)

    return {"bull": bull, "base": base_copy, "bear": bear}


def _adjust_scalar_or_list(value: float | list[float], delta: float) -> float | list[float]:
    if isinstance(value, list):
        return [v + delta for v in value]
    return value + delta
