"""2D sensitivity tables across any two assumptions"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import pandas as pd

from valuationengine.core import dcf
from valuationengine.core.models import Assumptions, Company, assumption_field_names


def run(
    company: Company,
    base_assumptions: Assumptions,
    x_field: str,
    x_values: list[float],
    y_field: str,
    y_values: list[float],
    output: str = "value_per_share",
    valuation_fn: Callable | None = None,
) -> pd.DataFrame:
    """
    Run a 2D sensitivity table over two Assumptions fields.

    Each cell runs valuation_fn(company, overridden_assumptions) and extracts `output`.

    Args:
        company: Company to value.
        base_assumptions: Base case assumptions copied per cell.
        x_field: Assumptions attribute name for column axis.
        x_values: Values to sweep on the x-axis.
        y_field: Assumptions attribute name for row axis.
        y_values: Values to sweep on the y-axis.
        output: Result attribute to capture (value_per_share, upside, irr, moic).
        valuation_fn: Valuation runner; defaults to dcf.run.

    Returns:
        DataFrame indexed by y_values with columns x_values.
    """
    if not x_values or not y_values:
        raise ValueError("x_values and y_values must be non-empty.")
    _validate_field(x_field)
    _validate_field(y_field)

    fn = valuation_fn or dcf.run
    table: dict[float, dict[float, float]] = {}

    for y in y_values:
        table[y] = {}
        for x in x_values:
            assumptions = deepcopy(base_assumptions)
            setattr(assumptions, x_field, x)
            setattr(assumptions, y_field, y)
            result = fn(company, assumptions)
            table[y][x] = _extract_output(result, output)

    return pd.DataFrame(table).T.reindex(y_values).reindex(columns=x_values)


def _validate_field(field: str) -> None:
    if field not in assumption_field_names():
        raise ValueError(f"Unknown Assumptions field '{field}'.")


def _extract_output(result: object, output: str) -> float:
    if not hasattr(result, output):
        raise ValueError(f"Result object has no attribute '{output}'.")
    return float(getattr(result, output))
