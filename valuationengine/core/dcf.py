"""DCF valuation engine"""

import numpy as np
import pandas as pd

from valuationengine.core.models import Assumptions, Company, DCFResult


def compute_wacc(assumptions: Assumptions, beta: float) -> float:
    """
    Compute WACC from CAPM cost of equity and after-tax cost of debt.

    Cost of equity = risk_free_rate + beta * equity_risk_premium
    Cost of debt (after tax) = cost_of_debt * (1 - tax_rate)
    WACC = (1 - D/V) * Re + (D/V) * Rd * (1 - T)

    Args:
        assumptions: WACC component assumptions.
        beta: Equity beta for CAPM.

    Returns:
        Weighted average cost of capital as a decimal.
    """
    if not 0 <= assumptions.target_debt_weight <= 1:
        raise ValueError(
            f"target_debt_weight must be between 0 and 1; got {assumptions.target_debt_weight}."
        )
    cost_of_equity = assumptions.risk_free_rate + beta * assumptions.equity_risk_premium
    cost_of_debt_after_tax = assumptions.cost_of_debt * (1 - assumptions.tax_rate)
    equity_weight = 1 - assumptions.target_debt_weight
    return equity_weight * cost_of_equity + assumptions.target_debt_weight * cost_of_debt_after_tax


def project_fcf(company: Company, assumptions: Assumptions) -> pd.DataFrame:
    """
    Project unlevered free cash flow over the forecast horizon.

    FCF = NOPAT + D&A - CapEx - ΔNWC, where NOPAT = EBIT * (1 - tax_rate).
    ΔNWC is modeled as nwc_pct_revenue times the incremental revenue change.

    Args:
        company: Company with latest revenue as the projection base.
        assumptions: Operating and tax assumptions.

    Returns:
        DataFrame with columns year, revenue, ebit, nopat, da, capex, change_nwc, fcf.
    """
    if company.latest_revenue <= 0:
        raise ValueError("Latest revenue must be positive to project FCF.")

    growth = assumptions.get_growth_path()
    margins = assumptions.get_margin_path()
    rows: list[dict] = []
    revenue = company.latest_revenue

    for year_idx, (g, margin) in enumerate(zip(growth, margins), start=1):
        prior_revenue = revenue
        revenue = revenue * (1 + g)
        ebit = revenue * margin
        nopat = ebit * (1 - assumptions.tax_rate)
        da = revenue * assumptions.da_pct_revenue
        capex = revenue * assumptions.capex_pct_revenue
        change_nwc = (revenue - prior_revenue) * assumptions.nwc_pct_revenue
        fcf = nopat + da - capex - change_nwc
        rows.append(
            {
                "year": year_idx,
                "revenue": revenue,
                "ebit": ebit,
                "nopat": nopat,
                "da": da,
                "capex": capex,
                "change_nwc": change_nwc,
                "fcf": fcf,
            }
        )

    return pd.DataFrame(rows)


def run(company: Company, assumptions: Assumptions) -> DCFResult:
    """
    Run a full DCF valuation.

    Terminal value (Gordon growth): TV = FCF_T * (1 + g) / (WACC - g)
    Terminal value (exit multiple): TV = EBITDA_T * exit_ev_ebitda_multiple
    Enterprise value = sum(PV of projected FCF) + PV of terminal value
    Equity value = EV - net debt; value per share = equity / shares outstanding

    Args:
        company: Company fundamentals and market data.
        assumptions: Valuation assumptions.

    Returns:
        DCFResult with projection, WACC, and valuation outputs.
    """
    if not company.revenue:
        raise ValueError("Company revenue history is empty.")
    if company.shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive.")
    if company.current_price <= 0:
        raise ValueError("current_price must be positive.")

    wacc = compute_wacc(assumptions, company.beta)
    projection = project_fcf(company, assumptions)
    terminal_fcf = projection.iloc[-1]["fcf"]

    if assumptions.use_exit_multiple:
        terminal_row = projection.iloc[-1]
        terminal_ebitda = terminal_row["ebit"] + terminal_row["da"]
        terminal_value = terminal_ebitda * assumptions.exit_ev_ebitda_multiple
    else:
        if wacc <= assumptions.terminal_growth:
            raise ValueError(
                f"WACC ({wacc:.4f}) must exceed terminal_growth ({assumptions.terminal_growth:.4f}) "
                "for Gordon growth terminal value."
            )
        terminal_value = terminal_fcf * (1 + assumptions.terminal_growth) / (
            wacc - assumptions.terminal_growth
        )

    years = np.arange(1, len(projection) + 1, dtype=float)
    discount_factors = 1 / (1 + wacc) ** years
    pv_fcf = projection["fcf"].to_numpy() * discount_factors
    projection = projection.copy()
    projection["discount_factor"] = discount_factors
    projection["pv_fcf"] = pv_fcf

    pv_terminal_value = terminal_value / (1 + wacc) ** len(projection)
    enterprise_value = float(pv_fcf.sum() + pv_terminal_value)
    equity_value = enterprise_value - company.net_debt
    value_per_share = equity_value / company.shares_outstanding
    upside = (value_per_share - company.current_price) / company.current_price

    return DCFResult(
        company=company,
        assumptions=assumptions,
        projection=projection,
        wacc=wacc,
        terminal_value=float(terminal_value),
        pv_terminal_value=float(pv_terminal_value),
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        upside=upside,
    )
