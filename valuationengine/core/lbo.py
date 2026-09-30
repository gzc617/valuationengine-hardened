"""LBO model with sponsor IRR and MOIC"""

import pandas as pd

from valuationengine.core.models import Assumptions, Company, LBOResult


def run(company: Company, assumptions: Assumptions) -> LBOResult:
    """
    Run a sponsor LBO model with debt paydown and exit valuation.

    MOIC = exit_equity / equity_check
    IRR ≈ MOIC^(1 / hold_period) - 1 (single in / single out approximation)

    Levered cash flow applies taxes on EBT (EBIT minus interest) so the interest
    tax shield flows through to cash available for debt service.

    Args:
        company: Target company fundamentals.
        assumptions: LBO and operating assumptions.

    Returns:
        LBOResult with sources/uses, projections, debt schedule, and returns.
    """
    if company.latest_ebitda <= 0:
        raise ValueError("Latest EBITDA must be positive for LBO entry valuation.")
    if assumptions.hold_period_years <= 0:
        raise ValueError("hold_period_years must be positive.")
    if company.latest_revenue <= 0:
        raise ValueError("Latest revenue must be positive for LBO projection.")

    purchase_price = company.latest_ebitda * assumptions.entry_ev_ebitda_multiple
    fees = purchase_price * assumptions.transaction_fees_pct
    debt_amount = purchase_price * assumptions.debt_pct_purchase
    equity_check = purchase_price - debt_amount + fees
    if equity_check <= 0:
        raise ValueError("Equity check must be positive; reduce debt_pct_purchase or fees.")

    sources_and_uses = {
        "purchase_price": purchase_price,
        "equity_check": equity_check,
        "debt": debt_amount,
        "fees": fees,
    }

    growth = _broadcast_hold_period(assumptions.revenue_growth, assumptions.hold_period_years)
    margins = _broadcast_hold_period(assumptions.operating_margin, assumptions.hold_period_years)

    proj_rows: list[dict] = []
    debt_rows: list[dict] = []
    revenue = company.latest_revenue
    balance = debt_amount
    rate = assumptions.lbo_debt_interest_rate
    mand_pct = assumptions.mandatory_amortization_pct
    sweep_pct = assumptions.cash_sweep_pct

    for year_idx, (g, margin) in enumerate(zip(growth, margins), start=1):
        prior_revenue = revenue
        revenue = revenue * (1 + g)
        ebit = revenue * margin
        da = revenue * assumptions.da_pct_revenue
        ebitda = ebit + da
        capex = revenue * assumptions.capex_pct_revenue
        change_nwc = (revenue - prior_revenue) * assumptions.nwc_pct_revenue

        beginning_balance = balance
        interest = beginning_balance * rate
        taxes = max(0.0, ebit - interest) * assumptions.tax_rate
        cfads = ebitda - taxes - capex - change_nwc

        mandatory_paydown = min(debt_amount * mand_pct, beginning_balance)
        excess_fcf = max(0.0, cfads - interest - mandatory_paydown)
        max_sweep = beginning_balance - mandatory_paydown
        sweep_paydown = min(excess_fcf * sweep_pct, max_sweep)
        ending_balance = beginning_balance - mandatory_paydown - sweep_paydown

        proj_rows.append(
            {
                "year": year_idx,
                "revenue": revenue,
                "ebitda": ebitda,
                "ebit": ebit,
                "interest": interest,
                "fcf": cfads - interest,
                "debt_paydown": mandatory_paydown + sweep_paydown,
            }
        )
        debt_rows.append(
            {
                "year": year_idx,
                "beginning_balance": beginning_balance,
                "interest_expense": interest,
                "mandatory_paydown": mandatory_paydown,
                "sweep_paydown": sweep_paydown,
                "ending_balance": ending_balance,
            }
        )
        balance = ending_balance

    projection = pd.DataFrame(proj_rows)
    debt_schedule = pd.DataFrame(debt_rows)

    exit_ebitda = projection.iloc[-1]["ebitda"]
    exit_ev = exit_ebitda * assumptions.exit_lbo_ev_ebitda_multiple
    exit_debt = float(debt_schedule.iloc[-1]["ending_balance"])
    exit_equity = exit_ev - exit_debt
    exit_info = {
        "exit_ev": exit_ev,
        "exit_debt": exit_debt,
        "exit_equity": exit_equity,
        "year": assumptions.hold_period_years,
    }

    moic = exit_equity / equity_check
    irr = moic ** (1 / assumptions.hold_period_years) - 1

    return LBOResult(
        company=company,
        assumptions=assumptions,
        sources_and_uses=sources_and_uses,
        projection=projection,
        debt_schedule=debt_schedule,
        exit=exit_info,
        irr=irr,
        moic=moic,
    )


def _broadcast_hold_period(value: float | list[float], years: int) -> list[float]:
    if isinstance(value, list):
        if len(value) >= years:
            return list(value[:years])
        return list(value) + [value[-1]] * (years - len(value))
    return [float(value)] * years
