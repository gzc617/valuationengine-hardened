"""Debt amortization schedule"""

import pandas as pd


def build_schedule(
    initial_debt: float,
    interest_rate: float,
    mandatory_amortization_pct: float,
    available_cash_flow: list[float],
    cash_sweep_pct: float,
    years: int,
) -> pd.DataFrame:
    """
    Build an LBO debt amortization schedule with mandatory paydown and cash sweep.

    For each year t:
        interest = beginning_balance * interest_rate
        mandatory = min(initial_debt * mandatory_amortization_pct, beginning_balance)
        excess_fcf = max(0, available_cash_flow[t-1] - interest - mandatory)
        sweep = min(excess_fcf * cash_sweep_pct, beginning_balance - mandatory)
        ending_balance = beginning_balance - mandatory - sweep

    Args:
        initial_debt: Opening debt balance at transaction close.
        interest_rate: Annual coupon rate as a decimal.
        mandatory_amortization_pct: Mandatory principal paydown as a fraction of original debt.
        available_cash_flow: FCF available for debt service, one entry per year.
        cash_sweep_pct: Fraction of excess FCF applied to optional paydown.
        years: Number of years in the schedule.

    Returns:
        DataFrame with columns year, beginning_balance, interest_expense,
        mandatory_paydown, sweep_paydown, ending_balance.
    """
    if initial_debt < 0:
        raise ValueError(f"initial_debt must be non-negative; got {initial_debt}.")
    if years <= 0:
        raise ValueError(f"years must be positive; got {years}.")
    if len(available_cash_flow) < years:
        raise ValueError(
            f"available_cash_flow must have at least {years} entries; got {len(available_cash_flow)}."
        )
    if not 0 <= cash_sweep_pct <= 1:
        raise ValueError(f"cash_sweep_pct must be between 0 and 1; got {cash_sweep_pct}.")
    if mandatory_amortization_pct < 0:
        raise ValueError(
            f"mandatory_amortization_pct must be non-negative; got {mandatory_amortization_pct}."
        )

    rows: list[dict] = []
    balance = initial_debt

    for year in range(1, years + 1):
        beginning_balance = balance
        interest_expense = beginning_balance * interest_rate
        mandatory_paydown = min(initial_debt * mandatory_amortization_pct, beginning_balance)
        excess_fcf = max(0.0, available_cash_flow[year - 1] - interest_expense - mandatory_paydown)
        max_sweep = beginning_balance - mandatory_paydown
        sweep_paydown = min(excess_fcf * cash_sweep_pct, max_sweep)
        ending_balance = beginning_balance - mandatory_paydown - sweep_paydown

        rows.append(
            {
                "year": year,
                "beginning_balance": beginning_balance,
                "interest_expense": interest_expense,
                "mandatory_paydown": mandatory_paydown,
                "sweep_paydown": sweep_paydown,
                "ending_balance": ending_balance,
            }
        )
        balance = ending_balance

    return pd.DataFrame(rows)
