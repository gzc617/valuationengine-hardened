"""Deterministic company fixtures for SAFE_MODE.

These figures are synthetic. They are not quotes, filings, or a cache of
market data. The same ticker always produces the same company, with no
clock, disk, or network access. Statement-model balances below are part of
that synthetic profile so the opt-in DCF can run offline.
"""

from __future__ import annotations

from valuationengine.core.models import Company
from valuationengine.validation import validate_history_years

# Named profiles used to exercise safe mode offline. Any other valid
# ticker gets a stable profile derived only from its symbol.
_NAMED: dict[str, dict[str, float]] = {
    "DEMO": {
        "revenue0": 1000.0,
        "growth": 0.08,
        "margin": 0.18,
        "price": 50.0,
        "shares": 100.0,
        "cash": 200.0,
        "debt": 300.0,
        "beta": 1.0,
        "tax": 0.21,
    },
    "PEER": {
        "revenue0": 800.0,
        "growth": 0.05,
        "margin": 0.14,
        "price": 32.0,
        "shares": 120.0,
        "cash": 90.0,
        "debt": 210.0,
        "beta": 1.2,
        "tax": 0.21,
    },
    "BETA": {
        "revenue0": 1500.0,
        "growth": 0.12,
        "margin": 0.22,
        "price": 75.0,
        "shares": 80.0,
        "cash": 400.0,
        "debt": 250.0,
        "beta": 1.4,
        "tax": 0.21,
    },
    "BULL": {
        "revenue0": 600.0,
        "growth": 0.16,
        "margin": 0.25,
        "price": 40.0,
        "shares": 90.0,
        "cash": 150.0,
        "debt": 80.0,
        "beta": 0.9,
        "tax": 0.21,
    },
    "BEAR": {
        "revenue0": 700.0,
        "growth": 0.03,
        "margin": 0.10,
        "price": 22.0,
        "shares": 140.0,
        "cash": 40.0,
        "debt": 360.0,
        "beta": 1.3,
        "tax": 0.21,
    },
}


def fixture_company(ticker: str, history_years: int = 5) -> Company:
    """Build a Company from a named fixture or a symbol-derived profile."""
    history_years = validate_history_years(history_years)

    spec = _NAMED.get(ticker) or _derived_profile(ticker)
    growth = float(spec["growth"])
    margin = float(spec["margin"])
    revenue0 = float(spec["revenue0"])
    revenue = [revenue0 * ((1.0 + growth) ** year) for year in range(history_years)]
    ebit = [value * margin for value in revenue]
    depreciation = [value * 0.03 for value in revenue]
    ebitda = [operating + da for operating, da in zip(ebit, depreciation)]
    capex = [value * 0.04 for value in revenue]
    change_in_nwc = [value * 0.01 for value in revenue]
    net_ppe = [revenue[0] * 0.40]
    for index in range(1, len(revenue)):
        net_ppe.append(net_ppe[-1] + capex[index] - depreciation[index])
    receivables = [value * 0.12 for value in revenue]
    inventory = [value * 0.08 for value in revenue]
    other_operating_current_assets = [value * 0.02 for value in revenue]
    payables = [value * 0.07 for value in revenue]
    other_operating_current_liabilities = [value * 0.03 for value in revenue]
    shares = float(spec["shares"])
    price = float(spec["price"])
    fiscal_year_ends = [f"Y{index}" for index in range(history_years)]

    return Company(
        ticker=ticker,
        name=f"{ticker} Fixture Co",
        revenue=revenue,
        ebit=ebit,
        ebitda=ebitda,
        depreciation_amortization=depreciation,
        capex=capex,
        change_in_nwc=change_in_nwc,
        effective_tax_rate=float(spec["tax"]),
        cash=float(spec["cash"]),
        total_debt=float(spec["debt"]),
        shares_outstanding=shares,
        current_price=price,
        market_cap=shares * price,
        beta=float(spec["beta"]),
        fiscal_year_ends=fiscal_year_ends,
        net_ppe=net_ppe,
        receivables=receivables,
        inventory=inventory,
        other_operating_current_assets=other_operating_current_assets,
        payables=payables,
        other_operating_current_liabilities=other_operating_current_liabilities,
        preferred_equity=20.0,
        noncontrolling_interest=10.0,
        other_adjustments=-5.0,
        diluted_shares=shares * 1.05,
        missing_fields=(),
    )


def _derived_profile(symbol: str) -> dict[str, float]:
    acc = 0
    for char in symbol:
        acc = (acc * 33 + ord(char)) % 1_000_003
    return {
        "revenue0": 100.0 + (acc % 400),
        "growth": 0.04 + (acc % 17) / 100.0,
        "margin": 0.08 + ((acc // 17) % 21) / 100.0,
        "price": 10.0 + (acc % 90),
        "shares": 50.0 + (acc % 50),
        "cash": float(acc % 80),
        "debt": float(50 + (acc % 120)),
        "beta": 0.8 + (acc % 8) / 10.0,
        "tax": 0.21,
    }
