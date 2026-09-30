"""Data models: Company, Assumptions, Result"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import pandas as pd


def _clip_or_fallback(value: float, lo: float, hi: float, fallback: float) -> float:
    """Clip value into [lo, hi]; if value is None, NaN, or infinite, return fallback instead."""
    if value is None or not math.isfinite(value):
        return fallback
    return max(lo, min(hi, value))


@dataclass
class Company:
    """Historical fundamentals and current market data for a public company."""

    ticker: str
    name: str
    revenue: list[float]
    ebit: list[float]
    ebitda: list[float]
    depreciation_amortization: list[float]
    capex: list[float]
    change_in_nwc: list[float]
    effective_tax_rate: float
    cash: float
    total_debt: float
    shares_outstanding: float
    current_price: float
    market_cap: float
    beta: float

    @property
    def net_debt(self) -> float:
        return self.total_debt - self.cash

    @property
    def latest_revenue(self) -> float:
        return self.revenue[-1]

    @property
    def latest_ebit(self) -> float:
        return self.ebit[-1]

    @property
    def latest_ebitda(self) -> float:
        return self.ebitda[-1]

    @property
    def historical_revenue_cagr(self) -> float:
        if len(self.revenue) < 2:
            raise ValueError("At least two years of revenue are required to compute CAGR.")
        start, end = self.revenue[0], self.revenue[-1]
        if start <= 0 or end <= 0:
            raise ValueError("Revenue must be positive to compute CAGR.")
        periods = len(self.revenue) - 1
        return (end / start) ** (1 / periods) - 1

    @property
    def avg_operating_margin(self) -> float:
        if not self.revenue:
            raise ValueError("Revenue history is empty.")
        margins = [e / r for e, r in zip(self.ebit, self.revenue) if r != 0]
        if not margins:
            raise ValueError("Cannot compute operating margin from zero revenue.")
        return sum(margins) / len(margins)

    @property
    def avg_capex_pct_revenue(self) -> float:
        return _mean_ratio(self.capex, self.revenue, "capex")

    @property
    def avg_da_pct_revenue(self) -> float:
        return _mean_ratio(self.depreciation_amortization, self.revenue, "D&A")

    @property
    def avg_nwc_pct_revenue(self) -> float:
        return _mean_ratio(self.change_in_nwc, self.revenue, "NWC change")


def _mean_ratio(numerator: list[float], denominator: list[float], label: str) -> float:
    if not denominator:
        raise ValueError(f"Revenue history is empty; cannot compute {label} ratio.")
    ratios = [n / d for n, d in zip(numerator, denominator) if d != 0]
    if not ratios:
        raise ValueError(f"Cannot compute {label} as a share of revenue.")
    return sum(ratios) / len(ratios)


@dataclass
class Assumptions:
    """DCF and LBO levers with sensible US large-cap defaults."""

    projection_years: int = 5
    revenue_growth: float | list[float] = 0.08
    operating_margin: float | list[float] = 0.20
    capex_pct_revenue: float = 0.05
    da_pct_revenue: float = 0.04
    nwc_pct_revenue: float = 0.02
    tax_rate: float = 0.25
    risk_free_rate: float = 0.045
    equity_risk_premium: float = 0.055
    cost_of_debt: float = 0.06
    target_debt_weight: float = 0.30
    terminal_growth: float = 0.025
    use_exit_multiple: bool = False
    exit_ev_ebitda_multiple: float = 10.0
    entry_ev_ebitda_multiple: float = 10.0
    debt_pct_purchase: float = 0.60
    lbo_debt_interest_rate: float = 0.08
    mandatory_amortization_pct: float = 0.05
    cash_sweep_pct: float = 0.75
    hold_period_years: int = 5
    exit_lbo_ev_ebitda_multiple: float = 10.0
    transaction_fees_pct: float = 0.02

    def get_growth_path(self) -> list[float]:
        return _broadcast_path(self.revenue_growth, self.projection_years, "revenue_growth")

    def get_margin_path(self) -> list[float]:
        return _broadcast_path(self.operating_margin, self.projection_years, "operating_margin")

    @classmethod
    def calibrated_for(cls, company: "Company", **overrides) -> "Assumptions":
        """
        Build an Assumptions instance whose revenue_growth, operating_margin,
        capex_pct_revenue, da_pct_revenue, and nwc_pct_revenue default to this
        specific company's own historical averages instead of generic constants.
        Any of those fields (or any other Assumptions field) passed as a keyword
        argument with a non-None value overrides the calibrated value.

        terminal_growth, risk_free_rate, equity_risk_premium, and
        target_debt_weight are intentionally NOT calibrated to company history;
        they remain at the generic dataclass defaults unless explicitly overridden.

        Each company-derived field is read defensively: if accessing the
        property raises, or returns None/NaN/inf, fall back to the generic
        dataclass default for that field instead of propagating bad data.
        """
        base = cls()

        try:
            base.revenue_growth = _clip_or_fallback(
                company.historical_revenue_cagr, -0.10, 0.40, base.revenue_growth
            )
        except Exception:
            pass

        try:
            base.operating_margin = _clip_or_fallback(
                company.avg_operating_margin, 0.01, 0.60, base.operating_margin
            )
        except Exception:
            pass

        try:
            base.capex_pct_revenue = _clip_or_fallback(
                company.avg_capex_pct_revenue, 0.0, 0.30, base.capex_pct_revenue
            )
        except Exception:
            pass

        try:
            base.da_pct_revenue = _clip_or_fallback(
                company.avg_da_pct_revenue, 0.0, 0.30, base.da_pct_revenue
            )
        except Exception:
            pass

        try:
            base.nwc_pct_revenue = _clip_or_fallback(
                company.avg_nwc_pct_revenue, -0.10, 0.20, base.nwc_pct_revenue
            )
        except Exception:
            pass

        valid_fields = {f.name for f in fields(cls)}
        for key, value in overrides.items():
            if key not in valid_fields:
                raise ValueError(f"'{key}' is not a valid Assumptions field.")
            if value is not None:
                setattr(base, key, value)
        return base


def _broadcast_path(value: float | list[float], length: int, name: str) -> list[float]:
    if length <= 0:
        raise ValueError(f"projection_years must be positive; got {length}.")
    if isinstance(value, list):
        if len(value) != length:
            raise ValueError(f"{name} list length {len(value)} must equal projection_years ({length}).")
        return list(value)
    return [float(value)] * length


@dataclass
class DCFResult:
    """Output of a discounted cash flow valuation."""

    company: Company
    assumptions: Assumptions
    projection: pd.DataFrame
    wacc: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    value_per_share: float
    upside: float

    def summary(self) -> str:
        lines = [
            f"DCF Valuation — {self.company.ticker} ({self.company.name})",
            f"  WACC:              {self.wacc:.2%}",
            f"  Enterprise Value:  ${self.enterprise_value:,.0f}",
            f"  Equity Value:      ${self.equity_value:,.0f}",
            f"  Value per Share:   ${self.value_per_share:,.2f}",
            f"  Current Price:     ${self.company.current_price:,.2f}",
            f"  Upside:            {self.upside:.1%}",
            f"  Terminal Value:    ${self.terminal_value:,.0f} (PV: ${self.pv_terminal_value:,.0f})",
        ]
        return "\n".join(lines)


@dataclass
class LBOResult:
    """Output of a leveraged buyout model."""

    company: Company
    assumptions: Assumptions
    sources_and_uses: dict
    projection: pd.DataFrame
    debt_schedule: pd.DataFrame
    exit: dict
    irr: float
    moic: float

    def summary(self) -> str:
        su = self.sources_and_uses
        ex = self.exit
        lines = [
            f"LBO Model — {self.company.ticker} ({self.company.name})",
            f"  Purchase EV:       ${su['purchase_price']:,.0f}",
            f"  Equity Check:      ${su['equity_check']:,.0f}",
            f"  Debt:              ${su['debt']:,.0f}",
            f"  Transaction Fees:  ${su['fees']:,.0f}",
            f"  Exit EV (Y{ex['year']}):     ${ex['exit_ev']:,.0f}",
            f"  Exit Equity:       ${ex['exit_equity']:,.0f}",
            f"  MOIC:              {self.moic:.2f}x",
            f"  IRR:               {self.irr:.1%}",
        ]
        return "\n".join(lines)


def assumption_field_names() -> set[str]:
    return {f.name for f in fields(Assumptions)}
