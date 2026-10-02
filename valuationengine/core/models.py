"""Data models: Company, Assumptions, Result"""

from __future__ import annotations

import datetime
import math
import statistics
from dataclasses import dataclass, field, fields

import pandas as pd

# How an assumption was chosen. Statement results and the CLI print these
# labels so a generic fallback is not described as reported history.
SOURCE_HISTORICAL_AVERAGE = "historical_average"
SOURCE_NORMALIZED = "normalized"
SOURCE_GENERIC_FALLBACK = "generic_fallback"
SOURCE_EXPLICIT = "explicit_override"
SOURCE_REPORTED = "reported"

# fiscal_year_ends observations are one fiscal year apart when the elapsed
# time falls in this band. A dropped year is about two years and is not
# treated as adjacent. Growth across that gap is annualized by the elapsed
# time. D&A uses only the immediately prior fiscal PPE.
IMMEDIATE_PRIOR_FISCAL_YEAR_MIN = 0.5
IMMEDIATE_PRIOR_FISCAL_YEAR_MAX = 1.5

_INTENSITY_DRIVER_FIELDS = (
    "revenue_growth",
    "operating_margin",
    "capex_pct_revenue",
    "da_pct_revenue",
    "nwc_pct_revenue",
    "tax_rate",
)
_STATEMENT_DRIVER_FIELDS = (
    "revenue_growth",
    "operating_margin",
    "capex_pct_revenue",
    "da_pct_opening_ppe",
    "receivables_pct_revenue",
    "inventory_pct_revenue",
    "other_operating_ca_pct_revenue",
    "payables_pct_revenue",
    "other_operating_cl_pct_revenue",
    "tax_rate",
)
NWC_RATIO_FIELDS = (
    "receivables_pct_revenue",
    "inventory_pct_revenue",
    "other_operating_ca_pct_revenue",
    "payables_pct_revenue",
    "other_operating_cl_pct_revenue",
)
OPERATING_NWC_COMPONENTS = (
    "receivables",
    "inventory",
    "other_operating_current_assets",
    "payables",
    "other_operating_current_liabilities",
)


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
    ebitda: list[float | None] | None
    depreciation_amortization: list[float | None] | None
    capex: list[float | None] | None
    change_in_nwc: list[float | None] | None
    effective_tax_rate: float
    # Non-operating cash: cash equivalents plus short-term investments.
    # None means the amount was not reported. It is not stored as zero.
    # The equity bridge adds this figure once. Operating NWC excludes it.
    cash: float | None
    # Book total debt. None means the claim is incomplete. It is not stored as zero.
    total_debt: float | None
    shares_outstanding: float
    current_price: float
    market_cap: float
    beta: float
    # Optional statement-model facts. None means the figure was not reported.
    # Historical lists, when present, line up with revenue (oldest fiscal period first).
    # Short-term investments are already included in ``cash`` when reported.
    # They are not added again in the equity bridge.
    short_term_investments: float | None = None
    fiscal_year_ends: list[str] | None = None
    net_ppe: list[float | None] | None = None
    receivables: list[float | None] | None = None
    inventory: list[float | None] | None = None
    other_operating_current_assets: list[float | None] | None = None
    payables: list[float | None] | None = None
    other_operating_current_liabilities: list[float | None] | None = None
    preferred_equity: float | None = None
    noncontrolling_interest: float | None = None
    other_adjustments: float | None = None
    diluted_shares: float | None = None
    missing_fields: tuple[str, ...] = ()

    @property
    def net_debt(self) -> float:
        if not _is_usable_number(self.total_debt) or not _is_usable_number(self.cash):
            raise ValueError(
                "Net debt requires reported cash and total debt. "
                "Missing cash or debt is not treated as zero."
            )
        return float(self.total_debt) - float(self.cash)

    @property
    def latest_net_ppe(self) -> float | None:
        """Closing net PPE of the latest revenue period, or None if it was not reported."""
        return _aligned_latest(self.net_ppe, len(self.revenue))

    @property
    def operating_nwc_components(self) -> dict[str, float | None]:
        """Latest operating-NWC component, or None when that component was not reported.

        Missing components stay missing. They are not replaced with zero, and the
        components that were reported are not discarded.
        """
        length = len(self.revenue)
        return {
            name: _aligned_latest(getattr(self, name), length)
            for name in OPERATING_NWC_COMPONENTS
        }

    @property
    def reported_operating_nwc(self) -> float | None:
        """
        Latest operating NWC from reported balances.

        Uses receivables, inventory, other operating current assets, payables, and
        other operating current liabilities. Returns None unless every component
        was reported for the latest revenue period. Missing components are not
        filled with zero, and reported components are not replaced by ratios.
        """
        latest = self.operating_nwc_components
        if any(value is None for value in latest.values()):
            return None
        receivables = latest["receivables"]
        inventory = latest["inventory"]
        other_assets = latest["other_operating_current_assets"]
        payables = latest["payables"]
        other_liabilities = latest["other_operating_current_liabilities"]
        return receivables + inventory + other_assets - payables - other_liabilities

    @property
    def latest_revenue(self) -> float:
        return self.revenue[-1]

    @property
    def latest_ebit(self) -> float:
        return self.ebit[-1]

    @property
    def latest_ebitda(self) -> float:
        if not self.ebitda or not _is_usable_number(self.ebitda[-1]):
            raise ValueError("Latest EBITDA was not reported.")
        return float(self.ebitda[-1])

    @property
    def historical_revenue_cagr(self) -> float:
        if len(self.revenue) < 2:
            raise ValueError("At least two years of revenue are required to compute CAGR.")
        start, end = self.revenue[0], self.revenue[-1]
        if start <= 0 or end <= 0:
            raise ValueError("Revenue must be positive to compute CAGR.")
        periods = _elapsed_years(self.fiscal_year_ends, len(self.revenue))
        if periods is None:
            periods = float(len(self.revenue) - 1)
        if periods <= 0:
            raise ValueError("Revenue history does not span a positive period.")
        return (end / start) ** (1 / periods) - 1

    @property
    def avg_operating_margin(self) -> float:
        if not self.revenue:
            raise ValueError("Revenue history is empty.")
        margins = [
            e / r
            for e, r in zip(self.ebit, self.revenue)
            if _is_usable_number(e) and _is_usable_number(r) and r != 0
        ]
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


def _mean_ratio(numerator: list[float | None] | None, denominator: list[float | None], label: str) -> float:
    if numerator is None or not denominator:
        raise ValueError(f"Revenue history is empty; cannot compute {label} ratio.")
    ratios = []
    for numerator_value, denominator_value in zip(numerator, denominator):
        if not _is_usable_number(numerator_value) or not _is_usable_number(denominator_value):
            continue
        if denominator_value == 0:
            continue
        ratios.append(numerator_value / denominator_value)
    if not ratios:
        raise ValueError(f"Cannot compute {label} as a share of revenue.")
    return sum(ratios) / len(ratios)


def _aligned_latest(series: list[float | None] | None, length: int) -> float | None:
    """Return the observation on the latest revenue period, without scanning backward."""
    if series is None or length <= 0 or len(series) != length:
        return None
    value = series[-1]
    if not _is_usable_number(value):
        return None
    return float(value)


def _is_usable_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


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
    # Opt-in school statement model. "intensity" keeps the historical default.
    valuation_model: str = "intensity"
    wacc_override: float | None = None
    da_pct_opening_ppe: float = 0.10
    receivables_pct_revenue: float = 0.10
    inventory_pct_revenue: float = 0.05
    other_operating_ca_pct_revenue: float = 0.0
    payables_pct_revenue: float = 0.08
    other_operating_cl_pct_revenue: float = 0.0
    # Sensitivity shortcut only. Not used to build a statement forecast.
    fcf_margin_override: float | None = None
    # Explicit starting operating-NWC stock. When set, it replaces the latest
    # reported component stock. It does not zero out reported components.
    starting_operating_nwc: float | None = None
    # Provenance for drivers. Empty means the caller built the assumptions explicitly.
    driver_sources: dict[str, str] = field(default_factory=dict)

    def format_driver_sources(self) -> str:
        """Label each tracked driver as normalized, reported, historical, generic, or explicit."""
        if not self.driver_sources:
            return "explicit assumptions"
        names = list(dict.fromkeys([*_STATEMENT_DRIVER_FIELDS, *_INTENSITY_DRIVER_FIELDS]))
        parts = [
            f"{name}={self.driver_sources[name]}"
            for name in names
            if name in self.driver_sources
        ]
        return ", ".join(parts) if parts else "explicit assumptions"

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
        _assign_historical_average(
            base, "revenue_growth", lambda: company.historical_revenue_cagr, -0.10, 0.40
        )
        _assign_historical_average(
            base, "operating_margin", lambda: company.avg_operating_margin, 0.01, 0.60
        )
        _assign_historical_average(
            base, "capex_pct_revenue", lambda: company.avg_capex_pct_revenue, 0.0, 0.30
        )
        _assign_historical_average(
            base, "da_pct_revenue", lambda: company.avg_da_pct_revenue, 0.0, 0.30
        )
        _assign_historical_average(
            base, "nwc_pct_revenue", lambda: company.avg_nwc_pct_revenue, -0.10, 0.20
        )
        return _apply_overrides(base, overrides)

    @classmethod
    def normalized_for(cls, company: "Company", **overrides) -> "Assumptions":
        """
        Build statement-model assumptions from medians of reported history.

        Revenue growth is the median annualized year-over-year rate. Elapsed
        time comes from ``fiscal_year_ends`` when those labels are present, so
        a dropped year is not treated as one year. EBIT margin, capex, the
        depreciation rate on the immediately prior fiscal year's net PPE, and
        operating-NWC component ratios are medians of reported observations.
        Each result is clipped to a fixed bound. If a median cannot be
        computed, that field keeps the generic dataclass default and is labeled
        ``generic_fallback``. The statement DCF refuses generic operating-NWC
        ratios unless the caller passes an explicit override. No sell-side or
        consensus estimate is used.

        The valuation model is the opt-in statement model. ``calibrated_for``
        is unchanged and still targets the intensity model. Terminal growth
        and the CAPM inputs stay at generic defaults unless overridden.
        Year-by-year revenue and EBIT-margin paths are the broadcast or
        caller-supplied paths on ``revenue_growth`` and ``operating_margin``.
        """
        base = cls()
        base.valuation_model = "statement"
        for name in _STATEMENT_DRIVER_FIELDS:
            base.driver_sources[name] = SOURCE_GENERIC_FALLBACK

        _assign_normalized(
            base,
            "revenue_growth",
            lambda: _median(_yoy_growth(company.revenue, company.fiscal_year_ends)),
            -0.10,
            0.40,
        )
        _assign_normalized(
            base,
            "operating_margin",
            lambda: _median(_paired_ratios(company.ebit, company.revenue)),
            0.01,
            0.60,
        )
        _assign_normalized(
            base,
            "capex_pct_revenue",
            lambda: _median(_paired_ratios(company.capex, company.revenue)),
            0.0,
            0.30,
        )
        _assign_normalized(
            base,
            "da_pct_opening_ppe",
            lambda: _median(_opening_ppe_da_rates(company)),
            0.0,
            0.50,
        )

        _assign_ratio(
            base,
            "receivables_pct_revenue",
            company.receivables,
            company.revenue,
            0.0,
            0.80,
        )
        _assign_ratio(
            base,
            "inventory_pct_revenue",
            company.inventory,
            company.revenue,
            0.0,
            0.80,
        )
        _assign_ratio(
            base,
            "other_operating_ca_pct_revenue",
            company.other_operating_current_assets,
            company.revenue,
            -0.20,
            0.80,
        )
        _assign_ratio(
            base,
            "payables_pct_revenue",
            company.payables,
            company.revenue,
            0.0,
            0.80,
        )
        _assign_ratio(
            base,
            "other_operating_cl_pct_revenue",
            company.other_operating_current_liabilities,
            company.revenue,
            -0.20,
            0.80,
        )

        _assign_tax_rate(base, company)
        return _apply_overrides(base, overrides)


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
    terminal_value_share: float = float("nan")
    valuation_model: str = "intensity"
    discount_convention: str = "year_end"
    wacc_note: str = ""
    share_count_basis: str = "basic"
    fcf_margin_shortcut: bool = False
    note: str = ""
    equity_bridge: dict | None = None

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
        if self.valuation_model == "statement":
            lines.extend(
                [
                    f"  Model:             statement (school DCF)",
                    f"  Discounting:       {self.discount_convention}",
                    f"  WACC basis:        {self.wacc_note}",
                    f"  TV share of EV:    {self.terminal_value_share:.1%}",
                    f"  Share count:       {self.share_count_basis}",
                ]
            )
            if self.equity_bridge is not None:
                lines.append(
                    f"  Cash in bridge:    ${self.equity_bridge['cash']:,.0f} "
                    f"({self.equity_bridge.get('cash_basis', '')})"
                )
            if self.note:
                lines.append(f"  Note:              {self.note}")
        lines.append(f"  Drivers:           {self.assumptions.format_driver_sources()}")
        if self.valuation_model == "statement" or self.company.missing_fields or self._bridge_gaps():
            lines.append(f"  Missing fields:    {self._missing_field_summary()}")
        return "\n".join(lines)

    def _bridge_gaps(self) -> list[str]:
        bridge = self.equity_bridge or {}
        gaps = []
        for flag, label in (
            ("preferred_equity_missing", "preferred_equity"),
            ("noncontrolling_interest_missing", "noncontrolling_interest"),
            ("other_adjustments_missing", "other_adjustments"),
        ):
            if bridge.get(flag):
                gaps.append(label)
        if self.valuation_model == "statement" and self.share_count_basis == "basic":
            gaps.append("diluted_shares")
        return gaps

    def _missing_field_summary(self) -> str:
        names = list(self.company.missing_fields)
        for label in self._bridge_gaps():
            if label not in names:
                names.append(label)
        return ", ".join(names) if names else "none"


def _apply_overrides(base: Assumptions, overrides: dict) -> Assumptions:
    valid_fields = {f.name for f in fields(type(base))}
    for key, value in overrides.items():
        if key not in valid_fields:
            raise ValueError(f"'{key}' is not a valid Assumptions field.")
        if value is not None:
            setattr(base, key, value)
            if key != "driver_sources":
                base.driver_sources[key] = SOURCE_EXPLICIT
    return base


def _assign_historical_average(assumptions: Assumptions, field_name: str, reader, low: float, high: float) -> None:
    fallback = getattr(assumptions, field_name)
    try:
        raw = reader()
    except Exception:
        assumptions.driver_sources[field_name] = SOURCE_GENERIC_FALLBACK
        return
    if not _is_usable_number(raw):
        assumptions.driver_sources[field_name] = SOURCE_GENERIC_FALLBACK
        return
    setattr(assumptions, field_name, _clip_or_fallback(raw, low, high, fallback))
    assumptions.driver_sources[field_name] = SOURCE_HISTORICAL_AVERAGE


def _assign_normalized(assumptions: Assumptions, field_name: str, reader, low: float, high: float) -> None:
    fallback = getattr(assumptions, field_name)
    try:
        raw = reader()
    except Exception:
        assumptions.driver_sources[field_name] = SOURCE_GENERIC_FALLBACK
        return
    if not _is_usable_number(raw):
        assumptions.driver_sources[field_name] = SOURCE_GENERIC_FALLBACK
        return
    setattr(assumptions, field_name, _clip_or_fallback(raw, low, high, fallback))
    assumptions.driver_sources[field_name] = SOURCE_NORMALIZED


def _assign_tax_rate(assumptions: Assumptions, company: Company) -> None:
    if "effective_tax_rate" in set(company.missing_fields):
        assumptions.driver_sources["tax_rate"] = SOURCE_GENERIC_FALLBACK
        if _is_usable_number(company.effective_tax_rate):
            assumptions.tax_rate = _clip_or_fallback(
                company.effective_tax_rate, 0.0, 0.40, assumptions.tax_rate
            )
        return
    if not _is_usable_number(company.effective_tax_rate):
        assumptions.driver_sources["tax_rate"] = SOURCE_GENERIC_FALLBACK
        return
    assumptions.tax_rate = _clip_or_fallback(company.effective_tax_rate, 0.0, 0.40, assumptions.tax_rate)
    assumptions.driver_sources["tax_rate"] = SOURCE_REPORTED


def _assign_ratio(
    assumptions: Assumptions,
    field: str,
    numerator: list[float | None] | None,
    revenue: list[float],
    low: float,
    high: float,
) -> None:
    try:
        median = _median(_paired_ratios(numerator or [], revenue))
    except Exception:
        assumptions.driver_sources[field] = SOURCE_GENERIC_FALLBACK
        return
    if not _is_usable_number(median):
        assumptions.driver_sources[field] = SOURCE_GENERIC_FALLBACK
        return
    setattr(assumptions, field, _clip_or_fallback(median, low, high, getattr(assumptions, field)))
    assumptions.driver_sources[field] = SOURCE_NORMALIZED


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("No observations.")
    return float(statistics.median(values))


def _yoy_growth(revenue: list[float], fiscal_year_ends: list[str] | None = None) -> list[float]:
    """Annualized growth between successive observations.

    When ``fiscal_year_ends`` is aligned with revenue, each step is scaled by
    the elapsed years. A two-year gap is not treated as a one-year change.
    Without usable dates, each list step counts as one year.
    """
    spans = _adjacent_spans(fiscal_year_ends, len(revenue))
    growth = []
    for index, (previous, current) in enumerate(zip(revenue, revenue[1:])):
        if not _is_usable_number(previous) or not _is_usable_number(current) or previous <= 0 or current <= 0:
            continue
        span = 1.0 if spans is None else spans[index]
        if span <= 0:
            continue
        growth.append((current / previous) ** (1 / span) - 1)
    return growth


def _paired_ratios(numerator: list[float | None], denominator: list[float | None]) -> list[float]:
    ratios = []
    for numerator_value, denominator_value in zip(numerator, denominator):
        if not _is_usable_number(numerator_value) or not _is_usable_number(denominator_value):
            continue
        if denominator_value == 0:
            continue
        ratios.append(float(numerator_value) / float(denominator_value))
    return ratios


def _opening_ppe_da_rates(company: Company) -> list[float]:
    """D&A divided by the immediately prior fiscal period's net PPE.

    A gap in ``fiscal_year_ends`` is not treated as an adjacent year, and an
    older PPE balance is not substituted for the missing prior year.
    """
    ppe = company.net_ppe
    depreciation = company.depreciation_amortization
    if not ppe or not depreciation:
        raise ValueError("Net PPE or D&A history is missing.")
    length = len(company.revenue)
    if len(ppe) != length or len(depreciation) != length:
        raise ValueError("Net PPE and D&A must line up with fiscal periods.")
    spans = _adjacent_spans(company.fiscal_year_ends, length)
    rates = []
    for index in range(1, length):
        if spans is not None and not _is_immediate_prior_fiscal_year(spans[index - 1]):
            continue
        opening_ppe = ppe[index - 1]
        charge = depreciation[index]
        if not _is_usable_number(opening_ppe) or not _is_usable_number(charge):
            continue
        if opening_ppe <= 0 or charge < 0:
            continue
        rates.append(float(charge) / float(opening_ppe))
    if not rates:
        raise ValueError("Cannot compute a D&A rate on opening net PPE.")
    return rates


def _elapsed_years(fiscal_year_ends: list[str] | None, length: int) -> float | None:
    spans = _adjacent_spans(fiscal_year_ends, length)
    if spans is None:
        return None
    return float(sum(spans))


def _adjacent_spans(fiscal_year_ends: list[str] | None, length: int) -> list[float] | None:
    """Elapsed years between successive fiscal labels, or None when dates are unusable."""
    if fiscal_year_ends is None or len(fiscal_year_ends) != length or length < 2:
        return None
    spans: list[float] = []
    for left, right in zip(fiscal_year_ends, fiscal_year_ends[1:]):
        span = _years_between(left, right)
        if span is None:
            return None
        spans.append(span)
    return spans


def _years_between(start: str, end: str) -> float | None:
    """Elapsed years between two fiscal labels.

    ISO dates use day count / 365.25. Sequential fixture labels such as ``Y0``
    and ``Y2`` use the index difference, so a skipped fixture year is two years.
    """
    left = _parse_fiscal_end(start)
    right = _parse_fiscal_end(end)
    if isinstance(left, datetime.date) and isinstance(right, datetime.date):
        return (right - left).days / 365.25
    if isinstance(left, int) and isinstance(right, int):
        return float(right - left)
    return None


def _parse_fiscal_end(label: object) -> datetime.date | int | None:
    if not isinstance(label, str):
        return None
    text = label.strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            return datetime.date.fromisoformat(text[:10])
        except ValueError:
            return None
    if len(text) >= 2 and text[0] == "Y" and text[1:].isdigit():
        return int(text[1:])
    return None


def _is_immediate_prior_fiscal_year(span_years: float) -> bool:
    return IMMEDIATE_PRIOR_FISCAL_YEAR_MIN <= span_years <= IMMEDIATE_PRIOR_FISCAL_YEAR_MAX


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
