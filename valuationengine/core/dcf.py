"""DCF valuation engine.

The default model is the historical intensity DCF: year-end discounting,
target capital-structure weights, and free cash flow from revenue intensities.
``Assumptions.valuation_model = "statement"`` selects the opt-in school
statement model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from valuationengine.core.models import (
    NWC_RATIO_FIELDS,
    SOURCE_EXPLICIT,
    SOURCE_GENERIC_FALLBACK,
    SOURCE_NORMALIZED,
    SOURCE_REPORTED,
    Assumptions,
    Company,
    DCFResult,
)

_AUDITED_NWC_SOURCES = {SOURCE_EXPLICIT, SOURCE_NORMALIZED, SOURCE_REPORTED}
CASH_BASIS = "cash equivalents plus short-term investments"
from valuationengine.validation import validate_fcf_margin, validate_wacc_override

# Printed on statement results and in the sensitivity CLI. The override
# replaces the statement build; it is not a second way to forecast the statements.
FCF_MARGIN_SENSITIVITY_LABEL = (
    "FCF-margin override is a sensitivity shortcut, not a full statement build."
)

BOOK_DEBT_WACC_NOTE = (
    "Equity weight is market capitalization. Debt weight is book debt, "
    "used as a proxy for the market value of debt. Preferred equity and "
    "noncontrolling interest are bridge claims, not WACC weights."
)

_NWC_SPEC = (
    ("receivables", "receivables_pct_revenue", 1.0),
    ("inventory", "inventory_pct_revenue", 1.0),
    ("other_operating_current_assets", "other_operating_ca_pct_revenue", 1.0),
    ("payables", "payables_pct_revenue", -1.0),
    ("other_operating_current_liabilities", "other_operating_cl_pct_revenue", -1.0),
)
_NWC_COLUMN = {
    "other_operating_current_assets": "other_operating_ca",
    "other_operating_current_liabilities": "other_operating_cl",
}


def compute_wacc(assumptions: Assumptions, beta: float, company: Company | None = None) -> float:
    """
    Compute WACC from CAPM cost of equity and after-tax cost of debt.

    Intensity model (default):
    Cost of equity = risk_free_rate + beta * equity_risk_premium
    Cost of debt (after tax) = cost_of_debt * (1 - tax_rate)
    WACC = (1 - D/V) * Re + (D/V) * Rd * (1 - T)
    D/V is ``target_debt_weight``.

    Statement model:
    Weights are market capitalization and book debt. Book debt is a proxy
    for the market value of debt. An explicit ``wacc_override`` replaces
    that calculation.

    Args:
        assumptions: WACC component assumptions.
        beta: Equity beta for CAPM.
        company: Required for statement-model market weights when no override is set.

    Returns:
        Weighted average cost of capital as a decimal.
    """
    if assumptions.valuation_model == "statement":
        return compute_statement_wacc(assumptions, beta, company)
    if assumptions.valuation_model != "intensity":
        raise ValueError("valuation_model must be 'intensity' or 'statement'.")
    if not 0 <= assumptions.target_debt_weight <= 1:
        raise ValueError(
            f"target_debt_weight must be between 0 and 1; got {assumptions.target_debt_weight}."
        )
    cost_of_equity = assumptions.risk_free_rate + beta * assumptions.equity_risk_premium
    cost_of_debt_after_tax = assumptions.cost_of_debt * (1 - assumptions.tax_rate)
    equity_weight = 1 - assumptions.target_debt_weight
    return equity_weight * cost_of_equity + assumptions.target_debt_weight * cost_of_debt_after_tax


def compute_statement_wacc(assumptions: Assumptions, beta: float, company: Company | None) -> float:
    """
    Statement-model WACC.

    Equity weight = market capitalization / (market capitalization + book debt).
    Debt weight = book debt / (market capitalization + book debt).
    Book debt stands in for the market value of debt because a market price
    for the debt is not part of the reported history. Preferred equity and
    noncontrolling interest stay out of these weights; the equity bridge
    deducts them instead. ``wacc_override``, when set, replaces this result.
    """
    if company is not None:
        _require_statement_capital(company)
    if assumptions.wacc_override is not None:
        return validate_wacc_override(assumptions.wacc_override)
    if company is None:
        raise ValueError(
            "Statement WACC needs the company for market-capitalization and book-debt weights."
        )
    equity_value = company.market_cap
    book_debt = float(company.total_debt)
    if (
        not math.isfinite(equity_value)
        or not math.isfinite(book_debt)
        or equity_value < 0
        or book_debt < 0
        or equity_value + book_debt <= 0
    ):
        raise ValueError(
            "Market capitalization and book debt must be non-negative finite numbers "
            "whose sum is positive."
        )
    cost_of_equity = assumptions.risk_free_rate + beta * assumptions.equity_risk_premium
    cost_of_debt_after_tax = assumptions.cost_of_debt * (1 - assumptions.tax_rate)
    capital = equity_value + book_debt
    equity_weight = equity_value / capital
    debt_weight = book_debt / capital
    return equity_weight * cost_of_equity + debt_weight * cost_of_debt_after_tax


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

    Intensity model (default):
    Terminal value (Gordon growth): TV = FCF_T * (1 + g) / (WACC - g)
    Terminal value (exit multiple): TV = EBITDA_T * exit_ev_ebitda_multiple
    Cash flows are discounted at year end.
    Enterprise value = sum(PV of projected FCF) + PV of terminal value
    Equity value = EV - net debt; value per share = equity / basic shares.

    Statement model:
    FCF = EBIT after tax + D&A - CapEx - change in operating NWC.
    D&A is a rate on opening net PPE, and PPE rolls forward.
    Discounting is mid-year. Equity value deducts book debt, preferred equity,
    and noncontrolling interest, then adds cash and a signed other adjustment.
    Diluted shares are used when they were reported.

    Args:
        company: Company fundamentals and market data.
        assumptions: Valuation assumptions. ``valuation_model`` selects the model.

    Returns:
        DCFResult with projection, WACC, and valuation outputs.
    """
    if assumptions.valuation_model == "statement":
        return _run_statement(company, assumptions)
    if assumptions.valuation_model != "intensity":
        raise ValueError("valuation_model must be 'intensity' or 'statement'.")

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
    terminal_value_share = _terminal_share(pv_terminal_value, enterprise_value)

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
        terminal_value_share=terminal_value_share,
        valuation_model="intensity",
        discount_convention="year_end",
        wacc_note="Target capital-structure weight.",
        share_count_basis="basic",
    )


def _run_statement(company: Company, assumptions: Assumptions) -> DCFResult:
    """School statement DCF. Intensity inputs on the company are left unused."""
    if not company.revenue:
        raise ValueError("Company revenue history is empty.")
    if company.latest_revenue <= 0:
        raise ValueError("Latest revenue must be positive to project FCF.")
    if company.shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive.")
    if company.current_price <= 0:
        raise ValueError("current_price must be positive.")

    wacc = compute_statement_wacc(assumptions, company.beta, company)
    shortcut = assumptions.fcf_margin_override is not None
    if shortcut and assumptions.use_exit_multiple:
        raise ValueError(
            "The FCF-margin sensitivity shortcut cannot support an exit multiple, "
            "because it does not build EBITDA from the statement. "
            + FCF_MARGIN_SENSITIVITY_LABEL
        )

    projection = _project_statement(company, assumptions, shortcut=shortcut)
    terminal_fcf = float(projection.iloc[-1]["fcf"])
    if assumptions.use_exit_multiple:
        terminal_row = projection.iloc[-1]
        terminal_ebitda = float(terminal_row["ebit"] + terminal_row["da"])
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

    exponents = projection["year"].to_numpy(dtype=float) - 0.5
    discount_factors = (1 + wacc) ** (-exponents)
    pv_fcf = projection["fcf"].to_numpy(dtype=float) * discount_factors
    projection = projection.copy()
    projection["discount_exponent"] = exponents
    projection["discount_factor"] = discount_factors
    projection["pv_fcf"] = pv_fcf

    horizon = float(len(projection))
    pv_terminal_value = float(terminal_value / (1 + wacc) ** (horizon - 0.5))
    enterprise_value = float(pv_fcf.sum() + pv_terminal_value)
    bridge = _equity_bridge(company, enterprise_value)
    equity_value = bridge["equity_value"]
    shares, share_basis = _share_count(company)
    value_per_share = equity_value / shares
    upside = (value_per_share - company.current_price) / company.current_price

    return DCFResult(
        company=company,
        assumptions=assumptions,
        projection=projection,
        wacc=float(wacc),
        terminal_value=float(terminal_value),
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        upside=upside,
        terminal_value_share=_terminal_share(pv_terminal_value, enterprise_value),
        valuation_model="statement",
        discount_convention="mid-year",
        wacc_note=BOOK_DEBT_WACC_NOTE,
        share_count_basis=share_basis,
        fcf_margin_shortcut=shortcut,
        note=_statement_note(shortcut, assumptions),
        equity_bridge=bridge,
    )


def _project_statement(company: Company, assumptions: Assumptions, shortcut: bool) -> pd.DataFrame:
    growth = assumptions.get_growth_path()
    margins = assumptions.get_margin_path()
    fcf_margin = validate_fcf_margin(assumptions.fcf_margin_override) if shortcut else None
    for ratio_name in (
        "capex_pct_revenue",
        "da_pct_opening_ppe",
        "tax_rate",
        "receivables_pct_revenue",
        "inventory_pct_revenue",
        "other_operating_ca_pct_revenue",
        "payables_pct_revenue",
        "other_operating_cl_pct_revenue",
    ):
        value = getattr(assumptions, ratio_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{ratio_name} must be a finite number.")

    opening_ppe = company.latest_net_ppe
    if not shortcut:
        if opening_ppe is None:
            raise ValueError(
                "Statement DCF requires reported latest net PPE as opening PPE. "
                "Missing PPE is left blank and is not filled in."
            )
        if opening_ppe < 0:
            raise ValueError(
                "Statement DCF rejects negative opening net PPE before calculating D&A."
            )
        prior_nwc = _starting_operating_nwc(company, assumptions)
        _require_auditable_nwc_ratios(assumptions)
    else:
        prior_nwc = None
    revenue = company.latest_revenue
    rows: list[dict] = []

    for year_idx, (growth_rate, margin) in enumerate(zip(growth, margins), start=1):
        revenue = revenue * (1 + growth_rate)
        ebit = revenue * margin
        nopat = ebit * (1 - assumptions.tax_rate)
        if shortcut:
            row = {
                "year": year_idx,
                "revenue": revenue,
                "revenue_growth": growth_rate,
                "ebit_margin": margin,
                "ebit": ebit,
                "nopat": nopat,
                "opening_ppe": math.nan,
                "da": math.nan,
                "capex": math.nan,
                "closing_ppe": math.nan,
                "receivables": math.nan,
                "inventory": math.nan,
                "other_operating_ca": math.nan,
                "payables": math.nan,
                "other_operating_cl": math.nan,
                "operating_nwc": math.nan,
                "change_nwc": math.nan,
                "fcf": revenue * fcf_margin,
                "fcf_build": "fcf_margin_shortcut",
            }
        else:
            if opening_ppe < 0:
                raise ValueError(
                    "Statement DCF rejects negative opening net PPE before calculating D&A."
                )
            da = opening_ppe * assumptions.da_pct_opening_ppe
            capex = revenue * assumptions.capex_pct_revenue
            closing_ppe = opening_ppe + capex - da
            if closing_ppe < 0:
                raise ValueError(
                    "Statement DCF produced negative closing net PPE. "
                    "Review the D&A-on-opening-PPE rate and CapEx assumptions."
                )
            components = _nwc_components(revenue, assumptions)
            change_nwc = components["operating_nwc"] - prior_nwc
            fcf = nopat + da - capex - change_nwc
            row = {
                "year": year_idx,
                "revenue": revenue,
                "revenue_growth": growth_rate,
                "ebit_margin": margin,
                "ebit": ebit,
                "nopat": nopat,
                "opening_ppe": opening_ppe,
                "da": da,
                "capex": capex,
                "closing_ppe": closing_ppe,
                "receivables": components["receivables"],
                "inventory": components["inventory"],
                "other_operating_ca": components["other_operating_ca"],
                "payables": components["payables"],
                "other_operating_cl": components["other_operating_cl"],
                "operating_nwc": components["operating_nwc"],
                "change_nwc": change_nwc,
                "fcf": fcf,
                "fcf_build": "statement",
            }
            opening_ppe = closing_ppe
            prior_nwc = components["operating_nwc"]
        rows.append(row)
    return pd.DataFrame(rows)


def _nwc_components(revenue: float, assumptions: Assumptions) -> dict[str, float]:
    amounts: dict[str, float] = {}
    operating_nwc = 0.0
    for attribute, ratio_name, sign in _NWC_SPEC:
        amount = revenue * getattr(assumptions, ratio_name)
        amounts[_NWC_COLUMN.get(attribute, attribute)] = amount
        operating_nwc += sign * amount
    amounts["operating_nwc"] = operating_nwc
    return amounts


def _starting_operating_nwc(company: Company, assumptions: Assumptions) -> float:
    """
    Prior operating NWC.

    The latest reported component stock is used when all five components were
    observed. An explicit ``starting_operating_nwc`` override is the other
    accepted source. A missing component is not replaced with zero, and the
    components that were reported are not discarded in favor of generic ratios.
    """
    override = assumptions.starting_operating_nwc
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, (int, float)) or not math.isfinite(override):
            raise ValueError("starting_operating_nwc must be a finite number.")
        return float(override)
    components = company.operating_nwc_components
    missing = [name for name, value in components.items() if value is None]
    if missing:
        reported = {name: value for name, value in components.items() if value is not None}
        raise ValueError(
            "Statement DCF requires a complete starting operating-NWC stock "
            "or an explicit starting_operating_nwc override. "
            f"Missing components: {', '.join(missing)}. "
            f"Reported components are kept for audit and are not replaced: {reported}. "
            "A missing component is not replaced with zero or a generic ratio."
        )
    reported = company.reported_operating_nwc
    if reported is None:
        raise ValueError("Statement DCF could not audit the starting operating-NWC stock.")
    return float(reported)


def _require_auditable_nwc_ratios(assumptions: Assumptions) -> None:
    """Refuse generic operating-NWC ratios on a classified statement case."""
    if not assumptions.driver_sources:
        return
    generic = [
        name
        for name in NWC_RATIO_FIELDS
        if assumptions.driver_sources.get(name) not in _AUDITED_NWC_SOURCES
    ]
    if generic:
        raise ValueError(
            "Statement DCF will not substitute generic operating-NWC ratios "
            f"for {', '.join(generic)}. "
            "Pass explicit overrides, or provide reported history that can be normalized. "
            "Missing components are not discarded or replaced with zeros."
        )


def _require_statement_capital(company: Company) -> tuple[float, float]:
    """Return cash and book debt, or fail when either claim is incomplete.

    Cash is cash equivalents plus short-term investments. Incomplete debt is
    not a valuation of zero. Explicit finite values are accepted only when the
    company no longer flags those claims as missing.
    """
    flags = set(company.missing_fields)
    unresolved: list[str] = []
    if "cash" in flags or not _is_finite_claim(company.cash):
        unresolved.append("cash")
    debt_flags = flags & {"total_debt", "current_debt", "long_term_debt"}
    if debt_flags or not _is_finite_claim(company.total_debt):
        unresolved.append("total_debt")
    if unresolved:
        listed = ", ".join(sorted(flags)) if flags else "none"
        raise ValueError(
            "Statement WACC and the equity bridge need complete cash and book debt. "
            f"Unresolved claims: {', '.join(unresolved)}. "
            f"missing_fields={listed}. "
            "Missing cash and incomplete debt are not valued as zero. "
            "Report cash equivalents plus short-term investments and complete total debt, "
            "or set company.cash and company.total_debt explicitly and clear the incomplete flags."
        )
    return float(company.cash), float(company.total_debt)


def _is_finite_claim(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _statement_note(shortcut: bool, assumptions: Assumptions) -> str:
    parts = []
    if shortcut:
        parts.append(FCF_MARGIN_SENSITIVITY_LABEL)
    else:
        if assumptions.starting_operating_nwc is not None:
            parts.append("Starting operating NWC is an explicit override.")
        else:
            parts.append("Starting operating NWC is the reported latest component stock.")
    return " ".join(parts)


def _equity_bridge(company: Company, enterprise_value: float) -> dict:
    preferred, preferred_missing = _signed_claim(company.preferred_equity, "preferred_equity")
    nci, nci_missing = _signed_claim(company.noncontrolling_interest, "noncontrolling_interest")
    other, other_missing = _signed_claim(company.other_adjustments, "other_adjustments")
    cash, debt = _require_statement_capital(company)
    # ``cash`` already includes short-term investments. Do not add them again.
    equity_value = enterprise_value - debt - preferred - nci + cash + other
    return {
        "enterprise_value": enterprise_value,
        "debt": debt,
        "preferred_equity": preferred,
        "noncontrolling_interest": nci,
        "cash": cash,
        "short_term_investments": company.short_term_investments,
        "other_adjustments": other,
        "equity_value": equity_value,
        "preferred_equity_missing": preferred_missing,
        "noncontrolling_interest_missing": nci_missing,
        "other_adjustments_missing": other_missing,
        "debt_basis": "book debt",
        "cash_basis": CASH_BASIS,
    }


def _signed_claim(value: float | None, label: str) -> tuple[float, bool]:
    """Return (amount, missing). A provided amount is added with its own sign."""
    if value is None:
        return 0.0, True
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number when it is provided.")
    return float(value), False


def _share_count(company: Company) -> tuple[float, str]:
    diluted = company.diluted_shares
    if diluted is None:
        return float(company.shares_outstanding), "basic"
    if isinstance(diluted, bool) or not isinstance(diluted, (int, float)) or not math.isfinite(diluted) or diluted <= 0:
        raise ValueError("diluted_shares must be positive when it is provided.")
    return float(diluted), "diluted"


def _terminal_share(pv_terminal_value: float, enterprise_value: float) -> float:
    if enterprise_value == 0:
        return float("nan")
    return float(pv_terminal_value / enterprise_value)
