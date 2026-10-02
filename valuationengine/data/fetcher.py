"""Company fundamentals.

SAFE_MODE=1 returns a deterministic fixture and does not import yfinance.
Otherwise a live fetch runs with a wall-clock timeout. Failures surface as
ValueError or TimeoutError for the CLI and library callers.

Live history uses the fiscal periods that actually line up across statements.
A cash-flow or balance-sheet column matches an income-statement period only
when the dates are the same or differ by at most FISCAL_PERIOD_TOLERANCE_DAYS.
Sharing a calendar year is not enough to pair them. A period without reported
revenue and EBIT is dropped. History may be shorter than requested. Optional
fields that were not reported stay None. The fetch does not repeat an older
year or insert zeros to pad the window. Missing cash and incomplete debt stay
missing; they are not stored as zero.
"""

from __future__ import annotations

import concurrent.futures
import math
import threading

import pandas as pd

from valuationengine.core.models import Company
from valuationengine.data.fixtures import fixture_company
from valuationengine.safe_mode import install_network_guard, safe_mode_enabled
from valuationengine.validation import FETCH_TIMEOUT_SECONDS, normalize_ticker, validate_history_years

# A column is the same fiscal period when its date matches, or is off by only
# a few days (weekend, timezone, or a one-day source difference). A June fiscal
# date is not paired with a December column from the same calendar year.
FISCAL_PERIOD_TOLERANCE_DAYS = 5

_LIVE_FETCH_WORKERS = 5
_LIVE_FETCH_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=_LIVE_FETCH_WORKERS)
_LIVE_FETCH_SLOTS = threading.BoundedSemaphore(_LIVE_FETCH_WORKERS)


def fetch_company(ticker: str, history_years: int = 5) -> Company:
    """
    Return a Company for one ticker.

    Args:
        ticker: Equity ticker symbol.
        history_years: Number of most recent complete fiscal years to include (oldest first).
            Fewer years are returned when fewer complete periods were reported.
            Missing years are not filled in.
    """
    symbol = normalize_ticker(ticker)
    history_years = validate_history_years(history_years)

    if safe_mode_enabled():
        install_network_guard()
        return fixture_company(symbol, history_years)

    try:
        return _fetch_live_with_timeout(symbol, history_years)
    except TimeoutError:
        raise
    except ValueError:
        raise
    except RuntimeError:
        raise
    except OSError as exc:
        raise TimeoutError(
            f"Could not fetch data for ticker '{symbol}' because the network call failed."
        ) from exc
    except Exception as exc:
        message = str(exc).strip() or type(exc).__name__
        lowered = message.lower()
        if any(token in lowered for token in ("timeout", "timed out", "connection", "network", "ssl")):
            raise TimeoutError(
                f"Timed out fetching data for ticker '{symbol}'."
            ) from exc
        raise ValueError(
            f"Could not fetch data for ticker '{symbol}'. Check the symbol or try again."
        ) from exc


def _load_yfinance():
    """Import yfinance only on the live path."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Live market data requires the yfinance package. "
            "Set SAFE_MODE=1 to use deterministic fixture companies."
        ) from exc
    return yf


def _fetch_live_with_timeout(symbol: str, history_years: int) -> Company:
    if not _LIVE_FETCH_SLOTS.acquire(blocking=False):
        raise TimeoutError("Live market-data worker limit reached. Try again after current fetches finish.")
    future = _LIVE_FETCH_POOL.submit(_fetch_live, symbol, history_years)
    future.add_done_callback(lambda _future: _LIVE_FETCH_SLOTS.release())
    try:
        return future.result(timeout=FETCH_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise TimeoutError(
            f"Timed out fetching data for ticker '{symbol}' "
            f"after {FETCH_TIMEOUT_SECONDS} seconds."
        ) from exc



def _fetch_live(symbol: str, history_years: int) -> Company:
    yf = _load_yfinance()
    yf_ticker = yf.Ticker(symbol)
    info = yf_ticker.info or {}

    if not info:
        raise ValueError(
            f"Could not fetch data for ticker '{symbol}'. Check the symbol or try again."
        )

    name = info.get("longName") or info.get("shortName") or symbol
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if current_price is None or current_price <= 0:
        raise ValueError(f"Could not determine a valid current price for '{symbol}'.")

    beta = info.get("beta")
    beta = 1.0 if beta is None else float(beta)

    shares_outstanding = info.get("sharesOutstanding")
    market_cap = info.get("marketCap")
    if shares_outstanding is None:
        if market_cap is None:
            raise ValueError(f"Could not determine shares outstanding for '{symbol}'.")
        shares_outstanding = market_cap / current_price
    shares_outstanding = float(shares_outstanding)
    market_cap = float(market_cap if market_cap is not None else shares_outstanding * current_price)

    history = _aligned_history(
        yf_ticker.financials,
        yf_ticker.cashflow,
        yf_ticker.balance_sheet,
        history_years,
    )
    if not history["revenue"]:
        raise ValueError(
            f"No complete fiscal periods with revenue and EBIT for ticker '{symbol}'."
        )

    diluted = info.get("impliedSharesOutstanding")
    diluted_shares = float(diluted) if _is_finite(diluted) and float(diluted) > 0 else None

    missing = set(history["missing_fields"])
    if diluted_shares is None:
        missing.add("diluted_shares")
    missing.add("other_adjustments")

    return Company(
        ticker=symbol,
        name=name,
        revenue=history["revenue"],
        ebit=history["ebit"],
        ebitda=history["ebitda"],
        depreciation_amortization=history["depreciation_amortization"],
        capex=history["capex"],
        change_in_nwc=history["change_in_nwc"],
        effective_tax_rate=history["effective_tax_rate"],
        cash=history["cash"],
        total_debt=history["total_debt"],
        short_term_investments=history["short_term_investments"],
        shares_outstanding=shares_outstanding,
        current_price=float(current_price),
        market_cap=market_cap,
        beta=beta,
        fiscal_year_ends=history["fiscal_year_ends"],
        net_ppe=history["net_ppe"],
        receivables=history["receivables"],
        inventory=history["inventory"],
        other_operating_current_assets=history["other_operating_current_assets"],
        payables=history["payables"],
        other_operating_current_liabilities=history["other_operating_current_liabilities"],
        preferred_equity=history["preferred_equity"],
        noncontrolling_interest=history["noncontrolling_interest"],
        other_adjustments=None,
        diluted_shares=diluted_shares,
        missing_fields=tuple(sorted(missing)),
    )


_REVENUE_NAMES = ["Total Revenue", "Revenue"]
_EBIT_NAMES = ["EBIT", "Operating Income"]
_EBITDA_NAMES = ["EBITDA", "Normalized EBITDA"]
_TAX_NAMES = ["Tax Provision", "Income Tax Expense"]
_PRETAX_NAMES = ["Pretax Income", "Income Before Tax", "Earnings Before Tax"]
_DA_NAMES = ["Depreciation And Amortization", "Depreciation", "Amortization"]
_CAPEX_NAMES = ["Capital Expenditure", "Capital Expenditures"]
_NWC_CHANGE_NAMES = ["Change In Working Capital", "Changes In Working Capital"]
_CASH_NAMES = ["Cash And Cash Equivalents", "Cash"]
_CASH_AND_STI_NAMES = ["Cash Cash Equivalents And Short Term Investments"]
_STI_NAMES = ["Other Short Term Investments", "Short Term Investments"]
_TOTAL_DEBT_NAMES = ["Total Debt"]
# A long-term aggregate is not total debt. It is used only as the long-term
# piece of a current-plus-long-term sum, and never by itself.
_LONG_TERM_AGGREGATE_NAMES = ["Long Term Debt And Capital Lease Obligation"]
_LONG_TERM_DEBT_NAMES = ["Long Term Debt"]
_CURRENT_DEBT_NAMES = ["Current Debt", "Current Debt And Capital Lease Obligation", "Short Long Term Debt"]
_PPE_NAMES = ["Net PPE", "Property Plant Equipment Net"]
_RECEIVABLE_NAMES = ["Receivables", "Accounts Receivable", "Net Receivables"]
_INVENTORY_NAMES = ["Inventory"]
_PAYABLE_NAMES = ["Accounts Payable", "Payables And Accrued Expenses", "Payables"]
_CURRENT_ASSET_NAMES = ["Current Assets"]
_CURRENT_LIABILITY_NAMES = ["Current Liabilities"]
_PREFERRED_NAMES = ["Preferred Stock", "Preferred Stock Equity", "Preferred Securities Outside Stock Equity"]
_NCI_NAMES = ["Minority Interest", "Noncontrolling Interest", "Non Controlling Interest"]


def _aligned_history(financials, cashflow, balance_sheet, history_years: int) -> dict:
    """Keep complete fiscal periods only, oldest first, with no padded years."""
    complete = []
    for period, column in _columns(financials):
        revenue = _cell(financials, column, _REVENUE_NAMES)
        ebit = _cell(financials, column, _EBIT_NAMES)
        if revenue is None or ebit is None:
            continue
        complete.append(period)
    complete = complete[-history_years:]

    revenue: list[float] = []
    ebit: list[float] = []
    ebitda: list[float | None] = []
    depreciation: list[float | None] = []
    capex: list[float | None] = []
    change_in_nwc: list[float | None] = []
    tax: list[float | None] = []
    pretax: list[float | None] = []
    net_ppe: list[float | None] = []
    receivables: list[float | None] = []
    inventory: list[float | None] = []
    other_ca: list[float | None] = []
    payables: list[float | None] = []
    other_cl: list[float | None] = []
    fiscal_year_ends: list[str] = []

    for period in complete:
        income_column = _column_for(financials, period)
        cash_column = _column_for(cashflow, period)
        balance_column = _column_for(balance_sheet, period)
        revenue_value = _cell(financials, income_column, _REVENUE_NAMES)
        ebit_value = _cell(financials, income_column, _EBIT_NAMES)
        depreciation_value = _cell(cashflow, cash_column, _DA_NAMES)
        reported_ebitda = _cell(financials, income_column, _EBITDA_NAMES)
        if reported_ebitda is None and depreciation_value is not None and ebit_value is not None:
            reported_ebitda = ebit_value + depreciation_value
        capex_value = _cell(cashflow, cash_column, _CAPEX_NAMES)
        nwc_change = _cell(cashflow, cash_column, _NWC_CHANGE_NAMES)
        receivables_value = _cell(balance_sheet, balance_column, _RECEIVABLE_NAMES)
        inventory_value = _cell(balance_sheet, balance_column, _INVENTORY_NAMES)
        payables_value = _cell(balance_sheet, balance_column, _PAYABLE_NAMES)

        revenue.append(float(revenue_value))
        ebit.append(float(ebit_value))
        fiscal_year_ends.append(period.strftime("%Y-%m-%d"))
        ebitda.append(reported_ebitda)
        depreciation.append(depreciation_value)
        capex.append(None if capex_value is None else abs(capex_value))
        change_in_nwc.append(None if nwc_change is None else -nwc_change)
        tax.append(_cell(financials, income_column, _TAX_NAMES))
        pretax.append(_cell(financials, income_column, _PRETAX_NAMES))
        net_ppe.append(_cell(balance_sheet, balance_column, _PPE_NAMES))
        receivables.append(receivables_value)
        inventory.append(inventory_value)
        payables.append(payables_value)
        other_ca.append(
            _other_operating_assets(financials, cashflow, balance_sheet, income_column, balance_column, receivables_value, inventory_value)
        )
        other_cl.append(
            _other_operating_liabilities(balance_sheet, balance_column, payables_value)
        )

    cash, short_term_investments, debt, preferred, nci, point_missing = _latest_claims(
        balance_sheet, complete[-1] if complete else None
    )
    effective_tax_rate, tax_missing = _effective_tax_rate(tax, pretax)
    missing = set(point_missing)
    if tax_missing:
        missing.add("effective_tax_rate")
    series_fields = {
        "depreciation_amortization": depreciation,
        "capex": capex,
        "change_in_nwc": change_in_nwc,
        "ebitda": ebitda,
        "net_ppe": net_ppe,
        "receivables": receivables,
        "inventory": inventory,
        "other_operating_current_assets": other_ca,
        "payables": payables,
        "other_operating_current_liabilities": other_cl,
    }
    for name, values in series_fields.items():
        if not values or all(value is None for value in values):
            missing.add(name)

    return {
        "revenue": revenue,
        "ebit": ebit,
        "ebitda": _blank_if_unreported(ebitda),
        "depreciation_amortization": _blank_if_unreported(depreciation),
        "capex": _blank_if_unreported(capex),
        "change_in_nwc": _blank_if_unreported(change_in_nwc),
        "effective_tax_rate": effective_tax_rate,
        "cash": cash,
        "short_term_investments": short_term_investments,
        "total_debt": debt,
        "fiscal_year_ends": fiscal_year_ends,
        "net_ppe": _blank_if_unreported(net_ppe),
        "receivables": _blank_if_unreported(receivables),
        "inventory": _blank_if_unreported(inventory),
        "other_operating_current_assets": _blank_if_unreported(other_ca),
        "payables": _blank_if_unreported(payables),
        "other_operating_current_liabilities": _blank_if_unreported(other_cl),
        "preferred_equity": preferred,
        "noncontrolling_interest": nci,
        "missing_fields": missing,
    }


def _other_operating_assets(financials, cashflow, balance_sheet, income_column, balance_column, receivables, inventory):
    """Residual current assets after removing the same non-operating cash used in the bridge."""
    del financials, cashflow, income_column
    current_assets = _cell(balance_sheet, balance_column, _CURRENT_ASSET_NAMES)
    non_operating_cash, _short_term, _sti_not_identified = _non_operating_cash(balance_sheet, balance_column)
    if current_assets is None or receivables is None or inventory is None or non_operating_cash is None:
        return None
    return current_assets - non_operating_cash - receivables - inventory


def _non_operating_cash(balance_sheet, column) -> tuple[float | None, float | None, bool]:
    """Cash equivalents plus short-term investments, the STI component, and a gap flag.

    The first value is the non-operating cash added in the equity bridge and
    subtracted from operating current assets. The same figure is used in both
    places so short-term investments are not omitted and are not counted twice.

    When both cash equivalents and short-term investments are reported, they
    are summed. The combined source line is used when either piece is absent,
    because that line already includes short-term investments. Cash equivalents
    alone are used only when neither short-term investments nor the combined
    line was reported. That last case sets the third value, so the missing
    breakout is visible instead of being treated as a confirmed zero.
    """
    cash_equivalents = _cell(balance_sheet, column, _CASH_NAMES)
    short_term_investments = _cell(balance_sheet, column, _STI_NAMES)
    combined = _cell(balance_sheet, column, _CASH_AND_STI_NAMES)
    if cash_equivalents is not None and short_term_investments is not None:
        return cash_equivalents + short_term_investments, short_term_investments, False
    if combined is not None:
        return combined, short_term_investments, False
    if cash_equivalents is not None:
        return cash_equivalents, None, True
    return None, short_term_investments, True


def _other_operating_liabilities(balance_sheet, balance_column, payables):
    current_liabilities = _cell(balance_sheet, balance_column, _CURRENT_LIABILITY_NAMES)
    current_debt = _cell(balance_sheet, balance_column, _CURRENT_DEBT_NAMES)
    if current_liabilities is None or payables is None or current_debt is None:
        return None
    return current_liabilities - payables - current_debt


def _latest_claims(
    balance_sheet, period
) -> tuple[float | None, float | None, float | None, float | None, float | None, set[str]]:
    missing: set[str] = set()
    column = _column_for(balance_sheet, period) if period is not None else None
    cash, short_term_investments, short_term_not_identified = _non_operating_cash(balance_sheet, column)
    if cash is None:
        missing.add("cash")
    if short_term_not_identified:
        missing.add("short_term_investments")

    debt, debt_missing = _book_debt(balance_sheet, column)
    missing.update(debt_missing)

    preferred = _cell(balance_sheet, column, _PREFERRED_NAMES)
    nci = _cell(balance_sheet, column, _NCI_NAMES)
    if preferred is None:
        missing.add("preferred_equity")
    if nci is None:
        missing.add("noncontrolling_interest")
    return cash, short_term_investments, debt, preferred, nci, missing


def _book_debt(balance_sheet, column) -> tuple[float | None, set[str]]:
    """Complete book debt, or None when the reported pieces do not add up.

    ``Total Debt`` is the complete claim. The fallback sums a current piece and
    a long-term piece only when both are available. A long-term aggregate such
    as long-term debt and capital leases is not treated as total debt on its own.
    """
    missing: set[str] = set()
    total = _cell(balance_sheet, column, _TOTAL_DEBT_NAMES)
    current = _cell(balance_sheet, column, _CURRENT_DEBT_NAMES)
    long_term = _cell(balance_sheet, column, _LONG_TERM_DEBT_NAMES)
    long_term_aggregate = _cell(balance_sheet, column, _LONG_TERM_AGGREGATE_NAMES)
    if total is not None:
        return total, missing

    long_piece = long_term_aggregate if long_term_aggregate is not None else long_term
    if current is not None and long_piece is not None:
        return current + long_piece, missing

    missing.add("total_debt")
    if current is None and long_piece is not None:
        missing.add("current_debt")
    if long_piece is None and current is not None:
        missing.add("long_term_debt")
    return None, missing


def _columns(frame) -> list[tuple[pd.Timestamp, object]]:
    if frame is None or getattr(frame, "empty", True):
        return []
    pairs = []
    seen: set[pd.Timestamp] = set()
    for column in frame.columns:
        period = _normalize_period(column)
        if period is None or period in seen:
            continue
        seen.add(period)
        pairs.append((period, column))
    pairs.sort(key=lambda item: item[0])
    return pairs


def _column_for(frame, period: pd.Timestamp | None):
    """Match one statement column to the same fiscal period.

    The match is the closest date within ``FISCAL_PERIOD_TOLERANCE_DAYS``.
    A shared calendar year is not a match. If two columns are equally close,
    the period is left unmatched rather than paired with an arbitrary date.
    """
    if period is None:
        return None
    matches: list[tuple[int, pd.Timestamp, object]] = []
    for key, column in _columns(frame):
        try:
            delta_days = abs(int((key - period).days))
        except (TypeError, ValueError):
            continue
        if delta_days <= FISCAL_PERIOD_TOLERANCE_DAYS:
            matches.append((delta_days, key, column))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]))
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        return None
    return matches[0][2]


def _cell(frame, column, names: list[str]) -> float | None:
    if frame is None or getattr(frame, "empty", True) or column is None or column not in getattr(frame, "columns", []):
        return None
    index = list(frame.index)
    for name in names:
        if name not in index:
            continue
        value = frame.loc[name, column]
        if isinstance(value, pd.Series):
            observed = [float(item) for item in value.tolist() if _is_finite(item)]
            if observed:
                return observed[0]
            continue
        if _is_finite(value):
            return float(value)
    return None


def _effective_tax_rate(tax: list[float | None], pretax: list[float | None]) -> tuple[float, bool]:
    rates = []
    for tax_value, pretax_value in zip(tax, pretax):
        if not _is_finite(tax_value) or not _is_finite(pretax_value) or pretax_value == 0:
            continue
        rates.append(max(0.0, min(0.40, float(tax_value) / float(pretax_value))))
    if not rates:
        return 0.25, True
    return sum(rates) / len(rates), False


def _blank_if_unreported(values: list[float | None]) -> list[float | None] | None:
    if not values or all(value is None for value in values):
        return None
    return values


def _normalize_period(column) -> pd.Timestamp | None:
    try:
        period = pd.Timestamp(column)
    except (TypeError, ValueError):
        return None
    if pd.isna(period):
        return None
    return period.normalize()


def _is_finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
