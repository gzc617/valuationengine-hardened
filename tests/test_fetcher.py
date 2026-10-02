"""Tests for yfinance data fetcher (mocked, no network)."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from valuationengine.data.fetcher import fetch_company


def _mock_ticker_factory(info=None, financials=None, cashflow=None, balance_sheet=None):
    """Helper that builds a MagicMock mirroring yf.Ticker's surface."""
    mock = MagicMock()
    mock.info = info or {
        "longName": "Mock Corp",
        "currentPrice": 100.0,
        "sharesOutstanding": 1_000_000,
        "marketCap": 100_000_000,
        "beta": 1.1,
    }
    years = pd.DatetimeIndex(
        ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]
    )
    mock.financials = (
        financials
        if financials is not None
        else pd.DataFrame(
            {
                y: {
                    "Total Revenue": 100e6 * (1.1**i),
                    "EBIT": 20e6 * (1.1**i),
                    "EBITDA": 24e6 * (1.1**i),
                    "Tax Provision": 5e6 * (1.1**i),
                    "Pretax Income": 20e6 * (1.1**i),
                }
                for i, y in enumerate(years)
            }
        )
    )
    mock.cashflow = (
        cashflow
        if cashflow is not None
        else pd.DataFrame(
            {
                y: {
                    "Depreciation": 4e6 * (1.1**i),
                    "Capital Expenditure": -5e6 * (1.1**i),
                    "Change In Working Capital": -2e6 * (1.1**i),
                }
                for i, y in enumerate(years)
            }
        )
    )
    mock.balance_sheet = (
        balance_sheet
        if balance_sheet is not None
        else pd.DataFrame(
            {y: {"Cash And Cash Equivalents": 10e6, "Total Debt": 30e6} for y in years}
        )
    )
    return mock


@patch("valuationengine.data.fetcher._load_yfinance")
def test_fetch_company_happy_path(mock_load):
    """Happy path: fetch_company returns a fully populated Company."""
    mock_load.return_value.Ticker.return_value = _mock_ticker_factory()
    c = fetch_company("TEST")
    assert c.ticker == "TEST"
    assert c.name == "Mock Corp"
    assert c.current_price == pytest.approx(100.0)
    assert len(c.revenue) > 0
    assert c.beta == pytest.approx(1.1)


@patch("valuationengine.data.fetcher._load_yfinance")
def test_fetch_company_missing_info_raises(mock_load):
    """Empty or missing ticker info raises ValueError."""
    mock = MagicMock()
    mock.info = {}
    mock.financials = pd.DataFrame()
    mock.cashflow = pd.DataFrame()
    mock.balance_sheet = pd.DataFrame()
    mock_load.return_value.Ticker.return_value = mock
    with pytest.raises(ValueError):
        fetch_company("BAD")


@patch("valuationengine.data.fetcher._load_yfinance")
def test_fetch_company_defaults_beta_to_one(mock_load):
    """When beta is None, fetcher defaults to 1.0."""
    info = {
        "longName": "X",
        "currentPrice": 50.0,
        "sharesOutstanding": 1e6,
        "marketCap": 50e6,
        "beta": None,
    }
    mock_load.return_value.Ticker.return_value = _mock_ticker_factory(info=info)
    c = fetch_company("X")
    assert c.beta == pytest.approx(1.0)


@patch("valuationengine.data.fetcher._load_yfinance")
def test_fetch_company_shares_fallback(mock_load):
    """When sharesOutstanding is None, fetcher falls back to market_cap / current_price."""
    info = {
        "longName": "X",
        "currentPrice": 50.0,
        "sharesOutstanding": None,
        "marketCap": 100e6,
        "beta": 1.0,
    }
    mock_load.return_value.Ticker.return_value = _mock_ticker_factory(info=info)
    c = fetch_company("X")
    assert c.shares_outstanding == pytest.approx(100e6 / 50.0)


def _statement_frames(income_years, income_rows, cash_years=None, cash_rows=None, balance_years=None, balance_rows=None):
    financials = pd.DataFrame({year: income_rows[year] for year in income_years})
    if cash_years is None:
        cashflow = pd.DataFrame()
    else:
        cashflow = pd.DataFrame({year: cash_rows[year] for year in cash_years})
    if balance_years is None:
        balance_sheet = pd.DataFrame()
    else:
        balance_sheet = pd.DataFrame({year: balance_rows[year] for year in balance_years})
    return financials, cashflow, balance_sheet


@patch("valuationengine.data.fetcher._load_yfinance")
def test_fetch_does_not_pad_short_or_gapped_history(mock_load):
    """Short history stays short, gaps are not filled, and columns come out oldest first."""
    income_rows = {
        "2024-12-31": {"Total Revenue": 300.0, "EBIT": 30.0, "Pretax Income": 30.0, "Tax Provision": 6.0},
        "2023-12-31": {"Total Revenue": 200.0, "EBIT": 20.0, "Pretax Income": 20.0, "Tax Provision": 4.0},
        "2022-12-31": {"Total Revenue": 100.0, "EBIT": 10.0, "Pretax Income": 10.0, "Tax Provision": 2.0},
        "2021-12-31": {"Total Revenue": 80.0, "EBIT": float("nan"), "Pretax Income": 8.0, "Tax Provision": 2.0},
    }
    cash_rows = {
        "2024-12-31": {"Depreciation": 12.0, "Capital Expenditure": -15.0, "Change In Working Capital": -3.0},
        "2022-12-30": {"Depreciation": 4.0, "Capital Expenditure": -5.0, "Change In Working Capital": -1.0},
    }
    balance_rows = {
        "2024-12-31": {"Cash And Cash Equivalents": 40.0, "Total Debt": 70.0},
        "2022-12-30": {
            "Cash And Cash Equivalents": 20.0,
            "Total Debt": 50.0,
            "Net PPE": 90.0,
        },
    }
    financials, cashflow, balance = _statement_frames(
        list(income_rows),
        income_rows,
        list(cash_rows),
        cash_rows,
        list(balance_rows),
        balance_rows,
    )
    mock_load.return_value.Ticker.return_value = _mock_ticker_factory(
        financials=financials,
        cashflow=cashflow,
        balance_sheet=balance,
    )
    company = fetch_company("TEST", history_years=5)
    assert company.revenue == [100.0, 200.0, 300.0]
    assert company.ebit == [10.0, 20.0, 30.0]
    assert 80.0 not in company.revenue
    assert company.fiscal_year_ends == ["2022-12-31", "2023-12-31", "2024-12-31"]
    assert len(company.revenue) == 3
    assert company.capex == [5.0, None, 15.0]
    assert company.depreciation_amortization == [4.0, None, 12.0]
    assert company.change_in_nwc == [1.0, None, 3.0]
    assert company.net_ppe == [90.0, None, None]
    assert company.receivables is None
    assert company.preferred_equity is None
    assert company.diluted_shares is None
    assert "receivables" in company.missing_fields
    assert "preferred_equity" in company.missing_fields
    assert "diluted_shares" in company.missing_fields
    assert 0.0 not in (company.net_ppe or [])
    assert company.cash == pytest.approx(40.0)
    assert company.total_debt == pytest.approx(70.0)
    assert "short_term_investments" in company.missing_fields


@patch("valuationengine.data.fetcher._load_yfinance")
def test_fetch_uses_diluted_shares_only_when_reported(mock_load):
    info = {
        "longName": "Mock Corp",
        "currentPrice": 100.0,
        "sharesOutstanding": 1_000_000,
        "impliedSharesOutstanding": 1_250_000,
        "marketCap": 100_000_000,
        "beta": 1.1,
    }
    mock_load.return_value.Ticker.return_value = _mock_ticker_factory(info=info)
    company = fetch_company("TEST")
    assert company.shares_outstanding == pytest.approx(1_000_000)
    assert company.diluted_shares == pytest.approx(1_250_000)
    assert "diluted_shares" not in company.missing_fields


def _income_year(revenue=100.0):
    return {
        "Total Revenue": revenue,
        "EBIT": revenue * 0.2,
        "Pretax Income": revenue * 0.2,
        "Tax Provision": revenue * 0.04,
    }


@patch("valuationengine.data.fetcher._load_yfinance")
def test_short_term_investments_join_cash_and_leave_operating_nwc(mock_load):
    income_rows = {"2024-12-31": _income_year()}
    cash_rows = {"2024-12-31": {"Depreciation": 8.0, "Capital Expenditure": -5.0, "Change In Working Capital": -1.0}}
    balance_rows = {
        "2024-12-31": {
            "Current Assets": 100.0,
            "Cash And Cash Equivalents": 20.0,
            "Other Short Term Investments": 30.0,
            "Cash Cash Equivalents And Short Term Investments": 50.0,
            "Receivables": 15.0,
            "Inventory": 10.0,
            "Total Debt": 40.0,
        }
    }
    financials, cashflow, balance = _statement_frames(
        list(income_rows), income_rows, list(cash_rows), cash_rows, list(balance_rows), balance_rows
    )
    mock_load.return_value.Ticker.return_value = _mock_ticker_factory(
        financials=financials, cashflow=cashflow, balance_sheet=balance
    )
    company = fetch_company("TEST", history_years=1)
    assert company.cash == pytest.approx(50.0)
    assert company.short_term_investments == pytest.approx(30.0)
    assert company.other_operating_current_assets == [25.0]
    excluded = 100.0 - company.other_operating_current_assets[-1] - 15.0 - 10.0
    assert excluded == pytest.approx(company.cash)
    assert excluded != pytest.approx(50.0 + 30.0)
    assert "short_term_investments" not in company.missing_fields


@patch("valuationengine.data.fetcher._load_yfinance")
def test_combined_cash_line_includes_short_term_investments_when_the_piece_is_absent(mock_load):
    income_rows = {"2024-12-31": _income_year()}
    balance_rows = {
        "2024-12-31": {
            "Current Assets": 100.0,
            "Cash And Cash Equivalents": 20.0,
            "Cash Cash Equivalents And Short Term Investments": 50.0,
            "Receivables": 15.0,
            "Inventory": 10.0,
            "Total Debt": 40.0,
        }
    }
    financials, cashflow, balance = _statement_frames(
        list(income_rows), income_rows, None, None, list(balance_rows), balance_rows
    )
    mock_load.return_value.Ticker.return_value = _mock_ticker_factory(
        financials=financials, cashflow=cashflow, balance_sheet=balance
    )
    company = fetch_company("TEST", history_years=1)
    assert company.cash == pytest.approx(50.0)
    assert company.other_operating_current_assets == [25.0]
    assert "short_term_investments" not in company.missing_fields


@patch("valuationengine.data.fetcher._load_yfinance")
def test_statements_do_not_pair_on_calendar_year_alone(mock_load):
    from valuationengine.data.fetcher import FISCAL_PERIOD_TOLERANCE_DAYS

    income_rows = {
        "2023-12-31": _income_year(80.0),
        "2024-12-31": _income_year(100.0),
    }
    cash_rows = {
        "2023-12-31": {"Depreciation": 4.0, "Capital Expenditure": -5.0},
        "2024-06-30": {"Depreciation": 99.0, "Capital Expenditure": -99.0},
        "2024-12-26": {"Depreciation": 7.0, "Capital Expenditure": -8.0},
    }
    balance_rows = {
        "2023-12-31": {"Cash And Cash Equivalents": 10.0, "Total Debt": 20.0, "Net PPE": 40.0},
        "2024-06-30": {"Cash And Cash Equivalents": 1.0, "Total Debt": 2.0, "Net PPE": 3.0},
        "2024-12-26": {"Cash And Cash Equivalents": 15.0, "Total Debt": 25.0, "Net PPE": 45.0},
    }
    assert (pd.Timestamp("2024-12-31") - pd.Timestamp("2024-12-26")).days == FISCAL_PERIOD_TOLERANCE_DAYS
    financials, cashflow, balance = _statement_frames(
        list(income_rows), income_rows, list(cash_rows), cash_rows, list(balance_rows), balance_rows
    )
    mock_load.return_value.Ticker.return_value = _mock_ticker_factory(
        financials=financials, cashflow=cashflow, balance_sheet=balance
    )
    company = fetch_company("TEST", history_years=5)
    assert company.fiscal_year_ends == ["2023-12-31", "2024-12-31"]
    assert company.depreciation_amortization == [4.0, 7.0]
    assert company.capex == [5.0, 8.0]
    assert company.net_ppe == [40.0, 45.0]
    assert company.cash == pytest.approx(15.0)
    assert 99.0 not in company.depreciation_amortization
    assert 3.0 not in company.net_ppe

    outside = {
        "2024-12-31": _income_year(),
    }
    outside_cash = {"2024-12-25": {"Depreciation": 4.0, "Capital Expenditure": -6.0}}
    outside_balance = {"2024-06-30": {"Cash And Cash Equivalents": 9.0, "Total Debt": 11.0, "Net PPE": 12.0}}
    financials, cashflow, balance = _statement_frames(
        list(outside), outside, list(outside_cash), outside_cash, list(outside_balance), outside_balance
    )
    mock_load.return_value.Ticker.return_value = _mock_ticker_factory(
        financials=financials, cashflow=cashflow, balance_sheet=balance
    )
    company = fetch_company("TEST", history_years=1)
    assert company.depreciation_amortization is None
    assert company.net_ppe is None
    assert company.cash is None
    assert "cash" in company.missing_fields
    assert company.cash != 0


@patch("valuationengine.data.fetcher._load_yfinance")
def test_debt_fallback_requires_both_pieces_and_does_not_invent_zero(mock_load):
    from valuationengine.core import dcf
    from valuationengine.core.models import Assumptions

    income_rows = {"2024-12-31": _income_year()}

    def fetch_with_balance(balance_row):
        balance_rows = {"2024-12-31": balance_row}
        financials, cashflow, balance = _statement_frames(
            list(income_rows), income_rows, None, None, list(balance_rows), balance_rows
        )
        mock_load.return_value.Ticker.return_value = _mock_ticker_factory(
            financials=financials, cashflow=cashflow, balance_sheet=balance
        )
        return fetch_company("TEST", history_years=1)

    aggregate_only = fetch_with_balance(
        {
            "Cash And Cash Equivalents": 12.0,
            "Long Term Debt And Capital Lease Obligation": 80.0,
        }
    )
    assert aggregate_only.total_debt is None
    assert aggregate_only.cash == pytest.approx(12.0)
    assert "total_debt" in aggregate_only.missing_fields
    assert "current_debt" in aggregate_only.missing_fields
    with pytest.raises(ValueError, match="not valued as zero"):
        dcf.run(aggregate_only, Assumptions(valuation_model="statement"))

    summed = fetch_with_balance(
        {
            "Cash And Cash Equivalents": 12.0,
            "Current Debt": 10.0,
            "Long Term Debt": 70.0,
        }
    )
    assert summed.total_debt == pytest.approx(80.0)
    assert "total_debt" not in summed.missing_fields

    aggregate_plus_current = fetch_with_balance(
        {
            "Cash And Cash Equivalents": 12.0,
            "Current Debt": 10.0,
            "Long Term Debt And Capital Lease Obligation": 70.0,
        }
    )
    assert aggregate_plus_current.total_debt == pytest.approx(80.0)

    current_only = fetch_with_balance({"Cash And Cash Equivalents": 12.0, "Current Debt": 10.0})
    assert current_only.total_debt is None
    assert "long_term_debt" in current_only.missing_fields

    reported_total = fetch_with_balance(
        {
            "Cash And Cash Equivalents": 12.0,
            "Total Debt": 100.0,
            "Long Term Debt And Capital Lease Obligation": 80.0,
        }
    )
    assert reported_total.total_debt == pytest.approx(100.0)

    missing_cash = fetch_with_balance({"Total Debt": 40.0})
    assert missing_cash.cash is None
    assert missing_cash.total_debt == pytest.approx(40.0)
    assert "cash" in missing_cash.missing_fields
    with pytest.raises(ValueError, match="Unresolved claims: cash"):
        dcf.run(missing_cash, Assumptions(valuation_model="statement"))
