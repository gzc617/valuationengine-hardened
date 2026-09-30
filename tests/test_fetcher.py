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
