"""Shared pytest fixtures for valuationengine tests."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from valuationengine.core.models import Assumptions, Company


@pytest.fixture(autouse=True)
def _live_fetch_unless_test_opts_into_safe_mode(monkeypatch):
    """Keep the suite on the live (mocked) fetch path unless a test sets SAFE_MODE."""
    monkeypatch.delenv("SAFE_MODE", raising=False)


@pytest.fixture
def sample_company() -> Company:
    """Synthetic company chosen so DCF and LBO numbers are easy to verify by hand."""
    return Company(
        ticker="TEST",
        name="Test Corp",
        revenue=[100.0, 110.0, 121.0, 133.1, 146.41],
        ebit=[20.0, 22.0, 24.2, 26.62, 29.282],
        ebitda=[24.0, 26.4, 29.04, 31.944, 35.1384],
        depreciation_amortization=[4.0, 4.4, 4.84, 5.324, 5.8564],
        capex=[5.0, 5.5, 6.05, 6.655, 7.3205],
        change_in_nwc=[2.0, 2.2, 2.42, 2.662, 2.9282],
        effective_tax_rate=0.25,
        cash=50.0,
        total_debt=100.0,
        shares_outstanding=100.0,
        current_price=5.0,
        market_cap=500.0,
        beta=1.0,
    )


@pytest.fixture
def base_assumptions() -> Assumptions:
    """Default Assumptions instance reused across tests."""
    return Assumptions()
