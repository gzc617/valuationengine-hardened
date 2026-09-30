"""SAFE_MODE uses fixtures and does not call the live market-data client."""

from unittest.mock import patch

import pytest

from valuationengine.data.fetcher import fetch_company
from valuationengine.safe_mode import (
    SafeModeNetworkError,
    _is_local,
    install_network_guard,
    safe_mode_enabled,
)


def test_safe_mode_flag(monkeypatch):
    monkeypatch.delenv("SAFE_MODE", raising=False)
    assert safe_mode_enabled() is False
    monkeypatch.setenv("SAFE_MODE", "1")
    assert safe_mode_enabled() is True
    monkeypatch.setenv("SAFE_MODE", "yes")
    assert safe_mode_enabled() is True
    monkeypatch.setenv("SAFE_MODE", "0")
    assert safe_mode_enabled() is False


def test_safe_mode_fixture_is_deterministic_and_offline(monkeypatch):
    monkeypatch.setenv("SAFE_MODE", "1")
    with patch("valuationengine.data.fetcher._load_yfinance") as load:
        first = fetch_company("demo")
        second = fetch_company("DEMO")
        peer = fetch_company("PEER")
    load.assert_not_called()
    assert first.ticker == "DEMO"
    assert first.name == "DEMO Fixture Co"
    assert first.revenue == second.revenue
    assert first.current_price == second.current_price
    assert peer.ticker == "PEER"
    assert peer.current_price != first.current_price


def test_safe_mode_rejects_invalid_ticker_without_live_client(monkeypatch):
    monkeypatch.setenv("SAFE_MODE", "1")
    with patch("valuationengine.data.fetcher._load_yfinance") as load:
        with pytest.raises(ValueError, match="Invalid ticker"):
            fetch_company("not a ticker")
    load.assert_not_called()


def test_network_guard_blocks_external_addresses(monkeypatch):
    monkeypatch.setenv("SAFE_MODE", "1")
    install_network_guard()
    import socket

    with pytest.raises(SafeModeNetworkError, match="SAFE_MODE"):
        socket.create_connection(("192.0.2.1", 9), timeout=0.2)
    with socket.socket() as sock:
        with pytest.raises(SafeModeNetworkError, match="SAFE_MODE"):
            sock.connect(("192.0.2.1", 9))
        with pytest.raises(SafeModeNetworkError, match="SAFE_MODE"):
            sock.connect_ex(("192.0.2.1", 9))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(SafeModeNetworkError, match="SAFE_MODE"):
            sock.sendto(b"blocked", ("192.0.2.1", 9))


def test_local_hosts_are_not_treated_as_external():
    assert _is_local("127.0.0.1") is True
    assert _is_local("::1") is True
    assert _is_local("localhost") is True
    assert _is_local("192.0.2.1") is False
    assert _is_local("example.test") is False
