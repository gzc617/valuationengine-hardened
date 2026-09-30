"""Ticker, projection-year, and sensitivity-step limits shared by the CLI and library."""

import pytest
from click.testing import CliRunner

from valuationengine.adapters.cli import cli
from valuationengine.validation import (
    MAX_PROJECTION_YEARS,
    MAX_SENSITIVITY_STEPS,
    MAX_TICKERS,
    parse_tickers,
    validate_projection_years,
    validate_history_years,
    validate_hold_years,
    validate_sensitivity_steps,
)


def test_parse_tickers_caps_at_five_even_if_caller_asks_for_more():
    raw = ",".join(f"T{i}" for i in range(8))
    tickers, warnings = parse_tickers(raw, max_tickers=100)
    assert len(tickers) == MAX_TICKERS == 5
    assert any("first **5**" in warning for warning in warnings)


def test_projection_years_bounds():
    assert validate_projection_years(1) == 1
    assert validate_projection_years(MAX_PROJECTION_YEARS) == MAX_PROJECTION_YEARS
    with pytest.raises(ValueError):
        validate_projection_years(0)
    with pytest.raises(ValueError):
        validate_projection_years(11)
    with pytest.raises(ValueError):
        validate_projection_years(True)


def test_sensitivity_steps_bounds():
    assert validate_sensitivity_steps(2) == 2
    assert validate_sensitivity_steps(MAX_SENSITIVITY_STEPS) == 20
    with pytest.raises(ValueError):
        validate_sensitivity_steps(1)
    with pytest.raises(ValueError):
        validate_sensitivity_steps(21)


def test_history_and_hold_bounds():
    assert validate_history_years(10) == 10
    assert validate_hold_years(10) == 10
    with pytest.raises(ValueError):
        validate_history_years(11)
    with pytest.raises(ValueError):
        validate_hold_years(11)


def test_cli_rejects_projection_years_outside_range():
    runner = CliRunner()
    result = runner.invoke(cli, ["dcf", "DEMO", "--years", "11"])
    assert result.exit_code != 0
    assert "10" in result.output


def test_cli_rejects_too_many_sensitivity_steps():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sensitivity",
            "DEMO",
            "--x-field",
            "revenue_growth",
            "--x-min",
            "0.04",
            "--x-max",
            "0.12",
            "--x-steps",
            "21",
            "--y-field",
            "operating_margin",
            "--y-min",
            "0.10",
            "--y-max",
            "0.20",
            "--y-steps",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "20" in result.output


def test_cli_rejects_invalid_ticker():
    runner = CliRunner()
    result = runner.invoke(cli, ["reverse", "NOT A TICKER"])
    assert result.exit_code != 0
    assert "Invalid ticker" in result.output


def test_cli_rejects_excessive_hold_period():
    runner = CliRunner()
    result = runner.invoke(cli, ["lbo", "DEMO", "--hold", "100000"])
    assert result.exit_code != 0


def test_cli_safe_mode_dcf_uses_fixture(monkeypatch):
    monkeypatch.setenv("SAFE_MODE", "1")
    runner = CliRunner()
    result = runner.invoke(cli, ["dcf", "demo"])
    assert result.exit_code == 0
    assert "DEMO Fixture Co" in result.output
    assert "SAFE_MODE=1" in result.output
