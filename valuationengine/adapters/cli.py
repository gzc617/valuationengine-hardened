"""Click-based command-line interface."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import click
import pandas as pd

from valuationengine.core import dcf as dcf_module
from valuationengine.core import lbo as lbo_module
from valuationengine.core import reverse as reverse_module
from valuationengine.core import scenario as scenario_module
from valuationengine.core import sensitivity as sensitivity_module
from valuationengine.core.models import Assumptions
from valuationengine.data.fetcher import fetch_company
from valuationengine.safe_mode import install_network_guard, safe_mode_enabled
from valuationengine.validation import (
    normalize_ticker,
    validate_hold_years,
    validate_projection_years,
    validate_sensitivity_steps,
)


@click.group()
def cli():
    """DCF, LBO, and reverse DCF valuation toolkit."""
    if safe_mode_enabled():
        install_network_guard()
        click.echo(
            "SAFE_MODE=1: using deterministic fixture companies. External network calls are blocked.",
            err=True,
        )


def _load_company(ticker: str):
    """Validate one ticker and load it (fixture data when SAFE_MODE is on)."""
    symbol = normalize_ticker(ticker)
    return fetch_company(symbol)


def _history_source(ticker: str) -> str:
    if safe_mode_enabled():
        return "deterministic fixture history (SAFE_MODE=1)"
    return f"{ticker}'s own 5yr history"


@cli.command()
@click.argument("ticker")
@click.option("--growth", type=float, default=None, help="Revenue growth rate as decimal (e.g. 0.10).")
@click.option("--margin", type=float, default=None, help="Operating (EBIT) margin as decimal.")
@click.option("--wacc-rf", type=float, default=None, help="Risk-free rate.")
@click.option("--wacc-erp", type=float, default=None, help="Equity risk premium.")
@click.option("--terminal-growth", type=float, default=None, help="Terminal growth rate.")
@click.option("--use-exit-multiple", is_flag=True, help="Use exit EV/EBITDA multiple for terminal value.")
@click.option("--exit-multiple", type=float, default=None, help="Terminal exit EV/EBITDA multiple.")
@click.option("--years", type=int, default=None, help="Projection years.")
@click.pass_context
def dcf(
    ctx,
    ticker,
    growth,
    margin,
    wacc_rf,
    wacc_erp,
    terminal_growth,
    use_exit_multiple,
    exit_multiple,
    years,
):
    """Run a DCF on TICKER."""
    try:
        if years is not None:
            years = validate_projection_years(years)
        company = _load_company(ticker)
        ticker = company.ticker
        assumptions = _build_assumptions(
            company,
            growth=growth,
            margin=margin,
            wacc_rf=wacc_rf,
            wacc_erp=wacc_erp,
            terminal_growth=terminal_growth,
            use_exit_multiple=use_exit_multiple,
            exit_multiple=exit_multiple,
            years=years,
        )
        result = dcf_module.run(company, assumptions)
        click.echo(
            f"Calibrated base case: {assumptions.revenue_growth*100:.1f}% growth, "
            f"{assumptions.operating_margin*100:.1f}% margin (from {_history_source(ticker)} "
            f"unless overridden by flags)"
        )
        click.echo(result.summary())
        click.echo("")
        click.echo("Projection:")
        click.echo(result.projection.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    except (ValueError, TimeoutError, OSError) as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


@cli.command()
@click.argument("ticker")
@click.option("--entry-multiple", type=float, default=None)
@click.option("--exit-multiple", type=float, default=None)
@click.option("--debt-pct", type=float, default=None)
@click.option("--interest-rate", type=float, default=None)
@click.option("--hold", type=click.IntRange(1, 10), default=None)
@click.pass_context
def lbo(ctx, ticker, entry_multiple, exit_multiple, debt_pct, interest_rate, hold):
    """Run an LBO on TICKER."""
    try:
        company = _load_company(ticker)
        ticker = company.ticker
        a = Assumptions.calibrated_for(company)
        a.tax_rate = company.effective_tax_rate
        if entry_multiple is not None:
            a.entry_ev_ebitda_multiple = entry_multiple
        if exit_multiple is not None:
            a.exit_lbo_ev_ebitda_multiple = exit_multiple
        if debt_pct is not None:
            a.debt_pct_purchase = debt_pct
        if interest_rate is not None:
            a.lbo_debt_interest_rate = interest_rate
        if hold is not None:
            a.hold_period_years = validate_hold_years(hold)
        result = lbo_module.run(company, a)
        click.echo(
            f"Calibrated base case: {a.revenue_growth*100:.1f}% growth, "
            f"{a.operating_margin*100:.1f}% margin (from {_history_source(ticker)} "
            f"unless overridden by flags)"
        )
        click.echo(result.summary())
        click.echo("")
        click.echo("Debt schedule:")
        click.echo(result.debt_schedule.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    except (ValueError, TimeoutError, OSError) as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


@cli.command()
@click.argument("ticker")
@click.option("--field", default="revenue_growth", type=click.Choice(["revenue_growth", "operating_margin", "terminal_growth"]), help="Assumption field to solve for.")
@click.option("--target", default="market_cap", type=click.Choice(["market_cap", "current_price"]))
@click.pass_context
def reverse(ctx, ticker, field, target):
    """Run a reverse DCF on TICKER: back-solve market-implied assumptions."""
    try:
        company = _load_company(ticker)
        ticker = company.ticker
        assumptions = Assumptions.calibrated_for(company)
        assumptions.tax_rate = company.effective_tax_rate
        result = reverse_module.solve(
            company, assumptions, field=field, target=target
        )
        click.echo(
            f"Calibrated base case: {assumptions.revenue_growth*100:.1f}% growth, "
            f"{assumptions.operating_margin*100:.1f}% margin (from {_history_source(ticker)} "
            f"unless overridden by flags)"
        )
        click.echo(f"\nReverse DCF for {ticker}")
        click.echo(f"Solving for: {result['field']}")
        click.echo(f"Target ({result['target']}): {result['target_value']:,.2f}")
        click.echo(f"Implied value: {result['implied_value']:.4f}")
        click.echo(f"\n{result['interpretation']}")
    except (ValueError, TimeoutError, OSError) as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


@cli.command()
@click.argument("ticker")
@click.option("--x-field", required=True)
@click.option("--x-min", type=float, required=True)
@click.option("--x-max", type=float, required=True)
@click.option("--x-steps", type=int, default=5)
@click.option("--y-field", required=True)
@click.option("--y-min", type=float, required=True)
@click.option("--y-max", type=float, required=True)
@click.option("--y-steps", type=int, default=5)
@click.option("--output", default="value_per_share", type=click.Choice(["value_per_share", "upside", "enterprise_value", "equity_value"]))
@click.pass_context
def sensitivity(
    ctx,
    ticker,
    x_field,
    x_min,
    x_max,
    x_steps,
    y_field,
    y_min,
    y_max,
    y_steps,
    output,
):
    """Run a 2D sensitivity table on TICKER."""
    import numpy as np

    try:
        x_steps = validate_sensitivity_steps(x_steps, axis="X steps")
        y_steps = validate_sensitivity_steps(y_steps, axis="Y steps")
        company = _load_company(ticker)
        ticker = company.ticker
        assumptions = Assumptions.calibrated_for(company)
        assumptions.tax_rate = company.effective_tax_rate
        x_values = list(np.linspace(x_min, x_max, x_steps))
        y_values = list(np.linspace(y_min, y_max, y_steps))
        table = sensitivity_module.run(
            company,
            assumptions,
            x_field,
            x_values,
            y_field,
            y_values,
            output=output,
        )
        click.echo(
            f"Calibrated base case: {assumptions.revenue_growth*100:.1f}% growth, "
            f"{assumptions.operating_margin*100:.1f}% margin (from {_history_source(ticker)} "
            f"unless overridden by flags)"
        )
        click.echo(f"\nSensitivity table for {ticker} ({output})")
        click.echo(f"Rows: {y_field}, Columns: {x_field}\n")
        click.echo(table.to_string(float_format=lambda x: f"{x:,.2f}"))
    except (ValueError, TimeoutError, OSError) as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


@cli.command()
@click.argument("ticker")
@click.option("--growth-delta", type=float, default=0.03)
@click.option("--margin-delta", type=float, default=0.03)
@click.pass_context
def scenario(ctx, ticker, growth_delta, margin_delta):
    """Run bull / base / bear scenario analysis on TICKER."""
    try:
        company = _load_company(ticker)
        ticker = company.ticker
        assumptions = Assumptions.calibrated_for(company)
        assumptions.tax_rate = company.effective_tax_rate
        scenarios = scenario_module.build_bull_base_bear(
            assumptions,
            growth_delta=growth_delta,
            margin_delta=margin_delta,
        )
        results = scenario_module.run(company, scenarios)
        click.echo(
            f"Calibrated base case: {assumptions.revenue_growth*100:.1f}% growth, "
            f"{assumptions.operating_margin*100:.1f}% margin (from {_history_source(ticker)} "
            f"unless overridden by flags)"
        )
        click.echo(f"\nScenario analysis for {ticker}\n")
        rows = []
        for name, r in results.items():
            rows.append(
                {
                    "scenario": name,
                    "value_per_share": r.value_per_share,
                    "current_price": company.current_price,
                    "upside_pct": r.upside * 100,
                }
            )
        click.echo(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    except (ValueError, TimeoutError, OSError) as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


def _build_assumptions(
    company,
    growth=None,
    margin=None,
    wacc_rf=None,
    wacc_erp=None,
    terminal_growth=None,
    use_exit_multiple=False,
    exit_multiple=None,
    years=None,
) -> Assumptions:
    a = Assumptions.calibrated_for(company)
    a.tax_rate = company.effective_tax_rate
    if growth is not None:
        a.revenue_growth = growth
    if margin is not None:
        a.operating_margin = margin
    if wacc_rf is not None:
        a.risk_free_rate = wacc_rf
    if wacc_erp is not None:
        a.equity_risk_premium = wacc_erp
    if terminal_growth is not None:
        a.terminal_growth = terminal_growth
    if use_exit_multiple:
        a.use_exit_multiple = True
    if exit_multiple is not None:
        a.exit_ev_ebitda_multiple = exit_multiple
    if years is not None:
        a.projection_years = validate_projection_years(years)
    return a


if __name__ == "__main__":
    cli()
