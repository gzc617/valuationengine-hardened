"""Opt-in school statement DCF, with the intensity model left in place."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from valuationengine.adapters.cli import cli
from valuationengine.core import dcf, reverse, scenario, sensitivity
from valuationengine.core.dcf import BOOK_DEBT_WACC_NOTE, FCF_MARGIN_SENSITIVITY_LABEL
from valuationengine.core.models import Assumptions, Company
from valuationengine.data.fetcher import fetch_company
from valuationengine.data.fixtures import fixture_company


def _statement_company(**overrides) -> Company:
    values = dict(
        ticker="STMT",
        name="Statement Co",
        revenue=[100.0],
        ebit=[20.0],
        ebitda=[28.0],
        depreciation_amortization=[8.0],
        capex=[5.0],
        change_in_nwc=[1.0],
        effective_tax_rate=0.25,
        cash=50.0,
        total_debt=100.0,
        shares_outstanding=100.0,
        current_price=5.0,
        market_cap=500.0,
        beta=1.0,
        net_ppe=[80.0],
        receivables=[10.0],
        inventory=[5.0],
        other_operating_current_assets=[1.0],
        payables=[8.0],
        other_operating_current_liabilities=[2.0],
        preferred_equity=10.0,
        noncontrolling_interest=5.0,
        other_adjustments=-3.0,
        diluted_shares=110.0,
    )
    values.update(overrides)
    return Company(**values)


def _statement_assumptions(**overrides) -> Assumptions:
    assumptions = Assumptions(
        valuation_model="statement",
        projection_years=2,
        revenue_growth=0.10,
        operating_margin=0.20,
        capex_pct_revenue=0.05,
        da_pct_opening_ppe=0.10,
        tax_rate=0.25,
        receivables_pct_revenue=0.10,
        inventory_pct_revenue=0.05,
        other_operating_ca_pct_revenue=0.01,
        payables_pct_revenue=0.08,
        other_operating_cl_pct_revenue=0.02,
        risk_free_rate=0.045,
        equity_risk_premium=0.055,
        cost_of_debt=0.06,
        terminal_growth=0.025,
    )
    for key, value in overrides.items():
        setattr(assumptions, key, value)
    return assumptions


def test_statement_fcf_identity_ppe_rollforward_and_nwc():
    company = _statement_company()
    assumptions = _statement_assumptions()
    result = dcf.run(company, assumptions)
    projection = result.projection

    assert list(projection["revenue_growth"]) == pytest.approx([0.10, 0.10])
    assert list(projection["ebit_margin"]) == pytest.approx([0.20, 0.20])
    assert list(projection["revenue"]) == pytest.approx([110.0, 121.0])
    assert projection.iloc[0]["opening_ppe"] == pytest.approx(80.0)
    assert projection.iloc[0]["da"] == pytest.approx(8.0)
    assert projection.iloc[0]["capex"] == pytest.approx(5.5)
    assert projection.iloc[0]["closing_ppe"] == pytest.approx(77.5)
    assert projection.iloc[1]["opening_ppe"] == pytest.approx(projection.iloc[0]["closing_ppe"])
    assert projection.iloc[0]["operating_nwc"] == pytest.approx(6.6)
    assert projection.iloc[0]["change_nwc"] == pytest.approx(0.6)
    assert projection.iloc[0]["fcf"] == pytest.approx(18.4)
    assert projection.iloc[1]["fcf"] == pytest.approx(19.19)

    for _, row in projection.iterrows():
        assert row["nopat"] == pytest.approx(row["ebit"] * (1 - assumptions.tax_rate))
        assert row["da"] == pytest.approx(row["opening_ppe"] * assumptions.da_pct_opening_ppe)
        assert row["closing_ppe"] == pytest.approx(row["opening_ppe"] + row["capex"] - row["da"])
        assert row["operating_nwc"] == pytest.approx(
            row["receivables"] + row["inventory"] + row["other_operating_ca"] - row["payables"] - row["other_operating_cl"]
        )
        assert row["fcf"] == pytest.approx(row["nopat"] + row["da"] - row["capex"] - row["change_nwc"])
        assert row["fcf_build"] == "statement"


def test_explicit_year_by_year_paths():
    assumptions = _statement_assumptions(
        projection_years=2,
        revenue_growth=[0.20, 0.10],
        operating_margin=[0.20, 0.30],
    )
    projection = dcf._project_statement(_statement_company(), assumptions, shortcut=False)
    assert list(projection["revenue"]) == pytest.approx([120.0, 132.0])
    assert list(projection["ebit"]) == pytest.approx([24.0, 39.6])
    assert list(projection["revenue_growth"]) == pytest.approx([0.20, 0.10])
    assert list(projection["ebit_margin"]) == pytest.approx([0.20, 0.30])


def test_mid_year_discount_factors_and_terminal_share():
    result = dcf.run(_statement_company(), _statement_assumptions())
    wacc = result.wacc
    for _, row in result.projection.iterrows():
        exponent = row["year"] - 0.5
        assert row["discount_exponent"] == pytest.approx(exponent)
        assert row["discount_factor"] == pytest.approx((1 + wacc) ** (-exponent))
    horizon = len(result.projection)
    assert result.pv_terminal_value == pytest.approx(result.terminal_value / (1 + wacc) ** (horizon - 0.5))
    assert result.terminal_value_share == pytest.approx(result.pv_terminal_value / result.enterprise_value)
    assert result.discount_convention == "mid-year"
    assert "TV share of EV" in result.summary()


def test_market_capital_wacc_uses_book_debt_as_proxy():
    company = _statement_company()
    assumptions = _statement_assumptions()
    result = dcf.run(company, assumptions)
    equity_weight = 500 / 600
    debt_weight = 100 / 600
    cost_of_equity = 0.045 + 1.0 * 0.055
    cost_of_debt_after_tax = 0.06 * (1 - 0.25)
    expected = equity_weight * cost_of_equity + debt_weight * cost_of_debt_after_tax
    assert result.wacc == pytest.approx(expected)
    assert result.wacc != pytest.approx(dcf.compute_wacc(Assumptions(), beta=1.0))
    assert "book debt" in result.wacc_note.lower()
    assert result.wacc_note == BOOK_DEBT_WACC_NOTE


def test_wacc_override_replaces_market_weights():
    result = dcf.run(_statement_company(), _statement_assumptions(wacc_override=0.09))
    assert result.wacc == pytest.approx(0.09)


def test_equity_bridge_signed_adjustment_and_diluted_shares():
    company = _statement_company()
    result = dcf.run(company, _statement_assumptions())
    bridge = result.equity_bridge
    assert bridge["other_adjustments"] == pytest.approx(-3.0)
    assert result.equity_value == pytest.approx(
        result.enterprise_value - 100.0 - 10.0 - 5.0 + 50.0 + (-3.0)
    )
    assert result.share_count_basis == "diluted"
    assert result.value_per_share == pytest.approx(result.equity_value / 110.0)

    basic = _statement_company(diluted_shares=None)
    basic_result = dcf.run(basic, _statement_assumptions())
    assert basic_result.share_count_basis == "basic"
    assert basic_result.value_per_share == pytest.approx(basic_result.equity_value / 100.0)
    assert basic_result.equity_bridge["preferred_equity_missing"] is False

    missing_claims = _statement_company(preferred_equity=None, noncontrolling_interest=None, other_adjustments=None, diluted_shares=None)
    missing_result = dcf.run(missing_claims, _statement_assumptions())
    assert missing_result.equity_bridge["preferred_equity"] == pytest.approx(0.0)
    assert missing_result.equity_bridge["preferred_equity_missing"] is True
    assert missing_result.equity_bridge["other_adjustments"] == pytest.approx(0.0)
    assert missing_result.equity_value == pytest.approx(missing_result.enterprise_value - 100.0 + 50.0)
    assert "preferred_equity" in missing_result.summary()
    assert "diluted_shares" in missing_result.summary()
    assert "cash equivalents plus short-term investments" in missing_result.summary()


def test_exit_multiple_uses_statement_ebitda():
    assumptions = _statement_assumptions(use_exit_multiple=True, exit_ev_ebitda_multiple=8.0)
    result = dcf.run(_statement_company(), assumptions)
    last = result.projection.iloc[-1]
    assert result.terminal_value == pytest.approx((last["ebit"] + last["da"]) * 8.0)


def test_fcf_margin_override_is_a_shortcut():
    assumptions = _statement_assumptions(fcf_margin_override=0.08)
    result = dcf.run(_statement_company(), assumptions)
    assert result.fcf_margin_shortcut is True
    assert result.note == FCF_MARGIN_SENSITIVITY_LABEL
    assert "sensitivity shortcut" in result.note
    assert "not a full statement build" in result.note
    assert list(result.projection["fcf"]) == pytest.approx([110.0 * 0.08, 121.0 * 0.08])
    assert result.projection.iloc[0]["fcf_build"] == "fcf_margin_shortcut"
    assert result.projection["opening_ppe"].isna().all()
    with pytest.raises(ValueError, match="sensitivity shortcut"):
        dcf.run(_statement_company(), _statement_assumptions(fcf_margin_override=0.08, use_exit_multiple=True))


def test_incomplete_operating_nwc_is_not_replaced_with_ratios():
    company = _statement_company(receivables=None)
    with pytest.raises(ValueError, match="not replaced") as exc_info:
        dcf.run(company, _statement_assumptions(projection_years=1))
    message = str(exc_info.value)
    assert "receivables" in message
    assert "5.0" in message
    assert "generic ratio" in message

    explicit = Assumptions.normalized_for(company, starting_operating_nwc=6.0, receivables_pct_revenue=0.11)
    result = dcf.run(company, explicit)
    row = result.projection.iloc[0]
    assert row["change_nwc"] == pytest.approx(row["operating_nwc"] - 6.0)
    assert result.note.startswith("Starting operating NWC is an explicit override.")

    still_generic = Assumptions.normalized_for(company, starting_operating_nwc=6.0)
    with pytest.raises(ValueError, match="generic operating-NWC ratios"):
        dcf.run(company, still_generic)


def test_missing_ppe_is_not_invented():
    company = _statement_company(net_ppe=None)
    with pytest.raises(ValueError, match="net PPE"):
        dcf.run(company, _statement_assumptions())


def test_negative_opening_ppe_is_rejected_before_da():
    company = _statement_company(net_ppe=[-5.0])
    with pytest.raises(ValueError, match="negative opening net PPE"):
        dcf.run(company, _statement_assumptions(projection_years=1, da_pct_opening_ppe=0.10))


def test_negative_closing_ppe_is_rejected():
    company = _statement_company(net_ppe=[1.0])
    assumptions = _statement_assumptions(
        projection_years=1,
        capex_pct_revenue=0.0,
        da_pct_opening_ppe=2.0,
    )
    with pytest.raises(ValueError, match="negative closing net PPE"):
        dcf.run(company, assumptions)


def test_reverse_rejects_non_numeric_assumption_fields():
    with pytest.raises(ValueError, match="continuous numeric DCF input"):
        reverse.solve(
            _statement_company(),
            _statement_assumptions(),
            field="valuation_model",
        )


def test_intensity_model_is_unchanged_by_statement_fields(sample_company, base_assumptions):
    enriched = deepcopy(sample_company)
    enriched.net_ppe = [40.0, 42.0, 44.0, 46.0, 48.0]
    enriched.preferred_equity = 25.0
    enriched.noncontrolling_interest = 7.0
    enriched.other_adjustments = -4.0
    enriched.diluted_shares = 140.0
    plain = dcf.run(sample_company, base_assumptions)
    with_statement_facts = dcf.run(enriched, base_assumptions)
    assert plain.valuation_model == "intensity"
    assert plain.discount_convention == "year_end"
    assert plain.share_count_basis == "basic"
    assert plain.equity_bridge is None
    assert with_statement_facts.value_per_share == pytest.approx(plain.value_per_share)
    assert with_statement_facts.equity_value == pytest.approx(plain.enterprise_value - sample_company.net_debt)
    year = plain.projection.iloc[0]["year"]
    assert plain.projection.iloc[0]["discount_factor"] == pytest.approx(1 / (1 + plain.wacc) ** year)
    assert "mid-year" not in plain.summary()


def test_normalized_for_uses_medians_and_calibrated_for_stays_on_averages():
    company = Company(
        ticker="MED",
        name="Median Co",
        revenue=[100.0, 150.0, 150.0],
        ebit=[10.0, 30.0, 90.0],
        ebitda=[14.0, 36.0, 96.0],
        depreciation_amortization=[4.0, 6.0, 6.0],
        capex=[1.0, 3.0, 15.0],
        change_in_nwc=[1.0, 2.0, 9.0],
        effective_tax_rate=0.21,
        cash=10.0,
        total_debt=20.0,
        shares_outstanding=10.0,
        current_price=8.0,
        market_cap=80.0,
        beta=1.0,
        net_ppe=[40.0, 50.0, 60.0],
        receivables=[5.0, 30.0, 15.0],
    )
    normalized = Assumptions.normalized_for(company)
    calibrated = Assumptions.calibrated_for(company)
    assert normalized.valuation_model == "statement"
    assert calibrated.valuation_model == "intensity"
    assert normalized.revenue_growth == pytest.approx(0.25)
    assert calibrated.revenue_growth == pytest.approx(company.historical_revenue_cagr)
    assert normalized.revenue_growth != pytest.approx(calibrated.revenue_growth)
    assert normalized.operating_margin == pytest.approx(0.20)
    assert calibrated.operating_margin == pytest.approx(company.avg_operating_margin)
    assert normalized.capex_pct_revenue == pytest.approx(0.02)
    assert calibrated.capex_pct_revenue == pytest.approx(company.avg_capex_pct_revenue)
    assert normalized.receivables_pct_revenue == pytest.approx(0.10)
    assert normalized.tax_rate == pytest.approx(0.21)
    assert normalized.terminal_growth == pytest.approx(Assumptions().terminal_growth)
    assert calibrated.da_pct_opening_ppe == pytest.approx(Assumptions().da_pct_opening_ppe)


def test_normalized_for_bounds_extreme_history():
    company = _statement_company(revenue=[100.0, 1000.0], ebit=[80.0, 950.0])
    assumptions = Assumptions.normalized_for(company)
    assert assumptions.revenue_growth == pytest.approx(0.40)
    assert assumptions.operating_margin == pytest.approx(0.60)


def test_normalized_for_keeps_default_when_history_cannot_support_a_median():
    company = _statement_company(revenue=[100.0], net_ppe=None, receivables=None)
    assumptions = Assumptions.normalized_for(company)
    assert assumptions.revenue_growth == pytest.approx(Assumptions().revenue_growth)
    assert assumptions.da_pct_opening_ppe == pytest.approx(Assumptions().da_pct_opening_ppe)
    assert assumptions.receivables_pct_revenue == pytest.approx(Assumptions().receivables_pct_revenue)


def test_reverse_scenario_and_sensitivity_follow_statement_mode():
    company = _statement_company()
    base = _statement_assumptions(projection_years=4, revenue_growth=0.08, wacc_override=0.09)
    planted = deepcopy(company)
    known = deepcopy(base)
    known.revenue_growth = 0.12
    target = dcf.run(planted, known)
    planted.market_cap = target.equity_value
    planted.current_price = target.value_per_share
    solved = reverse.solve(planted, base, field="revenue_growth", target="market_cap", bracket=(-0.05, 0.40))
    assert solved["implied_value"] == pytest.approx(0.12, abs=1e-3)

    scenarios = scenario.build_bull_base_bear(base, growth_delta=0.02, margin_delta=0.02)
    results = scenario.run(company, scenarios)
    assert results["bull"].value_per_share > results["base"].value_per_share > results["bear"].value_per_share
    assert results["base"].valuation_model == "statement"

    wacc_table = sensitivity.run(
        company,
        base,
        x_field="terminal_growth",
        x_values=[0.01, 0.02],
        y_field="wacc_override",
        y_values=[0.08, 0.10],
    )
    assert wacc_table.iloc[0, 1] > wacc_table.iloc[0, 0]
    assert wacc_table.iloc[0, 1] > wacc_table.iloc[1, 1]

    margin_table = sensitivity.run(
        company,
        base,
        x_field="fcf_margin_override",
        x_values=[0.05, 0.12],
        y_field="wacc_override",
        y_values=[0.09],
    )
    assert margin_table.iloc[0, 1] > margin_table.iloc[0, 0]
    shortcut = dcf.run(company, _statement_assumptions(fcf_margin_override=0.05, wacc_override=0.09))
    assert shortcut.fcf_margin_shortcut is True


def test_safe_mode_statement_fixtures_do_not_use_the_network(monkeypatch):
    monkeypatch.setenv("SAFE_MODE", "1")
    with patch("valuationengine.data.fetcher._load_yfinance") as load:
        company = fetch_company("DEMO")
        result = dcf.run(company, Assumptions.normalized_for(company))
    load.assert_not_called()
    assert company.diluted_shares == pytest.approx(company.shares_outstanding * 1.05)
    assert company.other_adjustments == pytest.approx(-5.0)
    assert company.net_ppe[1] == pytest.approx(company.net_ppe[0] + company.capex[1] - company.depreciation_amortization[1])
    assert result.valuation_model == "statement"
    assert result.value_per_share != 0


def test_cli_model_selection_keeps_intensity_as_the_default(monkeypatch):
    monkeypatch.setenv("SAFE_MODE", "1")
    from click.testing import CliRunner

    runner = CliRunner()
    default = runner.invoke(cli, ["dcf", "DEMO"])
    statement = runner.invoke(cli, ["dcf", "DEMO", "--model", "statement"])
    rejected = runner.invoke(cli, ["dcf", "DEMO", "--wacc", "0.09"])
    surface = runner.invoke(
        cli,
        [
            "sensitivity",
            "DEMO",
            "--model",
            "statement",
            "--x-field",
            "fcf_margin_override",
            "--x-min",
            "0.04",
            "--x-max",
            "0.08",
            "--x-steps",
            "2",
            "--y-field",
            "wacc_override",
            "--y-min",
            "0.08",
            "--y-max",
            "0.10",
            "--y-steps",
            "2",
        ],
    )
    reverse_case = runner.invoke(cli, ["reverse", "DEMO", "--model", "statement"])
    scenario_case = runner.invoke(cli, ["scenario", "DEMO", "--model", "statement"])
    assert default.exit_code == 0
    assert "mid-year" not in default.output
    assert "historical_average" in default.output
    assert statement.exit_code == 0
    assert "mid-year" in statement.output
    assert "book debt" in statement.output
    assert "TV share of EV" in statement.output
    assert rejected.exit_code != 0
    assert "statement" in rejected.output
    assert surface.exit_code == 0
    assert FCF_MARGIN_SENSITIVITY_LABEL in surface.output
    assert reverse_case.exit_code == 0
    assert "Reverse DCF" in reverse_case.output
    assert scenario_case.exit_code == 0
    assert "bull" in scenario_case.output


def test_cash_and_debt_must_be_complete_in_statement_mode():
    missing_cash = _statement_company(cash=None)
    with pytest.raises(ValueError, match="Unresolved claims: cash"):
        dcf.run(missing_cash, _statement_assumptions())
    with pytest.raises(ValueError, match="not valued as zero"):
        dcf.compute_statement_wacc(_statement_assumptions(wacc_override=0.09), 1.0, missing_cash)

    fake_zero = _statement_company(cash=0.0, missing_fields=("cash",))
    with pytest.raises(ValueError, match="not valued as zero"):
        dcf.run(fake_zero, _statement_assumptions())

    missing_debt = _statement_company(total_debt=None, missing_fields=("total_debt", "current_debt"))
    with pytest.raises(ValueError, match="Unresolved claims: total_debt"):
        dcf.run(missing_debt, _statement_assumptions())

    partial_flag = _statement_company(total_debt=80.0, missing_fields=("long_term_debt",))
    with pytest.raises(ValueError, match="total_debt"):
        dcf.run(partial_flag, _statement_assumptions())

    explicit = _statement_company(cash=0.0, total_debt=0.0, short_term_investments=0.0)
    result = dcf.run(explicit, _statement_assumptions())
    assert result.equity_bridge["cash"] == pytest.approx(0.0)
    assert result.equity_bridge["debt"] == pytest.approx(0.0)
    assert result.equity_value == pytest.approx(result.enterprise_value - 10.0 - 5.0 + 0.0 + (-3.0))


def test_short_term_investments_are_inside_bridge_cash_once():
    company = _statement_company(cash=50.0, short_term_investments=30.0)
    result = dcf.run(company, _statement_assumptions())
    assert result.equity_bridge["cash"] == pytest.approx(50.0)
    assert result.equity_bridge["short_term_investments"] == pytest.approx(30.0)
    assert result.equity_value == pytest.approx(result.enterprise_value - 100.0 - 10.0 - 5.0 + 50.0 + (-3.0))
    assert "generic_fallback" not in result.summary()


def test_gapped_years_are_annualized_and_da_uses_the_prior_fiscal_ppe():
    gapped_growth = Company(
        ticker="GAP",
        name="Gap Co",
        revenue=[100.0, 121.0],
        ebit=[20.0, 24.0],
        ebitda=[24.0, 30.0],
        depreciation_amortization=[4.0, 50.0],
        capex=[5.0, 6.0],
        change_in_nwc=[1.0, 1.0],
        effective_tax_rate=0.21,
        cash=10.0,
        total_debt=20.0,
        shares_outstanding=10.0,
        current_price=8.0,
        market_cap=80.0,
        beta=1.0,
        fiscal_year_ends=["2020-12-31", "2022-12-31"],
        net_ppe=[80.0, 70.0],
        receivables=[10.0, 12.0],
        inventory=[5.0, 6.0],
        other_operating_current_assets=[1.0, 1.0],
        payables=[8.0, 9.0],
        other_operating_current_liabilities=[2.0, 2.0],
    )
    assert gapped_growth.historical_revenue_cagr == pytest.approx(0.10, abs=1e-3)
    normalized = Assumptions.normalized_for(gapped_growth)
    assert normalized.revenue_growth == pytest.approx(0.10, abs=1e-3)
    assert normalized.driver_sources["revenue_growth"] == "normalized"
    assert normalized.driver_sources["da_pct_opening_ppe"] == "generic_fallback"
    assert normalized.da_pct_opening_ppe == pytest.approx(Assumptions().da_pct_opening_ppe)

    adjacent = Company(
        ticker="ADJ",
        name="Adjacent Co",
        revenue=[100.0, 110.0, 120.0],
        ebit=[10.0, 11.0, 12.0],
        ebitda=[15.0, 21.0, 62.0],
        depreciation_amortization=[5.0, 10.0, 50.0],
        capex=[4.0, 4.0, 4.0],
        change_in_nwc=[1.0, 1.0, 1.0],
        effective_tax_rate=0.21,
        cash=10.0,
        total_debt=20.0,
        shares_outstanding=10.0,
        current_price=8.0,
        market_cap=80.0,
        beta=1.0,
        fiscal_year_ends=["2020-12-31", "2021-12-31", "2023-12-31"],
        net_ppe=[100.0, 80.0, 70.0],
        receivables=[10.0, 11.0, 12.0],
        inventory=[5.0, 5.0, 5.0],
        other_operating_current_assets=[1.0, 1.0, 1.0],
        payables=[8.0, 8.0, 8.0],
        other_operating_current_liabilities=[2.0, 2.0, 2.0],
    )
    paired = Assumptions.normalized_for(adjacent)
    assert paired.da_pct_opening_ppe == pytest.approx(0.10)
    assert paired.driver_sources["da_pct_opening_ppe"] == "normalized"


def test_cli_labels_sources_and_missing_statement_fields(monkeypatch):
    monkeypatch.setenv("SAFE_MODE", "1")
    from click.testing import CliRunner

    company = _statement_company(
        revenue=[100.0],
        ebit=[20.0],
        depreciation_amortization=[8.0],
        net_ppe=[80.0],
        preferred_equity=None,
        missing_fields=("preferred_equity",),
    )

    monkeypatch.setattr("valuationengine.adapters.cli.fetch_company", lambda ticker: company)
    runner = CliRunner()
    statement = runner.invoke(cli, ["dcf", "STMT", "--model", "statement"])
    assert statement.exit_code == 0, statement.output
    assert "generic_fallback" in statement.output
    assert "normalized" in statement.output
    assert "Missing fields:" in statement.output
    assert "preferred_equity" in statement.output
    assert "reported latest component stock" in statement.output


def test_fixture_company_builder_has_statement_fields_offline():
    company = fixture_company("PEER", history_years=3)
    assert len(company.revenue) == 3
    assert company.net_ppe is not None
    assert len(company.net_ppe) == 3
    assert company.missing_fields == ()
