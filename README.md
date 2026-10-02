# valuationengine

**DCF and LBO valuation toolkit.** Shipped as a Python library and a command-line tool. It does not call a language-model API, start an MCP server, or run a web server. See [SECURITY.md](SECURITY.md).

---

## What it does

`valuationengine` runs the kind of valuation work an equity analyst or private-equity associate would do in Excel, but in seconds from the library or the CLI:

- **Discounted Cash Flow (DCF)** with Gordon growth or exit-multiple terminal value
- **Leveraged Buyout (LBO)** with sources and uses, debt schedules, sponsor IRR and MOIC
- **Reverse DCF** that back-solves the market-implied assumptions baked into the current price
- **Sensitivity tables** across any two assumption fields
- **Scenario analysis** with built-in bull, base, and bear cases
- **Fundamentals** via yfinance when `SAFE_MODE` is off, or deterministic fixtures when `SAFE_MODE=1`

The default DCF is an **intensity model**. It grows the latest reported revenue, applies an EBIT margin, and models capex, D&A, and the change in net working capital as shares of revenue. Discounting is year-end. WACC uses a target debt weight. Equity value is enterprise value minus net debt, and the share count is basic shares outstanding. That path is unchanged when you do not ask for the statement model.

An opt-in **statement model** (`--model statement`, or `Assumptions.normalized_for`) is a school-style DCF built only from normalized reported history. It does not use sell-side estimates.

- Revenue growth and EBIT margin are explicit year-by-year paths (a single rate is repeated across the forecast, or you pass a list with one rate per year).
- Free cash flow equals EBIT after tax, plus D&A, minus capex, minus the change in operating net working capital.
- D&A is a rate on opening net PPE. The historical rate uses the immediately prior fiscal year's net PPE. A gap in `fiscal_year_ends` is not treated as an adjacent year. Negative opening net PPE is rejected before D&A is calculated. PPE rolls forward as opening PPE plus capex minus D&A.
- Revenue growth and CAGR use the elapsed time between `fiscal_year_ends` when those dates are present, so a dropped year is annualized.
- Operating net working capital is receivables, inventory, and other operating current assets, minus payables and other operating current liabilities. Cash equivalents and short-term investments are one non-operating cash amount: the equity bridge adds that amount once, and operating NWC excludes the same amount. Forecast balances use revenue ratios. The starting balance uses the latest reported components when all five were reported, or an explicit `starting_operating_nwc` override. A missing component is not replaced with zero or a generic ratio, and the components that were reported are not discarded.
- WACC weights are market capitalization and book debt. Book debt is a proxy for the market value of debt. Pass `--wacc` to override that WACC. Preferred equity and noncontrolling interest are not in the weights. Missing cash and incomplete debt stop the statement WACC and the equity bridge instead of being valued as zero. Total debt is the reported total, or the sum of current and long-term pieces when both are present. A long-term aggregate alone is not treated as total debt.
- Forecast cash flows and the terminal value use mid-year discounting.
- The terminal value is Gordon growth or an exit EV/EBITDA multiple.
- The equity bridge starts from enterprise value, deducts book debt, preferred equity, and noncontrolling interest, and adds cash plus a signed other adjustment.
- Diluted shares are used when they were reported. Otherwise the model uses basic shares.
- The result reports the terminal value's share of enterprise value.

`Assumptions.normalized_for` sets the statement drivers from medians of reported ratios and clips each median to a fixed bound. `Assumptions.calibrated_for` is unchanged: it still uses historical averages for the intensity model.

## Statement-model data lineage and calculation

This section describes exactly what the statement model reads, how it normalizes those inputs, and how it turns them into value. Amounts remain in the units returned by the data source; ratios are decimals.

### 1. Data source

With `SAFE_MODE` off, `fetch_company()` uses the public fields exposed by **yfinance/Yahoo Finance**. It does not currently fetch SEC filings directly and it does not use sell-side estimates. The source objects are:

- `Ticker.info`: company name, current/regular-market price, beta, basic shares, implied diluted shares when available, and market capitalization.
- `Ticker.financials`: revenue, EBIT or operating income, EBITDA when available, tax provision, and pretax income.
- `Ticker.cashflow`: depreciation and amortization, capital expenditures, and reported change in working capital.
- `Ticker.balance_sheet`: cash, short-term investments, debt, net PPE, receivables, inventory, current assets, payables, current liabilities, preferred equity, and noncontrolling interest.

`SAFE_MODE=1` replaces those network inputs with deterministic synthetic fixtures. Fixture output is for testing only; it is not a quote or filing.

### 2. Fiscal-period alignment and missing data

The income statement defines the candidate fiscal periods. A cash-flow or balance-sheet column is paired only when its date is the same or within five days; sharing a calendar year is not enough. A period missing revenue or EBIT is dropped, and the requested history can therefore return fewer years. History is never padded by repeating an older year or inserting zero.

Optional fields remain missing when the source does not report them. Statement mode stops rather than treating missing cash, incomplete debt, opening net PPE, or an incomplete starting operating-NWC stock as zero. The CLI result labels driver provenance and reports missing fields so normalized history, reported values, generic fallbacks, and explicit overrides remain distinguishable.

### 3. Historical normalization

`Assumptions.normalized_for(company)` creates the statement-model base case from medians of the aligned reported observations:

- **Revenue growth:** median annualized year-over-year growth. Actual elapsed time between `fiscal_year_ends` is used, so a two-year data gap is annualized over two years.
- **EBIT margin:** median of `EBIT / revenue`.
- **CapEx intensity:** median of `abs(CapEx) / revenue`.
- **D&A rate:** median of `D&A / immediately prior fiscal year's net PPE`. A missing intervening fiscal year is not treated as adjacent.
- **Operating-NWC ratios:** medians of each reported balance divided by revenue: receivables, inventory, other operating current assets, payables, and other operating current liabilities.
- **Tax rate:** historical tax provision divided by pretax income, bounded to 0%–40%; if no valid observation exists, the generic 25% default is visibly labeled as a fallback.

Normalized values are clipped to fixed validation ranges to prevent malformed source data from producing unbounded assumptions. Revenue growth and EBIT margin can instead be supplied as explicit year-by-year lists. Explicit CLI/library overrides take precedence and are labeled as overrides.

### 4. Revenue and operating-profit forecast

For each forecast year `t`:

```text
Revenue_t = Revenue_(t-1) × (1 + growth_t)
EBIT_t    = Revenue_t × EBIT_margin_t
NOPAT_t   = EBIT_t × (1 - tax_rate)
```

A scalar growth or margin is repeated across the forecast horizon; a list must contain exactly one value per forecast year.

### 5. PPE, depreciation, and capital expenditures

The latest reported net PPE is the first opening balance. Each forecast year uses:

```text
D&A_t         = Opening_net_PPE_t × D&A_rate
CapEx_t       = Revenue_t × CapEx_percent_of_revenue
Closing_PPE_t = Opening_net_PPE_t + CapEx_t - D&A_t
```

The closing PPE balance becomes the next year's opening balance. Negative opening or closing PPE is rejected rather than allowed to create an invalid forecast.

### 6. Operating net working capital

Cash and short-term investments are non-operating and excluded from operating NWC. Other operating current assets are derived, when the source lines are available, as current assets less the same combined cash amount, receivables, and inventory. Other operating current liabilities are current liabilities less payables and current debt.

For each forecast year:

```text
Receivables_t        = Revenue_t × receivables_ratio
Inventory_t          = Revenue_t × inventory_ratio
Other_operating_CA_t = Revenue_t × other_operating_CA_ratio
Payables_t           = Revenue_t × payables_ratio
Other_operating_CL_t = Revenue_t × other_operating_CL_ratio

Operating_NWC_t = Receivables_t + Inventory_t + Other_operating_CA_t
                  - Payables_t - Other_operating_CL_t
Change_NWC_t    = Operating_NWC_t - Operating_NWC_(t-1)
```

The first change uses the complete latest reported component stock, unless the caller supplies an explicit `starting_operating_nwc`. A missing component is not silently replaced with zero or a generic ratio.

### 7. Unlevered free cash flow

```text
FCF_t = NOPAT_t + D&A_t - CapEx_t - Change_NWC_t
```

The optional `fcf_margin_override` replaces this build with `Revenue_t × FCF_margin` only for sensitivity analysis. It is labeled as a shortcut, is not a full statement forecast, and cannot be combined with the exit-multiple terminal method.

### 8. WACC

Unless `wacc_override` is supplied:

```text
Cost_of_equity     = risk_free_rate + beta × equity_risk_premium
After_tax_debt_cost = cost_of_debt × (1 - tax_rate)
Equity_weight       = market_cap / (market_cap + book_debt)
Debt_weight         = book_debt / (market_cap + book_debt)
WACC                = Equity_weight × Cost_of_equity
                      + Debt_weight × After_tax_debt_cost
```

Market capitalization comes from `Ticker.info`, falling back to current price times basic shares. Debt is reported total debt, or current debt plus a reported long-term debt component when both are available. Book debt is explicitly a proxy for debt market value. Preferred equity and noncontrolling interest are excluded from WACC weights and handled in the equity bridge.

The default macro assumptions are 4.5% risk-free rate, 5.5% equity-risk premium, and 6.0% pre-tax cost of debt. These are model defaults—not live market observations—and should be overridden for a dated investment case.

### 9. Terminal value and mid-year discounting

The model supports either:

```text
Gordon growth: TV = FCF_final × (1 + terminal_growth) / (WACC - terminal_growth)
Exit multiple: TV = (EBIT_final + D&A_final) × exit_EV/EBITDA_multiple
```

Gordon growth requires `WACC > terminal_growth`. Explicit forecast cash flows use mid-year exponents `t - 0.5`; the terminal value is discounted at `forecast_years - 0.5`:

```text
PV(FCF_t) = FCF_t / (1 + WACC)^(t - 0.5)
Enterprise_value = sum(PV(FCF_t)) + PV(TV)
```

The result reports present-value terminal value as a percentage of enterprise value so terminal dependence is visible.

### 10. Enterprise-to-equity bridge and per-share value

The latest non-operating cash amount is cash equivalents plus short-term investments. It is added exactly once here and excluded from operating NWC above:

```text
Equity_value = Enterprise_value
               + cash_and_short_term_investments
               - book_debt
               - preferred_equity
               - noncontrolling_interest
               + signed_other_adjustments

Value_per_share = Equity_value / diluted_shares
```

Diluted shares are used when reported; otherwise the model identifies that it fell back to basic shares. Missing cash or incomplete debt blocks statement-mode WACC and the bridge instead of becoming a zero-valued claim. `other_adjustments` is a signed, explicit caller input because the live fetcher cannot reliably infer every pension, lease, investment, option, or other non-operating claim.

### 11. Interpretation and reproducibility

The output is a mechanical valuation from the fetched history plus displayed assumptions. It is not an investment recommendation. For a decision-grade case, reconcile every material input to dated primary filings, specify the valuation date, update market assumptions, make operating forecasts explicit by year, and test WACC/terminal growth, operating scenarios, and reverse DCF. Re-running with identical `Company` data and `Assumptions` produces the same result.

Reverse DCF, scenarios, and sensitivity all call the same DCF. Pass `--model statement` to run them on the statement model. The default remains intensity.

Two school sensitivity surfaces are available on the statement model: `wacc_override` versus `terminal_growth`, and `wacc_override` versus `fcf_margin_override`. The FCF-margin axis is a sensitivity shortcut. It replaces free cash flow with revenue times that margin. It does not rebuild PPE or working capital, and it cannot be combined with an exit multiple.

Assumptions auto-calibrate to each company's own reported history when you use the intensity default. Revenue growth, operating margin, capex intensity, D&A intensity, and working-capital intensity all default to that specific company's trailing averages instead of one generic profile applied to every ticker. Explicit overrides always take precedence. Terminal growth and WACC inputs are deliberately left at conservative macro defaults rather than extrapolated from history, since perpetuity assumptions should reflect long-run sustainable growth, not a company's recent trajectory.

Live fetches align income, cash flow, and balance-sheet columns only when they are the same fiscal period, within 5 days. A shared calendar year is not a match. Periods missing revenue or EBIT are dropped. The history can be shorter than the requested window. Optional lines that are absent stay missing. The fetcher does not repeat an earlier year or insert zeros to pad the series. Missing cash and incomplete debt stay missing rather than being stored as zero.

Statement results label each driver as normalized, reported, a historical average, a generic fallback, or an explicit override, and they list missing bridge or statement fields.

The library and the CLI call the same pure-Python engine, so the same inputs produce the same result from either path.

---

## Installation

```bash
git clone https://github.com/rishabhkarnawat/valuationengine.git
cd valuationengine
pip install -r requirements.txt
pip install --no-deps .
```

Requires Python 3.12 or higher. Direct runtime dependencies are pinned in `requirements.txt`. The hashed lock `requirements-lock.txt` is regenerated from those pins for CPython 3.12 on linux/amd64. Test-only dependencies are in `requirements-test.txt`.

---

## Quick start

### As a Python library

```python
from valuationengine.data.fetcher import fetch_company
from valuationengine.core.models import Assumptions
from valuationengine.core import dcf, reverse

amzn = fetch_company("AMZN")
result = dcf.run(amzn, Assumptions())
print(result.summary())

# Reverse DCF: what is the market currently pricing in?
implied = reverse.solve(amzn, Assumptions(), field="revenue_growth")
print(implied["interpretation"])
```

### From the command line

```bash
python -m valuationengine dcf AMZN --growth 0.10 --margin 0.20
python -m valuationengine dcf AMZN --model statement
python -m valuationengine reverse AMZN
python -m valuationengine reverse AMZN --model statement
python -m valuationengine lbo AMZN --entry-multiple 10 --debt-pct 0.6
python -m valuationengine scenario AMZN
python -m valuationengine scenario AMZN --model statement
python -m valuationengine sensitivity AMZN \
    --x-field revenue_growth --x-min 0.04 --x-max 0.12 --x-steps 5 \
    --y-field operating_margin --y-min 0.15 --y-max 0.25 --y-steps 5
python -m valuationengine sensitivity AMZN --model statement \
    --x-field terminal_growth --x-min 0.01 --x-max 0.03 --x-steps 3 \
    --y-field wacc_override --y-min 0.08 --y-max 0.10 --y-steps 3
```

`ve` is the installed console script. `python ve.py` runs the same CLI from a checkout. Tickers are validated, projection years must be 1–10, and sensitivity steps must be 2–20 on each axis. `--model` defaults to `intensity` for DCF, reverse DCF, scenarios, and sensitivity. The LBO command stays on the intensity drivers.

Set `SAFE_MODE=1` to skip live market data and use deterministic fixture companies (`DEMO`, `PEER`, `BETA`, `BULL`, `BEAR`).

---

## Architecture

```
valuationengine/
├── core/                pure-Python valuation engine (no I/O)
│   ├── models.py            Company, Assumptions, DCFResult, LBOResult
│   ├── dcf.py               intensity DCF by default; opt-in statement DCF
│   ├── lbo.py               sources/uses, projection, sponsor IRR and MOIC
│   ├── debt.py              amortization schedule with mandatory paydown and cash sweep
│   ├── sensitivity.py       2D grid across any two assumption fields
│   ├── scenario.py          bull, base, bear runner
│   └── reverse.py           reverse DCF solver via scipy brentq
├── data/
│   ├── fetcher.py           yfinance when SAFE_MODE is off; fixtures when it is on
│   └── fixtures.py          deterministic companies for safe mode
├── validation.py            ticker, year, and sensitivity limits
├── safe_mode.py             SAFE_MODE flag and socket guard
└── adapters/
    ├── cli.py               Click CLI
    └── mcp_server.py        disabled; import fails closed
```

The core engine never touches the network, never reads files, and never knows about a user interface. The CLI calls the same functions as the library.

---

## Why reverse DCF

A standard DCF asks: *given these assumptions, what is this company worth?* That question has no clean answer because the assumptions are made up.

A reverse DCF flips it: *given the current price, what assumptions must you believe?* That question has exactly one answer. Compare it to your own view of what is realistic.

`valuationengine` treats reverse DCF as a first-class capability. Solve for implied revenue growth, operating margin, or terminal growth from the CLI or the library.

---

## Case studies

Real-world walkthroughs in `examples/`:

- [**Amazon**](examples/amzn_case_study.ipynb) — what growth does AMZN's current price require, and is that defensible?
- [**Nvidia**](examples/nvda_case_study.ipynb) — a reverse DCF on the most-watched stock of the cycle.
- [**LBO screen**](examples/lbo_case_study.ipynb) — take-private analysis on a candidate, with debt schedule and entry-vs-exit sensitivity.

---

## Testing

```bash
pip install -r requirements-test.txt
pytest
```

The suite covers the core engine, the data fetcher, safe mode, input limits, and the disabled model-API and MCP paths. yfinance calls are mocked. Safe-mode tests use fixtures. Test dependencies stay in `requirements-test.txt`.

---

## Roadmap

- Direct SEC EDGAR integration as a primary data source
- Trading-comps and transaction-comps modules
- Multi-currency support for international tickers

---

## License

MIT. See [LICENSE](LICENSE).

---

## Author

Built by [Rishabh Karnawat](https://github.com/rishabhkarnawat). Issues and pull requests welcome.
