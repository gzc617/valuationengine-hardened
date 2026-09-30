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

Assumptions auto-calibrate to each company's own five-year history. Revenue growth, operating margin, capex intensity, D&A intensity, and working-capital intensity all default to that specific company's trailing averages instead of one generic profile applied to every ticker. Explicit overrides always take precedence. Terminal growth and WACC inputs are deliberately left at conservative macro defaults rather than extrapolated from history, since perpetuity assumptions should reflect long-run sustainable growth, not a company's recent trajectory.

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
python -m valuationengine reverse AMZN
python -m valuationengine lbo AMZN --entry-multiple 10 --debt-pct 0.6
python -m valuationengine scenario AMZN
python -m valuationengine sensitivity AMZN \
    --x-field revenue_growth --x-min 0.04 --x-max 0.12 --x-steps 5 \
    --y-field operating_margin --y-min 0.15 --y-max 0.25 --y-steps 5
```

`ve` is the installed console script. `python ve.py` runs the same CLI from a checkout. Tickers are validated, projection years must be 1–10, and sensitivity steps must be 2–20 on each axis.

Set `SAFE_MODE=1` to skip live market data and use deterministic fixture companies (`DEMO`, `PEER`, `BETA`, `BULL`, `BEAR`).

---

## Architecture

```
valuationengine/
├── core/                pure-Python valuation engine (no I/O)
│   ├── models.py            Company, Assumptions, DCFResult, LBOResult
│   ├── dcf.py               WACC, FCF projection, terminal value, DCF
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
