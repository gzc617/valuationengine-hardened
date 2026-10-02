# Security model

`valuationengine` is a command-line tool and Python library. It computes DCF, LBO, and reverse DCF. It has no web server and no hosted deployment. The controls below keep the package from calling a language-model API, starting MCP, or depending on a credential or a disk.

## Trust boundary

- Callers can pass ticker strings and assumption values. Those inputs are untrusted.
- Fixture output is synthetic. It must not be treated as a quote or a filing.
- Live quotes, when `SAFE_MODE` is off, come from yfinance and Yahoo's service. That is an external data dependency, not an authentication secret stored here.

## Controls in this tree

- **No model API.** Library and CLI code do not import a tool-calling client and do not read `OPENAI_API_KEY`. The old client module is not in the package.
- **No MCP.** `valuationengine.adapters.mcp_server` raises on import. `fastmcp` is not a runtime or test dependency.
- **Safe mode.** `SAFE_MODE=1` returns fixture companies and installs a guard on `socket.socket.connect`, `connect_ex`, and `create_connection`. Loopback stays allowed. The live client is not imported on that path.
- **Input limits.** One shared validator (`valuationengine.validation`) is used by the CLI and library callers: tickers are 1–12 characters (`A–Z`, `0–9`, `.`, `-`, `^`), at most 5 tickers, projection years 1–10, sensitivity steps 2–20 per axis. A statement-model WACC override must be greater than 0 and less than 1. The FCF-margin sensitivity shortcut must be between -1 and 1.
- **No invented history.** The live fetcher keeps fiscal periods that report both revenue and EBIT. It aligns cash-flow and balance-sheet columns to those periods only when the dates match or differ by at most 5 days. A shared calendar year is not a match. History may be shorter than requested. It does not repeat an older observation or insert zeros to fill the window. Optional statement fields that were not reported are left missing. A zero is not written in their place. Missing cash and incomplete debt stay missing and are not valued as zero in the statement model. The effective tax rate still falls back to 0.25 when no tax line was reported, and that fallback is listed in `missing_fields`. Beta still defaults to 1.0 when a quote omits it.
- **Statement model.** The school statement DCF is opt-in. It uses the same CLI and library boundary, the same ticker and year limits, and no model API or MCP server. Book debt is a documented proxy in the WACC weights. It is not a new market-data credential. The statement model requires a complete operating-NWC stock and normalized component ratios, or explicit overrides, and it rejects negative opening net PPE before calculating D&A. Fixture companies used in safe mode include synthetic statement balances so the model runs offline; those figures are not filings.
- **Fetch failures.** Live fetches have a 20-second wait. Timeouts and connection errors become `TimeoutError` or `ValueError` instead of an uncaught hang on the caller.
- **Dependencies.** Runtime and test requirements are separate. Direct versions are exact pins. A hashed lock (`requirements-lock.txt`) is produced from those pins for review.

## Out of scope

- HTTP login, sessions, and a public hostname. This package does not serve HTTP.
- Multi-platform lock generation. The existing lock targets CPython 3.12 on linux/amd64 and is stale until it is regenerated from the current direct pins.
- Killing a yfinance worker thread that is still inside its own socket timeout after the 20-second wait returns. The caller stops waiting; the thread may continue until that library times out.
- Native sockets opened by curl-based HTTP clients. Safe mode avoids that stack by returning fixtures without calling it. With `SAFE_MODE` unset or off, live fetch uses that stack and places outbound network calls.

## Operator rules

- Keep `SAFE_MODE=1` when you want fixture companies and no outbound market-data calls.
- Do not set a market-data or model API key. The library does not need one.
