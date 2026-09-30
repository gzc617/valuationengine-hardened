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
- **Input limits.** One shared validator (`valuationengine.validation`) is used by the CLI and library callers: tickers are 1–12 characters (`A–Z`, `0–9`, `.`, `-`, `^`), at most 5 tickers, projection years 1–10, sensitivity steps 2–20 per axis.
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
