"""Shared input limits for the CLI and library callers.

Ticker text, ticker count, projection length, and sensitivity grid size are
checked here so every caller uses the same rules.
"""

from __future__ import annotations

import math
import re

MAX_TICKERS = 5
MAX_TICKER_LENGTH = 12
MIN_PROJECTION_YEARS = 1
MAX_PROJECTION_YEARS = 10
MIN_SENSITIVITY_STEPS = 2
MAX_SENSITIVITY_STEPS = 20
FETCH_TIMEOUT_SECONDS = 20
MAX_HISTORY_YEARS = 10
MIN_HOLD_YEARS = 1
MAX_HOLD_YEARS = 10

# Letters, digits, and the separators Yahoo-style symbols use (BRK.B, BRK-B).
_TICKER_RE = re.compile(rf"^[A-Z0-9.\-^]{{1,{MAX_TICKER_LENGTH}}}$")

_INVALID_TICKER_WARNING = "Invalid ticker removed (use letters, numbers, and . - ^ only)."


def normalize_ticker(raw: str) -> str:
    """Return one uppercase ticker, or raise ValueError."""
    ticker = (raw or "").strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        raise ValueError(
            "Invalid ticker. Use 1–12 characters: letters, numbers, and . - ^."
        )
    return ticker


def parse_tickers(raw: str, max_tickers: int = MAX_TICKERS) -> tuple[list[str], list[str]]:
    """
    Parse comma-separated tickers: trim, uppercase, deduplicate, cap count.

    The cap cannot be raised above MAX_TICKERS. A smaller limit is honored.

    Returns:
        (tickers, warnings) where warnings are user-facing info messages.
    """
    if isinstance(max_tickers, bool) or not isinstance(max_tickers, int):
        raise ValueError("max_tickers must be a whole number.")
    if max_tickers < 1:
        raise ValueError("max_tickers must be at least 1.")
    limit = min(max_tickers, MAX_TICKERS)

    seen: set[str] = set()
    tickers: list[str] = []
    warnings: list[str] = []

    for part in (raw or "").split(","):
        ticker = part.strip().upper()
        if not ticker:
            continue
        if not _TICKER_RE.fullmatch(ticker):
            warnings.append(_INVALID_TICKER_WARNING)
            continue
        if ticker in seen:
            warnings.append(f"Duplicate ticker **{ticker}** removed.")
            continue
        seen.add(ticker)
        tickers.append(ticker)

    if len(tickers) > limit:
        dropped = tickers[limit:]
        tickers = tickers[:limit]
        warnings.append(
            f"Showing the first **{limit}** tickers only "
            f"(dropped {', '.join(dropped)})."
        )

    return tickers, warnings


def validate_projection_years(years: int) -> int:
    """Require a projection length in the inclusive range 1–10."""
    if isinstance(years, bool) or not isinstance(years, int):
        raise ValueError("Projection years must be a whole number from 1 to 10.")
    if years < MIN_PROJECTION_YEARS or years > MAX_PROJECTION_YEARS:
        raise ValueError(
            f"Projection years must be between {MIN_PROJECTION_YEARS} and "
            f"{MAX_PROJECTION_YEARS}; got {years}."
        )
    return years


def validate_sensitivity_steps(steps: int, axis: str = "Sensitivity steps") -> int:
    """Require a grid axis length in the inclusive range 2–20."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError(f"{axis} must be a whole number from 2 to 20.")
    if steps < MIN_SENSITIVITY_STEPS or steps > MAX_SENSITIVITY_STEPS:
        raise ValueError(
            f"{axis} must be between {MIN_SENSITIVITY_STEPS} and "
            f"{MAX_SENSITIVITY_STEPS}; got {steps}."
        )
    return steps


def validate_history_years(years: int) -> int:
    if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= MAX_HISTORY_YEARS:
        raise ValueError(f"History years must be a whole number from 1 to {MAX_HISTORY_YEARS}.")
    return years


def validate_hold_years(years: int) -> int:
    if isinstance(years, bool) or not isinstance(years, int) or not MIN_HOLD_YEARS <= years <= MAX_HOLD_YEARS:
        raise ValueError(f"Hold years must be a whole number from {MIN_HOLD_YEARS} to {MAX_HOLD_YEARS}.")
    return years


DCF_MODEL_INTENSITY = "intensity"
DCF_MODEL_STATEMENT = "statement"
DCF_MODELS = (DCF_MODEL_INTENSITY, DCF_MODEL_STATEMENT)


def validate_dcf_model(model: str) -> str:
    """Accept the default intensity model or the opt-in statement model."""
    if model not in DCF_MODELS:
        raise ValueError("DCF model must be 'intensity' or 'statement'.")
    return model


def validate_wacc_override(wacc: float) -> float:
    """Bound an explicit WACC. The statement model is the caller that uses it."""
    if isinstance(wacc, bool) or not isinstance(wacc, (int, float)) or not math.isfinite(wacc):
        raise ValueError("WACC override must be a finite number.")
    if not 0 < wacc < 1:
        raise ValueError("WACC override must be greater than 0 and less than 1.")
    return float(wacc)


def validate_fcf_margin(margin: float) -> float:
    """Bound the FCF-margin sensitivity shortcut."""
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin):
        raise ValueError("FCF margin must be a finite number.")
    if not -1 <= margin <= 1:
        raise ValueError("FCF margin must be between -1 and 1.")
    return float(margin)
