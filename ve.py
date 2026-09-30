#!/usr/bin/env python3
"""Run the valuationengine CLI from the project directory."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from valuationengine.adapters.cli import cli

if __name__ == "__main__":
    cli()
