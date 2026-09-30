"""MCP is excluded from this CLI and library package.

The module fails on import so it cannot start a server or load an MCP framework.
DCF, LBO, and reverse DCF remain available through the CLI and library.
"""

raise RuntimeError(
    "MCP is disabled in this package and is not part of the runtime."
)
