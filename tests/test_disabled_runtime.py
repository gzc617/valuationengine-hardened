"""This CLI and library package does not read an OpenAI key or ship MCP."""

import importlib

import pytest

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PACKAGE = _REPO / "valuationengine"


def _package_sources() -> list[Path]:
    return sorted(_PACKAGE.rglob("*.py"))


def test_package_sources_do_not_read_openai_or_import_mcp():
    forbidden = (
        "OPENAI_API_KEY",
        "api.openai.com",
        "openai_tools_client",
        "fastmcp",
        "from openai",
        "import openai",
    )
    for path in _package_sources():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains {token}"
    assert not (_PACKAGE / "adapters" / "openai_tools_client.py").exists()


def test_runtime_dependencies_exclude_openai_and_mcp():
    runtime = (_REPO / "requirements.txt").read_text(encoding="utf-8").lower()
    project = (_REPO / "pyproject.toml").read_text(encoding="utf-8").lower()
    for name in ("fastmcp", "openai", "pytest"):
        assert name not in runtime
    assert "fastmcp" not in project
    assert "openai" not in project
    assert "pytest" in (_REPO / "requirements-test.txt").read_text(encoding="utf-8").lower()
    lock = (_REPO / "requirements-lock.txt").read_text(encoding="utf-8").lower()
    assert "openai" not in lock
    assert "fastmcp" not in lock


def test_mcp_adapter_is_disabled():
    with pytest.raises(RuntimeError, match="MCP is disabled"):
        importlib.import_module("valuationengine.adapters.mcp_server")
