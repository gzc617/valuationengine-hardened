"""SAFE_MODE blocks outbound network use and selects fixture companies.

SAFE_MODE is on when the environment variable is 1, true, yes, or on.
Unset means live fetch is allowed for this CLI and library process.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading

_TRUE = {"1", "true", "yes", "on"}
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


class SafeModeNetworkError(RuntimeError):
    """Raised when SAFE_MODE refuses an outbound socket connection."""


def safe_mode_enabled() -> bool:
    """Return True when this process must not place external network calls."""
    return os.environ.get("SAFE_MODE", "").strip().lower() in _TRUE


def install_network_guard() -> None:
    """Refuse non-local socket connections while SAFE_MODE is enabled.

    The check runs on each connect. Turning SAFE_MODE off later in the same
    process allows connections again. Loopback and local UNIX sockets stay
    available.

    This covers the Python standard-library sockets. Live market-data clients
    that open sockets in native code are not used while safe mode is on,
    because the fetch path returns fixtures without importing yfinance.
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return

        orig_connect = socket.socket.connect
        orig_connect_ex = socket.socket.connect_ex
        orig_create = socket.create_connection
        orig_sendto = socket.socket.sendto
        orig_sendmsg = getattr(socket.socket, "sendmsg", None)

        def connect(self, address):
            if safe_mode_enabled():
                _reject_external(address)
            return orig_connect(self, address)

        def connect_ex(self, address):
            if safe_mode_enabled():
                _reject_external(address)
            return orig_connect_ex(self, address)

        def create_connection(address, *args, **kwargs):
            if safe_mode_enabled():
                _reject_external(address)
            return orig_create(address, *args, **kwargs)

        def sendto(self, data, *args):
            address = args[-1] if args else None
            if safe_mode_enabled() and address is not None:
                _reject_external(address)
            return orig_sendto(self, data, *args)

        def sendmsg(self, buffers, *args, **kwargs):
            address = kwargs.get("address")
            if address is None and len(args) >= 3:
                address = args[2]
            if safe_mode_enabled() and address is not None:
                _reject_external(address)
            assert orig_sendmsg is not None
            return orig_sendmsg(self, buffers, *args, **kwargs)

        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
        socket.create_connection = create_connection
        socket.socket.sendto = sendto
        if orig_sendmsg is not None:
            socket.socket.sendmsg = sendmsg
        _INSTALLED = True


def _reject_external(address) -> None:
    host = _host_from_address(address)
    if host is None or _is_local(host):
        return
    raise SafeModeNetworkError(
        "SAFE_MODE=1 blocks external network calls. "
        "Use the built-in fixture companies instead of a market-data service."
    )


def _host_from_address(address) -> str | None:
    if isinstance(address, (str, bytes)):
        # UNIX sockets are local files, not remote hosts.
        return None
    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, bytes):
            return host.decode("ascii", errors="replace")
        return str(host)
    return None


def _is_local(host: str) -> bool:
    name = host.strip().lower().rstrip(".")
    if name in {"localhost"} or name.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return False
    return ip.is_loopback
