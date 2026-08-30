import ipaddress
import socket
from typing import Any

import aiohttp
from aiohttp.abc import AbstractResolver


def is_ip_private(ip_str: str) -> bool:
    """Check if an IP address is private, loopback, or otherwise restricted.

    Non-IP strings (i.e. hostnames) return False — they must be resolved first
    and the resulting addresses checked individually.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified


# ``aiohttp.DefaultResolver`` is an alias that resolves to AsyncResolver or
# ThreadedResolver depending on whether aiodns is installed, so type checkers
# see a union of classes rather than a single base. Bind it to a name typed as
# a plain class object to keep the subclass below checkable.
_DefaultResolver: type[AbstractResolver] = aiohttp.DefaultResolver


class SSRFProtectResolver(_DefaultResolver):  # type: ignore[misc,valid-type]
    """
    A custom aiohttp resolver that prevents Server-Side Request Forgery (SSRF)
    by blocking resolution to private, loopback, and restricted IP addresses.
    """

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> list[dict[str, Any]]:
        # Check if the host itself is a private IP string before resolution
        if is_ip_private(host):
            raise ValueError(f"Access to private/restricted IP blocked: {host}")

        # Resolve the hostname
        ips = await super().resolve(host, port, family)

        # Filter out any private/restricted IPs from the resolved list. This is
        # what stops obfuscated literals (e.g. "2130706433", which getaddrinfo
        # happily turns into 127.0.0.1) from slipping past the check above.
        valid_ips = [ip_info for ip_info in ips if ip_info.get("host") and not is_ip_private(ip_info["host"])]

        if not valid_ips:
            raise ValueError(f"Access to host '{host}' blocked because it resolved to a private/restricted IP.")

        return valid_ips
