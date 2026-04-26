import ipaddress
import socket
from typing import Any, Dict, List, Optional

import aiohttp

def is_ip_private(ip_str: str) -> bool:
    """Check if an IP address is private, loopback, or otherwise restricted."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return False

class SSRFProtectResolver(aiohttp.DefaultResolver):
    """
    A custom aiohttp resolver that prevents Server-Side Request Forgery (SSRF)
    by blocking resolution to private, loopback, and restricted IP addresses.
    """
    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> List[Dict[str, Any]]:
        # Check if the host itself is a private IP string before resolution
        if is_ip_private(host):
            raise ValueError(f"Access to private/restricted IP blocked: {host}")

        # Resolve the hostname
        ips = await super().resolve(host, port, family)

        # Filter out any private/restricted IPs from the resolved list
        valid_ips = []
        for ip_info in ips:
            ip = ip_info.get("host")
            if ip and not is_ip_private(ip):
                valid_ips.append(ip_info)
            else:
                pass # Optionally log that a private IP was filtered

        if not valid_ips:
            raise ValueError(f"Access to host '{host}' blocked because it resolved to a private/restricted IP.")

        return valid_ips
