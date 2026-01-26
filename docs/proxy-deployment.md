# Proxy Deployment Guide

This guide covers deploying the NodeTool async reverse proxy, with special focus on security considerations when running behind load balancers or reverse proxies.

## Overview

The NodeTool proxy is an async Docker reverse proxy that provides:
- On-demand container startup and lifecycle management
- HTTP/HTTPS proxying with streaming support
- Let's Encrypt ACME certificate management
- Bearer token authentication

## Trusted Proxies Configuration

When deploying behind a reverse proxy (e.g., nginx, HAProxy, AWS ALB, Cloudflare), you must configure `trusted_proxies` to securely extract the real client IP address from `X-Forwarded-For` headers.

### Why This Matters

The `X-Forwarded-For` header can be spoofed by malicious clients. Without proper configuration:
- Attackers can fake their IP address
- Rate limiting and IP-based access controls become ineffective
- Audit logs show incorrect client IPs

The `trusted_proxies` configuration ensures that `X-Forwarded-For` is only trusted when the request comes from a known proxy.

### Configuration Options

#### YAML Configuration

```yaml
global:
  domain: "example.com"
  email: "admin@example.com"
  bearer_token: "your-secure-token"
  
  # List of trusted proxy IP addresses or CIDR ranges
  trusted_proxies:
    - "10.0.0.0/8"        # Private network range
    - "172.16.0.0/12"     # Docker default bridge
    - "192.168.0.0/16"    # Private network range
    - "127.0.0.1"         # Localhost
    - "::1"               # IPv6 localhost

services:
  - name: app1
    path: /app1
    image: myapp:latest
```

#### Environment Variable

You can also set trusted proxies via environment variable:

```bash
# Comma-separated list of IPs or CIDR ranges
export PROXY_GLOBAL_TRUSTED_PROXIES="10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
```

### Common Deployment Scenarios

#### Behind nginx

```yaml
trusted_proxies:
  - "127.0.0.1"  # nginx on same host
```

nginx configuration:
```nginx
location / {
    proxy_pass http://localhost:8443;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header Host $host;
}
```

#### Behind AWS Application Load Balancer

```yaml
trusted_proxies:
  - "10.0.0.0/8"      # VPC CIDR
  - "172.31.0.0/16"   # Default VPC
```

Note: AWS ALB IPs change dynamically. Use VPC CIDR ranges.

#### Behind Cloudflare

```yaml
trusted_proxies:
  # Cloudflare IPv4 ranges (check docs.cloudflare.com for current list)
  - "173.245.48.0/20"
  - "103.21.244.0/22"
  - "103.22.200.0/22"
  - "103.31.4.0/22"
  - "141.101.64.0/18"
  - "108.162.192.0/18"
  - "190.93.240.0/20"
  - "188.114.96.0/20"
  - "197.234.240.0/22"
  - "198.41.128.0/17"
  - "162.158.0.0/15"
  - "104.16.0.0/13"
  - "104.24.0.0/14"
  - "172.64.0.0/13"
  - "131.0.72.0/22"
```

#### Docker Compose with Traefik

```yaml
trusted_proxies:
  - "172.16.0.0/12"   # Docker networks
  - "10.0.0.0/8"      # Internal services
```

#### Kubernetes with Ingress

```yaml
trusted_proxies:
  - "10.0.0.0/8"      # Pod network CIDR
  - "172.16.0.0/12"   # Service network
```

### Security Best Practices

1. **Minimize trusted ranges**: Only include IP ranges that actually contain your proxies.

2. **Use specific IPs when possible**: If your proxy has a static IP, use that instead of a CIDR range.

3. **Review regularly**: Proxy infrastructure changes. Review trusted_proxies configuration when deploying new load balancers.

4. **Don't trust public IPs**: Never add public IP ranges to trusted_proxies unless you control those IPs.

5. **Test your configuration**: Verify that real client IPs are being logged correctly.

### How IP Extraction Works

When a request arrives:

1. **Check connection IP**: Is the direct connection from a trusted proxy?
   - If NO: Use the connection IP (ignore X-Forwarded-For)
   - If YES: Continue to step 2

2. **Parse X-Forwarded-For**: Extract the list of IPs (format: `client, proxy1, proxy2`)

3. **Walk backwards**: Find the rightmost IP that is NOT a trusted proxy. This is the real client.

4. **Edge case**: If all IPs in the chain are trusted, use the leftmost (original client) IP.

Example:
```
Connection IP: 10.0.0.1 (trusted proxy)
X-Forwarded-For: 203.0.113.50, 10.0.0.5, 10.0.0.1
Trusted: 10.0.0.0/8

Result: 203.0.113.50 (first non-trusted IP from the right)
```

### Using the API

The `get_real_client_ip` function is available for use in custom middleware or logging:

```python
from nodetool.proxy.config import get_real_client_ip

# In a request handler
def handle_request(request):
    client_ip = get_real_client_ip(
        request_client_ip=request.client.host,
        x_forwarded_for=request.headers.get("x-forwarded-for"),
        trusted_proxies=config.global_.trusted_proxies,
    )
    log.info(f"Request from {client_ip}")
```

### Validation

The configuration validates IP addresses and CIDR ranges at startup:

```python
# Valid entries
trusted_proxies:
  - "10.0.0.1"         # Single IPv4
  - "192.168.0.0/24"   # IPv4 CIDR
  - "::1"              # IPv6 localhost
  - "2001:db8::/32"    # IPv6 CIDR

# Invalid entries (will raise ValueError)
trusted_proxies:
  - "not-an-ip"        # Invalid format
  - "192.168.1.0/99"   # Invalid prefix length
```

## Other Configuration Options

### TLS/HTTPS

```yaml
global:
  tls_certfile: "/etc/letsencrypt/live/example.com/fullchain.pem"
  tls_keyfile: "/etc/letsencrypt/live/example.com/privkey.pem"
  listen_https: 443
  listen_http: 80
  http_redirect_to_https: true
```

### Container Management

```yaml
global:
  idle_timeout: 300        # Stop containers after 5 minutes of inactivity
  docker_network: "mynet"  # Docker network name
  connect_mode: "docker_dns"  # or "host_port"
```

### Environment Variable Overrides

All global settings can be overridden via environment variables:

| Setting | Environment Variable |
|---------|---------------------|
| domain | `PROXY_GLOBAL_DOMAIN` |
| email | `PROXY_GLOBAL_EMAIL` |
| bearer_token | `PROXY_GLOBAL_BEARER_TOKEN` |
| idle_timeout | `PROXY_GLOBAL_IDLE_TIMEOUT` |
| docker_network | `PROXY_GLOBAL_DOCKER_NETWORK` |
| connect_mode | `PROXY_GLOBAL_CONNECT_MODE` |
| http_redirect_to_https | `PROXY_GLOBAL_HTTP_REDIRECT_TO_HTTPS` |
| trusted_proxies | `PROXY_GLOBAL_TRUSTED_PROXIES` |

## Troubleshooting

### Client IPs showing as proxy IPs

- Check that `trusted_proxies` includes your reverse proxy's IP
- Verify that your reverse proxy is setting `X-Forwarded-For` header
- Enable debug logging to see IP extraction decisions

### Configuration validation errors

- Ensure all IP addresses and CIDR ranges are valid
- Check for typos in IP addresses
- Verify CIDR prefix lengths (e.g., /24 not /99)

### Connection refused errors

- Verify the proxy is listening on the configured ports
- Check firewall rules
- Ensure Docker network is accessible
