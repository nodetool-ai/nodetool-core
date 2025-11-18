[← Back to Docs Index](index.md)

# Security Hardening

**Audience:** Operators and security reviewers.  
**What you will learn:** How to harden NodeTool across dev, staging, and production with focused checklists.

## Principles

- Enforce authentication (`AUTH_PROVIDER=static` or `supabase`) outside local dev.
- Terminate TLS at the proxy or API server; prefer TLS even on internal hops.
- Isolate execution environments (subprocess or Docker) for untrusted workloads.
- Keep storage private and rotate tokens regularly.

## Development Checklist

- `AUTH_PROVIDER=local` only on isolated machines.
- Terminal WebSocket disabled unless actively debugging (`NODETOOL_ENABLE_TERMINAL_WS` unset by default).
- Assets stored locally; ensure test buckets are non-public.
- Use `.env.development.local` for secrets; never commit keys.

## Staging Checklist

- `AUTH_PROVIDER=static` or `supabase`; distribute tokens via secrets manager.
- TLS terminating at proxy or load balancer; HSTS enabled.
- Proxy bearer token set and rotated; disable proxy directory listings.
- Docker or subprocess execution for workflows that touch external data.
- Terminal WebSocket disabled; proxy blocks `/terminal` path if unused.
- Storage buckets private; signed URLs or short-lived tokens for assets.

## Production Checklist

- `AUTH_PROVIDER=supabase` or `static` with long, random tokens; rotate quarterly.
- All non-public endpoints behind TLS; `/health` and `/ping` may remain public for liveness only.
- Terminal WebSocket disabled (`NODETOOL_ENABLE_TERMINAL_WS` unset) and blocked at proxy.
- Worker tokens stored in a secrets manager; never logged.
- Proxy bearer token rotated and distributed via infrastructure secrets.
- Asset buckets private with signed URLs; temp buckets automatically expired.
- Disable host networking for Docker jobs; run containers as non-root where possible.
- Enable structured logging and forward to your SIEM; alert on authentication failures.

## Features to Revisit

- WebSocket and SSE endpoints accept Bearer tokens; prefer header auth over query params.
- Storage exposure: audit mounted volumes and clean temp files between runs.
- Provider keys: set only the providers you use; avoid loading unused secrets into containers.
- TLS: prefer HTTP/2 at the proxy and enable OCSP stapling where supported.

## Related Guides

- [Authentication](authentication.md)
- [Proxy Reference](proxy.md)
- [Execution Strategies](execution-strategies.md#security)
- [Deployment Journeys](deployment-journeys.md)
