# Security Policy

This document outlines security practices, vulnerability reporting procedures, and a deployment security checklist for NodeTool Core.

## Reporting Security Vulnerabilities

If you discover a security vulnerability in NodeTool Core, please report it responsibly:

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers privately
3. Include a detailed description of the vulnerability
4. Provide steps to reproduce the issue if possible
5. Allow reasonable time for the issue to be addressed before public disclosure

We take security seriously and will respond to valid security reports promptly.

## Security Architecture Overview

### Authentication System

NodeTool supports multiple authentication providers, configured via the `AUTH_PROVIDER` environment variable:

| Provider | Use Case | Configuration |
|----------|----------|---------------|
| `none` | Development only | No authentication required |
| `local` | Local development | Simple local user management |
| `static` | API integrations | Static token-based auth |
| `supabase` | Production | Full Supabase JWT authentication |

### Secrets Management

NodeTool uses an encrypted secrets storage system with the following security features:

- **Per-user encryption**: Each user's secrets are encrypted with a unique derived key
- **Fernet symmetric encryption**: AES-128 in CBC mode with PKCS7 padding and HMAC-SHA256 for integrity
- **Key derivation**: PBKDF2-SHA256 with 100,000 iterations
- **Multiple storage backends**: System keychain, AWS Secrets Manager, or environment variables

For detailed secrets management documentation, see [`src/nodetool/security/README.md`](src/nodetool/security/README.md).

---

## Deployment Security Checklist

Use this checklist when deploying NodeTool to production environments.

### 1. Environment Configuration

- [ ] **Set `ENV=production`** - Ensures production-mode behaviors are enabled
- [ ] **Disable debug mode** - Set `DEBUG=false`
- [ ] **Review log level** - Set `LOG_LEVEL=INFO` or `WARNING` (avoid `DEBUG` in production)

### 2. Authentication & Authorization

- [ ] **Configure authentication provider**
  - Set `AUTH_PROVIDER=supabase` (or appropriate provider) for production
  - **Never use `AUTH_PROVIDER=none` in production**
- [ ] **Configure Supabase credentials** (if using Supabase auth)
  - Set `SUPABASE_URL` to your Supabase project URL
  - Set `SUPABASE_KEY` to your Supabase service key
- [ ] **Configure OAuth credentials** (if using GitHub OAuth)
  - Set `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`
  - Verify OAuth callback URLs are correctly configured

### 3. Secrets Management

- [ ] **Generate and secure master key**
  - Generate: `python -c "from nodetool.security.crypto import SecretCrypto; print(SecretCrypto.generate_master_key())"`
  - Store in AWS Secrets Manager for multi-instance deployments
  - Or set via `SECRETS_MASTER_KEY` environment variable
- [ ] **Configure AWS Secrets Manager** (recommended for production)
  ```bash
  python -m nodetool.security.aws_secrets_util generate \
    --secret-name nodetool-master-key \
    --region us-east-1
  export AWS_SECRETS_MASTER_KEY_NAME=nodetool-master-key
  ```
- [ ] **Backup master key securely** - Loss of master key means all encrypted secrets are unrecoverable
- [ ] **Never commit secrets to version control**
  - Use `.env.*.local` files (gitignored) for local secrets
  - Use environment variables or secrets managers in production

### 4. API Keys & External Services

- [ ] **Audit required API keys** - Only configure keys for services you use:
  - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` - LLM providers
  - `HF_TOKEN` - Hugging Face (for gated models)
  - `REPLICATE_API_TOKEN`, `FAL_API_KEY` - AI infrastructure
  - `ELEVENLABS_API_KEY` - Text-to-speech
- [ ] **Use separate API keys for production** - Don't share keys between environments
- [ ] **Set appropriate rate limits** on external service accounts

### 5. Database Security

- [ ] **Use PostgreSQL or Supabase in production** (not SQLite)
  ```
  POSTGRES_DB=nodetool
  POSTGRES_USER=nodetool_user
  POSTGRES_PASSWORD=<strong-password>
  POSTGRES_HOST=<database-host>
  POSTGRES_PORT=5432
  ```
- [ ] **Use strong database passwords** - Minimum 20 characters, random
- [ ] **Restrict database network access** - Only allow connections from application servers
- [ ] **Enable TLS for database connections** where supported
- [ ] **Configure regular database backups**

### 6. Storage Security

- [ ] **Configure S3 storage** for production assets:
  ```
  S3_ACCESS_KEY_ID=<access-key>
  S3_SECRET_ACCESS_KEY=<secret-key>
  S3_ENDPOINT_URL=https://s3.amazonaws.com
  S3_REGION=us-east-1
  ASSET_BUCKET=<bucket-name>
  ```
- [ ] **Use IAM roles** instead of access keys where possible (AWS)
- [ ] **Configure bucket policies** to restrict public access
- [ ] **Enable S3 bucket versioning** for asset recovery
- [ ] **Enable S3 server-side encryption**

### 7. Network & HTTPS

- [ ] **Use HTTPS only** - Configure TLS certificates
  - Place `cert.pem` and `key.pem` in the deployment directory
  - Or use a reverse proxy (nginx, Traefik) with Let's Encrypt
- [ ] **Configure CORS appropriately**
  - Restrict `Access-Control-Allow-Origin` to specific domains in production
  - Avoid wildcard (`*`) CORS in production for sensitive endpoints
- [ ] **Use a reverse proxy** (nginx recommended)
  - See `docker/nginx/conf.d` for example configuration
- [ ] **Enable HTTP Strict Transport Security (HSTS)**
- [ ] **Configure firewall rules** - Only expose necessary ports (80, 443)

### 8. Docker & Container Security

- [ ] **Use specific image tags** - Avoid `latest` tag in production
- [ ] **Review mounted volumes** - Minimize host filesystem access
- [ ] **Don't mount Docker socket** in production unless absolutely required
- [ ] **Run containers as non-root user** where possible
- [ ] **Set resource limits** on containers
- [ ] **Scan images for vulnerabilities** before deployment

### 9. Monitoring & Logging

- [ ] **Configure error tracking**
  ```
  SENTRY_DSN=<your-sentry-dsn>
  ```
- [ ] **Configure OpenTelemetry** for observability:
  ```
  OTEL_SERVICE_NAME=nodetool-api
  OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317
  ```
- [ ] **Set up log aggregation** - Forward logs to a central logging system
- [ ] **Monitor for security events** - Failed auth attempts, unusual API patterns
- [ ] **Set up alerting** for critical errors and security events

### 10. Infrastructure

- [ ] **Use private networks** for internal communication
- [ ] **Configure health checks** - See `docker-compose.yaml` for examples
- [ ] **Set up automatic restarts** for crashed services
- [ ] **Configure memcached** for caching (optional but recommended):
  ```
  MEMCACHE_HOST=memcached
  MEMCACHE_PORT=11211
  ```
- [ ] **Plan for backup and disaster recovery**

---

## Production Hardening Quick Reference

### Minimum Required Configuration

```bash
# Core settings
ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Authentication (choose appropriate provider)
AUTH_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-key

# Secrets encryption
SECRETS_MASTER_KEY=<generated-master-key>
# OR use AWS Secrets Manager:
# AWS_SECRETS_MASTER_KEY_NAME=nodetool-master-key

# Database (PostgreSQL recommended)
POSTGRES_DB=nodetool
POSTGRES_USER=nodetool_user
POSTGRES_PASSWORD=<strong-random-password>
POSTGRES_HOST=your-db-host
POSTGRES_PORT=5432

# Storage
S3_ACCESS_KEY_ID=<access-key>
S3_SECRET_ACCESS_KEY=<secret-key>
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_REGION=us-east-1
ASSET_BUCKET=your-asset-bucket
```

### Security Anti-Patterns to Avoid

| ❌ Don't | ✅ Do Instead |
|----------|---------------|
| Use `AUTH_PROVIDER=none` in production | Use `supabase` or `static` auth |
| Commit API keys to version control | Use environment variables or secrets managers |
| Use SQLite in production | Use PostgreSQL or Supabase |
| Use wildcard CORS (`*`) for sensitive endpoints | Restrict to specific origins |
| Expose Docker socket to containers | Use alternative Docker management |
| Use weak or default passwords | Generate strong random passwords |
| Disable HTTPS | Always use TLS in production |
| Log sensitive data (tokens, keys) | Configure appropriate log levels |

---

## Security Updates

Keep NodeTool and its dependencies updated:

```bash
# Check for dependency vulnerabilities
pip-audit

# Update dependencies
uv sync --upgrade

# Review CHANGELOG.md for security-related updates
```

## Additional Resources

- [Secrets Management Documentation](src/nodetool/security/README.md)
- [Environment Configuration](.env.example)
- [Docker Deployment](docker-compose.yaml)
- [Observability Guide](OBSERVABILITY.md)
