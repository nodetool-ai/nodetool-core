# Comprehensive Plan: Merge Worker into Server for Production Deployment

## Executive Summary

Merge the worker (`src/nodetool/deploy/worker.py`) functionality directly into the main server (`src/nodetool/api/server.py`) to create a unified production deployment. The proxy is not needed. Production requires admin token authentication for sensitive endpoints.

---

## Current State Analysis

### Main Server (`src/nodetool/api/server.py`)

**What it has:**
- 20+ API routers (assets, workflows, jobs, nodes, messages, threads, storage, etc.)
- WebSocket endpoints (`/ws`, `/ws/terminal`, `/ws/updates`, `/ws/download`)
- Database migrations on startup
- Job execution manager lifecycle
- Authentication middleware (static + user providers)
- CORS, resource scope middleware
- Sentry integration
- Health endpoint (`/health`)

**What it's missing for production:**
- Admin routes for model downloads, cache management
- Admin collection management routes
- Admin storage routes (upload/delete)
- OpenAI-compatible endpoints are only enabled in non-production mode
- No admin token requirement for sensitive operations

### Worker (`src/nodetool/deploy/worker.py`)

**What it has that server needs:**
- `create_admin_router()` - Model downloads (HuggingFace, Ollama), cache scan/size, DB operations, collection CRUD, asset management
- `create_collection_router()` - Legacy collection routes at `/collections/*`
- `create_admin_storage_router()` - File upload/delete at `/admin/storage/*`
- `create_public_storage_router()` - Public file access at `/storage/*`
- `create_workflow_router()` - Workflow execution routes
- OpenAI-compatible router always enabled
- Worker auth token from `WORKER_AUTH_TOKEN` or config file

---

## Detailed Implementation Plan

### Phase 1: Modify `create_app()` in `server.py`

#### 1.1 Add Worker Routers to Production

**File:** `src/nodetool/api/server.py`

**Current `_load_default_routers()` function (lines 133-185):**

```python name=src/nodetool/api/server.py
def _load_default_routers() -> list[APIRouter]:
    """
    Lazily import and assemble the default routers to avoid heavy imports at
    module import time.
    """
    from . import (
        admin_secrets,
        asset,
        collection,
        cost,
        debug,
        file,
        font,
        job,
        memory,
        message,
        model,
        node,
        oauth,
        settings,
        storage,
        thread,
        vibecoding,
        workflow,
        workspace,
    )

    routers: list[APIRouter] = [
        admin_secrets.router,
        asset.router,
        cost.router,
        message.router,
        thread.router,
        model.router,
        node.router,
        oauth.router,
        workflow.router,
        workspace.router,
        storage.router,
        storage.temp_router,
        font.router,
        debug.router,
        job.router,
        settings.router,
        memory.router,
        vibecoding.router,
    ]

    if not Environment.is_production():
        routers.append(file.router)
        routers.append(collection.router)

    return routers
```

**Changes needed:**

```python
def _load_default_routers() -> list[APIRouter]:
    """
    Lazily import and assemble the default routers to avoid heavy imports at
    module import time.
    """
    from . import (
        admin_secrets,
        asset,
        collection,
        cost,
        debug,
        file,
        font,
        job,
        memory,
        message,
        model,
        node,
        oauth,
        settings,
        storage,
        thread,
        vibecoding,
        workflow,
        workspace,
    )

    routers: list[APIRouter] = [
        admin_secrets.router,
        asset.router,
        cost.router,
        message.router,
        thread.router,
        model.router,
        node.router,
        oauth.router,
        workflow.router,
        workspace.router,
        storage.router,
        storage.temp_router,
        font.router,
        debug.router,
        job.router,
        settings.router,
        memory.router,
        vibecoding.router,
        collection.router,  # CHANGE: Enable in production too
    ]

    if not Environment.is_production():
        routers.append(file.router)

    return routers


def _load_deploy_routers() -> list[APIRouter]:
    """
    Load deployment/admin routers for production.
    These provide admin operations, storage management, and workflow execution.
    """
    from nodetool.deploy.admin_routes import create_admin_router
    from nodetool.deploy.collection_routes import create_collection_router
    from nodetool.deploy.storage_routes import (
        create_admin_storage_router,
        create_public_storage_router,
    )
    from nodetool.deploy.workflow_routes import create_workflow_router

    return [
        create_admin_router(),
        create_collection_router(),
        create_admin_storage_router(),
        create_public_storage_router(),
        create_workflow_router(),
    ]
```

#### 1.2 Modify `create_app()` to Include Deploy Routers

**Current code (around line 437-446):**
```python
    # Mount OpenAI-compatible endpoints with default provider set to "ollama"
    if not Environment.is_production():
        app.include_router(
            create_openai_compatible_router(
                provider=Provider.Ollama.value,
            )
        )

    for router in routers:
        app.include_router(router)
```

**Change to:**
```python
    # Mount OpenAI-compatible endpoints (always enabled for production deployments)
    default_provider = os.environ.get("CHAT_PROVIDER", Provider.Ollama.value)
    default_model = os.environ.get("DEFAULT_MODEL", "llama3.2:latest")
    tools_str = os.environ.get("NODETOOL_TOOLS", "")
    tools = [t.strip() for t in tools_str.split(",") if t.strip()] if tools_str else []
    
    app.include_router(
        create_openai_compatible_router(
            provider=default_provider,
            default_model=default_model,
            tools=tools,
        )
    )

    for router in routers:
        app.include_router(router)

    # Include deploy routers for admin operations
    for router in _load_deploy_routers():
        app.include_router(router)
```

#### 1.3 Add `/ping` Endpoint

**Add after the `/health` endpoint (around line 458):**

```python
    @app.get("/health")
    async def health_check() -> str:
        return "OK"

    @app.get("/ping")
    async def ping():
        """Health check with timestamp."""
        import datetime
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
        }
```

#### 1.4 Add Production Startup Validation

**Add to the lifespan function (around line 339), before migrations:**

```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Validate production requirements
        if Environment.is_production():
            if not os.environ.get("SECRETS_MASTER_KEY"):
                raise RuntimeError(
                    "SECRETS_MASTER_KEY environment variable must be set in production"
                )
            admin_token = os.environ.get("ADMIN_TOKEN")
            if not admin_token:
                log.warning(
                    "ADMIN_TOKEN not set - admin endpoints will use WORKER_AUTH_TOKEN only"
                )
        
        # Run database migrations before starting
        from nodetool.models.migrations import run_startup_migrations
        # ... rest of existing code
```

---

### Phase 2: Create Admin Token Authentication

#### 2.1 Create New File: `src/nodetool/security/admin_auth.py`

```python
"""
Admin token authentication for production admin endpoints.

In production, sensitive admin operations (/admin/*) require either:
1. ADMIN_TOKEN header (X-Admin-Token) - for admin-only access
2. WORKER_AUTH_TOKEN (Authorization: Bearer) - standard auth still required

This provides defense-in-depth: users need both valid user auth AND admin token
for sensitive operations.
"""

import os
from typing import Callable, Awaitable

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# Paths that require admin token in production
ADMIN_TOKEN_REQUIRED_PATHS = [
    "/admin/models/",
    "/admin/cache/",
    "/admin/db/",
    "/admin/collections/",
    "/admin/storage/",
    "/admin/assets/",
]

# Paths that are always public (no auth needed)
PUBLIC_PATHS = ["/health", "/ping"]


def get_admin_token() -> str | None:
    """Get admin token from environment."""
    return os.environ.get("ADMIN_TOKEN")


def requires_admin_token(path: str) -> bool:
    """Check if path requires admin token."""
    return any(path.startswith(p) for p in ADMIN_TOKEN_REQUIRED_PATHS)


def create_admin_auth_middleware(
    enforce_in_production: bool = True,
) -> Callable[[Request, Callable], Awaitable]:
    """
    Create middleware that enforces admin token for sensitive endpoints.
    
    This middleware runs AFTER the regular auth middleware, so request.state.user_id
    is already set. It adds an additional check for admin operations.
    """
    
    async def middleware(request: Request, call_next):
        path = request.url.path
        
        # Skip for non-admin paths
        if not requires_admin_token(path):
            return await call_next(request)
        
        # Skip enforcement in development
        if not Environment.is_production() and not enforce_in_production:
            return await call_next(request)
        
        admin_token = get_admin_token()
        
        # If no admin token configured, allow (with warning logged at startup)
        if not admin_token:
            return await call_next(request)
        
        # Check for admin token header
        provided_token = request.headers.get("X-Admin-Token")
        
        if not provided_token:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "detail": "Admin token required. Use X-Admin-Token header."
                },
            )
        
        if provided_token != admin_token:
            log.warning(f"Invalid admin token attempt for {path}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Invalid admin token."},
            )
        
        return await call_next(request)
    
    return middleware
```

#### 2.2 Add Admin Auth Middleware to `server.py`

**After the existing auth middleware (around line 432):**

```python
    app.middleware("http")(auth_middleware)

    # Add admin token middleware for production
    if Environment.is_production():
        from nodetool.security.admin_auth import create_admin_auth_middleware
        admin_middleware = create_admin_auth_middleware(enforce_in_production=True)
        app.middleware("http")(admin_middleware)
```

---


---

### Phase 4: Update Environment Configuration

#### 4.1 Document New Environment Variables

Add to `CLAUDE.md` in the Deployment & Infrastructure table:

| Variable | Description | Default |
|----------|-------------|---------|
| `ADMIN_TOKEN` | Token for admin endpoint authentication | - |
| `CHAT_PROVIDER` | Default AI provider (ollama, openai, etc.) | `ollama` |
| `DEFAULT_MODEL` | Default model for chat completions | `llama3.2:latest` |
| `NODETOOL_TOOLS` | Comma-separated list of enabled tools | - |

#### 4.2 Example Production `.env`

```env
# Production Environment
ENV=production
PORT=7777
LOG_LEVEL=INFO

# Authentication
AUTH_PROVIDER=static
ADMIN_TOKEN=your-secure-admin-token-here-min-32-chars
SECRETS_MASTER_KEY=your-32-byte-master-key-base64-encoded

# Worker auth (for API access)
WORKER_AUTH_TOKEN=your-api-access-token

# Database
DB_PATH=/data/nodetool.db
# Or for PostgreSQL:
# POSTGRES_HOST=localhost
# POSTGRES_DB=nodetool
# POSTGRES_USER=nodetool
# POSTGRES_PASSWORD=password

# Storage
ASSET_BUCKET=assets
# For S3:
# S3_ENDPOINT_URL=https://s3.amazonaws.com
# S3_ACCESS_KEY_ID=...
# S3_SECRET_ACCESS_KEY=...

# AI Providers
CHAT_PROVIDER=ollama
DEFAULT_MODEL=llama3.2:latest
OLLAMA_API_URL=http://localhost:11434
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Optional observability
# SENTRY_DSN=https://...
# OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317
```

---

### Phase 5: Full Code Changes Summary

#### 5.1 Files to Modify

| File | Changes |
|------|---------|
| `src/nodetool/api/server.py` | Add deploy routers, OpenAI router always on, `/ping` endpoint, production validation, admin middleware |
| `Dockerfile` | Change CMD to run server, update ENV defaults, add healthcheck |

#### 5.2 Files to Create

| File | Purpose |
|------|---------|
| `src/nodetool/security/admin_auth.py` | Admin token middleware |
| `src/nodetool/api/run_server.py` | Production entry point |

#### 5.3 Files to Keep (No Changes Needed)

- `src/nodetool/deploy/admin_routes.py` - Already has all admin functionality
- `src/nodetool/deploy/collection_routes.py` - Already has collection routes
- `src/nodetool/deploy/storage_routes.py` - Already has storage routes
- `src/nodetool/deploy/workflow_routes.py` - Already has workflow routes
- `src/nodetool/security/http_auth.py` - Already has auth middleware

#### 5.4 Files to Deprecate (Not Delete Yet)

- `src/nodetool/deploy/worker.py` - Keep for backward compatibility, but server is now preferred
- `Dockerfile.proxy` - Not needed

---

### Phase 6: Complete `server.py` Modifications

Here's the complete diff-style changes for `src/nodetool/api/server.py`:

```python
# === ADD after existing imports (around line 19) ===
import datetime

# === MODIFY _load_default_routers() (lines 133-185) ===
def _load_default_routers() -> list[APIRouter]:
    """
    Lazily import and assemble the default routers to avoid heavy imports at
    module import time.
    """
    from . import (
        admin_secrets,
        asset,
        collection,
        cost,
        debug,
        file,
        font,
        job,
        memory,
        message,
        model,
        node,
        oauth,
        settings,
        storage,
        thread,
        vibecoding,
        workflow,
        workspace,
    )

    routers: list[APIRouter] = [
        admin_secrets.router,
        asset.router,
        cost.router,
        message.router,
        thread.router,
        model.router,
        node.router,
        oauth.router,
        workflow.router,
        workspace.router,
        storage.router,
        storage.temp_router,
        font.router,
        debug.router,
        job.router,
        settings.router,
        memory.router,
        vibecoding.router,
        collection.router,  # CHANGED: Now always included
    ]

    if not Environment.is_production():
        routers.append(file.router)

    return routers


# === ADD new function after _load_default_routers() ===
def _load_deploy_routers() -> list[APIRouter]:
    """
    Load deployment/admin routers.
    
    These provide:
    - Admin operations (model downloads, cache management)
    - Collection management via /admin/collections/*
    - Storage management via /admin/storage/* and /storage/*
    - Workflow execution via /workflows/*
    """
    from nodetool.deploy.admin_routes import create_admin_router
    from nodetool.deploy.collection_routes import create_collection_router
    from nodetool.deploy.storage_routes import (
        create_admin_storage_router,
        create_public_storage_router,
    )
    from nodetool.deploy.workflow_routes import create_workflow_router

    return [
        create_admin_router(),
        create_collection_router(),
        create_admin_storage_router(),
        create_public_storage_router(),
        create_workflow_router(),
    ]


# === MODIFY create_app() lifespan function (add at start of lifespan, around line 339) ===
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Validate production requirements
        if Environment.is_production():
            if not os.environ.get("SECRETS_MASTER_KEY"):
                raise RuntimeError(
                    "SECRETS_MASTER_KEY environment variable must be set in production. "
                    "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
                )
            if not os.environ.get("ADMIN_TOKEN"):
                log.warning(
                    "ADMIN_TOKEN not set - admin endpoints (/admin/*) will not require "
                    "additional admin authentication beyond standard auth"
                )
        
        # Run database migrations before starting
        from nodetool.models.migrations import run_startup_migrations
        # ... rest of existing lifespan code ...


# === MODIFY router inclusion section (around lines 437-446) ===
    # Mount OpenAI-compatible endpoints
    # In production, use environment variables for configuration
    default_provider = os.environ.get("CHAT_PROVIDER", Provider.Ollama.value)
    default_model = os.environ.get("DEFAULT_MODEL", "llama3.2:latest")
    tools_str = os.environ.get("NODETOOL_TOOLS", "")
    tools_list = [t.strip() for t in tools_str.split(",") if t.strip()] if tools_str else []
    
    app.include_router(
        create_openai_compatible_router(
            provider=default_provider,
            default_model=default_model,
            tools=tools_list,
        )
    )

    for router in routers:
        app.include_router(router)

    # Include deploy routers for admin and production operations
    for router in _load_deploy_routers():
        app.include_router(router)

    for extension_router in ExtensionRouterRegistry().get_routers():
        app.include_router(extension_router)


# === ADD admin auth middleware after existing auth middleware (around line 432) ===
    app.middleware("http")(auth_middleware)

    # Add admin token middleware for production admin endpoints
    if Environment.is_production():
        from nodetool.security.admin_auth import create_admin_auth_middleware
        admin_auth = create_admin_auth_middleware()
        app.middleware("http")(admin_auth)

    if not RUNNING_PYTEST:
        app.add_middleware(ResourceScopeMiddleware)


# === MODIFY health endpoint and add ping (around line 458) ===
    @app.get("/health")
    async def health_check() -> str:
        return "OK"

    @app.get("/ping")
    async def ping():
        """Health check with system information."""
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    @app.get("/editor/{workflow_id}")
    async def editor_redirect(workflow_id: str):
        return RedirectResponse(url="/")
```

---

### Phase 7: Testing Checklist

#### 7.1 Unit Tests

- [ ] Test `_load_deploy_routers()` returns correct routers
- [ ] Test admin auth middleware allows requests without token in dev
- [ ] Test admin auth middleware requires token in production
- [ ] Test admin auth middleware rejects invalid tokens

#### 7.2 Integration Tests

- [ ] All existing API tests pass
- [ ] Admin endpoints accessible with proper auth
- [ ] OpenAI-compatible endpoints work
- [ ] WebSocket endpoints work
- [ ] Storage upload/download works

#### 7.3 Docker Tests

```bash
# Build
docker build -t nodetool-server:test .

# Run in development mode
docker run -p 7777:7777 \
  -e ENV=development \
  nodetool-server:test

# Run in production mode
docker run -p 7777:7777 \
  -e ENV=production \
  -e ADMIN_TOKEN=test-admin-token-32-chars-min \
  -e SECRETS_MASTER_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))") \
  -e WORKER_AUTH_TOKEN=test-worker-token \
  nodetool-server:test

# Test endpoints
curl http://localhost:7777/health
curl http://localhost:7777/ping
curl -H "Authorization: Bearer test-worker-token" http://localhost:7777/api/workflows
curl -H "Authorization: Bearer test-worker-token" \
     -H "X-Admin-Token: test-admin-token-32-chars-min" \
     http://localhost:7777/admin/cache/scan
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Entry Points** | `worker.py` for deploy, `server.py` for dev | Single `server.py` for all |
| **Admin Routes** | Worker only | Server (always) |
| **OpenAI API** | Dev only in server | Always enabled |
| **Admin Auth** | Worker token only | Worker token + Admin token |
| **Dockerfile CMD** | `nodetool.deploy.worker` | `nodetool.api.run_server` |
| **Production Validation** | Worker checks `SECRETS_MASTER_KEY` | Server checks same |

This approach:
1. ✅ Merges worker into server (no separate worker needed)
2. ✅ Ignores proxy (not needed)
3. ✅ Adds admin token for production
4. ✅ Keeps existing admin endpoints from `deploy/admin_routes.py`
5. ✅ Enables OpenAI-compatible endpoints in production
6. ✅ Maintains backward compatibility with existing API
