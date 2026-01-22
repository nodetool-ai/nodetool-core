# CORS Configuration in Deployment Routes

**Insight**: Multiple deployment routes in `deploy/workflow_routes.py` and `deploy/admin_routes.py` use `Access-Control-Allow-Origin: "*"` in their SSE (Server-Sent Events) responses.

**Location**:
- `src/nodetool/deploy/workflow_routes.py:184`
- `src/nodetool/deploy/admin_routes.py:155, 191`

**Context**: These routes return `StreamingResponse` with SSE and include CORS headers directly in the response headers rather than using FastAPI's CORSMiddleware.

**Security Consideration**: Wildcard CORS (`*`) allows any origin to make cross-origin requests. This is appropriate for:
- Public APIs that need to be accessed from web applications
- Development/test environments

**This may be a concern for**:
- Production deployments where API should only be accessed from specific origins
- Admin routes that expose sensitive operations

**Recommendation**:
1. Consider using FastAPI's CORSMiddleware consistently instead of inline headers
2. Restrict origins in production using environment-based configuration
3. For admin routes, consider more restrictive CORS or authentication requirements

**Why Not Changed**: These routes may be intentionally open for development convenience. The primary FastAPI app uses CORSMiddleware with configurable origins. Changing these SSE routes could break existing integrations without understanding the full deployment context.
