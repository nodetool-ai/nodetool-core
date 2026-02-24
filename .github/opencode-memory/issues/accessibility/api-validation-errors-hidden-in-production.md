# API Validation Errors Hidden in Production

**Problem**: Validation error handler was only registered in non-production environments, leaving users without helpful feedback in production.

**Solution**: Always register validation error handler with sanitization for production:
- In development: Show full error details for debugging
- In production: Show sanitized errors with actionable information
- Add request context (method, path, client) to logs for better traceability

**Example**:
```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.error(
        "Request validation error: %s | Method: %s | Path: %s | Client: %s",
        exc.errors(),
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )
    # Provide user-friendly validation errors
    if Environment.is_production():
        sanitized_errors = [{"msg": e["msg"], "type": e["type"]} for e in errors]
        return JSONResponse({"detail": "Request validation failed", "errors": sanitized_errors}, status_code=422)
    else:
        return JSONResponse({"detail": "Request validation failed", "errors": errors}, status_code=422)
```

**Files**: `src/nodetool/api/server.py:685-690`

**Date**: 2026-02-24
