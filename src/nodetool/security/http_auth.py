from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from fastapi.responses import JSONResponse

from nodetool.security.auth_provider import AuthProvider, TokenType

if TYPE_CHECKING:
    from fastapi import Request


def _make_response(detail: str, status_code: int = 401) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_http_auth_middleware(
    static_provider: AuthProvider,
    user_provider: Optional[AuthProvider],
    exempt_paths: Iterable[str] = ("/health", "/ping"),
    enforce_auth: bool = True,
):
    exempt_paths = set(exempt_paths)

    async def middleware(request: Request, call_next):
        path = request.url.path
        if path in exempt_paths:
            return await call_next(request)

        if not enforce_auth:
            return await call_next(request)

        token = static_provider.extract_token_from_headers(request.headers)
        if not token:
            return _make_response(
                "Authorization header required. Use 'Authorization: Bearer <token>'."
            )

        static_result = await static_provider.verify_token(token)
        if static_result.ok:
            request.state.user_id = static_result.user_id
            request.state.token_type = static_result.token_type or TokenType.STATIC
            return await call_next(request)

        if user_provider:
            user_result = await user_provider.verify_token(token)
            if user_result.ok:
                request.state.user_id = user_result.user_id
                request.state.token_type = user_result.token_type or TokenType.USER
                return await call_next(request)
            detail = user_result.error or "Invalid user authentication token."
            return _make_response(detail)

        detail = static_result.error or "Invalid authentication token."
        return _make_response(detail)

    return middleware
