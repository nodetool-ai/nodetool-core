from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Optional

from supabase import AsyncClient, create_async_client  # type: ignore

from nodetool.security.auth_provider import AuthProvider, AuthResult, TokenType


class SupabaseAuthProvider(AuthProvider):
    """Auth provider that validates Supabase JWT tokens."""

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        cache_ttl: int = 60,
        cache_max: int = 2000,
    ):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.cache_ttl = max(cache_ttl, 0)
        self.cache_max = max(cache_max, 0)
        self._client: Optional[AsyncClient] = None
        self._client_lock = asyncio.Lock()
        self._token_cache: dict[str, tuple[str, float]] = {}
        self._token_cache_order: "OrderedDict[str, None]" = OrderedDict()

    async def _get_client(self) -> AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = await create_async_client(
                    self.supabase_url, self.supabase_key
                )
        assert self._client is not None
        return self._client

    def _make_cache_key(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _get_cached_user(self, token: str) -> Optional[str]:
        if self.cache_ttl <= 0:
            return None
        key = self._make_cache_key(token)
        cached = self._token_cache.get(key)
        if not cached:
            return None
        user_id, expires_at = cached
        if time.monotonic() >= expires_at:
            # expired entry
            self._token_cache.pop(key, None)
            self._token_cache_order.pop(key, None)
            return None
        # refresh order for LRU eviction
        self._token_cache_order.pop(key, None)
        self._token_cache_order[key] = None
        return user_id

    def _cache_user(self, token: str, user_id: str) -> None:
        if self.cache_ttl <= 0 or self.cache_max <= 0:
            return
        key = self._make_cache_key(token)
        expires_at = time.monotonic() + self.cache_ttl
        self._token_cache[key] = (user_id, expires_at)
        self._token_cache_order.pop(key, None)
        self._token_cache_order[key] = None
        if len(self._token_cache_order) > self.cache_max:
            oldest_key, _ = self._token_cache_order.popitem(last=False)
            self._token_cache.pop(oldest_key, None)

    async def verify_token(self, token: str) -> AuthResult:

        if not token:
            return AuthResult(ok=False, error="Missing Supabase token")

        user_id = self._get_cached_user(token)
        if user_id:
            return AuthResult(ok=True, user_id=user_id, token_type=TokenType.USER)

        client = await self._get_client()
        try:
            user_response = await client.auth.get_user(jwt=token)
        except Exception as exc:  # noqa: BLE001
            return AuthResult(ok=False, error=str(exc))

        supabase_user = getattr(user_response, "user", None)
        supabase_user_id = getattr(supabase_user, "id", None) if supabase_user else None
        if not supabase_user_id:
            return AuthResult(ok=False, error="Invalid Supabase token")

        user_id_str = str(supabase_user_id)
        self._cache_user(token, user_id_str)
        return AuthResult(ok=True, user_id=user_id_str, token_type=TokenType.USER)

    def clear_caches(self) -> None:
        self._token_cache.clear()
        self._token_cache_order.clear()
        self._client = None

