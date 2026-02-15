# Hardcoded Local Token Cleanup

**Problem**: Hardcoded `"local_token"` values in collection API endpoints prevented proper authentication token propagation from clients.

**Solution**: Extract token from authorization header instead, with fallback to `"local_token"` for development environments.

**Why**: The hardcoded tokens were a security concern and prevented proper authentication in production deployments.

**Files**:
- `src/nodetool/api/collection.py:195`
- `src/nodetool/deploy/collection_routes.py:31`
- `src/nodetool/chat/chat_cli.py:631`

**Date**: 2026-02-13

**PR**: https://github.com/nodetool-ai/nodetool-core/pull/553
