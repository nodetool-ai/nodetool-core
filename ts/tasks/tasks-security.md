# Security / Auth Tasks — `packages/security`, `packages/auth`

Parity gaps between `src/nodetool/security/` (Python) and `ts/packages/security/` + `ts/packages/auth/`.

Core crypto (Fernet, PBKDF2, keychain) is fully ported. The gaps are the auth provider ecosystem and middleware.

---

## Phase 1 — Unblock real deployments

### T-SEC-1 · HTTP auth middleware
**Status:** 🟢 done
**Python source:** `security/http_auth.py` — extracts JWT / static token from `Authorization` header; injects `current_user` into FastAPI dependency chain.
**Gap:** TS API layer has no auth enforcement. Any caller gets full access.

- [ ] **TEST** — Write test: request without `Authorization` header to a protected endpoint returns 401.
- [ ] **TEST** — Write test: request with valid static token returns 200 and user object.
- [ ] **TEST** — Write test: request with expired/invalid JWT returns 401.
- [ ] **IMPL** — Create `ts/packages/auth/src/http-auth.ts`. Export `authenticateRequest(request, options): Promise<User | null>`. Called at the top of `handleApiRequest()` in `http-api.ts` for protected routes.

---

### T-SEC-2 · Local auth provider
**Status:** 🟢 done
**Python source:** `security/providers/local.py` — single-user mode, no credentials required; returns a hardcoded admin user.

- [ ] **TEST** — Write test: `LocalAuthProvider.authenticate(request)` always returns the default user with id "1".
- [ ] **IMPL** — Create `ts/packages/auth/src/providers/local-provider.ts`. Useful for development/single-user deployments.

---

### T-SEC-3 · Multi-user auth provider
**Status:** 🟢 done
**Python source:** `security/providers/multi_user.py` — validates JWT tokens, looks up user by ID from DB.

- [ ] **TEST** — Write test: valid JWT with user_id claim returns matching user from DB.
- [ ] **TEST** — Write test: JWT with unknown user_id returns null.
- [ ] **TEST** — Write test: malformed JWT returns null.
- [ ] **IMPL** — Create `ts/packages/auth/src/providers/multi-user-provider.ts`. Decode JWT (use `jose` npm package), look up user in DB.

---

### T-SEC-4 · Supabase auth provider
**Status:** 🔴 open
**Python source:** `security/providers/supabase.py` — validates Supabase JWTs using JWKS.

- [ ] **TEST** — Write test: Supabase JWT verified against JWKS endpoint; user metadata extracted.
- [ ] **IMPL** — Create `ts/packages/auth/src/providers/supabase-provider.ts`. Use `jose` JWKS fetching.

---

## Phase 2 — Admin and user management

### T-SEC-5 · Admin authentication
**Status:** 🟢 done
**Python source:** `security/admin_auth.py` — checks for admin role in user record; used to gate admin endpoints.

- [ ] **TEST** — Write test: non-admin user calling admin endpoint returns 403.
- [ ] **TEST** — Write test: admin user calling admin endpoint returns 200.
- [ ] **IMPL** — Add `isAdmin(user: User): boolean` helper to auth package. Gate admin routes in `http-api.ts`.

---

### T-SEC-6 · User management
**Status:** 🟢 done
**Python source:** `security/user_manager.py` — create, read, update, delete users; role management.

- [ ] **TEST** — Write test: `UserManager.create({ username, email, role })` persists user.
- [ ] **TEST** — Write test: `UserManager.findById(id)` returns user or null.
- [ ] **TEST** — Write test: `UserManager.setRole(userId, role)` updates role.
- [ ] **IMPL** — Create `ts/packages/auth/src/user-manager.ts`. Needs a `User` DB model (add to models package).

---

### T-SEC-7 · AWS Secrets Manager CLI
**Status:** ⚪ deferred
**Python source:** `security/aws_secrets_util.py` — CLI for generating/storing/retrieving master key in AWS.
Low priority; can use AWS CLI directly.

---

### T-SEC-8 · Startup security checks
**Status:** 🟢 done
**Python source:** `security/startup_checks.py` — validates master key available, DB accessible, required secrets configured.

- [ ] **TEST** — Write test: `runStartupChecks()` returns error list when OPENAI_API_KEY missing and it's required.
- [ ] **TEST** — Write test: `runStartupChecks()` returns empty list when all required secrets are present.
- [ ] **IMPL** — Create `ts/packages/security/src/startup-checks.ts`. Check: master key loadable, DB adapter connectable, required env vars present.

---

### T-SEC-9 · master_key — AWS Secrets Manager backend
**Status:** 🔴 open
**Python source:** `security/master_key.py` — loads master key from AWS Secrets Manager when `AWS_SECRETS_MASTER_KEY_NAME` env var is set.
**TS:** `master-key.ts` supports env var and system keychain, but not AWS.

- [ ] **TEST** — Write test: when `AWS_SECRETS_MASTER_KEY_NAME` is set, `MasterKeyManager.getKey()` calls AWS SDK to retrieve the secret value.
- [ ] **IMPL** — Add AWS Secrets Manager backend to `master-key.ts` using `@aws-sdk/client-secrets-manager`.
