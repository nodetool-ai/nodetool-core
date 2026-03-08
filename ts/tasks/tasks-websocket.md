# WebSocket / API Tasks — `packages/websocket`

Parity gaps between `src/nodetool/api/` (Python FastAPI) and `ts/packages/websocket/src/` (TypeScript manual routing).

---

## Phase 1 — Complete partial endpoints ✅ ALL DONE

### T-WS-1 · Workflow API
**Status:** 🟢 done — CRUD, autosave, names, public endpoints in `http-api.ts`

### T-WS-2 · Job API
**Status:** 🟢 done — list, get, delete, cancel, `running/all` in `http-api.ts`

### T-WS-3 · Asset API
**Status:** 🟢 done — CRUD, search, children in `http-api.ts`

### T-WS-4 · Message API
**Status:** 🟢 done — list, get, create, delete in `http-api.ts`

### T-WS-5 · Thread API
**Status:** 🟢 done — list, get, create, update, delete in `http-api.ts`

### T-WS-6 · Settings API
**Status:** 🟢 done — GET/PUT via `settings-api.ts` + `registerSetting()` / `getRegisteredSettings()`

### T-WS-7 · Node API — replicate status
**Status:** 🟢 done — `/api/nodes/replicate_status` in `http-api.ts`

### T-WS-8 · User API — username validation
**Status:** 🟢 done — `/api/users/validate_username` in `users-api.ts`

---

## Phase 2 — Modules ✅ MOSTLY DONE

### T-WS-9 · File browser API
**Status:** 🟢 done — `file-api.ts` with list, info, download

### T-WS-10 · Collections API
**Status:** 🟢 done — `collection-api.ts` (chromadb: CRUD + index stub)

### T-WS-11 · Storage key-value API
**Status:** 🟢 done — `storage-api.ts`

### T-WS-12 · Admin secrets API
**Status:** 🟢 done — `POST /admin/secrets/import` (admin-gated)

### T-WS-13 · Debug export API
**Status:** 🟢 done
**Python:** `api/debug.py` — exports debug bundle (logs, env info, version)

- [x] **IMPL** — `packages/websocket/src/debug-api.ts`: `POST /api/debug/export` with diagnostics, system info, providers, timestamp. Recursive secret redaction via regex patterns.
- [x] **TEST** — `packages/websocket/tests/debug-export.test.ts` (14 tests): redactSecrets, buildDebugExport, endpoint integration.

### T-WS-14 · Memory / model lifecycle API
**Status:** ⚪ N/A — GPU/PyTorch specific, not applicable to Node.js

### T-WS-15 · MCP server
**Status:** 🟢 done — `mcp-server.ts` (12 tools: workflows, assets, nodes, jobs, collections; stdio + HTTP/SSE)

---

## Additional API modules ✅ DONE

| Module | Status |
|--------|--------|
| `/v1/` OpenAI compat | 🟢 `openai-api.ts` |
| `/api/oauth/` | 🟢 `oauth-api.ts` |
| `/api/models/` | 🟢 `models-api.ts` |
| `/api/skills`, `/api/fonts` | 🟢 `skills-api.ts` |
| `/api/costs/` | 🟢 `cost-api.ts` |
| `/api/workspaces/` | 🟢 `workspace-api.ts` |
| `/api/users/` | 🟢 `users-api.ts` + `FileUserManager` |

---

## WebSocket runner gaps

### T-WS-18 · Job persistence in unified WebSocket runner
**Status:** 🟢 done — `Job.create()`, `markCompleted()`, `markFailed()`, `markCancelled()` integrated in `unified-websocket-runner.ts`.

---

### T-WS-19 · Model API — HuggingFace cache check
**Status:** ⚪ deferred — HuggingFace-specific, low priority for TS runtime

---

## Deferred

| Task | Reason |
|------|--------|
| T-WS-16 Dynamic schema resolution | FAL/Replicate providers not ported |
| T-WS-17 Vibecoding API | Depends on `vibecoding.py` in agents |
