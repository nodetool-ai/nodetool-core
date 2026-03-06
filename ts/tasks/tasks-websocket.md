# WebSocket / API Tasks — `packages/websocket`

Parity gaps between `src/nodetool/api/` (Python FastAPI) and `ts/packages/websocket/src/` (TypeScript manual routing).

---

## Phase 1 — Complete partial endpoints

### T-WS-1 · Workflow API — missing endpoints
**Status:** 🔴 open
**Python:** `api/workflow.py`

- [ ] **TEST** — `PUT /api/workflows/{id}/autosave` saves workflow without bumping version; returns 200.
- [ ] **IMPL** — Add autosave handler in `http-api.ts` (same as PUT but skips version increment).
- [ ] **TEST** — `POST /api/workflows/generate_name` returns AI-generated name from graph description.
- [ ] **IMPL** — Add `generate_name` endpoint. Can use a simple heuristic or call a provider.
- [ ] **TEST** — `GET /api/workflows/names` returns `{ id: name }` mapping for all user workflows.
- [ ] **IMPL** — Add `workflow_names` query endpoint.

---

### T-WS-2 · Job API — missing endpoints
**Status:** 🔴 open
**Python:** `api/job.py`

- [ ] **TEST** — `GET /api/jobs/running/all` returns all currently running jobs across all workflows.
- [ ] **IMPL** — Add `running/all` handler. Uses `Job.paginate` with status filter.
- [ ] **TEST** — `DELETE /api/jobs/{id}` hard-deletes a completed/failed job.
- [ ] **IMPL** — Add delete handler.

---

### T-WS-3 · Asset API — missing endpoints
**Status:** 🔴 open
**Python:** `api/asset.py`

- [ ] **TEST** — `GET /api/assets/search?query=foo` full-text search over asset names.
- [ ] **IMPL** — Add search handler using `Asset.search_assets_global()`.
- [ ] **TEST** — `GET /api/assets/packages` returns asset package index.
- [ ] **IMPL** — Add packages listing endpoint.
- [ ] **TEST** — `GET /api/assets/{id}/children` returns child assets for folder assets.
- [ ] **IMPL** — Add children endpoint using `Asset.get_children()`.

---

### T-WS-4 · Message API — delete endpoint
**Status:** 🔴 open
**Python:** `api/message.py`

- [ ] **TEST** — `DELETE /api/messages/{id}` returns 204; message no longer returned in list.
- [ ] **IMPL** — Add delete handler using `Message.delete()`.

---

### T-WS-5 · Thread API — update endpoint
**Status:** 🔴 open
**Python:** `api/thread.py`

- [ ] **TEST** — `PUT /api/threads/{id}` updates thread title; returns updated thread.
- [ ] **IMPL** — Add update handler.

---

### T-WS-6 · Settings API — profile endpoints
**Status:** 🔴 open
**Python:** `api/settings.py`

- [ ] **TEST** — `GET /api/settings` returns current user's settings profile (theme, defaults, etc.).
- [ ] **IMPL** — Add GET /api/settings handler.
- [ ] **TEST** — `PUT /api/settings` updates settings profile and persists.
- [ ] **IMPL** — Add PUT /api/settings handler. Needs a Settings model or JSON blob in DB.

---

### T-WS-7 · Node API — replicate status
**Status:** 🔴 open
**Python:** `api/node.py`

- [ ] **TEST** — `GET /api/nodes/replicate_status` returns whether Replicate API key is configured.
- [ ] **IMPL** — Add endpoint that checks for `REPLICATE_API_TOKEN` in env/secrets.

---

### T-WS-8 · User API — username validation
**Status:** 🔴 open
**Python:** `api/users.py`

- [ ] **TEST** — `GET /api/users/validate_username?username=foo` returns `{ valid: true, available: true }`.
- [ ] **IMPL** — Add username validation endpoint (regex check + DB uniqueness check).

---

## Phase 2 — Missing modules

### T-WS-9 · File browser API
**Status:** 🔴 open
**Python:** `api/file.py` — browse and download local filesystem files

- [ ] **TEST** — `GET /api/files/list?path=/some/dir` returns directory listing (name, size, is_dir, modified_at).
- [ ] **TEST** — Path traversal above allowed root is rejected with 403.
- [ ] **TEST** — `GET /api/files/info?path=/some/file` returns file metadata.
- [ ] **TEST** — `GET /api/files/download/{path}` streams file content with correct content-type.
- [ ] **IMPL** — Create `ts/packages/websocket/src/file-api.ts`. Route in `http-api.ts`. Enforce path sandbox.

---

### T-WS-10 · Collections API
**Status:** 🔴 open
**Python:** `api/collection.py` — asset collection CRUD

- [ ] **TEST** — `POST /api/collections` creates a named collection.
- [ ] **TEST** — `PUT /api/collections/{id}/assets` adds assets to collection.
- [ ] **TEST** — `GET /api/collections` lists user's collections.
- [ ] **TEST** — `DELETE /api/collections/{id}` removes collection (not assets).
- [ ] **IMPL** — Create `ts/packages/websocket/src/collection-api.ts`. Needs `Collection` model (deferred to tasks-models.md if needed).

---

### T-WS-11 · Storage key-value API
**Status:** 🔴 open
**Python:** `api/storage.py` — generic key-value storage via storage backends

- [ ] **TEST** — `GET /api/storage/{key}` retrieves stored value.
- [ ] **TEST** — `PUT /api/storage/{key}` stores value.
- [ ] **TEST** — `DELETE /api/storage/{key}` removes key.
- [ ] **IMPL** — Create `ts/packages/websocket/src/storage-api.ts`. Backed by SQLite or file storage.

---

### T-WS-12 · Admin secrets API
**Status:** 🔴 open
**Python:** `api/admin_secrets.py` — bulk import secrets from JSON/YAML

- [ ] **TEST** — `POST /api/admin/secrets/import` with JSON body imports multiple secrets atomically.
- [ ] **IMPL** — Create admin secrets import handler. Restricted to admin users.

---

### T-WS-13 · Debug export API
**Status:** 🔴 open
**Python:** `api/debug.py` — exports debug bundle (logs, env info, version)

- [ ] **TEST** — `POST /api/debug/export` returns ZIP with sanitized env info and recent logs.
- [ ] **IMPL** — Create debug export handler. Redact secrets from env before including.

---

### T-WS-14 · Memory / model lifecycle API
**Status:** 🔴 open
**Python:** `api/memory.py` — load/unload model instances, VRAM management

- [ ] **TEST** — `GET /api/memory/models` returns list of loaded models with memory usage.
- [ ] **TEST** — `DELETE /api/memory/{modelId}` unloads a model from memory.
- [ ] **IMPL** — Create `ts/packages/websocket/src/memory-api.ts`. Backed by a runtime model registry.

---

### T-WS-15 · MCP server
**Status:** 🔴 open
**Python:** `api/mcp_server.py` — Model Context Protocol server exposing workflows as MCP tools

- [ ] **TEST** — MCP server responds to `initialize` handshake with correct protocol version.
- [ ] **TEST** — MCP `tools/list` returns workflows marked as tools.
- [ ] **TEST** — MCP `tools/call` executes a workflow and returns result.
- [ ] **IMPL** — Create `ts/packages/websocket/src/mcp-server.ts` implementing MCP protocol. Use `@modelcontextprotocol/sdk` if available.

---

### T-WS-16 · Dynamic schema resolution
**Status:** ⚪ deferred
**Python:** `api/fal_schema.py`, `api/kie_schema.py`, `api/replicate_schema.py` — resolve dynamic schemas from third-party APIs.
Deferred until FAL/KIE/Replicate providers are ported to TS.

---

### T-WS-17 · Vibecoding API
**Status:** ⚪ deferred
**Python:** `api/vibecoding.py` — AI-generated HTML app from description.
Deferred; depends on `vibecoding.py` in agents.

---

## WebSocket runner gaps

### T-WS-18 · Job persistence in unified WebSocket runner
**Status:** 🔴 open
**Python:** `api/jobs.py` — persists job state to DB, marks completed/failed.
**TS:** `unified-websocket-runner.ts` runs jobs ephemerally; no DB writes.

- [ ] **TEST** — Write test: after `run_job` completes, a `Job` record exists in DB with status "completed".
- [ ] **TEST** — Write test: after `run_job` fails, `Job` record has status "failed" and error message.
- [ ] **IMPL** — Integrate `Job` model create/update into `UnifiedWebSocketRunner`. Use `Job.create()` at start, `job.mark_completed()` / `job.mark_failed()` at end.

---

### T-WS-19 · Model API — HuggingFace cache check
**Status:** 🔴 open
**Python:** `api/model.py` — checks local HuggingFace model cache.

- [ ] **TEST** — `GET /api/models/huggingface/cache?modelId=xxx` returns `{ cached: true/false, size }`.
- [ ] **IMPL** — Add HF cache check using local filesystem scan of `~/.cache/huggingface/`.
