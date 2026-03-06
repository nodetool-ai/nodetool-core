# Models Tasks — `packages/models`

Parity gaps between `src/nodetool/models/` + `src/nodetool/types/` (Python) and `ts/packages/models/src/` (TypeScript).

All schemas are ported. The gaps are **query and mutation methods** that the API layer depends on.

**Rule:** Write a failing test first, then implement.

---

## Phase 1 — Most-used models (unblock API layer)

### T-M-ASSET · Asset query methods
**Status:** 🔴 open
**Python source:** `models/asset.py`, `types/asset.py`

- [ ] **TEST** — `Asset.paginate(userId, { limit, startKey, contentType, parentId })` returns `[Asset[], nextKey]`
- [ ] **IMPL** — Port `paginate()` with filtering by user_id, optional content_type, optional parent_id
- [ ] **TEST** — `Asset.find(userId, assetId)` returns asset or null
- [ ] **IMPL** — Port `find()`
- [ ] **TEST** — `Asset.get_children(userId, parentId, limit)` returns child assets
- [ ] **IMPL** — Port `get_children()`
- [ ] **TEST** — `Asset.search_assets_global(query, limit)` full-text search across all users
- [ ] **IMPL** — Port `search_assets_global()` (LIKE query on name/content_type)
- [ ] **TEST** — `Asset.get_asset_path_info(assetId)` returns folder path array
- [ ] **IMPL** — Port `get_asset_path_info()` (recursive folder walk)
- [ ] **TEST** — `Asset.get_assets_recursive(userId, folderId)` returns all nested assets
- [ ] **IMPL** — Port `get_assets_recursive()`

---

### T-M-JOB · Job query and state methods
**Status:** 🔴 open
**Python source:** `models/job.py`, `types/job.py`

- [ ] **TEST** — `Job.paginate(userId, { limit, startKey, workflowId })` returns `[Job[], nextKey]`
- [ ] **IMPL** — Port `paginate()`
- [ ] **TEST** — `Job.find(userId, jobId)` returns job or null
- [ ] **IMPL** — Port `find()`
- [ ] **TEST** — `Job.claim(jobId, workerId)` atomically sets status to "running" and returns updated job
- [ ] **IMPL** — Port `claim()` using CAS (check-and-set on status)
- [ ] **TEST** — `Job.release(jobId)` resets to "pending"
- [ ] **IMPL** — Port `release()`
- [ ] **TEST** — `Job.update_heartbeat(jobId)` updates `updated_at` timestamp
- [ ] **IMPL** — Port `update_heartbeat()`
- [ ] **TEST** — `job.mark_completed(outputs)` sets status + stores output
- [ ] **TEST** — `job.mark_failed(error)` sets status + error message
- [ ] **TEST** — `job.mark_cancelled()` sets status
- [ ] **TEST** — `job.mark_suspended(nodeStates)` sets status + serializes states
- [ ] **IMPL** — Port all `mark_*()` instance methods
- [ ] **TEST** — `job.is_complete()`, `job.is_resumable()`, `job.is_paused()`, `job.is_suspended()`
- [ ] **IMPL** — Port all `is_*()` boolean helpers

---

### T-M-WORKFLOW · Workflow query methods
**Status:** 🔴 open
**Python source:** `models/workflow.py`, `types/workflow.py`

- [ ] **TEST** — `Workflow.paginate(userId, { limit, startKey })` returns `[Workflow[], nextKey]`
- [ ] **IMPL** — Port `paginate()`
- [ ] **TEST** — `Workflow.find(userId, workflowId)` returns workflow or null
- [ ] **IMPL** — Port `find()`
- [ ] **TEST** — `Workflow.paginate_tools(userId, limit)` returns workflows marked as tools
- [ ] **IMPL** — Port `paginate_tools()` (filter by `is_tool=true`)
- [ ] **TEST** — `Workflow.find_by_tool_name(userId, name)` returns matching tool workflow
- [ ] **IMPL** — Port `find_by_tool_name()`
- [ ] **TEST** — `workflow.has_trigger_nodes()` returns true if graph contains trigger nodes
- [ ] **IMPL** — Port `has_trigger_nodes()` (check node types in graph JSON)
- [ ] **TEST** — `workflow.get_api_graph()` returns simplified graph for API responses
- [ ] **IMPL** — Port `get_api_graph()` (strip internal metadata)
- [ ] **TEST** — `Workflow.from_dict(data)` creates workflow from raw dict with validation
- [ ] **IMPL** — Port `from_dict()` static factory

---

### T-M-MSG · Message query and creation methods
**Status:** 🔴 open
**Python source:** `models/message.py`

- [ ] **TEST** — `Message.paginate(threadId, { limit, startKey })` returns `[Message[], nextKey]`
- [ ] **IMPL** — Port `paginate()`
- [ ] **TEST** — `Message.create({ role, content, threadId, toolCalls })` normalizes content list → JSON string
- [ ] **IMPL** — Port `create()` with content serialization (MessageContent[] ↔ string)
- [ ] **TEST** — Deserialize stored content back to `MessageContent[]` on read
- [ ] **IMPL** — Port `_deserialize_str_list` and `_deserialize_obj_list` validator hooks
- [ ] **TEST** — `Message.delete(messageId)` hard-deletes message
- [ ] **IMPL** — Port `delete()`

---

### T-M-THREAD · Thread query methods
**Status:** 🔴 open
**Python source:** `models/thread.py`

- [ ] **TEST** — `Thread.find(userId, threadId)` returns thread or null
- [ ] **IMPL** — Port `find()`
- [ ] **TEST** — `Thread.create({ userId, title? })` returns new thread with generated id
- [ ] **IMPL** — Port `create()`
- [ ] **TEST** — `Thread.paginate(userId, { limit, startKey })` returns `[Thread[], nextKey]`
- [ ] **IMPL** — Port `paginate()`

---

## Phase 2 — Security-sensitive models

### T-M-SECRET · Secret encryption methods
**Status:** 🔴 open
**Python source:** `models/secret.py`

- [ ] **TEST** — `Secret.find(userId, key)` returns secret or null
- [ ] **IMPL** — Port `find()`
- [ ] **TEST** — `Secret.list_for_user(userId)` returns all secrets (keys only, no values)
- [ ] **IMPL** — Port `list_for_user()`
- [ ] **TEST** — `Secret.list_all()` returns all secrets across all users (admin only)
- [ ] **IMPL** — Port `list_all()`
- [ ] **TEST** — `secret.get_decrypted_value(masterKey)` decrypts stored value
- [ ] **IMPL** — Port `get_decrypted_value()` — integrate with `crypto.ts` Fernet decrypt
- [ ] **TEST** — `Secret.upsert_encrypted(userId, key, plainValue, masterKey)` stores encrypted
- [ ] **IMPL** — Port `upsert_encrypted()` — encrypt then save
- [ ] **TEST** — `secret.to_dict_safe()` returns dict without the encrypted value
- [ ] **IMPL** — Port `to_dict_safe()`
- [ ] **TEST** — `Secret.delete_secret(userId, key)` removes secret
- [ ] **IMPL** — Port `delete_secret()`

---

### T-M-OAUTH · OAuthCredential encryption methods
**Status:** 🔴 open
**Python source:** `models/oauth_credential.py`

- [ ] **TEST** — `OAuthCredential.create_encrypted({ userId, provider, accessToken, refreshToken, ... })` stores tokens encrypted
- [ ] **IMPL** — Port `create_encrypted()`
- [ ] **TEST** — `OAuthCredential.find_by_account(userId, provider, accountId)` returns credential
- [ ] **IMPL** — Port `find_by_account()`
- [ ] **TEST** — `OAuthCredential.list_for_user_and_provider(userId, provider)` returns list
- [ ] **IMPL** — Port `list_for_user_and_provider()`
- [ ] **TEST** — `credential.get_decrypted_access_token(masterKey)` decrypts token
- [ ] **IMPL** — Port `get_decrypted_access_token()`
- [ ] **TEST** — `credential.get_decrypted_refresh_token(masterKey)` decrypts refresh token
- [ ] **IMPL** — Port `get_decrypted_refresh_token()`
- [ ] **TEST** — `credential.update_tokens(accessToken, refreshToken, expiresAt, masterKey)` re-encrypts
- [ ] **IMPL** — Port `update_tokens()`
- [ ] **TEST** — `credential.to_dict_safe()` returns dict without encrypted values
- [ ] **IMPL** — Port `to_dict_safe()`

---

## Phase 3 — Operational models

### T-M-PREDICTION · Prediction query methods
**Status:** 🔴 open
**Python source:** `models/prediction.py`

- [ ] **TEST** — `Prediction.create({ userId, nodeId, provider, model, cost, ... })` stores prediction record
- [ ] **IMPL** — Port `create()`
- [ ] **TEST** — `Prediction.find(predictionId)` returns prediction or null
- [ ] **IMPL** — Port `find()`
*(aggregate methods already ported in `prediction.ts`)*

---

### T-M-RUNEVENT · RunEvent methods
**Status:** 🔴 open
**Python source:** `models/run_event.py`

- [ ] **TEST** — `RunEvent.create({ jobId, eventType, data })` stores event with auto seq
- [ ] **IMPL** — Port `create()`
- [ ] **TEST** — `RunEvent.get_next_seq(jobId)` returns next sequence number atomically
- [ ] **IMPL** — Port `get_next_seq()` (MAX(seq)+1 query)
- [ ] **TEST** — `RunEvent.append_event(jobId, eventType, data)` creates with next seq
- [ ] **IMPL** — Port `append_event()`
- [ ] **TEST** — `RunEvent.get_events(jobId, afterSeq?)` returns events in order
- [ ] **IMPL** — Port `get_events()`
- [ ] **TEST** — `RunEvent.get_last_event(jobId)` returns most recent event
- [ ] **IMPL** — Port `get_last_event()`
- [ ] **TEST** — `RunEvent.from_dict(data)` deserializes stored event with JSON data field
- [ ] **IMPL** — Port `from_dict()`

---

### T-M-RUNNODESTATE · RunNodeState methods
**Status:** 🔴 open
**Python source:** `models/run_node_state.py`

- [ ] **TEST** — `RunNodeState.get_node_state(jobId, nodeId)` returns state or null
- [ ] **IMPL** — Port `get_node_state()`
- [ ] **TEST** — `RunNodeState.get_or_create(jobId, nodeId)` returns existing or creates fresh
- [ ] **IMPL** — Port `get_or_create()`
- [ ] **TEST** — `state.mark_started()`, `state.mark_completed(result)`, `state.mark_failed(error)` update status
- [ ] **IMPL** — Port status transition methods

---

### T-M-RUNLEASE · RunLease methods
**Status:** 🔴 open
**Python source:** `models/run_lease.py`

- [ ] **TEST** — `RunLease.acquire(jobId, workerId, ttlSeconds)` creates lease or returns null if already held
- [ ] **IMPL** — Port `acquire()` with CAS semantics
- [ ] **TEST** — `lease.renew(ttlSeconds)` extends expiry
- [ ] **IMPL** — Port `renew()`
- [ ] **TEST** — `lease.release()` deletes lease
- [ ] **IMPL** — Port `release()`
- [ ] **TEST** — `lease.is_expired()` returns true if current time past expiry
- [ ] **IMPL** — Port `is_expired()`

---

## Not porting (intentional)

| Python File | Reason |
|-------------|--------|
| `postgres_adapter.py` | PostgreSQL-specific, not needed in TS initially |
| `supabase_adapter.py` | Supabase-specific |
| `migrations.py` | Auto-migration via `createTable()`; no manual migration needed |
| `run_inbox_message.py` | Durable inter-node inbox; deferred until needed |
| `trigger_input.py` | Trigger system; deferred |
| `workflow_version.py` | Workflow versioning; deferred |
