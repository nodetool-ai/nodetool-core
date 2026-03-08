# Models Tasks — `packages/models`

Parity gaps between `src/nodetool/models/` + `src/nodetool/types/` (Python) and `ts/packages/models/src/` (TypeScript).

All schemas are ported. All query and mutation methods are implemented.

---

## Phase 1 — Most-used models ✅ ALL COMPLETE

### T-M-ASSET · Asset query methods
**Status:** 🟢 done
All methods implemented: `paginate()`, `find()`, `getChildren()`, `searchAssetsGlobal()`, `getAssetPathInfo()`, `getAssetsRecursive()`

### T-M-JOB · Job query and state methods
**Status:** 🟢 done
All methods implemented: `paginate()`, `find()`, `claim()`, `release()`, `updateHeartbeat()`, `markRunning()`, `markCompleted()`, `markFailed()`, `markCancelled()`, `markSuspended()`, `isResumable()`, `isPaused()`

### T-M-WORKFLOW · Workflow query methods
**Status:** 🟢 done
Schema + inherited CRUD from DBModel

### T-M-MSG · Message query and creation methods
**Status:** 🟢 done
Schema + serialization helpers + inherited CRUD

### T-M-THREAD · Thread query methods
**Status:** 🟢 done
`find()`, `create()`, `paginate()` implemented

---

## Phase 2 — Security-sensitive models ✅ ALL COMPLETE

### T-M-SECRET · Secret encryption methods
**Status:** 🟢 done (20 tests passing)
All methods: `find()`, `listForUser()`, `listAll()`, `getDecryptedValue()`, `upsertEncrypted()`, `toSafeObject()`, `deleteSecret()`

### T-M-OAUTH · OAuthCredential encryption methods
**Status:** 🟢 done (168 tests passing)
All methods: `createEncrypted()`, `findByAccount()`, `listForUserAndProvider()`, token decryption, `updateTokens()`, `toDictSafe()`

---

## Phase 3 — Operational models ✅ ALL COMPLETE

### T-M-PREDICTION · Prediction query methods
**Status:** 🟢 done
Model + aggregate methods implemented

### T-M-RUNEVENT · RunEvent methods
**Status:** 🟢 done
Schema + audit-only design

### T-M-RUNNODESTATE · RunNodeState methods
**Status:** 🟢 done
Schema + status transitions

### T-M-RUNLEASE · RunLease methods
**Status:** 🟢 done
Model with lease semantics

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
