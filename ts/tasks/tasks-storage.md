# Storage Tasks — `packages/storage` (new package)

The entire storage abstraction layer is absent in TypeScript. Python's `src/nodetool/storage/` provides backends for asset storage (S3, local filesystem, Supabase, memory) and node result caching.

**This is a greenfield package.** Create `ts/packages/storage/` with `src/` and `tests/`.

---

## Phase 1 — Core abstraction + local backends

### T-ST-1 · AbstractStorage interface
**Status:** 🟢 done
**Python source:** `storage/abstract_storage.py`

Defines the contract all storage backends implement.

- [ ] **TEST** — Write interface conformance tests: any object implementing `AbstractStorage` must satisfy: `upload`, `download`, `delete`, `get_url`, `exists`.
- [ ] **IMPL** — Create `ts/packages/storage/src/abstract-storage.ts`. Export interface:
  ```typescript
  interface AbstractStorage {
    upload(key: string, data: Buffer | Readable, contentType?: string): Promise<void>;
    download(key: string): Promise<Buffer>;
    delete(key: string): Promise<void>;
    getUrl(key: string): string;
    exists(key: string): Promise<boolean>;
  }
  ```

---

### T-ST-2 · MemoryStorage backend
**Status:** 🟢 done
**Python source:** `storage/memory_storage.py`
Useful for tests; stores blobs in a `Map<string, Buffer>`.

- [ ] **TEST** — Write test: `MemoryStorage.upload("key", buf)` then `download("key")` returns same buffer.
- [ ] **TEST** — Write test: `MemoryStorage.delete("key")` then `exists("key")` returns false.
- [ ] **TEST** — Write test: `MemoryStorage.getUrl("key")` returns `memory://key`.
- [ ] **IMPL** — Create `ts/packages/storage/src/memory-storage.ts`.

---

### T-ST-3 · FileStorage backend
**Status:** 🟢 done
**Python source:** `storage/file_storage.py`
Stores blobs on local filesystem under a configured base directory.

- [ ] **TEST** — Write test: `FileStorage(basePath).upload("sub/key", buf)` writes file to `basePath/sub/key`.
- [ ] **TEST** — Write test: `download` reads back correct content.
- [ ] **TEST** — Write test: `delete` removes file; `exists` returns false after delete.
- [ ] **TEST** — Write test: `getUrl` returns `file://` or HTTP URL depending on config.
- [ ] **IMPL** — Create `ts/packages/storage/src/file-storage.ts`. Use `node:fs/promises`.

---

## Phase 2 — Cloud backends

### T-ST-4 · S3Storage backend
**Status:** 🟢 done — `s3-storage.ts` with dynamic `@aws-sdk/client-s3` import, PutObject/GetObject/DeleteObject/HeadObject commands.

---

### T-ST-5 · SupabaseStorage backend
**Status:** 🟢 done — `supabase-storage.ts` with dynamic `@supabase/supabase-js` import.

---

## Phase 3 — Node result caching

### T-ST-6 · AbstractNodeCache interface
**Status:** 🟢 done
**Python source:** `storage/abstract_node_cache.py`

- [ ] **TEST** — Write interface conformance tests: `get(key)`, `set(key, value, ttl?)`, `delete(key)`, `clear()`.
- [ ] **IMPL** — Create `ts/packages/storage/src/abstract-node-cache.ts`.

---

### T-ST-7 · MemoryNodeCache
**Status:** 🟢 done
**Python source:** `storage/memory_node_cache.py`

- [ ] **TEST** — Write test: `MemoryNodeCache.set("k", val, 1)` then after 1s `get("k")` returns null (TTL expired).
- [ ] **TEST** — Write test: `get` before expiry returns stored value.
- [ ] **IMPL** — Create `ts/packages/storage/src/memory-node-cache.ts`.

---

### T-ST-8 · MemcacheNodeCache
**Status:** ⚪ deferred
**Python source:** `storage/memcache_node_cache.py` — distributed caching via Memcache.
Deferred; only needed for multi-instance deployments.

---

### T-ST-9 · MemoryUriCache (asset URL caching)
**Status:** 🟢 done — `memory-uri-cache.ts` with TTL-based Map cache for signed URLs.

---

## Package setup

- [ ] Create `ts/packages/storage/package.json` with name `@nodetool/storage`
- [ ] Create `ts/packages/storage/tsconfig.json` extending `../../tsconfig.base.json`
- [ ] Add to `ts/package.json` workspaces
- [ ] Export all backends from `ts/packages/storage/src/index.ts`
