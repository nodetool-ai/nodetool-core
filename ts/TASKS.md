# TypeScript Parity Tasks

Master task index for closing all Python→TypeScript parity gaps.

**Rule: regression test first.** Every implementation task must be preceded by a test task that writes a failing test documenting the expected behavior. The test task is done when the test is committed. The implementation task is done when the test passes.

**Status legend:** 🔴 open · 🟡 in progress · 🟢 done · ⚪ N/A (infra gap, intentionally skipped)

---

## Sub-files by package

| Package | File | Open | Done |
|---------|------|------|------|
| Kernel (workflow engine) | [tasks/tasks-kernel.md](tasks/tasks-kernel.md) | 7 | 8 |
| Models (ORM methods) | [tasks/tasks-models.md](tasks/tasks-models.md) | 27 | 11 |
| Agents (tools + core) | [tasks/tasks-agents.md](tasks/tasks-agents.md) | 11 | 0 |
| Runtime (providers) | [tasks/tasks-runtime.md](tasks/tasks-runtime.md) | 16 | 2 |
| WebSocket / API | [tasks/tasks-websocket.md](tasks/tasks-websocket.md) | 19 | 0 |
| Security / Auth | [tasks/tasks-security.md](tasks/tasks-security.md) | 7 | 2 |
| Storage | [tasks/tasks-storage.md](tasks/tasks-storage.md) | 6 | 3 |
| Config | [tasks/tasks-config.md](tasks/tasks-config.md) | 4 | 0 |
| Metadata / Messaging | [tasks/tasks-metadata.md](tasks/tasks-metadata.md) | 7 | 0 |

**Total open: 104 · Total done: 26**

---

## Phase order (recommended)

### Phase 1 — Correctness (unblocks real workflows)
Kernel gaps that cause hangs or wrong results → Models query methods → Auth middleware

See [tasks-kernel.md](tasks/tasks-kernel.md) §Phase 1
See [tasks-models.md](tasks/tasks-models.md) §Phase 1
See [tasks-security.md](tasks/tasks-security.md) §T-SEC-1

### Phase 2 — Production basics
Storage layer · Config · Vision/Embeddings

See [tasks-storage.md](tasks/tasks-storage.md)
See [tasks-config.md](tasks/tasks-config.md)
See [tasks-runtime.md](tasks/tasks-runtime.md) §T-RT-1, §T-RT-2

### Phase 3 — Feature completeness
Remaining API endpoints · Agent tools · Provider coverage · Metadata reflection

See [tasks-websocket.md](tasks/tasks-websocket.md)
See [tasks-agents.md](tasks/tasks-agents.md)
See [tasks-metadata.md](tasks/tasks-metadata.md)
