# TypeScript Parity Tasks

Master task index for closing all Python→TypeScript parity gaps.

**Status legend:** 🔴 open · 🟡 in progress · 🟢 done · ⚪ N/A (infra gap, intentionally skipped)

---

## Sub-files by package

| Package | File | Open | Done |
|---------|------|------|------|
| Kernel (workflow engine) | [tasks/tasks-kernel.md](tasks/tasks-kernel.md) | 0 | 14 |
| Models (ORM methods) | [tasks/tasks-models.md](tasks/tasks-models.md) | 0 | 11 |
| Agents (tools + core) | [tasks/tasks-agents.md](tasks/tasks-agents.md) | 2 | 9 |
| Runtime (providers) | [tasks/tasks-runtime.md](tasks/tasks-runtime.md) | 0 | 19 |
| WebSocket / API | [tasks/tasks-websocket.md](tasks/tasks-websocket.md) | 0 | 19 |
| Security / Auth | [tasks/tasks-security.md](tasks/tasks-security.md) | 0 | 11 |
| Storage | [tasks/tasks-storage.md](tasks/tasks-storage.md) | 0 | 8 |
| Config | [tasks/tasks-config.md](tasks/tasks-config.md) | 0 | 6 |
| Metadata / Messaging | [tasks/tasks-metadata.md](tasks/tasks-metadata.md) | 1 | 12 |

**Total open: 3 · Total done: 108**

---

## Priority order

### High — Correctness & production basics ✅ ALL DONE
- ~~Kernel: T-K-10 (list type validation), T-K-15 (edge counters)~~
- ~~Storage: T-ST-4 (S3), T-ST-5 (Supabase)~~
- ~~Security: T-SEC-4 (Supabase auth), T-SEC-9 (AWS Secrets Manager)~~
- ~~WebSocket: T-WS-18 (job persistence)~~

### Medium — Feature completeness ✅ ALL DONE
- ~~Runtime: T-RT-1 (embeddings), T-RT-2 (vision), T-RT-3 (image gen), T-RT-4 (TTS), T-RT-5 (thinking)~~
- ~~Agents: T-AG-6 (SERP abstraction)~~
- ~~Metadata: T-META-2 (node introspection), T-MSG-6 (API graph)~~
- Agents: T-AG-2 (help tools) — deferred, depends on example index

### Low — Nice to have (remaining 3 tasks)
- ~~Runtime: T-RT-8/9/10 (minor provider parity), T-RT-11 (HuggingFace)~~ ✅
- ~~Config: T-CFG-4 (diagnostics)~~ ✅
- ~~WebSocket: T-WS-13 (debug export)~~ ✅
- Agents: T-AG-2 (help tools) — blocked on example index
- ~~Agents: T-AG-4 (asset tools)~~ ✅
- Agents: T-AG-5 (control tool) — blocked on agent loop modification
- Metadata: T-MSG-2 (help processor) — blocked on example index
