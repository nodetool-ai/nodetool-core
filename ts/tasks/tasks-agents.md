# Agents Tasks — `packages/agents`

Parity gaps in `ts/packages/agents/src/` vs `src/nodetool/agents/` and `src/nodetool/agents/tools/`.

Core agent system is ~85% complete. Tool coverage is ~70%.

---

## Core agent system gaps

### T-AG-1 · workspace_tools
**Status:** 🟢 done
**Python source:** `agents/tools/workspace_tools.py`
**Dependency:** Agent workspace directory (filesystem-based, available in Node.js)

The Python `workspace_tools` gives agents read/write access to a sandboxed workspace directory. The TS `filesystem-tools.ts` covers general filesystem ops; what's missing is the workspace-scoped version that resolves paths relative to `agent.workspace`.

- [ ] **TEST** — Write tests: `WorkspaceReadTool`, `WorkspaceWriteTool`, `WorkspaceListTool` resolve paths relative to agent workspace root; path traversal outside workspace is rejected.
- [ ] **IMPL** — Create `ts/packages/agents/src/tools/workspace-tools.ts`. Wrap filesystem-tools with workspace-root scoping. Validate `path.resolve(workspace, filePath).startsWith(workspace)`.

---

### T-AG-2 · help_tools
**Status:** ⚪ N/A — Python module deleted (obsolete). `SearchNodesTool` moved inline to `graph_planner.py`; `SearchExamplesTool` removed.

---

### T-AG-3 · model_tools
**Status:** 🟢 done
**Python source:** `agents/tools/model_tools.py`
**Dependency:** Provider model registry

- [ ] **TEST** — Write failing test: `ListModelsTool.execute({ provider: "openai" })` returns available models for that provider.
- [ ] **IMPL** — Port `ListModelsTool`, `GetModelInfoTool`. Use the runtime provider's `listModels()` method (needs to be added to `BaseProvider` interface — see [tasks-runtime.md](tasks-runtime.md) T-RT-9).

---

### T-AG-4 · asset_tools
**Status:** 🟢 done
**Python source:** `agents/tools/asset_tools.py`
**Dependency:** Storage layer (see [tasks-storage.md](tasks-storage.md))

- [x] **TEST** — 19 tests in `packages/agents/tests/tools/asset-tools.test.ts` covering save, read, round-trip, error handling, and edge cases.
- [x] **IMPL** — `SaveAssetTool` and `ReadAssetTool` in `packages/agents/src/tools/asset-tools.ts`. Uses `StorageAdapter` from `ProcessingContext`.

---

### T-AG-5 · control_tool
**Status:** 🟢 done
**Python source:** `agents/tools/control_tool.py`

- [x] **IMPL** — `ControlNodeTool` in `packages/agents/src/tools/control-tool.ts`. Builds JSON schema from node info, creates `RunEvent` from tool args. `StepExecutor` intercepts control tool calls and records events via `getControlEvents()`.
- [x] **TEST** — 21 tests in `packages/agents/tests/tools/control-tool.test.ts` covering sanitizeToolName, schema building, event creation, userMessage, process stub.

---

### T-AG-6 · SERP provider abstraction
**Status:** 🟢 done — `serp-providers/` with `SerpProvider` interface, `SerpApiProvider`, `DataForSeoProvider`. `search-tools.ts` and `dataseo-tools.ts` refactored to use providers.

---

### T-AG-7 · _remove_base64_images utility
**Status:** 🟢 done
**Python source:** `agents/tools/_remove_base64_images.py`

- [ ] **TEST** — Write test: utility removes base64 image strings from nested message content while preserving other content.
- [ ] **IMPL** — Add `removeBase64Images(content: MessageContent[]): MessageContent[]` to a utils file in agents package.

---

### T-AG-8 · wrap_generators_parallel utility
**Status:** 🟢 done
**Python source:** `agents/wrap_generators_parallel.py`

- [ ] **TEST** — Write test: `wrapGeneratorsParallel([gen1, gen2, gen3])` yields items from all generators concurrently, in arrival order.
- [ ] **IMPL** — Create `ts/packages/agents/src/utils/wrap-generators-parallel.ts`. Use `Promise.race` over an array of pending generator nexts.

---

### T-AG-9 · GraphPlanner (deferred)
**Status:** ⚪ deferred
**Python source:** `agents/graph_planner.py`
**Dependency:** Full workflow graph system + node type registry

Converts agent objectives into workflow node graphs. Deferred until the TS kernel is production-ready and node type metadata is available.

---

### T-AG-10 · AgentEvaluator (deferred)
**Status:** ⚪ deferred
**Python source:** `agents/agent_evaluator.py`
CLI/batch evaluation harness for comparing agent outputs. Python-specific tooling.

---

### T-AG-11 · DockerRunner (N/A)
**Status:** ⚪ N/A — infrastructure-specific to Python deployment.

---

## Top-level `nodetool/tools/` (all deferred)

These 9 tools all depend on the Node framework (BaseNode, ProcessingContext, model registry, job system). Deferred until TS has a matching node execution layer.

| Tool | Depends on |
|------|-----------|
| `agent_tools.py` | Node framework |
| `asset_tools.py` | Storage layer |
| `collection_tools.py` | Collection model |
| `hf_tools.py` | HuggingFace provider |
| `job_tools.py` | Job model methods |
| `model_tools.py` | Provider registry |
| `node_tools.py` | Node framework |
| `storage_tools.py` | Storage layer |
| `workflow_tools.py` | Workflow model methods |
