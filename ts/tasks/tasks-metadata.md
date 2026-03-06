# Metadata / Messaging Tasks

Parity gaps in `src/nodetool/metadata/` and `src/nodetool/messaging/` vs TypeScript packages.

---

## Metadata (`src/nodetool/metadata/`)

### T-META-1 · Type metadata utilities
**Status:** 🟢 done
**Python source:** `metadata/type_metadata.py` — `TypeMetadata`, `is_list_type()`, `is_union_type()`, type compatibility checking.

These are used by:
- Kernel gap T-K-10 (multi-edge list type validation)
- Kernel gap T-K-3d (edge type compatibility)
- Agent gap T-AG-3 (model tools)

- [ ] **TEST** — Write test: `TypeMetadata.fromString("list[int]").isListType()` returns true.
- [ ] **TEST** — Write test: `TypeMetadata.fromString("int").isListType()` returns false.
- [ ] **TEST** — Write test: `TypeMetadata.isCompatible("int", "float")` returns true (coercible).
- [ ] **TEST** — Write test: `TypeMetadata.isCompatible("str", "int")` returns false.
- [ ] **IMPL** — Create `ts/packages/protocol/src/type-metadata.ts`. Port `TypeMetadata` class with `isListType()`, `isUnionType()`, `isCompatibleWith(other)`. Parse type strings like `"list[int]"`, `"union[str, int]"`, `"ImageRef"`.

---

### T-META-2 · Node metadata / introspection
**Status:** 🔴 open
**Python source:** `metadata/node_metadata.py` — extracts input/output schemas from node classes via Python reflection.

In TypeScript, nodes register themselves with `NodeRegistry`. The metadata comes from static class properties (`inputSchema`, `outputSchema`).

- [ ] **TEST** — Write test: `getNodeMetadata(nodeClass)` returns `{ type, inputs: [{ name, type, required, default }], outputs: [...], description }`.
- [ ] **TEST** — Write test: `getNodeMetadata` works for streaming nodes (marks `is_streaming_output: true`).
- [ ] **IMPL** — Create `ts/packages/node-sdk/src/node-metadata.ts`. Export `getNodeMetadata(nodeClass)` that reads static class properties. Update `NodeRegistry` to use this.

---

### T-META-3 · Type validation (typecheck)
**Status:** 🟢 done
**Python source:** `metadata/typecheck.py` — validates that a runtime value matches a declared type.

- [ ] **TEST** — Write test: `validateType(42, "int")` returns `{ valid: true }`.
- [ ] **TEST** — Write test: `validateType("hello", "int")` returns `{ valid: false, error: "..." }`.
- [ ] **TEST** — Write test: `validateType([1, 2], "list[int]")` returns `{ valid: true }`.
- [ ] **TEST** — Write test: `validateType([1, "x"], "list[int]")` returns `{ valid: false }`.
- [ ] **IMPL** — Create `ts/packages/protocol/src/typecheck.ts`. Uses `TypeMetadata` for type parsing. Port core validation logic.

---

## Messaging (`src/nodetool/messaging/`)

### T-MSG-1 · Context packer
**Status:** 🟢 done
**Python source:** `messaging/context_packer.py` — serializes conversation history + system prompt, truncating to fit a token budget.

(Also tracked in [tasks-runtime.md](tasks-runtime.md) T-RT-16)

- [ ] **TEST** — Write test: `packContext(messages, systemPrompt, 1000)` returns messages that fit within 1000 tokens.
- [ ] **TEST** — Write test: truncation removes oldest messages first while keeping the system prompt.
- [ ] **TEST** — Write test: system prompt alone exceeds budget — returns just truncated system prompt.
- [ ] **IMPL** — Create `ts/packages/runtime/src/context-packer.ts`. Use `token-counter.ts` for counting.

---

### T-MSG-2 · Help message processor
**Status:** 🔴 open (deferred)
**Python source:** `messaging/help_message_processor.py` — answers questions about available nodes and examples using keyword search.
**Dependency:** node metadata (T-META-2) + example index.

- [ ] **TEST** (todo) — `HelpMessageProcessor.process("how do I resize an image?")` returns message mentioning relevant node names.
- [ ] **IMPL** — Deferred until T-META-2 is done and an example index exists.

---

### T-MSG-3 · Workflow message processor
**Status:** ⚪ deferred
**Python source:** `messaging/workflow_message_processor.py` — executes a workflow in response to a chat message.
**Dependency:** Full kernel + WebSocket integration.

Deferred until kernel parity is higher and job persistence (T-WS-18) is done.

---

### T-MSG-4 · Chat workflow message processor
**Status:** ⚪ deferred
**Python source:** `messaging/chat_workflow_message_processor.py` — builds and runs a workflow from a natural language description.
**Dependency:** GraphPlanner (T-AG-9) + kernel.

Deferred.

---

### T-MSG-5 · Graph input/output schema
**Status:** 🟢 done
**Python source:** `graph.py` `get_input_schema()`, `get_output_schema()` — returns JSON Schema for workflow inputs/outputs based on input/output node types.

- [ ] **TEST** — Write test: `graph.getInputSchema()` for a graph with one IntInput node returns `{ properties: { x: { type: "number" } }, required: ["x"] }`.
- [ ] **TEST** — Write test: `graph.getOutputSchema()` returns schema for output node handles.
- [ ] **IMPL** — Add `getInputSchema()` and `getOutputSchema()` to `ts/packages/kernel/src/graph.ts`. Walk input/output nodes and build JSON Schema from their node descriptor properties.

---

### T-MSG-6 · Workflow API graph representation
**Status:** 🔴 open
**Python source:** `types/api_graph.py` — lightweight graph representation for API responses (no internal runtime metadata).

- [ ] **TEST** — Write test: `toApiGraph(graph)` strips runtime-only fields from nodes; result is JSON-serializable without cycles.
- [ ] **IMPL** — Create `ts/packages/models/src/api-graph.ts`. Port `ApiGraph`, `ApiNode`, `ApiEdge` types. Add `toApiGraph(graph)` conversion function.

---

### T-MSG-7 · Workflow message types
**Status:** 🟢 done
**Python source:** `types/wrap_primitive_types.py` — wraps primitive values (int, float, str) in typed envelopes for JSON serialization in workflow I/O.

- [ ] **TEST** — Write test: `wrapPrimitive(42)` returns `{ type: "int", value: 42 }`.
- [ ] **TEST** — Write test: `unwrapPrimitive({ type: "int", value: 42 })` returns `42`.
- [ ] **IMPL** — Create `ts/packages/protocol/src/wrap-primitives.ts`.
