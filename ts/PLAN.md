# Decorator-Based Node Metadata Plan

## Goal

Make TypeScript node classes in `nodetool-core/ts/packages/base-nodes` the source of truth for TS node property metadata, using decorators like:

```ts
class Add extends BaseNode {
  @prop({ type: "int", default: 0, description: "First number" })
  a!: number;

  @prop({ type: "int", default: 0, description: "Second number" })
  b!: number;
}
```

The end state is:

- TS node property metadata comes from TS class definitions.
- Python node library metadata continues to come from Python package metadata.
- Metadata extraction is explicit and deterministic.
- Existing runtime behavior keeps working during the migration.
- The HTTP metadata endpoint returns a unified TS+Python metadata catalog.

---

## Current State

### What exists now

- [`packages/node-sdk/src/base-node.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/src/base-node.ts) defines `BaseNode`.
- [`packages/node-sdk/src/node-metadata.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/src/node-metadata.ts) builds metadata by:
  - instantiating the node,
  - reading `defaults()`,
  - inferring property types from runtime default values,
  - reading outputs from `toDescriptor().outputs`.
- [`packages/node-sdk/src/registry.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/src/registry.ts) currently prefers Python-loaded metadata.
- [`packages/websocket/src/http-api.ts`](/Users/mg/workspace/nodetool-core/ts/packages/websocket/src/http-api.ts) serves `/api/nodes/metadata` directly from Python JSON.

### Problems with the current model

- Property metadata is implicit and lossy.
  - `defaults()` gives a default value, but not description, title, bounds, enum values, or required-ness.
- Runtime inference is weak.
- `[]` only yields `list`, not `list[str]`.
- `{}` only yields `dict`, not a specific domain type.
- `null` yields `any`.
- The metadata endpoint is still Python-first.
- TS classes are not authoritative for TS node libraries, which creates drift.
- There is no explicit permanent coexistence model for TS and Python node libraries.

### Constraint that matters

Plain TypeScript type annotations like `a!: number` do not exist at runtime. That means the runtime metadata source must be the decorator payload, not the TS type itself. The TS type annotation is still valuable for editor and compiler checks.

---

## Target Design

### Authoring model

Use decorators on instance fields for properties:

```ts
class Add extends BaseNode {
  static readonly nodeType = "nodetool.math.Add";
  static readonly title = "Add";
  static readonly description = "Adds two integers";

  @prop({ type: "int", default: 0, description: "First number" })
  a!: number;

  @prop({ type: "int", default: 0, description: "Second number" })
  b!: number;
}
```

Optional typed helpers can be added later, but the canonical API should be the object form:

```ts
@prop({ type: "list[str]", default: [], description: "Items to join" })
items!: string[];
```

### Metadata rules

- `@prop(...)` is the runtime source of truth for TS node properties.
- TS field types like `a!: number` are compile-time only.
- `BaseNode` should be able to derive `defaults()` from decorated properties.
- `node-metadata.ts` should read decorator metadata first.
- Legacy `defaults()` inference remains only as a migration fallback.
- TS metadata and Python metadata must coexist permanently.
- Source of truth is per library:
  - TS libraries: TS class metadata is authoritative.
  - Python libraries: Python package metadata is authoritative.

### Output metadata

Properties are the main ask, but outputs cannot remain Python-only if TS classes are the source of truth. The plan should include a TS output declaration path, for example one of:

1. `@returns({ result: "int" })` on `process()`.
2. `static readonly outputTypes = { result: "int" }`.

Recommendation:

- Implement property decorators first.
- Add a simple explicit TS output declaration in the same migration.
- Do not rely on return type reflection, because TS return annotations are also erased at runtime.

### Unified metadata catalog

The system should build one metadata catalog from two permanent sources:

1. TS-derived metadata for registered TS node classes.
2. Python-derived metadata for discovered Python node libraries.

Recommended precedence by `node_type`:

- if a TS node class is registered for a given `node_type`, TS metadata wins for that node type;
- otherwise, if only Python metadata exists, Python metadata is used;
- if both exist intentionally during migration, Python may backfill missing TS fields, but TS remains authoritative for the TS implementation.

---

## Proposed API Surface

## 1. Property decorator

Add to `@nodetool/node-sdk`:

```ts
export interface PropOptions {
  type: string;
  default?: unknown;
  title?: string;
  description?: string;
  min?: number;
  max?: number;
  required?: boolean;
  values?: Array<string | number>;
  json_schema_extra?: Record<string, unknown>;
}

export declare function prop(options: PropOptions): ClassFieldDecorator;
```

Behavior:

- Registers property metadata on the node class.
- Works on instance fields only.
- Supports inherited properties.
- Does not require `reflect-metadata`.

## 2. Internal metadata store

Add an internal metadata registry in `node-sdk`, likely a new module such as:

- `packages/node-sdk/src/decorators.ts`
- or `packages/node-sdk/src/class-metadata.ts`

It should store:

- declared properties by class,
- optionally declared outputs by class,
- enough source markers to know whether metadata was explicit or inferred.

## 3. BaseNode helpers

Extend `BaseNode` with internal helpers such as:

- `static getDeclaredProperties()`
- `static getDeclaredOutputs()`
- `protected getDefaultPropsFromDecorators()`

The public API does not need to expose all of these immediately, but `BaseNode` should become the central place that understands decorated fields.

## 4. Optional output declaration

Add one explicit TS-native output API:

```ts
static readonly outputTypes = {
  result: "int",
};
```

This is simpler than a method decorator and easier to roll out fast. A method decorator can be added later if we want parity with Python return annotations.

Recommendation:

- Use `@prop(...)` for properties.
- Use `static readonly outputTypes` for outputs in phase 1.

That gives a clean path without over-designing the first migration.

---

## Implementation Plan

## Phase 1. Build decorator infrastructure in `node-sdk`

### Files

- [`packages/node-sdk/src/base-node.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/src/base-node.ts)
- [`packages/node-sdk/src/node-metadata.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/src/node-metadata.ts)
- [`packages/node-sdk/src/index.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/src/index.ts)
- new file: `packages/node-sdk/src/decorators.ts`
- possibly new file: `packages/node-sdk/src/class-metadata.ts`

### Tasks

- Add `prop(options)` decorator.
- Add internal class metadata storage.
- Add support for inherited decorated properties.
- Add `static readonly outputTypes: Record<string, string>` to `BaseNode`.
- Add a `BaseNode.defaults()` implementation that:
  - uses decorated property defaults when present,
  - still allows subclasses to override for special cases.
- Update `BaseNode.toDescriptor()` so property types can come from decorated fields, not just `propertyTypes`.

### Important rule

Do not break existing nodes that still use `defaults()` and do not use decorators.

That means `BaseNode` needs hybrid behavior:

- decorator metadata first,
- legacy `defaults()` and `propertyTypes` fallback second.

---

## Phase 2. Rewrite node metadata extraction to be decorator-first

### Files

- [`packages/node-sdk/src/node-metadata.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/src/node-metadata.ts)

### Tasks

- Replace the current `defaults()`-only extraction logic.
- Build `PropertyMetadata[]` from declared decorators first.
- Use `outputTypes` for outputs.
- Normalize type names to match the existing NodeTool metadata schema.

### Type normalization

The TS metadata should use the same type naming conventions the UI already expects:

- `str`, not `string`
- `int`
- `float`
- `bool`
- `list[...]`
- `dict[...]` or `dict` as appropriate

This avoids regressions in the existing web UI and metadata consumers.

### Merge strategy during migration

Until every base node is migrated, `getNodeMetadata()` should support a transitional merge model:

- explicit TS decorator metadata overrides Python metadata,
- missing TS metadata can be backfilled from Python metadata,
- once a node is fully migrated, Python should no longer be needed for it.

This merge logic is especially important for:

- descriptions,
- enum values,
- min/max,
- outputs,
- any domain-specific types that were previously only present in Python JSON.

Important:

- this merge logic only applies to overlapping node types during migration;
- it must not collapse the permanent distinction between TS-only and Python-only node libraries.

---

## Phase 3. Make the registry source-aware

### Files

- [`packages/node-sdk/src/registry.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/src/registry.ts)

### Tasks

- Change `NodeRegistry.register()` to generate metadata from the TS class at registration time.
- Store generated metadata for registered TS nodes.
- Keep Python metadata loaded in parallel for Python node libraries.
- Treat Python metadata as:
  - the canonical source for Python-only node types,
  - compatibility input for overlapping TS node types during migration,
  - parity validation source where relevant,
  - but not the canonical source for registered TS node classes.

### Desired registration behavior

When `register(SomeNode)` is called:

1. Extract TS metadata from `SomeNode`.
2. If Python metadata exists for the same `node_type`, merge only as transitional backfill for fields not yet declared in TS.
3. Store the resulting metadata in the registry.

At the same time, the registry must continue to expose Python-only metadata entries for node libraries that do not have TS implementations.

### Strict mode update

Current strict mode means “missing Python metadata is an error.” That should change to something closer to:

- “missing resolved metadata is an error”

Resolved metadata means:

- explicit TS metadata,
- or a valid merged TS/Python metadata object during migration,
- or authoritative Python metadata for Python-only node types.

### Recommended registry model

Internally, the registry should track metadata provenance:

- `ts`
- `python`
- `merged`

That provenance does not need to be exposed to the UI immediately, but it will make overlap behavior deterministic and testable.

---

## Phase 4. Change the HTTP metadata endpoint

### Files

- [`packages/websocket/src/http-api.ts`](/Users/mg/workspace/nodetool-core/ts/packages/websocket/src/http-api.ts)
- [`packages/websocket/src/server.ts`](/Users/mg/workspace/nodetool-core/ts/packages/websocket/src/server.ts)
- likely [`packages/websocket/src/test-ui-server.ts`](/Users/mg/workspace/nodetool-core/ts/packages/websocket/src/test-ui-server.ts)

### Tasks

- Stop serving `/api/nodes/metadata` from raw Python JSON only.
- Pass registry-derived metadata into the HTTP layer.
- Return:
  - TS-derived metadata for registered TS nodes,
  - Python-derived metadata for Python node libraries,
  - one unified list for clients.

### Result

The UI should see the same node metadata shape as before, but the backend should now assemble that metadata from both ecosystems:

- TS decorators for TS nodes,
- Python package metadata for Python nodes.

The endpoint should remain source-agnostic from the client perspective.

---

## Phase 5. Migrate base nodes incrementally

### First migration tranche

Start with low-risk, high-volume files:

- [`packages/base-nodes/src/nodes/input.ts`](/Users/mg/workspace/nodetool-core/ts/packages/base-nodes/src/nodes/input.ts)
- [`packages/base-nodes/src/nodes/constant.ts`](/Users/mg/workspace/nodetool-core/ts/packages/base-nodes/src/nodes/constant.ts)
- [`packages/node-sdk/src/nodes/test-nodes.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/src/nodes/test-nodes.ts)

These are good first targets because they:

- have simple defaults,
- are heavily used,
- cover scalar, list, dict, and null defaults,
- are easy to test.

### Migration rule for each node

For each migrated node:

1. Add decorated property declarations.
2. Add `outputTypes` if needed.
3. Remove duplicated `defaults()` when the decorator defaults fully replace it.
4. Keep `process()` logic unchanged unless cleanup is trivial.

### Example conversion

Before:

```ts
defaults() {
  return { a: 0, b: 0 };
}
```

After:

```ts
@prop({ type: "int", default: 0, description: "First number" })
a!: number;

@prop({ type: "int", default: 0, description: "Second number" })
b!: number;

static readonly outputTypes = {
  result: "int",
};
```

### Second tranche

After the simple files are stable, migrate:

- `text.ts`
- `boolean.ts`
- `numbers.ts`
- `list.ts`
- `dictionary.ts`

### Last tranche

Migrate the more complex nodes with richer domain types and optional settings:

- AI/provider nodes
- media nodes
- workspace/file nodes
- dynamic nodes

---

## Compatibility Strategy

## During migration

We need both old and new declarations to work.

Rules:

- If a node has decorators, use decorator metadata.
- If a node has `outputTypes`, use them.
- If a node lacks decorators, fall back to existing `defaults()` inference.
- If Python metadata exists, use it only to fill gaps while a node is not fully migrated.
- Python-only node libraries remain fully supported through Python metadata.

## After migration

Once all TS base nodes are migrated:

- remove Python-first assumptions from `registry.ts`,
- stop depending on Python JSON for TS nodes,
- keep Python metadata loading as a permanent first-class path for Python node libraries.

This is not a temporary compatibility layer. The intended steady state is mixed-source metadata:

- TS metadata for TS node libraries,
- Python metadata for Python node libraries.

---

## Testing Plan

## 1. Unit tests for decorators

Add tests in `packages/node-sdk/tests` for:

- registering decorated fields,
- inheritance of decorated fields,
- default extraction,
- required properties,
- title/description/min/max/enum propagation,
- `outputTypes` extraction,
- `toDescriptor()` propertyTypes generation from decorators.

## 2. Node metadata tests

Extend [`packages/node-sdk/tests/node-metadata.test.ts`](/Users/mg/workspace/nodetool-core/ts/packages/node-sdk/tests/node-metadata.test.ts) to cover:

- decorator-first extraction,
- legacy fallback,
- merge behavior with Python metadata,
- `str` vs `string` normalization,
- list/dict/custom type preservation.

## 3. Registry tests

Extend registry tests to verify:

- `register()` produces metadata from TS class definitions,
- strict mode validates resolved metadata, not just Python metadata,
- TS metadata overrides Python metadata where expected.
- Python-only metadata remains available when no TS node class exists.
- mixed catalogs containing TS and Python nodes are returned correctly.

## 4. HTTP API tests

Add or extend websocket/http tests to verify:

- `/api/nodes/metadata` includes TS-derived metadata,
- `/api/nodes/metadata` also includes Python-derived metadata for Python-only nodes,
- migrated nodes return decorator descriptions,
- non-migrated nodes still appear.

## 5. Parity tests

Keep a parity test temporarily, but change its purpose:

- compare Python JSON and TS-derived metadata for overlap,
- flag mismatches while migration is ongoing,
- do not allow Python metadata to silently replace TS metadata.

---

## Risks and Mitigations

## Risk: decorator metadata and runtime defaults diverge

Mitigation:

- make `BaseNode.defaults()` derive from decorator metadata by default,
- remove duplicated `defaults()` methods when migrating nodes.

## Risk: UI type names regress

Mitigation:

- normalize emitted types to existing NodeTool naming (`str`, `int`, `float`, `bool`),
- add direct tests for metadata payload shape.

## Risk: partial migration causes missing descriptions or outputs

Mitigation:

- use merge/backfill during transition,
- migrate `input.ts` and `constant.ts` early,
- add explicit `outputTypes`.

## Risk: decorators are applied but registry/http path still serves Python-only metadata

Mitigation:

- treat HTTP endpoint migration as a first-class phase, not follow-up cleanup.

## Risk: coexistence rules become ambiguous when the same `node_type` exists in both TS and Python

Mitigation:

- make precedence explicit and test it,
- track provenance internally,
- allow overlap only intentionally during migration or when there is a clear TS implementation for that same node type.

## Risk: dynamic nodes need extra behavior

Mitigation:

- keep dynamic node metadata logic separate,
- do not block the decorator rollout on dynamic outputs,
- handle static properties first, dynamic outputs second.

---

## Recommended Execution Order

1. Add `prop(...)` decorator infrastructure in `node-sdk`.
2. Add `outputTypes` support in `BaseNode` and metadata extraction.
3. Make `node-metadata.ts` decorator-first with migration fallback.
4. Make `NodeRegistry` register TS-derived metadata.
5. Change `/api/nodes/metadata` to serve registry-backed metadata.
6. Migrate `node-sdk` test nodes.
7. Migrate `base-nodes/input.ts`.
8. Migrate `base-nodes/constant.ts`.
9. Migrate the remaining simple base-node files in batches.
10. Remove Python-first assumptions for TS nodes while preserving Python metadata as a permanent source for Python libraries.

---

## Definition of Done

This work is done when all of the following are true:

- `@prop(...)` is supported in `@nodetool/node-sdk`.
- `BaseNode` can derive defaults and property types from decorators.
- `getNodeMetadata()` is decorator-first.
- `NodeRegistry` stores TS-derived metadata for registered TS nodes and Python-derived metadata for Python node libraries.
- `/api/nodes/metadata` returns a unified TS+Python metadata catalog.
- At least the first migration tranche is converted.
- Tests cover both decorated and legacy nodes.
- Python metadata is no longer the canonical source for TS base nodes.
- Python metadata remains the canonical source for Python node libraries.

---

## Open Questions

1. Should outputs use `static readonly outputTypes` first, or should we add a `@returns(...)` decorator immediately?
2. Do we want the object form `@prop({ type: "int", ... })` only, or also convenience helpers like `@intProp(...)` later?
3. Should we preserve Python metadata loading long-term for non-TS packages, or move all metadata generation behind a unified registry path?
4. Do we want a codemod for simple `defaults()` migrations once the decorator API is stable?

My recommendation is:

- ship `@prop(...)`,
- use `static readonly outputTypes` in phase 1,
- keep Python metadata as a permanent first-class source for Python node libraries,
- use Python metadata as transitional backfill only for overlapping TS node types during migration,
- add a codemod after the first two files are migrated successfully.

---

## Implementation Checklist By File

## `packages/node-sdk/src/decorators.ts`

- Define `PropOptions`.
- Implement `prop(options)` as a stage-3 field decorator.
- Register property metadata against the class constructor.
- Preserve declaration order for stable UI field ordering.
- Support inheritance by storing only class-local declarations here and merging later.
- Export any internal metadata readers needed by `BaseNode` and `node-metadata.ts`.

## `packages/node-sdk/src/class-metadata.ts`

- Add internal metadata structures for:
  - declared properties,
  - declared outputs,
  - provenance flags where useful.
- Implement helpers to:
  - read class-local property declarations,
  - read inherited property declarations,
  - read declared output types,
  - distinguish explicit vs inferred metadata.
- Keep this internal; do not expose it as the main public API.

## `packages/node-sdk/src/base-node.ts`

- Add `static readonly outputTypes: Record<string, string> = {}`.
- Add helpers like:
  - `static getDeclaredProperties()`
  - `static getAllDeclaredProperties()`
  - `static getDeclaredOutputs()`
- Change `defaults()` base behavior so it can derive defaults from decorated properties.
- Preserve subclass override support for complex nodes.
- Update `assign()` so it continues to merge defaults plus incoming properties.
- Update `toDescriptor()` so `propertyTypes` can be emitted from decorator metadata.
- Optionally emit `outputs` from `outputTypes` in `toDescriptor()`.

## `packages/node-sdk/src/node-metadata.ts`

- Replace `defaults()`-only property extraction with decorator-first extraction.
- Build `PropertyMetadata[]` from declared decorators.
- Normalize emitted types to NodeTool names:
  - `str`
  - `int`
  - `float`
  - `bool`
  - `list[...]`
  - `dict`
- Read outputs from `outputTypes`.
- Add migration-time overlap merge logic:
  - TS explicit metadata wins,
  - Python backfills missing fields only for overlapping node types.
- Keep legacy fallback for undeclared nodes that still rely on `defaults()`.

## `packages/node-sdk/src/registry.ts`

- Keep separate stores for:
  - registered TS classes,
  - loaded Python metadata,
  - resolved metadata returned to callers.
- On `register(nodeClass)`:
  - derive TS metadata,
  - merge with overlapping Python metadata only if needed,
  - record provenance as `ts` or `merged`.
- Keep Python-only entries available even when no TS class is registered.
- Update `getMetadata()` and `listMetadata()` to return unified resolved metadata.
- Update strict mode so it validates resolved metadata, not just Python metadata.

## `packages/node-sdk/src/index.ts`

- Export `prop`.
- Export any new public metadata types needed for node authors.
- Do not expose internal-only metadata registry helpers unless necessary.

## `packages/node-sdk/tests/node-metadata.test.ts`

- Add tests for decorated scalar properties.
- Add tests for decorated list and dict properties.
- Add tests for titles, descriptions, min/max, enum values, and required flags.
- Add tests for inheritance of decorated properties.
- Add tests for `outputTypes`.
- Add tests for fallback behavior when decorators are absent.
- Add tests for TS-over-Python merge behavior on overlapping node types.

## `packages/node-sdk/tests/metadata.test.ts`

- Add tests that Python-only metadata still loads and resolves.
- Add tests that TS-registered nodes resolve to TS-derived metadata.
- Add tests for mixed catalogs containing both TS and Python nodes.
- Add tests for strict mode using resolved metadata.

## `packages/node-sdk/tests/base-node.test.ts`

- Add tests that base `defaults()` derives from decorators.
- Add tests that subclass `defaults()` overrides still work.
- Add tests that `toDescriptor()` emits `propertyTypes` and outputs from declarations.

## `packages/node-sdk/src/nodes/test-nodes.ts`

- Convert a small set of test nodes to `@prop(...)`.
- Add `outputTypes` to those nodes.
- Remove redundant `defaults()` implementations where the decorator defaults fully replace them.
- Keep at least one legacy non-decorated node for fallback coverage.

## `packages/base-nodes/src/nodes/input.ts`

- Migrate simple input nodes first.
- Replace trivial `defaults()` with decorated fields.
- Add `outputTypes` where outputs are implicit today.
- Keep node behavior unchanged.
- Confirm special cases like `StringInputNode` still support extra fields such as `max_length` and `line_mode`.

## `packages/base-nodes/src/nodes/constant.ts`

- Migrate constant nodes to decorated `value` fields.
- Use explicit TS metadata for list, dict, and domain object constants.
- Add `outputTypes` for `output`, `image_size`, `width`, `height`, and similar explicit outputs.
- Remove redundant `defaults()` methods where safe.

## `packages/base-nodes/src/nodes/text.ts`

- Migrate simple text manipulation nodes.
- Declare string/list properties explicitly.
- Add output metadata for `result` and other named outputs.

## `packages/base-nodes/src/nodes/boolean.ts`

- Migrate predicate/configuration fields to decorators.
- Add explicit output type declarations for boolean outputs.

## `packages/base-nodes/src/nodes/numbers.ts`

- Migrate numeric thresholds, compare values, and modes.
- Declare min/max where already implied by logic.

## `packages/base-nodes/src/nodes/list.ts`

- Migrate sequence/list properties.
- Use explicit list element type strings where known.

## `packages/base-nodes/src/nodes/dictionary.ts`

- Migrate dict/string/filter properties.
- Declare outputs explicitly where multiple named outputs exist.

## `packages/base-nodes/src/nodes/output.ts`

- Review output-node behavior carefully.
- Ensure explicit output declarations do not regress current runtime routing semantics.

## `packages/base-nodes/src/nodes/control.ts`

- Review controlled nodes separately.
- Confirm decorator-based property metadata does not interfere with control-edge behavior.

## `packages/base-nodes/src/nodes/*` complex providers and media nodes

- Migrate after simple nodes are stable.
- Prefer explicit decorator metadata over inferred defaults for:
  - provider/model fields,
  - media references,
  - nested config objects,
  - enum-like selections.
- Use overlap merge with Python metadata during rollout when TS declarations are incomplete.

## `packages/base-nodes/src/index.ts`

- No major metadata logic should live here.
- Keep registration centralized.
- Verify that all migrated nodes still register exactly once.

## `packages/websocket/src/server.ts`

- Ensure the registry loads Python metadata and registers TS nodes into one unified catalog.
- Pass the unified registry or resolved metadata provider into HTTP/server layers.

## `packages/websocket/src/http-api.ts`

- Stop serving `/api/nodes/metadata` from raw Python JSON only.
- Source metadata from the unified registry-backed catalog.
- Return:
  - TS-derived metadata for TS nodes,
  - Python-derived metadata for Python-only nodes.
- Keep response shape unchanged for clients.

## `packages/websocket/src/test-ui-server.ts`

- Mirror the same unified metadata behavior used by the real server.
- Ensure test UI mode does not drift from production metadata behavior.

## `packages/base-nodes/tests/metadata-parity.test.ts`

- Reframe this test from “Python must exist for TS nodes” to:
  - overlap comparisons where both sides exist,
  - mismatch detection during migration,
  - coexistence validation for mixed catalogs.
- Do not require Python metadata to remain canonical for TS nodes.

## `packages/websocket/tests/*`

- Add endpoint tests for unified metadata responses.
- Add tests that Python-only node metadata remains present.
- Add tests that migrated TS nodes expose decorator descriptions/defaults/types.

## Follow-up cleanup

- Remove redundant `defaults()` methods from migrated simple nodes.
- Remove redundant static `propertyTypes` where decorators fully replace them.
- Keep legacy fallback code until migration coverage is high enough.
- Once stable, consider adding a codemod for bulk conversion of trivial nodes.
