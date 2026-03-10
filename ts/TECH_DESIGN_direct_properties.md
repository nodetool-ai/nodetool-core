# Technical Design: Direct Property Access for Node Fields

**Author:** AI Assistant
**Date:** 2026-03-09
**Status:** Proposal / Final

## Summary

Remove `_props` entirely from BaseNode. All node properties use direct instance access (`this.value`). This is a hard cutover — no migration period, no deprecated shims. One atomic change across the SDK and all 61 node files.

## Current State

### Problem

The current design uses a separation between metadata (decorators) and runtime storage (`_props`):

```typescript
class MyNode extends BaseNode {
  @prop({ type: "str", default: "hello" })
  value!: string;  // Only type annotation, no runtime value

  async process() {
    return { output: this._props.value };  // Awkward indirection
  }
}
```

**Issues:**
- Properties declared with `@prop()` aren't actually set on the instance
- Must access values through `this._props.propertyName`
- Two sources of truth: decorator metadata vs `_props` storage
- Unintuitive for developers familiar with standard TypeScript classes

### How It Works Now

1. Decorators register property metadata in a WeakMap (`declaredPropsByClass`)
2. `defaults()` extracts defaults from metadata and returns them
3. `assign()` merges defaults with incoming data into `_props`
4. All property access goes through `_props`

### Scale of Change

- **1456** `this._props.` references across **61** node files
- **678** `defaults()` overrides across **64** node files
- Hydration path: `registry.resolve()` → `new NodeClass()` → `assign(descriptorProps)` → `toExecutor()`
- `_props` is never serialized to disk — workflows use `NodeDescriptor.properties`

## Proposed Design

### Direct Property Access

Properties are initialized on the instance and accessed directly:

```typescript
class MyNode extends BaseNode {
  @prop({ type: "str", default: "hello" })
  value!: string;

  @prop({ type: "int", default: 0, min: 0, max: 100 })
  count!: number;

  async process() {
    return { output: `${this.value} x${this.count}` };
  }
}

const node = new MyNode();
console.log(node.value);  // "hello" (from decorator default)
node.value = "world";     // Direct assignment

// Or with properties at construction time
const node2 = new MyNode({ value: "world", count: 5 });
```

### Constructor and `assign()`

The constructor accepts an optional properties object and delegates to `assign()`:

```typescript
constructor(properties: Record<string, unknown> = {}) {
  this.assign(properties);
}
```

`assign()` merges decorator defaults with provided properties and writes directly to instance fields:

```typescript
assign(properties: Record<string, unknown>): void {
  const ctor = this.constructor as typeof BaseNode;
  const defaults: Record<string, unknown> = {};
  for (const p of ctor.getDeclaredProperties()) {
    if (Object.prototype.hasOwnProperty.call(p.options, "default")) {
      defaults[p.name] = p.options.default;
    }
  }
  const merged = { ...defaults, ...properties };
  for (const { name } of ctor.getDeclaredProperties()) {
    if (name in merged) {
      (this as any)[name] = merged[name];
    }
  }
}
```

`assign()` remains public — it's useful for re-hydrating a node after construction (e.g., `deserialize` scenarios, test setup).

The registry simplifies to:

```typescript
// registry.ts resolve()
const instance = new NodeClass(descriptor.properties ?? {});
instance.__node_id = descriptor.id;
instance.__node_name = descriptor.name ?? descriptor.type;
return instance.toExecutor();
```

The `NodeClass` type signature widens to accept optional constructor properties:

```typescript
export type NodeClass = {
  new (properties?: Record<string, unknown>): BaseNode;
  // ... static fields unchanged
};
```

### `defaults()` Elimination

All 678 `defaults()` overrides become `@prop({ default: ... })` declarations. No `defaults()` method on BaseNode. No override mechanism.

```typescript
// Before
class ConcatTextNode extends BaseNode {
  defaults() { return { a: "", b: "" }; }
  async process(inputs) {
    const a = String(inputs.a ?? this._props.a ?? "");
    return { output: a };
  }
}

// After
class ConcatTextNode extends BaseNode {
  @prop({ type: "str", default: "" }) a!: string;
  @prop({ type: "str", default: "" }) b!: string;
  async process(inputs) {
    const a = String(inputs.a ?? this.a);
    return { output: a };
  }
}
```

If a node needs computed defaults, it does so in `initialize()`:

```typescript
class DynamicDefaultNode extends BaseNode {
  @prop({ type: "str", default: "" }) path!: string;

  async initialize() {
    if (!this.path) this.path = process.cwd();
  }
}
```

### Input Precedence (Breaking Change)

The conversion flips the precedence of inputs vs. stored properties:

```typescript
// Before: _props takes priority, inputs are fallback
return { output: this._props.value ?? inputs.value };

// After: inputs take priority, instance property is fallback
return { output: inputs.value ?? this.value };
```

This is intentional — runtime inputs from upstream edges should override static defaults.
Nodes that deliberately prefer stored state over inputs must be audited individually.

### Dynamic Properties

Nodes that add properties at runtime (e.g., `kie-dynamic.ts`) use explicit storage:

```typescript
protected dynamicProps = new Map<string, unknown>();

setDynamic(key: string, value: unknown): void {
  this.dynamicProps.set(key, value);
}

getDynamic<T = unknown>(key: string): T | undefined {
  return this.dynamicProps.get(key) as T | undefined;
}
```

### Serialization

`serialize()` and `deserialize()` cover `@prop`-decorated fields only.
`NodeDescriptor` remains the canonical format for workflow persistence.

```typescript
serialize(): Record<string, unknown> {
  const ctor = this.constructor as typeof BaseNode;
  const result: Record<string, unknown> = {};
  for (const { name } of ctor.getDeclaredProperties()) {
    result[name] = (this as any)[name];
  }
  return result;
}

deserialize(data: Record<string, unknown>): void {
  const ctor = this.constructor as typeof BaseNode;
  for (const { name, options } of ctor.getDeclaredProperties()) {
    if (name in data) {
      (this as any)[name] = data[name];
    } else if (options.default !== undefined) {
      (this as any)[name] = options.default;
    }
  }
}
```

## Benefits

### 1. Developer Experience
- **Intuitive:** Standard TypeScript property access patterns
- **Type-safe:** TypeScript checks property types at compile time
- **IDE-friendly:** Auto-completion works naturally
- **Familiar:** No learning curve for TypeScript developers

### 2. Cleaner Code
```typescript
// Before
return { output: this._props.value ?? this._props.defaultValue };

// After
return { output: this.value };
```

### 3. Reduced Complexity
- Single source of truth: instance properties
- No separate `_props` object to maintain
- No `defaults()` override pattern to reason about
- Eliminates potential sync issues between `_props` and actual state

### 4. Better Encapsulation
- No exposed `_props` object that could be mutated externally
- Clear separation: `@prop` for declared fields, `dynamicProps` for runtime fields

## Implementation Plan

### Step 1: BaseNode Changes

**`packages/node-sdk/src/base-node.ts`:**
- Add `constructor(properties?: Record<string, unknown>)` that calls `assign()`
- Remove `_props` field
- Remove `get props` getter
- Remove `defaults()` method
- Remove `getDefaultPropsFromDecorators()` method
- Update `assign()` to write directly to instance fields (inline default extraction)
- Add `serialize()` / `deserialize()` methods
- Add `dynamicProps` Map for runtime properties
- Add `__node_id` and `__node_name` as regular instance fields
- Widen `NodeClass` type: `new (properties?: Record<string, unknown>) => BaseNode`

**`packages/node-sdk/src/registry.ts`:**
- Pass `descriptor.properties` to constructor: `new NodeClass(descriptor.properties ?? {})`
- Set `__node_id` and `__node_name` directly on instance
- Remove separate `assign()` call

**`packages/node-sdk/src/node-metadata.ts`:**
- Remove any `_props` dependencies in metadata extraction

### Step 2: Codemod All Nodes

Mechanical transforms (can be scripted):

| Pattern | Replacement |
|---------|-------------|
| `this._props.X` | `this.X` |
| `this._props["X"]` | `this.X` |
| `defaults() { return { X: val }; }` | `@prop({ type: "...", default: val })` on field declarations |

Manual audit required for:
- Nodes that spread `_props`: `{ ...this._props }` — replace with `this.serialize()`
- Nodes with `this._props.X ?? inputs.X` — flip to `inputs.X ?? this.X`
- Nodes with computed `defaults()` — move logic to `initialize()`
- `propertyTypes` static field usage — convert to `@prop` decorators

**Files (61 node files + SDK):**
All files under `packages/base-nodes/src/nodes/` plus test files in `packages/node-sdk/`.

### Step 3: Testing

1. **Unit Tests** — Property initialization, assignment, serialization round-trip, inheritance
2. **Integration Tests** — Workflow execution, registry metadata, all 20 example workflows
3. **Build** — `npm run build` must pass across all 8 packages

### Validation

Keep validation centralized in `assign()`. No per-property setters or `@validate()` decorators.

## Edge Cases

### 1. Property Name Conflicts
The constructor calls `assign()` which runs after field initializers, so assigned values win over `value = "initializer"`.

### 2. Inheritance
`getDeclaredProperties()` already walks the prototype chain. `assign()` sets all inherited `@prop` fields.

### 3. Undefined vs Default
Explicit `undefined` stays `undefined`. Defaults only apply when no value is provided to `assign()`.

### 4. `propertyTypes` Static Field
Redundant with `@prop` metadata. Remove it in this change. `toDescriptor()` reads from `getDeclaredProperties()` only.

## Code Examples

### Before (Current)
```typescript
export class ConstantStringNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.String";
  static readonly title = "String";
  static readonly description = "String constant";

  defaults() {
    return { value: "" };
  }

  async process(inputs) {
    if ("value" in inputs) {
      return { output: inputs.value };
    }
    return { output: this._props.value ?? "" };
  }
}
```

### After
```typescript
export class ConstantStringNode extends BaseNode {
  static readonly nodeType = "nodetool.constant.String";
  static readonly title = "String";
  static readonly description = "Represents a string constant in the workflow.";
  static readonly outputTypes = { output: "str" };

  @prop({ type: "str", default: "", description: "The string value" })
  value!: string;

  async process(inputs: Record<string, unknown>) {
    return { output: inputs.value ?? this.value };
  }
}
```

## Appendix: Full BaseNode After

```typescript
export type NodeClass = {
  new (properties?: Record<string, unknown>): BaseNode;
  nodeType: string;
  title: string;
  description: string;
  isStreamingInput: boolean;
  isStreamingOutput: boolean;
  syncMode: SyncMode;
  isControlled: boolean;
  outputTypes: DeclaredOutputTypes;
  getDeclaredProperties(): Array<{ name: string; options: PropOptions }>;
  getDeclaredOutputs(): Record<string, string>;
  toDescriptor(id?: string): NodeDescriptor;
};

export abstract class BaseNode {
  static readonly nodeType: string = "";
  static readonly title: string = "";
  static readonly description: string = "";
  static readonly isStreamingInput: boolean = false;
  static readonly isStreamingOutput: boolean = false;
  static readonly syncMode: SyncMode = "zip_all";
  static readonly isControlled: boolean = false;
  static readonly outputTypes: DeclaredOutputTypes = {};

  __node_id: string = "";
  __node_name: string = "";

  protected dynamicProps = new Map<string, unknown>();

  constructor(properties: Record<string, unknown> = {}) {
    this.assign(properties);
  }

  static getDeclaredProperties() {
    return getDeclaredPropertiesForClass(this);
  }

  static getDeclaredOutputs(): Record<string, string> {
    return { ...(this.outputTypes ?? {}) };
  }

  assign(properties: Record<string, unknown>): void {
    const ctor = this.constructor as typeof BaseNode;
    const defaults: Record<string, unknown> = {};
    for (const p of ctor.getDeclaredProperties()) {
      if (Object.prototype.hasOwnProperty.call(p.options, "default")) {
        defaults[p.name] = p.options.default;
      }
    }
    const merged = { ...defaults, ...properties };
    for (const { name } of ctor.getDeclaredProperties()) {
      if (name in merged) {
        (this as any)[name] = merged[name];
      }
    }
  }

  serialize(): Record<string, unknown> {
    const ctor = this.constructor as typeof BaseNode;
    const result: Record<string, unknown> = {};
    for (const { name } of ctor.getDeclaredProperties()) {
      result[name] = (this as any)[name];
    }
    return result;
  }

  deserialize(data: Record<string, unknown>): void {
    const ctor = this.constructor as typeof BaseNode;
    for (const { name, options } of ctor.getDeclaredProperties()) {
      if (name in data) {
        (this as any)[name] = data[name];
      } else if (options.default !== undefined) {
        (this as any)[name] = options.default;
      }
    }
  }

  async initialize(): Promise<void> {}
  async preProcess(): Promise<void> {}
  async finalize(): Promise<void> {}

  abstract process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>>;

  async *genProcess(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): AsyncGenerator<Record<string, unknown>> {
    yield await this.process(inputs, context);
  }

  toExecutor(): NodeExecutor {
    return {
      process: (inputs, context?) => this.process(inputs, context),
      genProcess: (inputs, context?) => this.genProcess(inputs, context),
      preProcess: () => this.preProcess(),
      finalize: () => this.finalize(),
      initialize: () => this.initialize(),
    };
  }

  static toDescriptor(id?: string): NodeDescriptor {
    const cls = this as unknown as typeof BaseNode;
    const propertyTypes = Object.fromEntries(
      cls.getDeclaredProperties().map((entry) => [entry.name, entry.options.type]),
    );
    const desc: NodeDescriptor = {
      id: id ?? cls.nodeType,
      type: cls.nodeType,
      name: cls.title,
      is_streaming_input: cls.isStreamingInput,
      is_streaming_output: cls.isStreamingOutput,
      sync_mode: cls.syncMode,
      is_controlled: cls.isControlled,
    };
    if (Object.keys(propertyTypes).length > 0) {
      desc.propertyTypes = propertyTypes;
    }
    const outputs = cls.getDeclaredOutputs();
    if (Object.keys(outputs).length > 0) {
      desc.outputs = outputs;
    }
    return desc;
  }
}
```
