import type { NodeDescriptor } from "@nodetool/protocol";
import type { NodeExecutor } from "@nodetool/kernel";
import type { NodeClass } from "./base-node.js";
import type {
  NodeMetadata,
  PythonMetadataLoadOptions,
  PythonMetadataLoadResult,
} from "./metadata.js";
import { loadPythonPackageMetadata } from "./metadata.js";

export interface NodeRegistryOptions {
  metadataByType?: Map<string, NodeMetadata>;
  strictMetadata?: boolean;
}

export interface RegisterNodeOptions {
  metadata?: NodeMetadata;
}

export class NodeRegistry {
  private _classes = new Map<string, NodeClass>();
  private _loadedMetadataByType = new Map<string, NodeMetadata>();
  private _registeredMetadataByType = new Map<string, NodeMetadata>();
  private _strictMetadata: boolean;

  constructor(options: NodeRegistryOptions = {}) {
    this._strictMetadata = options.strictMetadata ?? false;
    if (options.metadataByType) {
      for (const [nodeType, metadata] of options.metadataByType.entries()) {
        this._loadedMetadataByType.set(nodeType, metadata);
      }
    }
  }

  register(nodeClass: NodeClass, options: RegisterNodeOptions = {}): void {
    if (!nodeClass.nodeType) {
      throw new Error(
        `Cannot register node class without nodeType: ${nodeClass.name}`
      );
    }
    const metadata = options.metadata ?? this._resolveLoadedMetadata(nodeClass.nodeType);
    if (metadata) {
      this._registeredMetadataByType.set(nodeClass.nodeType, metadata);
    } else if (this._strictMetadata) {
      throw new Error(`Missing Python metadata for node type: ${nodeClass.nodeType}`);
    }
    this._classes.set(nodeClass.nodeType, nodeClass);
  }

  resolve(descriptor: NodeDescriptor): NodeExecutor {
    const NodeClass = this._classes.get(descriptor.type);
    if (!NodeClass) {
      throw new Error(`Unknown node type: ${descriptor.type}`);
    }
    const instance = new NodeClass();
    const descriptorProps: Record<string, unknown> = {
      ...(descriptor.properties as Record<string, unknown> | undefined),
      __node_id: descriptor.id,
      __node_name: descriptor.name ?? descriptor.type,
    };
    instance.assign(descriptorProps);
    return instance.toExecutor();
  }

  has(nodeType: string): boolean {
    return this._classes.has(nodeType);
  }

  list(): string[] {
    return [...this._classes.keys()];
  }

  getMetadata(nodeType: string): NodeMetadata | undefined {
    return this._registeredMetadataByType.get(nodeType) ?? this._loadedMetadataByType.get(nodeType);
  }

  listMetadata(): NodeMetadata[] {
    return this.list()
      .map((nodeType) => this.getMetadata(nodeType))
      .filter((md): md is NodeMetadata => md !== undefined);
  }

  listRegisteredNodeTypesWithoutMetadata(): string[] {
    return this.list().filter((nodeType) => this.getMetadata(nodeType) === undefined);
  }

  loadPythonMetadata(options: PythonMetadataLoadOptions = {}): PythonMetadataLoadResult {
    const loaded = loadPythonPackageMetadata(options);
    for (const [nodeType, metadata] of loaded.nodesByType.entries()) {
      this._loadedMetadataByType.set(nodeType, metadata);
    }
    return loaded;
  }

  private _resolveLoadedMetadata(nodeType: string): NodeMetadata | undefined {
    const exact = this._loadedMetadataByType.get(nodeType);
    if (exact) return exact;
    if (nodeType.endsWith("Node")) {
      return this._loadedMetadataByType.get(nodeType.slice(0, -4));
    }
    return undefined;
  }

  static readonly global = new NodeRegistry();
}

export function register(nodeClass: NodeClass): void {
  NodeRegistry.global.register(nodeClass);
}
