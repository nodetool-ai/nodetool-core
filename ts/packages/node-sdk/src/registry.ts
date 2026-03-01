import type { NodeDescriptor } from "@nodetool/protocol";
import type { NodeExecutor } from "@nodetool/kernel";
import type { NodeClass } from "./base-node.js";

export class NodeRegistry {
  private _classes = new Map<string, NodeClass>();

  register(nodeClass: NodeClass): void {
    if (!nodeClass.nodeType) {
      throw new Error(
        `Cannot register node class without nodeType: ${nodeClass.name}`
      );
    }
    this._classes.set(nodeClass.nodeType, nodeClass);
  }

  resolve(descriptor: NodeDescriptor): NodeExecutor {
    const NodeClass = this._classes.get(descriptor.type);
    if (!NodeClass) {
      throw new Error(`Unknown node type: ${descriptor.type}`);
    }
    const instance = new NodeClass();
    if (descriptor.properties) {
      instance.assign(descriptor.properties as Record<string, unknown>);
    }
    return instance.toExecutor();
  }

  has(nodeType: string): boolean {
    return this._classes.has(nodeType);
  }

  list(): string[] {
    return [...this._classes.keys()];
  }

  static readonly global = new NodeRegistry();
}

export function register(nodeClass: NodeClass): void {
  NodeRegistry.global.register(nodeClass);
}
