import { BaseNode } from "@nodetool/node-sdk";
import type { ProcessingContext } from "@nodetool/runtime";

export class OutputNode extends BaseNode {
  static readonly nodeType = "nodetool.output.Output";
  static readonly title = "Output";
  static readonly description = "Generic output sink node";

  defaults() {
    return { value: null };
  }

  private inferOutputType(value: unknown): string {
    if (value === null || value === undefined) return "any";
    if (typeof value === "string") return "str";
    if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
    if (typeof value === "boolean") return "bool";
    if (Array.isArray(value)) return "list";
    if (value && typeof value === "object") return "dict";
    return "any";
  }

  private async normalize(value: unknown, context?: ProcessingContext): Promise<unknown> {
    if (!context || typeof context.normalizeOutputValue !== "function") return value;
    return context.normalizeOutputValue(value);
  }

  private emitOutputUpdate(value: unknown, context?: ProcessingContext): void {
    if (!context || typeof context.emit !== "function") return;
    const nodeId = String(this._props.__node_id ?? this._props.name ?? this._props.__node_name ?? "");
    const nodeName = String(this._props.__node_name ?? this._props.name ?? nodeId);
    const outputName =
      typeof this._props.name === "string" && this._props.name.trim().length > 0
        ? this._props.name
        : "output";
    context.emit({
      type: "output_update",
      node_id: nodeId,
      node_name: nodeName,
      output_name: outputName,
      value,
      output_type: this.inferOutputType(value),
      metadata: {},
    });
  }

  async process(inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    let value: unknown = this._props.value ?? null;
    if ("value" in inputs) {
      value = inputs.value;
    } else if ("input_value" in inputs) {
      value = inputs.input_value;
    } else if ("output" in inputs) {
      value = inputs.output;
    } else {
      const keys = Object.keys(inputs);
      if (keys.length === 1) value = inputs[keys[0]];
    }

    const normalized = await this.normalize(value, context);
    this.emitOutputUpdate(normalized, context);
    return { output: normalized };
  }
}

export class PreviewNode extends BaseNode {
  static readonly nodeType = "nodetool.workflows.base_node.Preview";
  static readonly title = "Preview";
  static readonly description = "Preview values inside the workflow graph";

  defaults() {
    return { value: null, name: "" };
  }

  private async normalize(value: unknown, context?: ProcessingContext): Promise<unknown> {
    if (!context || typeof context.normalizeOutputValue !== "function") return value;
    return context.normalizeOutputValue(value);
  }

  private emitPreview(value: unknown, context?: ProcessingContext): void {
    if (!context || typeof context.emit !== "function") return;
    const nodeId = String(this._props.__node_id ?? this._props.name ?? this._props.__node_name ?? "");
    context.emit({
      type: "preview_update",
      node_id: nodeId,
      value,
    });
  }

  async process(inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    let value: unknown = null;
    if ("value" in inputs) {
      value = inputs.value;
    } else {
      const keys = Object.keys(inputs);
      if (keys.length === 1) value = inputs[keys[0]];
      else value = this._props.value ?? null;
    }

    const normalized = await this.normalize(value, context);
    this.emitPreview(normalized, context);
    return { output: normalized };
  }
}

export const OUTPUT_NODES = [OutputNode, PreviewNode] as const;
