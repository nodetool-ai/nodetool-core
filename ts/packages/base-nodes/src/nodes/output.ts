import { BaseNode } from "@nodetool/node-sdk";
import type { ProcessingContext } from "@nodetool/runtime";

export class OutputNode extends BaseNode {
  static readonly nodeType = "nodetool.output.Output";
  static readonly title = "Output";
  static readonly description = "Generic output sink node";

  defaults() {
    return { value: null };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("value" in inputs) {
      return { output: inputs.value };
    }
    if ("input_value" in inputs) {
      return { output: inputs.input_value };
    }
    if ("output" in inputs) {
      return { output: inputs.output };
    }

    const keys = Object.keys(inputs);
    if (keys.length === 1) {
      return { output: inputs[keys[0]] };
    }

    return { output: this._props.value ?? null };
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
