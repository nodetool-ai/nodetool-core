import { BaseNode } from "@nodetool/node-sdk";

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

export const OUTPUT_NODES = [OutputNode] as const;
