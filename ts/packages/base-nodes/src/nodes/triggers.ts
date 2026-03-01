import { BaseNode } from "@nodetool/node-sdk";

export class WaitNode extends BaseNode {
  static readonly nodeType = "nodetool.triggers.WaitNode";
  static readonly title = "Wait Node";
  static readonly description = "Pause execution for timeout_seconds and return wait metadata";

  defaults() {
    return {
      timeout_seconds: 0,
      input: "",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const timeoutSeconds = Number(
      inputs.timeout_seconds ?? this._props.timeout_seconds ?? 0
    );
    const inputData = inputs.input ?? this._props.input ?? "";

    const start = Date.now();
    if (timeoutSeconds > 0) {
      await new Promise((resolve) =>
        setTimeout(resolve, Math.floor(timeoutSeconds * 1000))
      );
    }
    const waitedSeconds = (Date.now() - start) / 1000;

    return {
      data: inputData,
      resumed_at: new Date().toISOString(),
      waited_seconds: waitedSeconds,
    };
  }
}

export const TRIGGER_NODES = [WaitNode] as const;
