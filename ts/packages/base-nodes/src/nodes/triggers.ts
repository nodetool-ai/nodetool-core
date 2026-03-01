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

export class WaitAliasNode extends BaseNode {
  static readonly nodeType = "nodetool.triggers.Wait";
  static readonly title = "Wait";
  static readonly description = "Alias for WaitNode.";

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

export class ManualTriggerNode extends BaseNode {
  static readonly nodeType = "nodetool.triggers.ManualTrigger";
  static readonly title = "Manual Trigger";
  static readonly description = "Emit manually provided payload.";

  defaults() {
    return { payload: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const payload = inputs.payload ?? this._props.payload ?? {};
    return { output: payload, payload };
  }
}

export class IntervalTriggerNode extends BaseNode {
  static readonly nodeType = "nodetool.triggers.IntervalTrigger";
  static readonly title = "Interval Trigger";
  static readonly description = "Emit interval trigger metadata.";

  defaults() {
    return { interval_seconds: 60 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const intervalSeconds = Number(inputs.interval_seconds ?? this._props.interval_seconds ?? 60);
    return {
      output: {
        interval_seconds: intervalSeconds,
        triggered_at: new Date().toISOString(),
      },
    };
  }
}

export class WebhookTriggerNode extends BaseNode {
  static readonly nodeType = "nodetool.triggers.WebhookTrigger";
  static readonly title = "Webhook Trigger";
  static readonly description = "Forward webhook request payload.";

  defaults() {
    return { body: {}, headers: {}, method: "POST", path: "/" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {
      output: {
        method: String(inputs.method ?? this._props.method ?? "POST"),
        path: String(inputs.path ?? this._props.path ?? "/"),
        headers: inputs.headers ?? this._props.headers ?? {},
        body: inputs.body ?? this._props.body ?? {},
      },
    };
  }
}

export class FileWatchTriggerNode extends BaseNode {
  static readonly nodeType = "nodetool.triggers.FileWatchTrigger";
  static readonly title = "File Watch Trigger";
  static readonly description = "Emit file watch metadata.";

  defaults() {
    return { path: "", event: "modified" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {
      output: {
        path: String(inputs.path ?? this._props.path ?? ""),
        event: String(inputs.event ?? this._props.event ?? "modified"),
        detected_at: new Date().toISOString(),
      },
    };
  }
}

export const TRIGGER_NODES = [
  WaitNode,
  WaitAliasNode,
  ManualTriggerNode,
  IntervalTriggerNode,
  WebhookTriggerNode,
  FileWatchTriggerNode,
] as const;
