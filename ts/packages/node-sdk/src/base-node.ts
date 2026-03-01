import type { NodeDescriptor, SyncMode } from "@nodetool/protocol";
import type { NodeExecutor } from "@nodetool/kernel";

export type NodeClass = {
  new (): BaseNode;
  nodeType: string;
  title: string;
  description: string;
  isStreamingInput: boolean;
  isStreamingOutput: boolean;
  syncMode: SyncMode;
  isControlled: boolean;
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

  protected _props: Record<string, unknown> = {};

  defaults(): Record<string, unknown> {
    return {};
  }

  assign(properties: Record<string, unknown>): void {
    this._props = { ...this.defaults(), ...properties };
  }

  get props(): Record<string, unknown> {
    return this._props;
  }

  async initialize(): Promise<void> {}
  async preProcess(): Promise<void> {}
  async finalize(): Promise<void> {}

  abstract process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>>;

  async *genProcess(
    inputs: Record<string, unknown>
  ): AsyncGenerator<Record<string, unknown>> {
    yield await this.process(inputs);
  }

  toExecutor(): NodeExecutor {
    const self = this;
    return {
      async process(inputs: Record<string, unknown>) {
        return self.process(inputs);
      },
      async *genProcess(inputs: Record<string, unknown>) {
        yield* self.genProcess(inputs);
      },
      async preProcess() {
        return self.preProcess();
      },
      async finalize() {
        return self.finalize();
      },
      async initialize() {
        return self.initialize();
      },
    };
  }

  static toDescriptor(id?: string): NodeDescriptor {
    const cls = this as unknown as typeof BaseNode;
    return {
      id: id ?? cls.nodeType,
      type: cls.nodeType,
      name: cls.title,
      is_streaming_input: cls.isStreamingInput,
      is_streaming_output: cls.isStreamingOutput,
      sync_mode: cls.syncMode,
      is_controlled: cls.isControlled,
    };
  }
}
