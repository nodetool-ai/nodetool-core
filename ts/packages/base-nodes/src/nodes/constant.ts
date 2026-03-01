import { BaseNode } from "@nodetool/node-sdk";

abstract class ConstantNode extends BaseNode {
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("value" in inputs) {
      return { output: inputs.value };
    }
    return { output: this._props.value ?? null };
  }
}

export class ConstantBoolNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Bool";
  static readonly title = "Bool";
  static readonly description = "Boolean constant";

  defaults() {
    return { value: false };
  }
}

export class ConstantIntegerNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Integer";
  static readonly title = "Integer";
  static readonly description = "Integer constant";

  defaults() {
    return { value: 0 };
  }
}

export class ConstantFloatNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Float";
  static readonly title = "Float";
  static readonly description = "Float constant";

  defaults() {
    return { value: 0.0 };
  }
}

export class ConstantStringNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.String";
  static readonly title = "String";
  static readonly description = "String constant";

  defaults() {
    return { value: "" };
  }
}

export class ConstantListNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.List";
  static readonly title = "List";
  static readonly description = "List constant";

  defaults() {
    return { value: [] as unknown[] };
  }
}

export class ConstantTextListNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.TextList";
  static readonly title = "Text List";
  static readonly description = "List of text constants";

  defaults() {
    return { value: [] as string[] };
  }
}

export class ConstantDictNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Dict";
  static readonly title = "Dict";
  static readonly description = "Dictionary constant";

  defaults() {
    return { value: {} as Record<string, unknown> };
  }
}

export const CONSTANT_NODES = [
  ConstantBoolNode,
  ConstantIntegerNode,
  ConstantFloatNode,
  ConstantStringNode,
  ConstantListNode,
  ConstantTextListNode,
  ConstantDictNode,
] as const;
