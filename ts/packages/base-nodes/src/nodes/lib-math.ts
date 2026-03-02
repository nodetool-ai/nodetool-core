import { BaseNode } from "@nodetool/node-sdk";

type MathOperation =
  | "negate"
  | "absolute"
  | "square"
  | "cube"
  | "square_root"
  | "cube_root"
  | "sine"
  | "cosine"
  | "tangent"
  | "arcsin"
  | "arccos"
  | "arctan"
  | "log";

export class AddLibNode extends BaseNode {
  static readonly nodeType = "lib.math.Add";
  static readonly title = "Add";
  static readonly description = "Adds two numbers together.";

  defaults() {
    return { a: 0, b: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: Number(inputs.a ?? this._props.a ?? 0) + Number(inputs.b ?? this._props.b ?? 0) };
  }
}

export class SubtractLibNode extends BaseNode {
  static readonly nodeType = "lib.math.Subtract";
  static readonly title = "Subtract";
  static readonly description = "Subtracts B from A.";

  defaults() {
    return { a: 0, b: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: Number(inputs.a ?? this._props.a ?? 0) - Number(inputs.b ?? this._props.b ?? 0) };
  }
}

export class MultiplyLibNode extends BaseNode {
  static readonly nodeType = "lib.math.Multiply";
  static readonly title = "Multiply";
  static readonly description = "Multiplies two numbers together.";

  defaults() {
    return { a: 0, b: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: Number(inputs.a ?? this._props.a ?? 0) * Number(inputs.b ?? this._props.b ?? 0) };
  }
}

export class DivideLibNode extends BaseNode {
  static readonly nodeType = "lib.math.Divide";
  static readonly title = "Divide";
  static readonly description = "Divides A by B to calculate the quotient.";

  defaults() {
    return { a: 0, b: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: Number(inputs.a ?? this._props.a ?? 0) / Number(inputs.b ?? this._props.b ?? 1) };
  }
}

export class ModulusLibNode extends BaseNode {
  static readonly nodeType = "lib.math.Modulus";
  static readonly title = "Modulus";
  static readonly description = "Computes A modulo B to find the remainder after division.";

  defaults() {
    return { a: 0, b: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: Number(inputs.a ?? this._props.a ?? 0) % Number(inputs.b ?? this._props.b ?? 1) };
  }
}

export class MathFunctionLibNode extends BaseNode {
  static readonly nodeType = "lib.math.MathFunction";
  static readonly title = "Math Function";
  static readonly description = "Performs a selected unary math operation on an input.";

  defaults() {
    return { input: 0, operation: "negate" as MathOperation };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const input = Number(inputs.input ?? this._props.input ?? 0);
    const operation = String(
      inputs.operation ?? this._props.operation ?? "negate"
    ) as MathOperation;

    switch (operation) {
      case "negate":
        return { output: -input };
      case "absolute":
        return { output: Math.abs(input) };
      case "square":
        return { output: input * input };
      case "cube":
        return { output: input * input * input };
      case "square_root":
        return { output: Math.sqrt(input) };
      case "cube_root":
        return { output: Math.sign(input) * Math.pow(Math.abs(input), 1 / 3) };
      case "sine":
        return { output: Math.sin(input) };
      case "cosine":
        return { output: Math.cos(input) };
      case "tangent":
        return { output: Math.tan(input) };
      case "arcsin":
        return { output: Math.asin(input) };
      case "arccos":
        return { output: Math.acos(input) };
      case "arctan":
        return { output: Math.atan(input) };
      case "log":
        return { output: Math.log(input) };
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
  }
}

export class SineLibNode extends BaseNode {
  static readonly nodeType = "lib.math.Sine";
  static readonly title = "Sine";
  static readonly description = "Computes sine of the given angle in radians.";

  defaults() {
    return { angle_rad: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: Math.sin(Number(inputs.angle_rad ?? this._props.angle_rad ?? 0)) };
  }
}

export class CosineLibNode extends BaseNode {
  static readonly nodeType = "lib.math.Cosine";
  static readonly title = "Cosine";
  static readonly description = "Computes cosine of the given angle in radians.";

  defaults() {
    return { angle_rad: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: Math.cos(Number(inputs.angle_rad ?? this._props.angle_rad ?? 0)) };
  }
}

export class PowerLibNode extends BaseNode {
  static readonly nodeType = "lib.math.Power";
  static readonly title = "Power";
  static readonly description = "Raises base to the given exponent.";

  defaults() {
    return { base: 0, exponent: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {
      output: Math.pow(
        Number(inputs.base ?? this._props.base ?? 0),
        Number(inputs.exponent ?? this._props.exponent ?? 1)
      ),
    };
  }
}

export class SqrtLibNode extends BaseNode {
  static readonly nodeType = "lib.math.Sqrt";
  static readonly title = "Sqrt";
  static readonly description = "Computes square root of x.";

  defaults() {
    return { x: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: Math.sqrt(Number(inputs.x ?? this._props.x ?? 0)) };
  }
}

export const LIB_MATH_NODES = [
  AddLibNode,
  SubtractLibNode,
  MultiplyLibNode,
  DivideLibNode,
  ModulusLibNode,
  MathFunctionLibNode,
  SineLibNode,
  CosineLibNode,
  PowerLibNode,
  SqrtLibNode,
] as const;
