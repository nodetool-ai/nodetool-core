import { BaseNode } from "@nodetool/node-sdk";
import type { ProcessingContext } from "@nodetool/runtime";
import sharp from "sharp";

// ---------------------------------------------------------------------------
// NdArray helpers
// ---------------------------------------------------------------------------

type NdArray = { data: number[]; shape: number[] };

function totalSize(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function asNdArray(v: unknown): NdArray {
  if (v && typeof v === "object" && "data" in v && "shape" in v) {
    return v as NdArray;
  }
  if (typeof v === "number") {
    return { data: [v], shape: [1] };
  }
  if (Array.isArray(v)) {
    return { data: (v as number[]).map(Number), shape: [v.length] };
  }
  return { data: [], shape: [0] };
}

function elementwiseUnary(arr: NdArray, fn: (x: number) => number): NdArray {
  return { data: arr.data.map(fn), shape: [...arr.shape] };
}

function padArrays(a: NdArray, b: NdArray): [NdArray, NdArray] {
  if (a.data.length === 1 || b.data.length === 1) return [a, b];
  if (a.data.length === b.data.length) return [a, b];
  const maxLen = Math.max(a.data.length, b.data.length);
  const padA = a.data.length < maxLen
    ? { data: [...a.data, ...new Array(maxLen - a.data.length).fill(0)], shape: [maxLen] }
    : a;
  const padB = b.data.length < maxLen
    ? { data: [...b.data, ...new Array(maxLen - b.data.length).fill(0)], shape: [maxLen] }
    : b;
  return [padA, padB];
}

function elementwiseBinary(
  a: NdArray,
  b: NdArray,
  fn: (x: number, y: number) => number
): NdArray {
  const [pa, pb] = padArrays(a, b);
  if (pa.data.length === 1) {
    return { data: pb.data.map((v) => fn(pa.data[0], v)), shape: [...pb.shape] };
  }
  if (pb.data.length === 1) {
    return { data: pa.data.map((v) => fn(v, pb.data[0])), shape: [...pa.shape] };
  }
  return {
    data: pa.data.map((v, i) => fn(v, pb.data[i])),
    shape: [...pa.shape],
  };
}

function convertOutput(arr: NdArray): Record<string, unknown> {
  if (arr.data.length === 1) {
    return { output: arr.data[0] };
  }
  return { output: { data: arr.data, shape: arr.shape } };
}

// Reduction helpers for axis-based operations
function reduceAlongAxis(
  arr: NdArray,
  axis: number,
  reduceFn: (values: number[]) => number
): NdArray {
  if (arr.shape.length === 0 || arr.data.length === 0) {
    return { data: [], shape: [] };
  }

  const ndim = arr.shape.length;
  const clampedAxis = ((axis % ndim) + ndim) % ndim;

  if (ndim === 1) {
    return { data: [reduceFn(arr.data)], shape: [1] };
  }

  const newShape = arr.shape.filter((_, i) => i !== clampedAxis);
  const newSize = totalSize(newShape);
  const result: number[] = new Array(newSize);

  const axisLen = arr.shape[clampedAxis];
  const outerStride = arr.shape.slice(clampedAxis + 1).reduce((a, b) => a * b, 1);
  const axisStride = outerStride;
  const outerLen = arr.shape.slice(0, clampedAxis).reduce((a, b) => a * b, 1);
  const innerLen = outerStride;

  let idx = 0;
  for (let o = 0; o < outerLen; o++) {
    for (let inner = 0; inner < innerLen; inner++) {
      const values: number[] = [];
      for (let a = 0; a < axisLen; a++) {
        values.push(arr.data[o * axisLen * axisStride + a * axisStride + inner]);
      }
      result[idx++] = reduceFn(values);
    }
  }

  return { data: result, shape: newShape };
}

// WAV encoding (from lib-synthesis.ts pattern)
function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(Math.round(s * 0x7fff), 44 + i * 2);
  }
  return new Uint8Array(buffer);
}

function audioRefFromWav(wav: Uint8Array): Record<string, unknown> {
  return { uri: "", data: Buffer.from(wav).toString("base64") };
}

// ---------------------------------------------------------------------------
// Arithmetic (5 nodes)
// ---------------------------------------------------------------------------

export class AddArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.arithmetic.AddArray";
  static readonly title = "Add Array";
  static readonly description = "Performs addition on two arrays.";

  defaults() { return { a: 0, b: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asNdArray(inputs.a ?? this._props.a ?? 0);
    const b = asNdArray(inputs.b ?? this._props.b ?? 0);
    return convertOutput(elementwiseBinary(a, b, (x, y) => x + y));
  }
}

export class SubtractArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.arithmetic.SubtractArray";
  static readonly title = "Subtract Array";
  static readonly description = "Subtracts the second array from the first.";

  defaults() { return { a: 0, b: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asNdArray(inputs.a ?? this._props.a ?? 0);
    const b = asNdArray(inputs.b ?? this._props.b ?? 0);
    return convertOutput(elementwiseBinary(a, b, (x, y) => x - y));
  }
}

export class MultiplyArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.arithmetic.MultiplyArray";
  static readonly title = "Multiply Array";
  static readonly description = "Multiplies two arrays.";

  defaults() { return { a: 0, b: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asNdArray(inputs.a ?? this._props.a ?? 0);
    const b = asNdArray(inputs.b ?? this._props.b ?? 0);
    return convertOutput(elementwiseBinary(a, b, (x, y) => x * y));
  }
}

export class DivideArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.arithmetic.DivideArray";
  static readonly title = "Divide Array";
  static readonly description = "Divides the first array by the second.";

  defaults() { return { a: 0, b: 1 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asNdArray(inputs.a ?? this._props.a ?? 0);
    const b = asNdArray(inputs.b ?? this._props.b ?? 1);
    return convertOutput(elementwiseBinary(a, b, (x, y) => x / y));
  }
}

export class ModulusArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.arithmetic.ModulusArray";
  static readonly title = "Modulus Array";
  static readonly description = "Calculates the element-wise remainder of division.";

  defaults() { return { a: 0, b: 1 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asNdArray(inputs.a ?? this._props.a ?? 0);
    const b = asNdArray(inputs.b ?? this._props.b ?? 1);
    return convertOutput(elementwiseBinary(a, b, (x, y) => x % y));
  }
}

// ---------------------------------------------------------------------------
// Math (7 nodes)
// ---------------------------------------------------------------------------

export class AbsArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.math.AbsArray";
  static readonly title = "Abs Array";
  static readonly description = "Compute the absolute value of each element in a array.";

  defaults() { return { values: { data: [], shape: [0] } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    return convertOutput(elementwiseUnary(arr, Math.abs));
  }
}

export class SineArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.math.SineArray";
  static readonly title = "Sine Array";
  static readonly description = "Computes the sine of input angles in radians.";

  defaults() { return { angle_rad: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.angle_rad ?? this._props.angle_rad ?? 0);
    return convertOutput(elementwiseUnary(arr, Math.sin));
  }
}

export class CosineArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.math.CosineArray";
  static readonly title = "Cosine Array";
  static readonly description = "Computes the cosine of input angles in radians.";

  defaults() { return { angle_rad: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.angle_rad ?? this._props.angle_rad ?? 0);
    return convertOutput(elementwiseUnary(arr, Math.cos));
  }
}

export class ExpArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.math.ExpArray";
  static readonly title = "Exp Array";
  static readonly description = "Calculate the exponential of each element in a array.";

  defaults() { return { values: { data: [], shape: [0] } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    return convertOutput(elementwiseUnary(arr, Math.exp));
  }
}

export class LogArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.math.LogArray";
  static readonly title = "Log Array";
  static readonly description = "Calculate the natural logarithm of each element in a array.";

  defaults() { return { values: { data: [], shape: [0] } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    return convertOutput(elementwiseUnary(arr, Math.log));
  }
}

export class SqrtArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.math.SqrtArray";
  static readonly title = "Sqrt Array";
  static readonly description = "Calculates the square root of the input array element-wise.";

  defaults() { return { values: { data: [], shape: [0] } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    return convertOutput(elementwiseUnary(arr, Math.sqrt));
  }
}

export class PowerArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.math.PowerArray";
  static readonly title = "Power Array";
  static readonly description = "Raises the base array to the power of the exponent element-wise.";

  defaults() { return { base: 1, exponent: 2 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const base = asNdArray(inputs.base ?? this._props.base ?? 1);
    const exp = asNdArray(inputs.exponent ?? this._props.exponent ?? 2);
    return convertOutput(elementwiseBinary(base, exp, Math.pow));
  }
}

// ---------------------------------------------------------------------------
// Statistics (6 nodes)
// ---------------------------------------------------------------------------

export class SumArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.statistics.SumArray";
  static readonly title = "Sum Array";
  static readonly description = "Calculate the sum of values along a specified axis of a array.";

  defaults() { return { values: { data: [], shape: [0] }, axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);
    const res = reduceAlongAxis(arr, axis, (vals) => vals.reduce((a, b) => a + b, 0));
    return convertOutput(res);
  }
}

export class MeanArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.statistics.MeanArray";
  static readonly title = "Mean Array";
  static readonly description = "Compute the mean value along a specified axis of a array.";

  defaults() { return { values: { data: [], shape: [0] }, axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);
    const res = reduceAlongAxis(arr, axis, (vals) =>
      vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
    );
    return convertOutput(res);
  }
}

export class MinArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.statistics.MinArray";
  static readonly title = "Min Array";
  static readonly description = "Calculate the minimum value along a specified axis of a array.";

  defaults() { return { values: { data: [], shape: [0] }, axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);
    const res = reduceAlongAxis(arr, axis, (vals) => Math.min(...vals));
    return convertOutput(res);
  }
}

export class MaxArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.statistics.MaxArray";
  static readonly title = "Max Array";
  static readonly description = "Compute the maximum value along a specified axis of a array.";

  defaults() { return { values: { data: [], shape: [0] }, axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);
    const res = reduceAlongAxis(arr, axis, (vals) => Math.max(...vals));
    return convertOutput(res);
  }
}

export class ArgMinArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.statistics.ArgMinArray";
  static readonly title = "Arg Min Array";
  static readonly description = "Find indices of minimum values along a specified axis of a array.";

  defaults() { return { values: { data: [], shape: [0] }, axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);
    const res = reduceAlongAxis(arr, axis, (vals) => {
      let minIdx = 0;
      for (let i = 1; i < vals.length; i++) {
        if (vals[i] < vals[minIdx]) minIdx = i;
      }
      return minIdx;
    });
    return convertOutput(res);
  }
}

export class ArgMaxArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.statistics.ArgMaxArray";
  static readonly title = "Arg Max Array";
  static readonly description = "Find indices of maximum values along a specified axis of a array.";

  defaults() { return { values: { data: [], shape: [0] }, axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);
    const res = reduceAlongAxis(arr, axis, (vals) => {
      let maxIdx = 0;
      for (let i = 1; i < vals.length; i++) {
        if (vals[i] > vals[maxIdx]) maxIdx = i;
      }
      return maxIdx;
    });
    return convertOutput(res);
  }
}

// ---------------------------------------------------------------------------
// Manipulation (6 nodes)
// ---------------------------------------------------------------------------

export class SliceArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.manipulation.SliceArray";
  static readonly title = "Slice Array";
  static readonly description = "Extract a slice of an array along a specified axis.";

  defaults() { return { values: { data: [], shape: [0] }, start: 0, stop: 0, step: 1, axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const start = Number(inputs.start ?? this._props.start ?? 0);
    const stop = Number(inputs.stop ?? this._props.stop ?? 0);
    const step = Number(inputs.step ?? this._props.step ?? 1);
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);

    const ndim = arr.shape.length;
    if (ndim === 0) return { output: { data: [], shape: [] } };

    const clampedAxis = ((axis % ndim) + ndim) % ndim;
    const axisLen = arr.shape[clampedAxis];
    const effectiveStop = stop === 0 ? axisLen : Math.min(stop, axisLen);

    const indices: number[] = [];
    for (let i = start; i < effectiveStop; i += step) {
      indices.push(i);
    }

    return this.takeAlongAxis(arr, indices, clampedAxis);
  }

  private takeAlongAxis(arr: NdArray, indices: number[], axis: number): Record<string, unknown> {
    const newShape = [...arr.shape];
    newShape[axis] = indices.length;
    const newData: number[] = new Array(totalSize(newShape));

    const outerLen = arr.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const innerLen = arr.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const axisLen = arr.shape[axis];

    let idx = 0;
    for (let o = 0; o < outerLen; o++) {
      for (const selIdx of indices) {
        for (let inner = 0; inner < innerLen; inner++) {
          newData[idx++] = arr.data[o * axisLen * innerLen + selIdx * innerLen + inner];
        }
      }
    }

    return convertOutput({ data: newData, shape: newShape });
  }
}

export class IndexArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.manipulation.IndexArray";
  static readonly title = "Index Array";
  static readonly description = "Select specific indices from an array along a specified axis.";

  defaults() { return { values: { data: [], shape: [0] }, indices: "", axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const indicesStr = String(inputs.indices ?? this._props.indices ?? "");
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);

    const indices = indicesStr.split(",").filter((s) => s.trim()).map((s) => parseInt(s.trim(), 10));
    if (indices.length === 0) return { output: { data: [], shape: [0] } };

    const ndim = arr.shape.length;
    const clampedAxis = ((axis % ndim) + ndim) % ndim;
    const newShape = [...arr.shape];
    newShape[clampedAxis] = indices.length;

    const outerLen = arr.shape.slice(0, clampedAxis).reduce((a, b) => a * b, 1);
    const innerLen = arr.shape.slice(clampedAxis + 1).reduce((a, b) => a * b, 1);
    const axisLen = arr.shape[clampedAxis];

    const newData: number[] = [];
    for (let o = 0; o < outerLen; o++) {
      for (const selIdx of indices) {
        for (let inner = 0; inner < innerLen; inner++) {
          newData.push(arr.data[o * axisLen * innerLen + selIdx * innerLen + inner]);
        }
      }
    }

    return { output: { data: newData, shape: newShape } };
  }
}

export class TransposeArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.manipulation.TransposeArray";
  static readonly title = "Transpose Array";
  static readonly description = "Transpose the dimensions of the input array.";

  defaults() { return { values: { data: [], shape: [0] } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const ndim = arr.shape.length;

    if (ndim <= 1) return { output: { data: [...arr.data], shape: [...arr.shape] } };

    if (ndim === 2) {
      const [rows, cols] = arr.shape;
      const newData: number[] = new Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          newData[c * rows + r] = arr.data[r * cols + c];
        }
      }
      return { output: { data: newData, shape: [cols, rows] } };
    }

    // General n-dim transpose (reverse axes)
    const newShape = [...arr.shape].reverse();
    const newData: number[] = new Array(arr.data.length);
    const strides: number[] = new Array(ndim);
    strides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) strides[i] = strides[i + 1] * arr.shape[i + 1];

    const newStrides: number[] = new Array(ndim);
    newStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) newStrides[i] = newStrides[i + 1] * newShape[i + 1];

    for (let flatIdx = 0; flatIdx < arr.data.length; flatIdx++) {
      let remaining = flatIdx;
      let newFlatIdx = 0;
      for (let d = 0; d < ndim; d++) {
        const coord = Math.floor(remaining / strides[d]);
        remaining %= strides[d];
        newFlatIdx += coord * newStrides[ndim - 1 - d];
      }
      newData[newFlatIdx] = arr.data[flatIdx];
    }

    return { output: { data: newData, shape: newShape } };
  }
}

export class MatMulNode extends BaseNode {
  static readonly nodeType = "lib.numpy.manipulation.MatMul";
  static readonly title = "Mat Mul";
  static readonly description = "Perform matrix multiplication on two input arrays.";

  defaults() { return { a: { data: [], shape: [0, 0] }, b: { data: [], shape: [0, 0] } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asNdArray(inputs.a ?? this._props.a);
    const b = asNdArray(inputs.b ?? this._props.b);

    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("MatMul requires 2D arrays");
    }

    const [m, k1] = a.shape;
    const [k2, n] = b.shape;
    if (k1 !== k2) throw new Error(`Shape mismatch: ${k1} vs ${k2}`);

    const result: number[] = new Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < k1; k++) {
          sum += a.data[i * k1 + k] * b.data[k * n + j];
        }
        result[i * n + j] = sum;
      }
    }

    return { output: { data: result, shape: [m, n] } };
  }
}

export class StackNode extends BaseNode {
  static readonly nodeType = "lib.numpy.manipulation.Stack";
  static readonly title = "Stack";
  static readonly description = "Stack multiple arrays along a specified axis.";

  defaults() { return { arrays: [], axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rawArrays = (inputs.arrays ?? this._props.arrays ?? []) as unknown[];
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);
    if (rawArrays.length === 0) return { output: { data: [], shape: [0] } };

    const arrays = rawArrays.map(asNdArray);
    const refShape = arrays[0].shape;
    const newShape = [...refShape];
    newShape.splice(axis, 0, arrays.length);

    const outerLen = refShape.slice(0, axis).reduce((a, b) => a * b, 1);
    const innerLen = refShape.slice(axis).reduce((a, b) => a * b, 1);

    const newData: number[] = [];
    for (let o = 0; o < outerLen; o++) {
      for (const arr of arrays) {
        for (let inner = 0; inner < innerLen; inner++) {
          newData.push(arr.data[o * innerLen + inner]);
        }
      }
    }

    return { output: { data: newData, shape: newShape } };
  }
}

export class SplitArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.manipulation.SplitArray";
  static readonly title = "Split Array";
  static readonly description = "Split an array into multiple sub-arrays along a specified axis.";

  defaults() { return { values: { data: [], shape: [0] }, num_splits: 1, axis: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const numSplits = Number(inputs.num_splits ?? this._props.num_splits ?? 1);
    const axis = Number(inputs.axis ?? this._props.axis ?? 0);

    const ndim = arr.shape.length;
    const clampedAxis = ((axis % ndim) + ndim) % ndim;
    const axisLen = arr.shape[clampedAxis];

    const splitSizes: number[] = [];
    const base = Math.floor(axisLen / numSplits);
    const remainder = axisLen % numSplits;
    for (let i = 0; i < numSplits; i++) {
      splitSizes.push(base + (i < remainder ? 1 : 0));
    }

    const outerLen = arr.shape.slice(0, clampedAxis).reduce((a, b) => a * b, 1);
    const innerLen = arr.shape.slice(clampedAxis + 1).reduce((a, b) => a * b, 1);

    const results: NdArray[] = [];
    let offset = 0;
    for (const size of splitSizes) {
      const newShape = [...arr.shape];
      newShape[clampedAxis] = size;
      const newData: number[] = [];
      for (let o = 0; o < outerLen; o++) {
        for (let s = 0; s < size; s++) {
          for (let inner = 0; inner < innerLen; inner++) {
            newData.push(arr.data[o * axisLen * innerLen + (offset + s) * innerLen + inner]);
          }
        }
      }
      results.push({ data: newData, shape: newShape });
      offset += size;
    }

    return { output: results };
  }
}

// ---------------------------------------------------------------------------
// Reshaping (4 nodes)
// ---------------------------------------------------------------------------

export class Reshape1DNode extends BaseNode {
  static readonly nodeType = "lib.numpy.reshaping.Reshape1D";
  static readonly title = "Reshape 1D";
  static readonly description = "Reshape an array to a 1D shape without changing its data.";

  defaults() { return { values: { data: [], shape: [0] }, num_elements: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const numElements = Number(inputs.num_elements ?? this._props.num_elements ?? arr.data.length);
    const n = numElements || arr.data.length;
    return { output: { data: arr.data.slice(0, n), shape: [n] } };
  }
}

export class Reshape2DNode extends BaseNode {
  static readonly nodeType = "lib.numpy.reshaping.Reshape2D";
  static readonly title = "Reshape 2D";
  static readonly description = "Reshape an array to a new shape without changing its data.";

  defaults() { return { values: { data: [], shape: [0] }, num_rows: 0, num_cols: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const rows = Number(inputs.num_rows ?? this._props.num_rows ?? 0);
    const cols = Number(inputs.num_cols ?? this._props.num_cols ?? 0);
    return { output: { data: [...arr.data], shape: [rows, cols] } };
  }
}

export class Reshape3DNode extends BaseNode {
  static readonly nodeType = "lib.numpy.reshaping.Reshape3D";
  static readonly title = "Reshape 3D";
  static readonly description = "Reshape an array to a 3D shape without changing its data.";

  defaults() { return { values: { data: [], shape: [0] }, num_rows: 0, num_cols: 0, num_depths: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const rows = Number(inputs.num_rows ?? this._props.num_rows ?? 0);
    const cols = Number(inputs.num_cols ?? this._props.num_cols ?? 0);
    const depths = Number(inputs.num_depths ?? this._props.num_depths ?? 0);
    return { output: { data: [...arr.data], shape: [rows, cols, depths] } };
  }
}

export class Reshape4DNode extends BaseNode {
  static readonly nodeType = "lib.numpy.reshaping.Reshape4D";
  static readonly title = "Reshape 4D";
  static readonly description = "Reshape an array to a 4D shape without changing its data.";

  defaults() {
    return { values: { data: [], shape: [0] }, num_rows: 0, num_cols: 0, num_depths: 0, num_channels: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const rows = Number(inputs.num_rows ?? this._props.num_rows ?? 0);
    const cols = Number(inputs.num_cols ?? this._props.num_cols ?? 0);
    const depths = Number(inputs.num_depths ?? this._props.num_depths ?? 0);
    const channels = Number(inputs.num_channels ?? this._props.num_channels ?? 0);
    return { output: { data: [...arr.data], shape: [rows, cols, depths, channels] } };
  }
}

// ---------------------------------------------------------------------------
// Conversion (7 nodes)
// ---------------------------------------------------------------------------

export class ListToArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.conversion.ListToArray";
  static readonly title = "List To Array";
  static readonly description = "Convert a list of values to a array.";

  defaults() { return { values: [] }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = (inputs.values ?? this._props.values ?? []) as unknown[];
    const flat = flattenNestedList(values);
    const shape = inferShape(values);
    return { output: { data: flat, shape } };
  }
}

function flattenNestedList(val: unknown): number[] {
  if (Array.isArray(val)) {
    const result: number[] = [];
    for (const item of val) {
      result.push(...flattenNestedList(item));
    }
    return result;
  }
  return [Number(val)];
}

function inferShape(val: unknown): number[] {
  if (!Array.isArray(val)) return [];
  const shape: number[] = [val.length];
  if (val.length > 0 && Array.isArray(val[0])) {
    shape.push(...inferShape(val[0]));
  }
  return shape;
}

export class ArrayToListNode extends BaseNode {
  static readonly nodeType = "lib.numpy.conversion.ArrayToList";
  static readonly title = "Array To List";
  static readonly description = "Convert a array to a nested list structure.";

  defaults() { return { values: { data: [], shape: [0] } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    return { output: toNestedList(arr.data, arr.shape, 0, 0).value };
  }
}

function toNestedList(
  data: number[],
  shape: number[],
  dim: number,
  offset: number
): { value: unknown; consumed: number } {
  if (dim >= shape.length) {
    return { value: data[offset] ?? 0, consumed: 1 };
  }
  if (dim === shape.length - 1) {
    const slice = data.slice(offset, offset + shape[dim]);
    return { value: slice, consumed: shape[dim] };
  }
  const result: unknown[] = [];
  let pos = offset;
  for (let i = 0; i < shape[dim]; i++) {
    const sub = toNestedList(data, shape, dim + 1, pos);
    result.push(sub.value);
    pos += sub.consumed;
  }
  return { value: result, consumed: pos - offset };
}

export class ScalarToArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.conversion.ScalarToArray";
  static readonly title = "Scalar To Array";
  static readonly description = "Convert a scalar value to a single-element array.";

  defaults() { return { value: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = Number(inputs.value ?? this._props.value ?? 0);
    return { output: { data: [value], shape: [1] } };
  }
}

export class ArrayToScalarNode extends BaseNode {
  static readonly nodeType = "lib.numpy.conversion.ArrayToScalar";
  static readonly title = "Array To Scalar";
  static readonly description = "Convert a single-element array to a scalar value.";

  defaults() { return { values: { data: [0], shape: [1] } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    return { output: arr.data[0] ?? 0 };
  }
}

export class ConvertToImageNode extends BaseNode {
  static readonly nodeType = "lib.numpy.conversion.ConvertToImage";
  static readonly title = "Convert To Image";
  static readonly description = "Convert array data to PIL Image format.";

  defaults() { return { values: { data: [], shape: [0] } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    if (arr.data.length === 0) throw new Error("The input array is not connected.");

    const ndim = arr.shape.length;
    if (ndim !== 2 && ndim !== 3) throw new Error("The array should have 2 or 3 dimensions (HxW or HxWxC).");

    let height: number, width: number, channels: number;
    if (ndim === 2) {
      [height, width] = arr.shape;
      channels = 1;
    } else {
      [height, width, channels] = arr.shape;
      if (channels !== 1 && channels !== 3 && channels !== 4) {
        throw new Error("The array channels should be either 1, 3, or 4.");
      }
    }

    const effectiveChannels = channels === 1 ? 1 : channels;
    const pixelData = new Uint8Array(height * width * effectiveChannels);
    for (let i = 0; i < arr.data.length; i++) {
      pixelData[i] = Math.round(Math.max(0, Math.min(1, arr.data[i])) * 255);
    }

    const pngBuffer = await sharp(pixelData, {
      raw: { width, height, channels: effectiveChannels as 1 | 3 | 4 },
    })
      .png()
      .toBuffer();

    return {
      output: {
        type: "image",
        uri: "",
        data: pngBuffer.toString("base64"),
      },
    };
  }
}

export class ConvertToAudioNode extends BaseNode {
  static readonly nodeType = "lib.numpy.conversion.ConvertToAudio";
  static readonly title = "Convert To Audio";
  static readonly description = "Converts a array object back to an audio file.";

  defaults() { return { values: { data: [], shape: [0] }, sample_rate: 44100 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const sampleRate = Number(inputs.sample_rate ?? this._props.sample_rate ?? 44100);
    const samples = new Float32Array(arr.data);
    const wav = encodeWav(samples, sampleRate);
    return { output: audioRefFromWav(wav) };
  }
}

export class ConvertToArrayNumpyNode extends BaseNode {
  static readonly nodeType = "lib.numpy.conversion.ConvertToArray";
  static readonly title = "Convert To Array";
  static readonly description = "Convert PIL Image to normalized tensor representation.";

  defaults() { return { image: { uri: "" } }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as Record<string, unknown>;

    let rawData: Uint8Array | null = null;
    if (image.data && typeof image.data === "string") {
      rawData = Uint8Array.from(Buffer.from(image.data, "base64"));
    }

    if (!rawData) throw new Error("The input image is not connected.");

    const metadata = await sharp(rawData).metadata();
    const { width, height, channels } = metadata;
    if (!width || !height) throw new Error("Could not read image dimensions.");

    const ch = channels ?? 3;
    const pixelBuf = await sharp(rawData)
      .ensureAlpha(ch === 4 ? undefined : undefined)
      .raw()
      .toBuffer();

    const actualChannels = pixelBuf.length / (width * height);
    const data: number[] = new Array(pixelBuf.length);
    for (let i = 0; i < pixelBuf.length; i++) {
      data[i] = pixelBuf[i] / 255.0;
    }

    const shape = actualChannels === 1
      ? [height, width, 1]
      : [height, width, actualChannels];

    return { output: { data, shape } };
  }
}

// ---------------------------------------------------------------------------
// IO (1 node)
// ---------------------------------------------------------------------------

export class SaveArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.io.SaveArray";
  static readonly title = "Save Array";
  static readonly description = "Save a numpy array to a file in the specified folder.";

  defaults() {
    return {
      values: { data: [], shape: [0] },
      folder: { asset_id: "" },
      name: "%Y-%m-%d_%H-%M-%S.json",
    };
  }

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    const nameTemplate = String(inputs.name ?? this._props.name ?? "array.json");

    const now = new Date();
    const filename = nameTemplate
      .replace(/%Y/g, String(now.getFullYear()))
      .replace(/%m/g, String(now.getMonth() + 1).padStart(2, "0"))
      .replace(/%d/g, String(now.getDate()).padStart(2, "0"))
      .replace(/%H/g, String(now.getHours()).padStart(2, "0"))
      .replace(/%M/g, String(now.getMinutes()).padStart(2, "0"))
      .replace(/%S/g, String(now.getSeconds()).padStart(2, "0"))
      .replace(/\.npy$/, ".json");

    const json = JSON.stringify({ data: arr.data, shape: arr.shape });

    if (context?.storage) {
      const uri = await context.storage.store(filename, new TextEncoder().encode(json), "application/json");
      return { output: { data: arr.data, shape: arr.shape, uri } };
    }

    return { output: { data: arr.data, shape: arr.shape } };
  }
}

// ---------------------------------------------------------------------------
// Utils (1 node)
// ---------------------------------------------------------------------------

export class BinaryOperationNode extends BaseNode {
  static readonly nodeType = "lib.numpy.utils.BinaryOperation";
  static readonly title = "Binary Operation";
  static readonly description = "";

  defaults() { return { a: 0, b: 0 }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asNdArray(inputs.a ?? this._props.a ?? 0);
    const b = asNdArray(inputs.b ?? this._props.b ?? 0);
    return convertOutput(elementwiseBinary(a, b, (x, y) => x + y));
  }
}

// ---------------------------------------------------------------------------
// Visualization (1 node)
// ---------------------------------------------------------------------------

export class PlotArrayNode extends BaseNode {
  static readonly nodeType = "lib.numpy.visualization.PlotArray";
  static readonly title = "Plot Array";
  static readonly description = "Create a plot visualization of array data.";

  defaults() { return { values: { data: [], shape: [0] }, plot_type: "line" }; }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const arr = asNdArray(inputs.values ?? this._props.values);
    if (arr.data.length === 0) throw new Error("Empty array");

    const ndim = arr.shape.length;
    let height: number, width: number;

    if (ndim === 2) {
      [height, width] = arr.shape;
    } else {
      height = 256;
      width = arr.data.length;
    }

    // Normalize data to 0-255 for grayscale visualization
    let min = Infinity, max = -Infinity;
    for (const v of arr.data) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;

    if (ndim === 2) {
      // 2D: render as grayscale image
      const pixels = new Uint8Array(height * width);
      for (let i = 0; i < arr.data.length; i++) {
        pixels[i] = Math.round(((arr.data[i] - min) / range) * 255);
      }
      const pngBuffer = await sharp(pixels, {
        raw: { width, height, channels: 1 },
      }).png().toBuffer();

      return {
        output: {
          type: "image",
          uri: "",
          data: pngBuffer.toString("base64"),
        },
      };
    }

    // 1D: simple line plot as image
    const imgWidth = Math.max(width, 400);
    const imgHeight = 256;
    const pixels = new Uint8Array(imgHeight * imgWidth).fill(255);

    for (let x = 0; x < arr.data.length && x < imgWidth; x++) {
      const normalized = (arr.data[x] - min) / range;
      const y = Math.round((1 - normalized) * (imgHeight - 1));
      const clampedY = Math.max(0, Math.min(imgHeight - 1, y));
      pixels[clampedY * imgWidth + x] = 0;
      // Draw a vertical line from baseline to point for visibility
      const baseline = imgHeight - 1;
      const startY = Math.min(clampedY, baseline);
      const endY = Math.max(clampedY, baseline);
      for (let py = startY; py <= endY; py++) {
        const existing = pixels[py * imgWidth + x];
        pixels[py * imgWidth + x] = Math.min(existing, 128);
      }
    }

    const pngBuffer = await sharp(pixels, {
      raw: { width: imgWidth, height: imgHeight, channels: 1 },
    }).png().toBuffer();

    return {
      output: {
        type: "image",
        uri: "",
        data: pngBuffer.toString("base64"),
      },
    };
  }
}

// ---------------------------------------------------------------------------
// Export all 38 nodes
// ---------------------------------------------------------------------------

export const LIB_NUMPY_NODES = [
  // arithmetic (5)
  AddArrayNode,
  SubtractArrayNode,
  MultiplyArrayNode,
  DivideArrayNode,
  ModulusArrayNode,
  // math (7)
  AbsArrayNode,
  SineArrayNode,
  CosineArrayNode,
  ExpArrayNode,
  LogArrayNode,
  SqrtArrayNode,
  PowerArrayNode,
  // statistics (6)
  SumArrayNode,
  MeanArrayNode,
  MinArrayNode,
  MaxArrayNode,
  ArgMinArrayNode,
  ArgMaxArrayNode,
  // manipulation (6)
  SliceArrayNode,
  IndexArrayNode,
  TransposeArrayNode,
  MatMulNode,
  StackNode,
  SplitArrayNode,
  // reshaping (4)
  Reshape1DNode,
  Reshape2DNode,
  Reshape3DNode,
  Reshape4DNode,
  // conversion (7)
  ListToArrayNode,
  ArrayToListNode,
  ScalarToArrayNode,
  ArrayToScalarNode,
  ConvertToImageNode,
  ConvertToAudioNode,
  ConvertToArrayNumpyNode,
  // io (1)
  SaveArrayNode,
  // utils (1)
  BinaryOperationNode,
  // visualization (1)
  PlotArrayNode,
] as const;
