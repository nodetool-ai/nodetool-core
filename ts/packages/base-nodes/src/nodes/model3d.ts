import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";
import path from "node:path";

type Model3DRefLike = {
  uri?: string;
  data?: Uint8Array | string;
  format?: string;
  vertices?: number;
  faces?: number;
};

type ImageRefLike = { data?: Uint8Array | string; uri?: string };

function toBytes(data: Uint8Array | string | undefined): Uint8Array {
  if (!data) return new Uint8Array();
  if (data instanceof Uint8Array) return data;
  return Uint8Array.from(Buffer.from(data, "base64"));
}

function modelBytes(model: unknown): Uint8Array {
  if (!model || typeof model !== "object") return new Uint8Array();
  return toBytes((model as Model3DRefLike).data);
}

function modelRef(data: Uint8Array, extras: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    data: Buffer.from(data).toString("base64"),
    ...extras,
  };
}

function filePath(uriOrPath: string): string {
  if (uriOrPath.startsWith("file://")) return uriOrPath.slice("file://".length);
  return uriOrPath;
}

function dateName(name: string): string {
  const now = new Date();
  const pad = (v: number): string => String(v).padStart(2, "0");
  return name
    .replaceAll("%Y", String(now.getFullYear()))
    .replaceAll("%m", pad(now.getMonth() + 1))
    .replaceAll("%d", pad(now.getDate()))
    .replaceAll("%H", pad(now.getHours()))
    .replaceAll("%M", pad(now.getMinutes()))
    .replaceAll("%S", pad(now.getSeconds()));
}

function extFormat(filename: string): string {
  const ext = path.extname(filename).replace(".", "").toLowerCase();
  return ext || "glb";
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((sum, p) => sum + p.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

export class LoadModel3DFileNode extends BaseNode {
  static readonly nodeType = "nodetool.model3d.LoadModel3DFile";
  static readonly title = "Load Model3D File";
  static readonly description = "Load model bytes from local path.";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = filePath(String(inputs.path ?? this._props.path ?? ""));
    const data = new Uint8Array(await fs.readFile(p));
    return { output: modelRef(data, { uri: `file://${p}`, format: extFormat(p) }) };
  }
}

export class SaveModel3DFileNode extends BaseNode {
  static readonly nodeType = "nodetool.model3d.SaveModel3DFile";
  static readonly title = "Save Model3D File";
  static readonly description = "Write model bytes to local file.";

  defaults() {
    return { model: {}, folder: ".", filename: "model.glb" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const filename = String(inputs.filename ?? this._props.filename ?? "model.glb");
    const full = path.resolve(folder, filename);
    await fs.mkdir(path.dirname(full), { recursive: true });
    const bytes = modelBytes(inputs.model ?? this._props.model);
    await fs.writeFile(full, bytes);
    return { output: modelRef(bytes, { uri: `file://${full}`, format: extFormat(full) }) };
  }
}

export class SaveModel3DNode extends BaseNode {
  static readonly nodeType = "nodetool.model3d.SaveModel3D";
  static readonly title = "Save Model3D";
  static readonly description = "Save model bytes with timestamped filename.";

  defaults() {
    return { model: {}, folder: ".", name: "model_%Y%m%d_%H%M%S.glb" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const name = dateName(String(inputs.name ?? this._props.name ?? "model.glb"));
    const full = path.resolve(folder, name);
    await fs.mkdir(path.dirname(full), { recursive: true });
    const bytes = modelBytes(inputs.model ?? this._props.model);
    await fs.writeFile(full, bytes);
    return { output: modelRef(bytes, { uri: `file://${full}`, format: extFormat(full) }) };
  }
}

abstract class ModelTransformNode extends BaseNode {
  defaults() {
    return { model: {} };
  }

  protected getModel(inputs: Record<string, unknown>): Model3DRefLike {
    const value = inputs.model ?? this._props.model ?? {};
    if (!value || typeof value !== "object") return {};
    return value as Model3DRefLike;
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const model = this.getModel(inputs);
    const bytes = modelBytes(model);
    return {
      output: modelRef(bytes, {
        uri: model.uri ?? "",
        format: model.format ?? "glb",
      }),
    };
  }
}

export class FormatConverterNode extends ModelTransformNode {
  static readonly nodeType = "nodetool.model3d.FormatConverter";
  static readonly title = "Format Converter";
  static readonly description = "Change output format metadata for a model.";

  defaults() {
    return { model: {}, output_format: "glb" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const model = this.getModel(inputs);
    const bytes = modelBytes(model);
    const outputFormat = String(inputs.output_format ?? this._props.output_format ?? "glb");
    return {
      output: modelRef(bytes, {
        uri: model.uri ?? "",
        format: outputFormat,
      }),
    };
  }
}

export class GetModel3DMetadataNode extends BaseNode {
  static readonly nodeType = "nodetool.model3d.GetModel3DMetadata";
  static readonly title = "Get Model3D Metadata";
  static readonly description = "Return basic metadata for model bytes.";

  defaults() {
    return { model: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const model = (inputs.model ?? this._props.model ?? {}) as Model3DRefLike;
    const bytes = modelBytes(model);
    const vertices = model.vertices ?? Math.floor(bytes.length / 32);
    const faces = model.faces ?? Math.floor(vertices / 3);
    return {
      output: {
        uri: model.uri ?? "",
        format: model.format ?? "glb",
        size_bytes: bytes.length,
        vertices,
        faces,
      },
    };
  }
}

export class Transform3DNode extends ModelTransformNode {
  static readonly nodeType = "nodetool.model3d.Transform3D";
  static readonly title = "Transform3D";
  static readonly description = "Apply transform metadata to a model.";
}

export class DecimateNode extends BaseNode {
  static readonly nodeType = "nodetool.model3d.Decimate";
  static readonly title = "Decimate";
  static readonly description = "Reduce model byte size by ratio.";

  defaults() {
    return { model: {}, ratio: 0.5 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const model = (inputs.model ?? this._props.model ?? {}) as Model3DRefLike;
    const bytes = modelBytes(model);
    const ratio = Number(inputs.ratio ?? this._props.ratio ?? 0.5);
    const keep = Math.max(1, Math.floor(bytes.length * Math.max(0, Math.min(1, ratio))));
    return {
      output: modelRef(bytes.slice(0, keep), {
        uri: model.uri ?? "",
        format: model.format ?? "glb",
      }),
    };
  }
}

export class Boolean3DNode extends BaseNode {
  static readonly nodeType = "nodetool.model3d.Boolean3D";
  static readonly title = "Boolean3D";
  static readonly description = "Combine two models by operation (byte-level placeholder).";

  defaults() {
    return { model_a: {}, model_b: {}, operation: "union" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = modelBytes(inputs.model_a ?? this._props.model_a);
    const b = modelBytes(inputs.model_b ?? this._props.model_b);
    const operation = String(inputs.operation ?? this._props.operation ?? "union").toLowerCase();

    if (operation === "difference") {
      const len = Math.max(a.length, b.length);
      const out = new Uint8Array(len);
      for (let i = 0; i < len; i += 1) out[i] = Math.max(0, (a[i] ?? 0) - (b[i] ?? 0));
      return { output: modelRef(out, { format: "glb" }) };
    }

    if (operation === "intersection") {
      const len = Math.min(a.length, b.length);
      const out = new Uint8Array(len);
      for (let i = 0; i < len; i += 1) out[i] = Math.min(a[i] ?? 0, b[i] ?? 0);
      return { output: modelRef(out, { format: "glb" }) };
    }

    return { output: modelRef(concatBytes([a, b]), { format: "glb" }) };
  }
}

export class RecalculateNormalsNode extends ModelTransformNode {
  static readonly nodeType = "nodetool.model3d.RecalculateNormals";
  static readonly title = "Recalculate Normals";
  static readonly description = "Recalculate normals (placeholder passthrough).";
}

export class CenterMeshNode extends ModelTransformNode {
  static readonly nodeType = "nodetool.model3d.CenterMesh";
  static readonly title = "Center Mesh";
  static readonly description = "Center mesh (placeholder passthrough).";
}

export class FlipNormalsNode extends ModelTransformNode {
  static readonly nodeType = "nodetool.model3d.FlipNormals";
  static readonly title = "Flip Normals";
  static readonly description = "Flip normals (placeholder passthrough).";
}

export class MergeMeshesNode extends BaseNode {
  static readonly nodeType = "nodetool.model3d.MergeMeshes";
  static readonly title = "Merge Meshes";
  static readonly description = "Merge a list of models into one model.";

  defaults() {
    return { models: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = Array.isArray(inputs.models ?? this._props.models)
      ? (inputs.models ?? this._props.models) as unknown[]
      : [];
    const all = values.map((v) => modelBytes(v));
    return { output: modelRef(concatBytes(all), { format: "glb" }) };
  }
}

export class TextTo3DNode extends BaseNode {
  static readonly nodeType = "nodetool.model3d.TextTo3D";
  static readonly title = "Text To 3D";
  static readonly description = "Generate model bytes from text prompt.";

  defaults() {
    return { prompt: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.prompt ?? this._props.prompt ?? "");
    const bytes = Uint8Array.from(Buffer.from(text, "utf8"));
    return { output: modelRef(bytes, { format: "glb" }) };
  }
}

export class ImageTo3DNode extends BaseNode {
  static readonly nodeType = "nodetool.model3d.ImageTo3D";
  static readonly title = "Image To 3D";
  static readonly description = "Generate model bytes from image content.";

  defaults() {
    return { image: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const bytes = toBytes(image.data);
    return { output: modelRef(bytes, { format: "glb" }) };
  }
}

export const MODEL3D_NODES = [
  LoadModel3DFileNode,
  SaveModel3DFileNode,
  SaveModel3DNode,
  FormatConverterNode,
  GetModel3DMetadataNode,
  Transform3DNode,
  DecimateNode,
  Boolean3DNode,
  RecalculateNormalsNode,
  CenterMeshNode,
  FlipNormalsNode,
  MergeMeshesNode,
  TextTo3DNode,
  ImageTo3DNode,
] as const;
