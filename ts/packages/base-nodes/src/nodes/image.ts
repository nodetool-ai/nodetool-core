import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";
import path from "node:path";

type ImageRefLike = {
  uri?: string;
  data?: Uint8Array | string;
  mimeType?: string;
  width?: number;
  height?: number;
};

function toBytes(data: Uint8Array | string | undefined): Uint8Array {
  if (!data) return new Uint8Array();
  if (data instanceof Uint8Array) return data;
  return Uint8Array.from(Buffer.from(data, "base64"));
}

function imageBytes(image: unknown): Uint8Array {
  if (!image || typeof image !== "object") return new Uint8Array();
  return toBytes((image as ImageRefLike).data);
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

function imageRef(data: Uint8Array, extras: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    data: Buffer.from(data).toString("base64"),
    ...extras,
  };
}

export class LoadImageFileNode extends BaseNode {
  static readonly nodeType = "nodetool.image.LoadImageFile";
  static readonly title = "Load Image File";
  static readonly description = "Load image bytes from local file";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = filePath(String(inputs.path ?? this._props.path ?? ""));
    const data = new Uint8Array(await fs.readFile(p));
    return { output: imageRef(data, { uri: `file://${p}` }) };
  }
}

export class LoadImageFolderNode extends BaseNode {
  static readonly nodeType = "nodetool.image.LoadImageFolder";
  static readonly title = "Load Image Folder";
  static readonly description = "Stream image files from folder";
  static readonly isStreamingOutput = true;

  defaults() {
    return { folder: "." };
  }

  async process(): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const entries = await fs.readdir(folder, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      const ext = path.extname(entry.name).toLowerCase();
      if (![".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"].includes(ext)) continue;
      const full = path.join(folder, entry.name);
      const data = new Uint8Array(await fs.readFile(full));
      yield { image: imageRef(data, { uri: `file://${full}` }), name: entry.name };
    }
  }
}

export class SaveImageFileImageNode extends BaseNode {
  static readonly nodeType = "nodetool.image.SaveImageFile";
  static readonly title = "Save Image File";
  static readonly description = "Save image to local file";

  defaults() {
    return { image: {}, path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = filePath(String(inputs.path ?? this._props.path ?? ""));
    await fs.mkdir(path.dirname(p), { recursive: true });
    await fs.writeFile(p, imageBytes(inputs.image ?? this._props.image));
    return { output: p };
  }
}

export class LoadImageAssetsNode extends BaseNode {
  static readonly nodeType = "nodetool.image.LoadImageAssets";
  static readonly title = "Load Image Assets";
  static readonly description = "Alias for folder image loading";
  static readonly isStreamingOutput = true;

  defaults() {
    return { folder: "." };
  }

  async process(): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const loader = new LoadImageFolderNode();
    loader.assign({ folder: inputs.folder ?? this._props.folder ?? "." });
    for await (const item of loader.genProcess({})) {
      yield item;
    }
  }
}

export class SaveImageNode extends BaseNode {
  static readonly nodeType = "nodetool.image.SaveImage";
  static readonly title = "Save Image";
  static readonly description = "Save image bytes using folder/name";

  defaults() {
    return { image: {}, folder: ".", name: "image_%Y%m%d_%H%M%S.png" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const name = dateName(String(inputs.name ?? this._props.name ?? "image.png"));
    const full = path.resolve(folder, name);
    await fs.mkdir(path.dirname(full), { recursive: true });
    const bytes = imageBytes(inputs.image ?? this._props.image);
    await fs.writeFile(full, bytes);
    return { output: imageRef(bytes, { uri: `file://${full}` }) };
  }
}

export class GetMetadataNode extends BaseNode {
  static readonly nodeType = "nodetool.image.GetMetadata";
  static readonly title = "Get Metadata";
  static readonly description = "Return basic image metadata";

  defaults() {
    return { image: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const bytes = imageBytes(image);
    return {
      output: {
        uri: image.uri ?? "",
        mime_type: image.mimeType ?? "image/unknown",
        size_bytes: bytes.length,
        width: image.width ?? null,
        height: image.height ?? null,
      },
    };
  }
}

export class BatchToListNode extends BaseNode {
  static readonly nodeType = "nodetool.image.BatchToList";
  static readonly title = "Batch To List";
  static readonly description = "Convert batch input to image list";

  defaults() {
    return { batch: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const batch = inputs.batch ?? this._props.batch ?? [];
    if (Array.isArray(batch)) return { output: batch };
    return { output: [batch] };
  }
}

export class ImagesToListNode extends BaseNode {
  static readonly nodeType = "nodetool.image.ImagesToList";
  static readonly title = "Images To List";
  static readonly description = "Collect image inputs into a list";

  defaults() {
    return { image_a: null, image_b: null, images: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const explicit = Array.isArray(inputs.images ?? this._props.images)
      ? (inputs.images ?? this._props.images) as unknown[]
      : [];
    const out = [...explicit];
    const a = inputs.image_a ?? this._props.image_a;
    const b = inputs.image_b ?? this._props.image_b;
    if (a) out.push(a);
    if (b) out.push(b);
    return { output: out };
  }
}

abstract class TransformImageNode extends BaseNode {
  defaults() {
    return { image: {}, width: null, height: null };
  }

  protected transformMeta(inputs: Record<string, unknown>): Record<string, unknown> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    return {
      width: Number(inputs.width ?? this._props.width ?? image.width ?? 0) || null,
      height: Number(inputs.height ?? this._props.height ?? image.height ?? 0) || null,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const bytes = imageBytes(image);
    return {
      output: imageRef(bytes, {
        uri: image.uri ?? "",
        ...this.transformMeta(inputs),
      }),
    };
  }
}

export class PasteNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Paste";
  static readonly title = "Paste";
  static readonly description = "Paste image (metadata-level placeholder)";
}

export class ScaleNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Scale";
  static readonly title = "Scale";
  static readonly description = "Scale image (metadata-level placeholder)";
}

export class ResizeNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Resize";
  static readonly title = "Resize";
  static readonly description = "Resize image (metadata-level placeholder)";
}

export class CropNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Crop";
  static readonly title = "Crop";
  static readonly description = "Crop image (metadata-level placeholder)";
}

export class FitNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Fit";
  static readonly title = "Fit";
  static readonly description = "Fit image (metadata-level placeholder)";
}

export class TextToImageNode extends BaseNode {
  static readonly nodeType = "nodetool.image.TextToImage";
  static readonly title = "Text To Image";
  static readonly description = "Generate placeholder image bytes from text";

  defaults() {
    return { prompt: "", width: 512, height: 512 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const width = Number(inputs.width ?? this._props.width ?? 512);
    const height = Number(inputs.height ?? this._props.height ?? 512);
    const bytes = Uint8Array.from(Buffer.from(prompt, "utf8"));
    return {
      output: imageRef(bytes, {
        width,
        height,
      }),
    };
  }
}

export class ImageToImageNode extends BaseNode {
  static readonly nodeType = "nodetool.image.ImageToImage";
  static readonly title = "Image To Image";
  static readonly description = "Transform image with prompt (placeholder passthrough)";

  defaults() {
    return { image: {}, prompt: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const bytes = imageBytes(image);
    return {
      output: imageRef(bytes, {
        uri: image.uri ?? "",
        prompt: String(inputs.prompt ?? this._props.prompt ?? ""),
      }),
    };
  }
}

export const IMAGE_NODES = [
  LoadImageFileNode,
  LoadImageFolderNode,
  SaveImageFileImageNode,
  LoadImageAssetsNode,
  SaveImageNode,
  GetMetadataNode,
  BatchToListNode,
  ImagesToListNode,
  PasteNode,
  ScaleNode,
  ResizeNode,
  CropNode,
  FitNode,
  TextToImageNode,
  ImageToImageNode,
] as const;
