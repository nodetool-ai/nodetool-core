import { BaseNode } from "@nodetool/node-sdk";
import type { ProcessingContext } from "@nodetool/runtime";
import { promises as fs } from "node:fs";
import path from "node:path";
import sharp from "sharp";

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

async function imageBytesAsync(image: unknown): Promise<Uint8Array> {
  if (!image || typeof image !== "object") return new Uint8Array();
  const ref = image as ImageRefLike;
  if (ref.data) return toBytes(ref.data);
  if (typeof ref.uri === "string" && ref.uri) {
    if (ref.uri.startsWith("file://")) {
      return new Uint8Array(await fs.readFile(filePath(ref.uri)));
    }
    const response = await fetch(ref.uri);
    return new Uint8Array(await response.arrayBuffer());
  }
  return new Uint8Array();
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

function inferImageMime(uri: string | undefined, bytes: Uint8Array): string {
  const lower = (uri ?? "").toLowerCase();
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".webp")) return "image/webp";
  if (lower.endsWith(".gif")) return "image/gif";
  if (lower.endsWith(".bmp")) return "image/bmp";
  if (bytes.length < 4) return "image/unknown";
  if (bytes[0] === 0xff && bytes[1] === 0xd8) return "image/jpeg";
  if (bytes[0] === 0x47 && bytes[1] === 0x49) return "image/gif";
  if (bytes[0] === 0x42 && bytes[1] === 0x4d) return "image/bmp";
  if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[8] === 0x57) return "image/webp";
  if (
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  ) {
    return "image/png";
  }
  return "image/unknown";
}

function getModelConfig(
  inputs: Record<string, unknown>,
  props: Record<string, unknown>
): { providerId: string; modelId: string } {
  const model = (inputs.model ?? props.model ?? {}) as Record<string, unknown>;
  return {
    providerId: typeof model.provider === "string" ? model.provider : "",
    modelId: typeof model.id === "string" ? model.id : "",
  };
}

function hasProviderSupport(
  context: ProcessingContext | undefined,
  providerId: string,
  modelId: string
): context is ProcessingContext & { runProviderPrediction: (req: Record<string, unknown>) => Promise<unknown> } {
  return !!context && typeof context.runProviderPrediction === "function" && !!providerId && !!modelId;
}

async function metadataFor(bytes: Uint8Array): Promise<{ width: number | null; height: number | null }> {
  try {
    const md = await sharp(bytes).metadata();
    return {
      width: md.width ?? null,
      height: md.height ?? null,
    };
  } catch {
    return { width: null, height: null };
  }
}

async function transformImage(
  image: ImageRefLike,
  operation: (instance: sharp.Sharp, bytes: Uint8Array) => sharp.Sharp
): Promise<Record<string, unknown>> {
  const bytes = await imageBytesAsync(image);
  if (bytes.length === 0) {
    return imageRef(bytes, {
      uri: image.uri ?? "",
      width: image.width ?? null,
      height: image.height ?? null,
    });
  }

  try {
    const outputBytes = await operation(sharp(bytes, { failOn: "none" }), bytes).toBuffer();
    const meta = await metadataFor(outputBytes);
    return imageRef(outputBytes, {
      uri: image.uri ?? "",
      mimeType: inferImageMime(image.uri, outputBytes),
      width: meta.width,
      height: meta.height,
    });
  } catch {
    return imageRef(bytes, {
      uri: image.uri ?? "",
      width: image.width ?? null,
      height: image.height ?? null,
    });
  }
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
    const meta = await metadataFor(data);
    return {
      output: imageRef(data, {
        uri: `file://${p}`,
        mimeType: inferImageMime(p, data),
        width: meta.width,
        height: meta.height,
      }),
    };
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
      const meta = await metadataFor(data);
      yield {
        image: imageRef(data, {
          uri: `file://${full}`,
          mimeType: inferImageMime(full, data),
          width: meta.width,
          height: meta.height,
        }),
        name: entry.name,
      };
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
    const bytes = await imageBytesAsync(image);
    const meta = await metadataFor(bytes);
    return {
      output: {
        uri: image.uri ?? "",
        mime_type: image.mimeType ?? inferImageMime(image.uri, bytes),
        size_bytes: bytes.length,
        width: image.width ?? meta.width,
        height: image.height ?? meta.height,
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
    return {
      image: {},
      width: null,
      height: null,
      paste: {},
      left: 0,
      top: 0,
      scale: 1,
      right: null,
      bottom: null,
    };
  }

  protected transformMeta(inputs: Record<string, unknown>): Record<string, unknown> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    return {
      width: Number(inputs.width ?? this._props.width ?? image.width ?? 0) || null,
      height: Number(inputs.height ?? this._props.height ?? image.height ?? 0) || null,
    };
  }
}

export class PasteNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Paste";
  static readonly title = "Paste";
  static readonly description = "Paste one image onto another at specified coordinates";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const paste = (inputs.paste ?? this._props.paste ?? {}) as ImageRefLike;
    const left = Math.max(0, Number(inputs.left ?? this._props.left ?? 0));
    const top = Math.max(0, Number(inputs.top ?? this._props.top ?? 0));
    const baseBytes = await imageBytesAsync(image);
    const overlayBytes = await imageBytesAsync(paste);

    if (baseBytes.length === 0 || overlayBytes.length === 0) {
      return {
        output: imageRef(baseBytes, {
          uri: image.uri ?? "",
          ...this.transformMeta(inputs),
        }),
      };
    }

    try {
      const outputBytes = await sharp(baseBytes, { failOn: "none" })
        .composite([{ input: Buffer.from(overlayBytes), left, top }])
        .toBuffer();
      const meta = await metadataFor(outputBytes);
      return {
        output: imageRef(outputBytes, {
          uri: image.uri ?? "",
          mimeType: inferImageMime(image.uri, outputBytes),
          width: meta.width,
          height: meta.height,
        }),
      };
    } catch {
      return {
        output: imageRef(baseBytes, {
          uri: image.uri ?? "",
          ...this.transformMeta(inputs),
        }),
      };
    }
  }
}

export class ScaleNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Scale";
  static readonly title = "Scale";
  static readonly description = "Scale image by a factor";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const requestedScale = Number(inputs.scale ?? this._props.scale ?? 0);
    const targetWidth = Number(inputs.width ?? this._props.width ?? 0);
    const targetHeight = Number(inputs.height ?? this._props.height ?? 0);
    const scale =
      requestedScale > 0
        ? requestedScale
        : targetWidth > 0 && (image.width ?? 0) > 0
          ? targetWidth / Number(image.width)
          : targetHeight > 0 && (image.height ?? 0) > 0
            ? targetHeight / Number(image.height)
            : 1;
    const output = (await transformImage(image, (instance) => {
      const fallbackWidth = image.width ?? 1;
      const fallbackHeight = image.height ?? 1;
      return instance.resize({
        width: Math.max(1, Math.round(fallbackWidth * scale)),
        height: Math.max(1, Math.round(fallbackHeight * scale)),
      });
    })) as Record<string, unknown>;
    const fallbackWidth =
      targetWidth > 0
        ? targetWidth
        : image.width != null
          ? Math.max(1, Math.round(Number(image.width) * scale))
          : null;
    const fallbackHeight =
      targetHeight > 0
        ? targetHeight
        : image.height != null
          ? Math.max(1, Math.round(Number(image.height) * scale))
          : null;
    return {
      output: {
        ...output,
        width: fallbackWidth ?? output.width,
        height: fallbackHeight ?? output.height,
      },
    };
  }
}

export class ResizeNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Resize";
  static readonly title = "Resize";
  static readonly description = "Resize image to target dimensions";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const width = Number(inputs.width ?? this._props.width ?? image.width ?? 0) || null;
    const height = Number(inputs.height ?? this._props.height ?? image.height ?? 0) || null;
    const output = (await transformImage(image, (instance) =>
      instance.resize(width ?? undefined, height ?? undefined)
    )) as Record<string, unknown>;
    return {
      output: {
        ...output,
        width: output.width ?? width,
        height: output.height ?? height,
      },
    };
  }
}

export class CropNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Crop";
  static readonly title = "Crop";
  static readonly description = "Crop image to specified bounds";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const left = Math.max(0, Number(inputs.left ?? this._props.left ?? 0));
    const top = Math.max(0, Number(inputs.top ?? this._props.top ?? 0));
    const right = Number(
      inputs.right ?? this._props.right ?? inputs.width ?? this._props.width ?? image.width ?? 0
    );
    const bottom = Number(
      inputs.bottom ?? this._props.bottom ?? inputs.height ?? this._props.height ?? image.height ?? 0
    );
    const width = Math.max(1, right - left);
    const height = Math.max(1, bottom - top);
    const output = (await transformImage(image, (instance) =>
      instance.extract({ left, top, width, height })
    )) as Record<string, unknown>;
    return {
      output: {
        ...output,
        width: output.width ?? width,
        height: output.height ?? height,
      },
    };
  }
}

export class FitNode extends TransformImageNode {
  static readonly nodeType = "nodetool.image.Fit";
  static readonly title = "Fit";
  static readonly description = "Fit image to target dimensions while preserving aspect ratio";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const width = Math.max(1, Number(inputs.width ?? this._props.width ?? image.width ?? 512));
    const height = Math.max(1, Number(inputs.height ?? this._props.height ?? image.height ?? 512));
    const output = (await transformImage(image, (instance) =>
      instance.resize(width, height, { fit: "cover", position: "centre" })
    )) as Record<string, unknown>;
    return {
      output: {
        ...output,
        width: output.width ?? width,
        height: output.height ?? height,
      },
    };
  }
}

export class TextToImageNode extends BaseNode {
  static readonly nodeType = "nodetool.image.TextToImage";
  static readonly title = "Text To Image";
  static readonly description = "Generate placeholder image bytes from text";

  defaults() {
    return { prompt: "", width: 512, height: 512 };
  }

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const width = Number(inputs.width ?? this._props.width ?? 512);
    const height = Number(inputs.height ?? this._props.height ?? 512);
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const output = (await context.runProviderPrediction({
        provider: providerId,
        capability: "text_to_image",
        model: modelId,
        params: {
          prompt,
          width,
          height,
          negative_prompt: inputs.negative_prompt ?? this._props.negative_prompt,
          quality: inputs.quality ?? this._props.quality,
        },
      })) as Uint8Array;
      const meta = await metadataFor(output);
      return {
        output: imageRef(output, {
          mimeType: inferImageMime(undefined, output),
          width: meta.width ?? width,
          height: meta.height ?? height,
        }),
      };
    }
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

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const image = (inputs.image ?? this._props.image ?? {}) as ImageRefLike;
    const bytes = await imageBytesAsync(image);
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const output = (await context.runProviderPrediction({
        provider: providerId,
        capability: "image_to_image",
        model: modelId,
        params: {
          image: bytes,
          prompt: String(inputs.prompt ?? this._props.prompt ?? ""),
          negative_prompt: inputs.negative_prompt ?? this._props.negative_prompt,
          target_width: inputs.target_width ?? this._props.target_width,
          target_height: inputs.target_height ?? this._props.target_height,
          quality: inputs.quality ?? this._props.quality,
        },
      })) as Uint8Array;
      const meta = await metadataFor(output);
      return {
        output: imageRef(output, {
          uri: image.uri ?? "",
          mimeType: inferImageMime(image.uri, output),
          width: meta.width,
          height: meta.height,
        }),
      };
    }
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
