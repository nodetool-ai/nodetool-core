import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";
import path from "node:path";

type VideoRefLike = { uri?: string; data?: Uint8Array | string };
type ImageRefLike = { uri?: string; data?: Uint8Array | string };
type AudioRefLike = { uri?: string; data?: Uint8Array | string };

function toBytes(data: Uint8Array | string | undefined): Uint8Array {
  if (!data) return new Uint8Array();
  if (data instanceof Uint8Array) return data;
  return Uint8Array.from(Buffer.from(data, "base64"));
}

function videoBytes(video: unknown): Uint8Array {
  if (!video || typeof video !== "object") return new Uint8Array();
  return toBytes((video as VideoRefLike).data);
}

function imageBytes(image: unknown): Uint8Array {
  if (!image || typeof image !== "object") return new Uint8Array();
  return toBytes((image as ImageRefLike).data);
}

function audioBytes(audio: unknown): Uint8Array {
  if (!audio || typeof audio !== "object") return new Uint8Array();
  return toBytes((audio as AudioRefLike).data);
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

function bytesConcat(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

function videoRef(data: Uint8Array, extras: Record<string, unknown> = {}): Record<string, unknown> {
  return { data: Buffer.from(data).toString("base64"), ...extras };
}

export class TextToVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.TextToVideo";
  static readonly title = "Text To Video";
  static readonly description = "Generate placeholder video bytes from text";

  defaults() {
    return { prompt: "", fps: 24, duration: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.prompt ?? this._props.prompt ?? "");
    const bytes = Uint8Array.from(Buffer.from(text, "utf8"));
    return { output: videoRef(bytes) };
  }
}

export class ImageToVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.ImageToVideo";
  static readonly title = "Image To Video";
  static readonly description = "Generate placeholder video from image bytes";

  defaults() {
    return { image: {}, prompt: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const img = imageBytes(inputs.image ?? this._props.image);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    return { output: videoRef(bytesConcat([img, Uint8Array.from(Buffer.from(prompt))])) };
  }
}

export class LoadVideoFileNode extends BaseNode {
  static readonly nodeType = "nodetool.video.LoadVideoFile";
  static readonly title = "Load Video File";
  static readonly description = "Load video bytes from local path";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = filePath(String(inputs.path ?? this._props.path ?? ""));
    const data = new Uint8Array(await fs.readFile(p));
    return { output: videoRef(data, { uri: `file://${p}` }) };
  }
}

export class SaveVideoFileVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.SaveVideoFile";
  static readonly title = "Save Video File";
  static readonly description = "Save video bytes to local path";

  defaults() {
    return { video: {}, path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = filePath(String(inputs.path ?? this._props.path ?? ""));
    await fs.mkdir(path.dirname(p), { recursive: true });
    await fs.writeFile(p, videoBytes(inputs.video ?? this._props.video));
    return { output: p };
  }
}

export class LoadVideoAssetsNode extends BaseNode {
  static readonly nodeType = "nodetool.video.LoadVideoAssets";
  static readonly title = "Load Video Assets";
  static readonly description = "Stream video files from folder";
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
      if (![".mp4", ".mov", ".webm", ".mkv", ".avi"].includes(ext)) continue;
      const full = path.join(folder, entry.name);
      const data = new Uint8Array(await fs.readFile(full));
      yield { video: videoRef(data, { uri: `file://${full}` }), name: entry.name };
    }
  }
}

export class SaveVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.SaveVideo";
  static readonly title = "Save Video";
  static readonly description = "Save video with folder/filename template";

  defaults() {
    return { video: {}, folder: ".", filename: "video_%Y%m%d_%H%M%S.mp4" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const filename = dateName(String(inputs.filename ?? this._props.filename ?? "video.mp4"));
    const full = path.resolve(folder, filename);
    await fs.mkdir(path.dirname(full), { recursive: true });
    const bytes = videoBytes(inputs.video ?? this._props.video);
    await fs.writeFile(full, bytes);
    return { output: videoRef(bytes, { uri: `file://${full}` }) };
  }
}

export class FrameIteratorNode extends BaseNode {
  static readonly nodeType = "nodetool.video.FrameIterator";
  static readonly title = "Frame Iterator";
  static readonly description = "Stream pseudo-frames from video bytes";
  static readonly isStreamingOutput = true;

  defaults() {
    return { video: {}, frame_size: 1024 };
  }

  async process(): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const bytes = videoBytes(inputs.video ?? this._props.video);
    const frameSize = Math.max(1, Number(inputs.frame_size ?? this._props.frame_size ?? 1024));
    let index = 0;
    for (let i = 0; i < bytes.length; i += frameSize) {
      const frame = bytes.slice(i, i + frameSize);
      yield { frame: { data: Buffer.from(frame).toString("base64") }, index };
      index += 1;
    }
  }
}

export class FpsNode extends BaseNode {
  static readonly nodeType = "nodetool.video.Fps";
  static readonly title = "FPS";
  static readonly description = "Attach FPS metadata (placeholder)";

  defaults() {
    return { video: {}, fps: 24 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const bytes = videoBytes(inputs.video ?? this._props.video);
    const fps = Number(inputs.fps ?? this._props.fps ?? 24);
    return { output: videoRef(bytes, { fps }) };
  }
}

export class FrameToVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.FrameToVideo";
  static readonly title = "Frame To Video";
  static readonly description = "Combine frame list into video bytes";

  defaults() {
    return { frames: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const frames = Array.isArray(inputs.frames ?? this._props.frames)
      ? (inputs.frames ?? this._props.frames) as unknown[]
      : [];
    const parts = frames.map((f) => {
      if (!f || typeof f !== "object") return new Uint8Array();
      return toBytes((f as { data?: Uint8Array | string }).data);
    });
    return { output: videoRef(bytesConcat(parts)) };
  }
}

abstract class VideoTransformNode extends BaseNode {
  defaults() {
    return { video: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const bytes = videoBytes(inputs.video ?? this._props.video);
    return { output: videoRef(bytes) };
  }
}

export class ConcatVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.Concat";
  static readonly title = "Concat";
  static readonly description = "Concatenate two videos";

  defaults() {
    return { video_a: {}, video_b: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = videoBytes(inputs.video_a ?? this._props.video_a);
    const b = videoBytes(inputs.video_b ?? this._props.video_b);
    return { output: videoRef(bytesConcat([a, b])) };
  }
}

export class TrimVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.Trim";
  static readonly title = "Trim";
  static readonly description = "Trim bytes from start/end";

  defaults() {
    return { video: {}, start: 0, end: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const bytes = videoBytes(inputs.video ?? this._props.video);
    const start = Math.max(0, Number(inputs.start ?? this._props.start ?? 0));
    const end = Math.max(0, Number(inputs.end ?? this._props.end ?? 0));
    return { output: videoRef(bytes.slice(start, Math.max(start, bytes.length - end))) };
  }
}

export class ResizeVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.ResizeNode";
  static readonly title = "Resize";
  static readonly description = "Resize video (placeholder)";
}

export class RotateVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.Rotate";
  static readonly title = "Rotate";
  static readonly description = "Rotate video (placeholder)";
}

export class SetSpeedVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.SetSpeed";
  static readonly title = "Set Speed";
  static readonly description = "Set playback speed (placeholder)";
}

export class OverlayVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.Overlay";
  static readonly title = "Overlay";
  static readonly description = "Overlay video (placeholder)";
}

export class ColorBalanceVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.ColorBalance";
  static readonly title = "Color Balance";
  static readonly description = "Color balance transform (placeholder)";
}

export class DenoiseVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.Denoise";
  static readonly title = "Denoise";
  static readonly description = "Denoise video (placeholder)";
}

export class StabilizeVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.Stabilize";
  static readonly title = "Stabilize";
  static readonly description = "Stabilize video (placeholder)";
}

export class SharpnessVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.Sharpness";
  static readonly title = "Sharpness";
  static readonly description = "Sharpness transform (placeholder)";
}

export class BlurVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.Blur";
  static readonly title = "Blur";
  static readonly description = "Blur transform (placeholder)";
}

export class SaturationVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.Saturation";
  static readonly title = "Saturation";
  static readonly description = "Saturation transform (placeholder)";
}

export class AddSubtitlesVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.AddSubtitles";
  static readonly title = "Add Subtitles";
  static readonly description = "Add subtitles (placeholder)";
}

export class ReverseVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.Reverse";
  static readonly title = "Reverse";
  static readonly description = "Reverse video bytes";

  defaults() {
    return { video: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const bytes = videoBytes(inputs.video ?? this._props.video);
    return { output: videoRef(new Uint8Array([...bytes].reverse())) };
  }
}

export class TransitionVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.Transition";
  static readonly title = "Transition";
  static readonly description = "Apply transition (placeholder)";
}

export class AddAudioVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.AddAudio";
  static readonly title = "Add Audio";
  static readonly description = "Mux video and audio bytes";

  defaults() {
    return { video: {}, audio: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const v = videoBytes(inputs.video ?? this._props.video);
    const a = audioBytes(inputs.audio ?? this._props.audio);
    return { output: videoRef(bytesConcat([v, a])) };
  }
}

export class ChromaKeyVideoNode extends VideoTransformNode {
  static readonly nodeType = "nodetool.video.ChromaKey";
  static readonly title = "Chroma Key";
  static readonly description = "Apply chroma key (placeholder)";
}

export class ExtractAudioVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.ExtractAudio";
  static readonly title = "Extract Audio";
  static readonly description = "Extract pseudo-audio from video bytes";

  defaults() {
    return { video: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const bytes = videoBytes(inputs.video ?? this._props.video);
    const half = Math.floor(bytes.length / 2);
    return {
      output: {
        data: Buffer.from(bytes.slice(0, half)).toString("base64"),
      },
    };
  }
}

export class ExtractFrameVideoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.ExtractFrame";
  static readonly title = "Extract Frame";
  static readonly description = "Extract pseudo-image frame from video bytes";

  defaults() {
    return { video: {}, frame_index: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const bytes = videoBytes(inputs.video ?? this._props.video);
    const frameSize = 1024;
    const index = Math.max(0, Number(inputs.frame_index ?? this._props.frame_index ?? 0));
    const start = index * frameSize;
    const frame = bytes.slice(start, start + frameSize);
    return {
      output: {
        data: Buffer.from(frame).toString("base64"),
      },
    };
  }
}

export class GetVideoInfoNode extends BaseNode {
  static readonly nodeType = "nodetool.video.GetVideoInfo";
  static readonly title = "Get Video Info";
  static readonly description = "Return basic video metadata";

  defaults() {
    return { video: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const video = (inputs.video ?? this._props.video ?? {}) as VideoRefLike;
    const bytes = videoBytes(video);
    return {
      output: {
        uri: video.uri ?? "",
        size_bytes: bytes.length,
        fps: 24,
        duration_seconds: bytes.length / 24000,
      },
    };
  }
}

export const VIDEO_NODES = [
  TextToVideoNode,
  ImageToVideoNode,
  LoadVideoFileNode,
  SaveVideoFileVideoNode,
  LoadVideoAssetsNode,
  SaveVideoNode,
  FrameIteratorNode,
  FpsNode,
  FrameToVideoNode,
  ConcatVideoNode,
  TrimVideoNode,
  ResizeVideoNode,
  RotateVideoNode,
  SetSpeedVideoNode,
  OverlayVideoNode,
  ColorBalanceVideoNode,
  DenoiseVideoNode,
  StabilizeVideoNode,
  SharpnessVideoNode,
  BlurVideoNode,
  SaturationVideoNode,
  AddSubtitlesVideoNode,
  ReverseVideoNode,
  TransitionVideoNode,
  AddAudioVideoNode,
  ChromaKeyVideoNode,
  ExtractAudioVideoNode,
  ExtractFrameVideoNode,
  GetVideoInfoNode,
] as const;
