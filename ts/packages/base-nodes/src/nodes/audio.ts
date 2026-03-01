import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";
import path from "node:path";

type AudioRefLike = {
  uri?: string;
  data?: Uint8Array | string;
};

type ImageLike = {
  data?: Uint8Array | string;
  uri?: string;
};

function toBytes(value: Uint8Array | string | undefined): Uint8Array {
  if (!value) return new Uint8Array();
  if (value instanceof Uint8Array) return value;
  return Uint8Array.from(Buffer.from(value, "base64"));
}

function audioBytes(audio: unknown): Uint8Array {
  if (!audio || typeof audio !== "object") return new Uint8Array();
  const ref = audio as AudioRefLike;
  if (ref.data) return toBytes(ref.data);
  return new Uint8Array();
}

function uriToPath(uriOrPath: string): string {
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

function audioRefFromBytes(data: Uint8Array, uri?: string): Record<string, unknown> {
  return {
    uri: uri ?? "",
    data: Buffer.from(data).toString("base64"),
  };
}

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

export class LoadAudioAssetsNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.LoadAudioAssets";
  static readonly title = "Load Audio Assets";
  static readonly description = "Load audio files from folder";
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
      if (![".wav", ".mp3", ".m4a", ".flac", ".ogg"].includes(ext)) continue;
      const full = path.join(folder, entry.name);
      const data = new Uint8Array(await fs.readFile(full));
      yield {
        audio: audioRefFromBytes(data, `file://${full}`),
        name: entry.name,
      };
    }
  }
}

export class LoadAudioFileNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.LoadAudioFile";
  static readonly title = "Load Audio File";
  static readonly description = "Load audio from local file";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = uriToPath(String(inputs.path ?? this._props.path ?? ""));
    const data = new Uint8Array(await fs.readFile(p));
    return { output: audioRefFromBytes(data, `file://${p}`) };
  }
}

export class LoadAudioFolderNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.LoadAudioFolder";
  static readonly title = "Load Audio Folder";
  static readonly description = "Alias for streaming audio folder load";
  static readonly isStreamingOutput = true;

  defaults() {
    return { folder: "." };
  }

  async process(): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const loader = new LoadAudioAssetsNode();
    loader.assign({ folder: inputs.folder ?? this._props.folder ?? "." });
    for await (const item of loader.genProcess({})) {
      yield item;
    }
  }
}

export class SaveAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.SaveAudio";
  static readonly title = "Save Audio";
  static readonly description = "Save audio data to folder with generated name";

  defaults() {
    return { audio: {}, folder: ".", name: "audio_%Y%m%d_%H%M%S.wav" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = inputs.audio ?? this._props.audio;
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const name = dateName(String(inputs.name ?? this._props.name ?? "audio.wav"));
    const full = path.resolve(folder, name);
    await fs.mkdir(path.dirname(full), { recursive: true });
    await fs.writeFile(full, audioBytes(audio));
    return { output: audioRefFromBytes(audioBytes(audio), `file://${full}`) };
  }
}

export class SaveAudioFileNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.SaveAudioFile";
  static readonly title = "Save Audio File";
  static readonly description = "Save audio data to file path";

  defaults() {
    return { audio: {}, path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = inputs.audio ?? this._props.audio;
    const p = uriToPath(String(inputs.path ?? this._props.path ?? ""));
    await fs.mkdir(path.dirname(p), { recursive: true });
    await fs.writeFile(p, audioBytes(audio));
    return { output: p };
  }
}

export class NormalizeAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.Normalize";
  static readonly title = "Normalize";
  static readonly description = "Return normalized audio (placeholder passthrough)";

  defaults() {
    return { audio: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = inputs.audio ?? this._props.audio;
    return { output: audioRefFromBytes(audioBytes(audio)) };
  }
}

export class OverlayAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.OverlayAudio";
  static readonly title = "Overlay Audio";
  static readonly description = "Overlay audio by byte-wise max mix";

  defaults() {
    return { audio_a: {}, audio_b: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = audioBytes(inputs.audio_a ?? this._props.audio_a);
    const b = audioBytes(inputs.audio_b ?? this._props.audio_b);
    const len = Math.max(a.length, b.length);
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) {
      out[i] = Math.max(a[i] ?? 0, b[i] ?? 0);
    }
    return { output: audioRefFromBytes(out) };
  }
}

export class RemoveSilenceNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.RemoveSilence";
  static readonly title = "Remove Silence";
  static readonly description = "Remove zero bytes";

  defaults() {
    return { audio: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = audioBytes(inputs.audio ?? this._props.audio);
    const filtered = data.filter((v) => v !== 0);
    return { output: audioRefFromBytes(filtered) };
  }
}

export class SliceAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.SliceAudio";
  static readonly title = "Slice Audio";
  static readonly description = "Slice by byte range";

  defaults() {
    return { audio: {}, start: 0, end: -1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = audioBytes(inputs.audio ?? this._props.audio);
    const start = Number(inputs.start ?? this._props.start ?? 0);
    let end = Number(inputs.end ?? this._props.end ?? -1);
    if (end < 0) end = data.length;
    return { output: audioRefFromBytes(data.slice(start, end)) };
  }
}

export class MonoToStereoNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.MonoToStereo";
  static readonly title = "Mono To Stereo";
  static readonly description = "Duplicate bytes as stereo pairs";

  defaults() {
    return { audio: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const mono = audioBytes(inputs.audio ?? this._props.audio);
    const out = new Uint8Array(mono.length * 2);
    for (let i = 0; i < mono.length; i += 1) {
      out[i * 2] = mono[i];
      out[i * 2 + 1] = mono[i];
    }
    return { output: audioRefFromBytes(out) };
  }
}

export class StereoToMonoNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.StereoToMono";
  static readonly title = "Stereo To Mono";
  static readonly description = "Take left channel from stereo pairs";

  defaults() {
    return { audio: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const stereo = audioBytes(inputs.audio ?? this._props.audio);
    const out = new Uint8Array(Math.ceil(stereo.length / 2));
    for (let i = 0, j = 0; i < stereo.length; i += 2, j += 1) {
      out[j] = stereo[i];
    }
    return { output: audioRefFromBytes(out) };
  }
}

export class ReverseAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.Reverse";
  static readonly title = "Reverse";
  static readonly description = "Reverse audio bytes";

  defaults() {
    return { audio: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = audioBytes(inputs.audio ?? this._props.audio);
    return { output: audioRefFromBytes(new Uint8Array([...data].reverse())) };
  }
}

export class FadeInAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.FadeIn";
  static readonly title = "Fade In";
  static readonly description = "Simple linear fade-in over first bytes";

  defaults() {
    return { audio: {}, duration: 1024 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = new Uint8Array(audioBytes(inputs.audio ?? this._props.audio));
    const duration = Math.max(1, Number(inputs.duration ?? this._props.duration ?? 1024));
    for (let i = 0; i < Math.min(duration, data.length); i += 1) {
      data[i] = Math.floor(data[i] * (i / duration));
    }
    return { output: audioRefFromBytes(data) };
  }
}

export class FadeOutAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.FadeOut";
  static readonly title = "Fade Out";
  static readonly description = "Simple linear fade-out over last bytes";

  defaults() {
    return { audio: {}, duration: 1024 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = new Uint8Array(audioBytes(inputs.audio ?? this._props.audio));
    const duration = Math.max(1, Number(inputs.duration ?? this._props.duration ?? 1024));
    const start = Math.max(0, data.length - duration);
    for (let i = start; i < data.length; i += 1) {
      const factor = (data.length - i) / Math.max(1, data.length - start);
      data[i] = Math.floor(data[i] * factor);
    }
    return { output: audioRefFromBytes(data) };
  }
}

export class RepeatAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.Repeat";
  static readonly title = "Repeat";
  static readonly description = "Repeat audio bytes";

  defaults() {
    return { audio: {}, count: 2 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = audioBytes(inputs.audio ?? this._props.audio);
    const count = Math.max(1, Number(inputs.count ?? this._props.count ?? 2));
    return { output: audioRefFromBytes(concatBytes(Array.from({ length: count }, () => data))) };
  }
}

export class AudioMixerNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.AudioMixer";
  static readonly title = "Audio Mixer";
  static readonly description = "Mix list of audio refs by averaging bytes";

  defaults() {
    return { audios: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audios = Array.isArray(inputs.audios ?? this._props.audios)
      ? (inputs.audios ?? this._props.audios) as unknown[]
      : [];
    const all = audios.map((a) => audioBytes(a));
    if (all.length === 0) return { output: audioRefFromBytes(new Uint8Array()) };
    const len = Math.max(...all.map((x) => x.length));
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) {
      let total = 0;
      for (const a of all) total += a[i] ?? 0;
      out[i] = Math.floor(total / all.length);
    }
    return { output: audioRefFromBytes(out) };
  }
}

export class AudioToNumpyNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.AudioToNumpy";
  static readonly title = "Audio To Numpy";
  static readonly description = "Convert audio bytes to number list";

  defaults() {
    return { audio: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = audioBytes(inputs.audio ?? this._props.audio);
    return { output: Array.from(data) };
  }
}

export class NumpyToAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.NumpyToAudio";
  static readonly title = "Numpy To Audio";
  static readonly description = "Convert number list to audio bytes";

  defaults() {
    return { values: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = Array.isArray(inputs.values ?? this._props.values)
      ? (inputs.values ?? this._props.values) as unknown[]
      : [];
    const bytes = new Uint8Array(values.map((v) => Number(v) & 0xff));
    return { output: audioRefFromBytes(bytes) };
  }
}

export class TrimAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.Trim";
  static readonly title = "Trim";
  static readonly description = "Trim bytes from start and end";

  defaults() {
    return { audio: {}, start: 0, end: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = audioBytes(inputs.audio ?? this._props.audio);
    const start = Math.max(0, Number(inputs.start ?? this._props.start ?? 0));
    const end = Math.max(0, Number(inputs.end ?? this._props.end ?? 0));
    return { output: audioRefFromBytes(data.slice(start, Math.max(start, data.length - end))) };
  }
}

export class ConvertToArrayNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.ConvertToArray";
  static readonly title = "Convert To Array";
  static readonly description = "Wrap audio in list";

  defaults() {
    return { audio: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: [inputs.audio ?? this._props.audio ?? {}] };
  }
}

export class CreateSilenceNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.CreateSilence";
  static readonly title = "Create Silence";
  static readonly description = "Create silent audio byte buffer";

  defaults() {
    return { length: 16000 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const length = Math.max(0, Number(inputs.length ?? this._props.length ?? 16000));
    return { output: audioRefFromBytes(new Uint8Array(length)) };
  }
}

export class ConcatAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.Concat";
  static readonly title = "Concat";
  static readonly description = "Concatenate two audio refs";

  defaults() {
    return { audio_a: {}, audio_b: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = audioBytes(inputs.audio_a ?? this._props.audio_a);
    const b = audioBytes(inputs.audio_b ?? this._props.audio_b);
    return { output: audioRefFromBytes(concatBytes([a, b])) };
  }
}

export class ConcatAudioListNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.ConcatList";
  static readonly title = "Concat List";
  static readonly description = "Concatenate list of audio refs";

  defaults() {
    return { audios: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audios = Array.isArray(inputs.audios ?? this._props.audios)
      ? (inputs.audios ?? this._props.audios) as unknown[]
      : [];
    const merged = concatBytes(audios.map((a) => audioBytes(a)));
    return { output: audioRefFromBytes(merged) };
  }
}

export class TextToSpeechNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.TextToSpeech";
  static readonly title = "Text To Speech";
  static readonly description = "Convert text to placeholder audio bytes";

  defaults() {
    return { text: "", model: null, voice: "", speed: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const bytes = Uint8Array.from(Buffer.from(text, "utf8"));
    return { output: audioRefFromBytes(bytes) };
  }
}

export class ChunkToAudioNode extends BaseNode {
  static readonly nodeType = "nodetool.audio.ChunkToAudio";
  static readonly title = "Chunk To Audio";
  static readonly description = "Convert chunk/image-like data to audio ref";

  defaults() {
    return { chunk: null };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const chunk = inputs.chunk ?? this._props.chunk ?? {};
    if (chunk && typeof chunk === "object") {
      const image = chunk as ImageLike;
      if (image.data || image.uri) {
        return { output: audioRefFromBytes(toBytes(image.data)) };
      }
    }
    return { output: audioRefFromBytes(new Uint8Array()) };
  }
}

export const AUDIO_NODES = [
  LoadAudioAssetsNode,
  LoadAudioFileNode,
  LoadAudioFolderNode,
  SaveAudioNode,
  SaveAudioFileNode,
  NormalizeAudioNode,
  OverlayAudioNode,
  RemoveSilenceNode,
  SliceAudioNode,
  MonoToStereoNode,
  StereoToMonoNode,
  ReverseAudioNode,
  FadeInAudioNode,
  FadeOutAudioNode,
  RepeatAudioNode,
  AudioMixerNode,
  AudioToNumpyNode,
  NumpyToAudioNode,
  TrimAudioNode,
  ConvertToArrayNode,
  CreateSilenceNode,
  ConcatAudioNode,
  ConcatAudioListNode,
  TextToSpeechNode,
  ChunkToAudioNode,
] as const;
