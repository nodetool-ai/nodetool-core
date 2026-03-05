import { BaseNode } from "@nodetool/node-sdk";
import sharp from "sharp";

// ── WAV helpers (shared with lib-synthesis.ts pattern) ──────────────

function encodeWav(samples: Float32Array, sampleRate: number, numChannels = 1): Uint8Array {
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
  buffer.writeUInt16LE(1, 20); // PCM
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

interface WavData {
  samples: Float32Array;
  sampleRate: number;
  numChannels: number;
}

function decodeWav(audio: Record<string, unknown>): WavData {
  let rawData: Uint8Array;
  if (typeof audio.data === "string") {
    rawData = Uint8Array.from(Buffer.from(audio.data, "base64"));
  } else if (audio.data instanceof Uint8Array) {
    rawData = audio.data;
  } else {
    throw new Error("Invalid audio data");
  }

  const buf = Buffer.from(rawData);
  if (buf.toString("ascii", 0, 4) !== "RIFF" || buf.length < 44) {
    throw new Error("Invalid WAV file");
  }

  const sampleRate = buf.readUInt32LE(24);
  const bitsPerSample = buf.readUInt16LE(34);
  const numChannels = buf.readUInt16LE(22);

  let dataOffset = 36;
  while (dataOffset < buf.length - 8) {
    const chunkId = buf.toString("ascii", dataOffset, dataOffset + 4);
    const chunkSize = buf.readUInt32LE(dataOffset + 4);
    if (chunkId === "data") {
      dataOffset += 8;
      break;
    }
    dataOffset += 8 + chunkSize;
  }

  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = Math.floor((buf.length - dataOffset) / bytesPerSample);
  const samples = new Float32Array(totalSamples);

  for (let i = 0; i < totalSamples; i++) {
    const pos = dataOffset + i * bytesPerSample;
    if (bitsPerSample === 16) {
      samples[i] = buf.readInt16LE(pos) / 0x7fff;
    } else if (bitsPerSample === 8) {
      samples[i] = (buf.readUInt8(pos) - 128) / 128;
    }
  }

  return { samples, sampleRate, numChannels };
}

// ── Part A: dB math nodes (pure TS, no deps) ──────────────────────

export class AmplitudeToDBNode extends BaseNode {
  static readonly nodeType = "lib.librosa.analysis.AmplitudeToDB";
  static readonly title = "Amplitude To DB";
  static readonly description =
    "Converts an amplitude spectrogram to a dB-scaled spectrogram.";

  defaults() {
    return { tensor: { data: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const tensor = (inputs.tensor ?? this._props.tensor ?? { data: [] }) as { data: number[] | number[][] };
    const data = tensor.data;

    const convert = (arr: number[]): number[] =>
      arr.map((x) => 20 * Math.log10(Math.max(x, 1e-10)));

    let result: number[] | number[][];
    if (Array.isArray(data[0])) {
      result = (data as number[][]).map(convert);
    } else {
      result = convert(data as number[]);
    }

    return { output: { data: result } };
  }
}

export class DBToAmplitudeNode extends BaseNode {
  static readonly nodeType = "lib.librosa.analysis.DBToAmplitude";
  static readonly title = "DB To Amplitude";
  static readonly description =
    "Converts a dB-scaled spectrogram to an amplitude spectrogram.";

  defaults() {
    return { tensor: { data: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const tensor = (inputs.tensor ?? this._props.tensor ?? { data: [] }) as { data: number[] | number[][] };
    const data = tensor.data;

    const convert = (arr: number[]): number[] =>
      arr.map((x) => Math.pow(10, x / 20));

    let result: number[] | number[][];
    if (Array.isArray(data[0])) {
      result = (data as number[][]).map(convert);
    } else {
      result = convert(data as number[]);
    }

    return { output: { data: result } };
  }
}

export class DBToPowerNode extends BaseNode {
  static readonly nodeType = "lib.librosa.analysis.DBToPower";
  static readonly title = "DB To Power";
  static readonly description =
    "Converts a decibel (dB) spectrogram back to power scale.";

  defaults() {
    return { tensor: { data: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const tensor = (inputs.tensor ?? this._props.tensor ?? { data: [] }) as { data: number[] | number[][] };
    const data = tensor.data;

    const convert = (arr: number[]): number[] =>
      arr.map((x) => Math.pow(10, x / 10));

    let result: number[] | number[][];
    if (Array.isArray(data[0])) {
      result = (data as number[][]).map(convert);
    } else {
      result = convert(data as number[]);
    }

    return { output: { data: result } };
  }
}

export class PowerToDBNode extends BaseNode {
  static readonly nodeType = "lib.librosa.analysis.PowertToDB";
  static readonly title = "Power To DB";
  static readonly description =
    "Converts a power spectrogram to decibel (dB) scale.";

  defaults() {
    return { tensor: { data: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const tensor = (inputs.tensor ?? this._props.tensor ?? { data: [] }) as { data: number[] | number[][] };
    const data = tensor.data;

    const convert = (arr: number[]): number[] =>
      arr.map((x) => 10 * Math.log10(Math.max(x, 1e-10)));

    let result: number[] | number[][];
    if (Array.isArray(data[0])) {
      result = (data as number[][]).map(convert);
    } else {
      result = convert(data as number[]);
    }

    return { output: { data: result } };
  }
}

export class PlotSpectrogramNode extends BaseNode {
  static readonly nodeType = "lib.librosa.analysis.PlotSpectrogram";
  static readonly title = "Plot Spectrogram";
  static readonly description =
    "Generates a visual representation of the spectrum of frequencies in an audio signal as they vary with time.";

  defaults() {
    return { tensor: { data: [] }, fmax: 8000 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const tensor = (inputs.tensor ?? this._props.tensor ?? { data: [] }) as { data: number[][] };
    const spec = tensor.data;

    if (!spec.length || !spec[0]?.length) {
      return { output: { uri: "", data: "" } };
    }

    const rows = spec.length;
    const cols = spec[0].length;

    // Find min/max for normalization
    let min = Infinity;
    let max = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = spec[r][c];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }

    const range = max - min || 1;

    // Create grayscale image buffer (transposed: freq on y, time on x)
    // Output image: width = cols (time), height = rows (freq, flipped)
    const imgBuf = Buffer.alloc(cols * rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const normalized = ((spec[r][c] - min) / range) * 255;
        // Flip vertically so low freq is at bottom
        imgBuf[(rows - 1 - r) * cols + c] = Math.round(normalized);
      }
    }

    const pngBuffer = await sharp(imgBuf, {
      raw: { width: cols, height: rows, channels: 1 },
    })
      .png()
      .toBuffer();

    return {
      output: {
        uri: "",
        data: pngBuffer.toString("base64"),
      },
    };
  }
}

// ── Part B: Audio filter/effect nodes (node-web-audio-api) ─────────

async function processAudioWithEffect(
  audio: Record<string, unknown>,
  setupEffect: (ctx: any, source: any) => void,
  extraLength = 0
): Promise<Record<string, unknown>> {
  const { OfflineAudioContext } = await import("node-web-audio-api");
  const wav = decodeWav(audio);
  const frameSamples = Math.floor(wav.samples.length / wav.numChannels);
  const totalFrames = frameSamples + extraLength;

  const ctx = new OfflineAudioContext(wav.numChannels, totalFrames, wav.sampleRate);
  const buffer = ctx.createBuffer(wav.numChannels, frameSamples, wav.sampleRate);

  // Fill buffer channels
  for (let ch = 0; ch < wav.numChannels; ch++) {
    const channelData = buffer.getChannelData(ch);
    for (let i = 0; i < frameSamples; i++) {
      channelData[i] = wav.samples[i * wav.numChannels + ch];
    }
  }

  const source = ctx.createBufferSource();
  source.buffer = buffer;

  setupEffect(ctx, source);
  source.start();

  const renderedBuffer = await ctx.startRendering();

  // Interleave channels back
  const outLength = renderedBuffer.length * wav.numChannels;
  const outSamples = new Float32Array(outLength);
  for (let ch = 0; ch < wav.numChannels; ch++) {
    const channelData = renderedBuffer.getChannelData(ch);
    for (let i = 0; i < renderedBuffer.length; i++) {
      outSamples[i * wav.numChannels + ch] = channelData[i];
    }
  }

  return audioRefFromWav(encodeWav(outSamples, wav.sampleRate, wav.numChannels));
}

export class GainNode_ extends BaseNode {
  static readonly nodeType = "lib.pedalboard.Gain";
  static readonly title = "Gain";
  static readonly description =
    "Applies a gain (volume adjustment) to an audio file.";

  defaults() {
    return { audio: {}, gain_db: 0.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const gainDb = Number(inputs.gain_db ?? this._props.gain_db ?? 0);

    if (!audio.data) return { output: audio };

    const output = await processAudioWithEffect(audio, (ctx: any, source: any) => {
      const gainNode = ctx.createGain();
      gainNode.gain.value = Math.pow(10, gainDb / 20);
      source.connect(gainNode);
      gainNode.connect(ctx.destination);
    });

    return { output };
  }
}

export class DelayNode_ extends BaseNode {
  static readonly nodeType = "lib.pedalboard.Delay";
  static readonly title = "Delay";
  static readonly description = "Applies a delay effect to an audio file.";

  defaults() {
    return { audio: {}, delay_seconds: 0.5, feedback: 0.3, mix: 0.5 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const delaySec = Number(inputs.delay_seconds ?? this._props.delay_seconds ?? 0.5);
    const feedback = Number(inputs.feedback ?? this._props.feedback ?? 0.3);
    const mix = Number(inputs.mix ?? this._props.mix ?? 0.5);

    if (!audio.data) return { output: audio };

    const wav = decodeWav(audio);
    const frameSamples = Math.floor(wav.samples.length / wav.numChannels);
    const delaySamples = Math.floor(delaySec * wav.sampleRate);

    // Process delay manually for better control over feedback
    const outLength = frameSamples + delaySamples * 4; // extra space for echoes
    const outSamples = new Float32Array(outLength * wav.numChannels);

    for (let ch = 0; ch < wav.numChannels; ch++) {
      const dry = new Float32Array(outLength);
      const wet = new Float32Array(outLength);

      // Copy dry signal
      for (let i = 0; i < frameSamples; i++) {
        dry[i] = wav.samples[i * wav.numChannels + ch];
      }

      // Apply delay with feedback
      for (let i = 0; i < outLength; i++) {
        const dryVal = i < frameSamples ? dry[i] : 0;
        const delayedVal = i >= delaySamples ? wet[i - delaySamples] : 0;
        wet[i] = dryVal + delayedVal * feedback;
      }

      // Mix dry and wet
      for (let i = 0; i < outLength; i++) {
        const dryVal = i < frameSamples ? dry[i] : 0;
        outSamples[i * wav.numChannels + ch] = dryVal * (1 - mix) + wet[i] * mix;
      }
    }

    return { output: audioRefFromWav(encodeWav(outSamples, wav.sampleRate, wav.numChannels)) };
  }
}

export class HighPassFilterNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.HighPassFilter";
  static readonly title = "High Pass Filter";
  static readonly description =
    "Applies a high-pass filter to attenuate frequencies below a cutoff point.";

  defaults() {
    return { audio: {}, cutoff_frequency_hz: 80.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const cutoff = Number(inputs.cutoff_frequency_hz ?? this._props.cutoff_frequency_hz ?? 80);

    if (!audio.data) return { output: audio };

    const output = await processAudioWithEffect(audio, (ctx: any, source: any) => {
      const filter = ctx.createBiquadFilter();
      filter.type = "highpass";
      filter.frequency.value = cutoff;
      source.connect(filter);
      filter.connect(ctx.destination);
    });

    return { output };
  }
}

export class LowPassFilterNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.LowPassFilter";
  static readonly title = "Low Pass Filter";
  static readonly description =
    "Applies a low-pass filter to attenuate frequencies above a cutoff point.";

  defaults() {
    return { audio: {}, cutoff_frequency_hz: 5000.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const cutoff = Number(inputs.cutoff_frequency_hz ?? this._props.cutoff_frequency_hz ?? 5000);

    if (!audio.data) return { output: audio };

    const output = await processAudioWithEffect(audio, (ctx: any, source: any) => {
      const filter = ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.value = cutoff;
      source.connect(filter);
      filter.connect(ctx.destination);
    });

    return { output };
  }
}

export class HighShelfFilterNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.HighShelfFilter";
  static readonly title = "High Shelf Filter";
  static readonly description =
    "Applies a high shelf filter to boost or cut high frequencies.";

  defaults() {
    return { audio: {}, cutoff_frequency_hz: 5000.0, gain_db: 0.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const cutoff = Number(inputs.cutoff_frequency_hz ?? this._props.cutoff_frequency_hz ?? 5000);
    const gainDb = Number(inputs.gain_db ?? this._props.gain_db ?? 0);

    if (!audio.data) return { output: audio };

    const output = await processAudioWithEffect(audio, (ctx: any, source: any) => {
      const filter = ctx.createBiquadFilter();
      filter.type = "highshelf";
      filter.frequency.value = cutoff;
      filter.gain.value = gainDb;
      source.connect(filter);
      filter.connect(ctx.destination);
    });

    return { output };
  }
}

export class LowShelfFilterNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.LowShelfFilter";
  static readonly title = "Low Shelf Filter";
  static readonly description =
    "Applies a low shelf filter to boost or cut low frequencies.";

  defaults() {
    return { audio: {}, cutoff_frequency_hz: 200.0, gain_db: 0.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const cutoff = Number(inputs.cutoff_frequency_hz ?? this._props.cutoff_frequency_hz ?? 200);
    const gainDb = Number(inputs.gain_db ?? this._props.gain_db ?? 0);

    if (!audio.data) return { output: audio };

    const output = await processAudioWithEffect(audio, (ctx: any, source: any) => {
      const filter = ctx.createBiquadFilter();
      filter.type = "lowshelf";
      filter.frequency.value = cutoff;
      filter.gain.value = gainDb;
      source.connect(filter);
      filter.connect(ctx.destination);
    });

    return { output };
  }
}

export class PeakFilterNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.PeakFilter";
  static readonly title = "Peak Filter";
  static readonly description =
    "Applies a peak filter to boost or cut a specific frequency range.";

  defaults() {
    return { audio: {}, cutoff_frequency_hz: 1000.0, q_factor: 1.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const cutoff = Number(inputs.cutoff_frequency_hz ?? this._props.cutoff_frequency_hz ?? 1000);
    const q = Number(inputs.q_factor ?? this._props.q_factor ?? 1.0);

    if (!audio.data) return { output: audio };

    const output = await processAudioWithEffect(audio, (ctx: any, source: any) => {
      const filter = ctx.createBiquadFilter();
      filter.type = "peaking";
      filter.frequency.value = cutoff;
      filter.Q.value = q;
      filter.gain.value = 0;
      source.connect(filter);
      filter.connect(ctx.destination);
    });

    return { output };
  }
}

export const LIB_AUDIO_DSP_NODES = [
  AmplitudeToDBNode,
  DBToAmplitudeNode,
  DBToPowerNode,
  PowerToDBNode,
  PlotSpectrogramNode,
  GainNode_,
  DelayNode_,
  HighPassFilterNode,
  LowPassFilterNode,
  HighShelfFilterNode,
  LowShelfFilterNode,
  PeakFilterNode,
] as const;
