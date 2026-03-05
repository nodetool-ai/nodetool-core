import { BaseNode } from "@nodetool/node-sdk";

// ── WAV helpers (duplicated from lib-audio-dsp.ts) ─────────────────

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

// ── DSP helpers ────────────────────────────────────────────────────

function processPerChannel(
  wav: WavData,
  fn: (channel: Float32Array, sampleRate: number) => Float32Array
): { samples: Float32Array; sampleRate: number; numChannels: number } {
  const { samples, sampleRate, numChannels } = wav;
  const frameSamples = Math.floor(samples.length / numChannels);

  const channels: Float32Array[] = [];
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = new Float32Array(frameSamples);
    for (let i = 0; i < frameSamples; i++) {
      channelData[i] = samples[i * numChannels + ch];
    }
    channels.push(fn(channelData, sampleRate));
  }

  const outFrames = channels[0].length;
  const outSamples = new Float32Array(outFrames * numChannels);
  for (let ch = 0; ch < numChannels; ch++) {
    for (let i = 0; i < outFrames; i++) {
      outSamples[i * numChannels + ch] = channels[ch][i];
    }
  }

  return { samples: outSamples, sampleRate, numChannels };
}

// ── Bitcrush ──────────────────────────────────────────────────────

export class BitcrushNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.Bitcrush";
  static readonly title = "Bitcrush";
  static readonly description =
    "Applies a bitcrushing effect to an audio file, reducing bit depth and/or sample rate.";

  defaults() {
    return { audio: {}, bit_depth: 8, sample_rate_reduction: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const bitDepth = Number(inputs.bit_depth ?? this._props.bit_depth ?? 8);
    const srrFactor = Number(inputs.sample_rate_reduction ?? this._props.sample_rate_reduction ?? 1);

    if (!audio.data) return { output: audio };

    const wav = decodeWav(audio);
    const result = processPerChannel(wav, (ch) => {
      const levels = Math.pow(2, bitDepth - 1) - 1;
      const out = new Float32Array(ch.length);
      for (let i = 0; i < ch.length; i++) {
        const idx = Math.floor(i / srrFactor) * srrFactor;
        const srcIdx = Math.min(idx, ch.length - 1);
        out[i] = Math.round(ch[srcIdx] * levels) / levels;
      }
      return out;
    });

    return { output: audioRefFromWav(encodeWav(result.samples, result.sampleRate, result.numChannels)) };
  }
}

// ── Compress ──────────────────────────────────────────────────────

export class CompressNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.Compress";
  static readonly title = "Compress";
  static readonly description =
    "Applies dynamic range compression to an audio file.";

  defaults() {
    return { audio: {}, threshold: -20.0, ratio: 4.0, attack: 5.0, release: 50.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const thresholdDb = Number(inputs.threshold ?? this._props.threshold ?? -20);
    const ratio = Number(inputs.ratio ?? this._props.ratio ?? 4);
    const attackMs = Number(inputs.attack ?? this._props.attack ?? 5);
    const releaseMs = Number(inputs.release ?? this._props.release ?? 50);

    if (!audio.data) return { output: audio };

    const wav = decodeWav(audio);
    const result = processPerChannel(wav, (ch, sr) => {
      const out = new Float32Array(ch.length);
      const attackCoeff = Math.exp(-1 / (sr * attackMs / 1000));
      const releaseCoeff = Math.exp(-1 / (sr * releaseMs / 1000));
      const thresholdLin = Math.pow(10, thresholdDb / 20);

      let envelope = 0;
      for (let i = 0; i < ch.length; i++) {
        const absVal = Math.abs(ch[i]);
        if (absVal > envelope) {
          envelope = attackCoeff * envelope + (1 - attackCoeff) * absVal;
        } else {
          envelope = releaseCoeff * envelope + (1 - releaseCoeff) * absVal;
        }

        if (envelope > thresholdLin) {
          const dbOver = 20 * Math.log10(envelope / thresholdLin);
          const dbReduction = dbOver * (1 - 1 / ratio);
          const gainReduction = Math.pow(10, -dbReduction / 20);
          out[i] = ch[i] * gainReduction;
        } else {
          out[i] = ch[i];
        }
      }
      return out;
    });

    return { output: audioRefFromWav(encodeWav(result.samples, result.sampleRate, result.numChannels)) };
  }
}

// ── Distortion ────────────────────────────────────────────────────

export class DistortionNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.Distortion";
  static readonly title = "Distortion";
  static readonly description =
    "Applies a distortion effect to an audio file.";

  defaults() {
    return { audio: {}, drive_db: 25.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const driveDb = Number(inputs.drive_db ?? this._props.drive_db ?? 25);

    if (!audio.data) return { output: audio };

    const wav = decodeWav(audio);
    const drive = Math.pow(10, driveDb / 20);
    const result = processPerChannel(wav, (ch) => {
      const out = new Float32Array(ch.length);
      for (let i = 0; i < ch.length; i++) {
        const driven = ch[i] * drive;
        out[i] = (2 / Math.PI) * Math.atan(driven);
      }
      return out;
    });

    return { output: audioRefFromWav(encodeWav(result.samples, result.sampleRate, result.numChannels)) };
  }
}

// ── Limiter ───────────────────────────────────────────────────────

export class LimiterNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.Limiter";
  static readonly title = "Limiter";
  static readonly description =
    "Applies a limiter effect to an audio file.";

  defaults() {
    return { audio: {}, threshold_db: -2.0, release_ms: 250.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const thresholdDb = Number(inputs.threshold_db ?? this._props.threshold_db ?? -2);
    const releaseMs = Number(inputs.release_ms ?? this._props.release_ms ?? 250);

    if (!audio.data) return { output: audio };

    const wav = decodeWav(audio);
    const threshold = Math.pow(10, thresholdDb / 20);
    const result = processPerChannel(wav, (ch, sr) => {
      const out = new Float32Array(ch.length);
      const releaseCoeff = Math.exp(-1 / (sr * releaseMs / 1000));
      let gainReduction = 1;

      for (let i = 0; i < ch.length; i++) {
        const absVal = Math.abs(ch[i]);
        if (absVal > threshold) {
          const targetGain = threshold / absVal;
          if (targetGain < gainReduction) {
            gainReduction = targetGain;
          }
        } else {
          gainReduction = releaseCoeff * gainReduction + (1 - releaseCoeff) * 1.0;
        }
        out[i] = ch[i] * gainReduction;
      }
      return out;
    });

    return { output: audioRefFromWav(encodeWav(result.samples, result.sampleRate, result.numChannels)) };
  }
}

// ── Reverb (Schroeder) ───────────────────────────────────────────

export class ReverbNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.Reverb";
  static readonly title = "Reverb";
  static readonly description =
    "Applies a reverb effect to an audio file.";

  defaults() {
    return { audio: {}, room_scale: 0.5, damping: 0.5, wet_level: 0.15, dry_level: 0.5 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const roomScale = Number(inputs.room_scale ?? this._props.room_scale ?? 0.5);
    const damping = Number(inputs.damping ?? this._props.damping ?? 0.5);
    const wetLevel = Number(inputs.wet_level ?? this._props.wet_level ?? 0.15);
    const dryLevel = Number(inputs.dry_level ?? this._props.dry_level ?? 0.5);

    if (!audio.data) return { output: audio };

    const wav = decodeWav(audio);
    const result = processPerChannel(wav, (ch, sr) => {
      // Schroeder reverb: 4 parallel comb filters -> 2 series allpass filters
      const baseCombDelays = [1557, 1617, 1491, 1422];
      const baseAllpassDelays = [225, 556];
      const scale = sr / 44100;

      // Comb filter
      function combFilter(input: Float32Array, delaySamples: number, feedback: number, damp: number): Float32Array {
        const out = new Float32Array(input.length);
        const buf = new Float32Array(delaySamples);
        let bufIdx = 0;
        let filterStore = 0;

        for (let i = 0; i < input.length; i++) {
          const delayed = buf[bufIdx];
          filterStore = delayed * (1 - damp) + filterStore * damp;
          buf[bufIdx] = input[i] + filterStore * feedback;
          out[i] = delayed;
          bufIdx = (bufIdx + 1) % delaySamples;
        }
        return out;
      }

      // Allpass filter
      function allpassFilter(input: Float32Array, delaySamples: number, feedback: number): Float32Array {
        const out = new Float32Array(input.length);
        const buf = new Float32Array(delaySamples);
        let bufIdx = 0;

        for (let i = 0; i < input.length; i++) {
          const delayed = buf[bufIdx];
          buf[bufIdx] = input[i] + delayed * feedback;
          out[i] = delayed - input[i] * feedback;
          bufIdx = (bufIdx + 1) % delaySamples;
        }
        return out;
      }

      const feedback = roomScale * 0.28 + 0.7;

      // Sum comb filters
      const combOut = new Float32Array(ch.length);
      for (const baseDelay of baseCombDelays) {
        const delay = Math.round(baseDelay * scale);
        const filtered = combFilter(ch, delay, feedback, damping);
        for (let i = 0; i < ch.length; i++) {
          combOut[i] += filtered[i];
        }
      }

      // Series allpass filters
      let apOut: Float32Array = combOut;
      for (const baseDelay of baseAllpassDelays) {
        const delay = Math.round(baseDelay * scale);
        apOut = allpassFilter(apOut, delay, 0.5);
      }

      // Mix dry/wet
      const out = new Float32Array(ch.length);
      for (let i = 0; i < ch.length; i++) {
        out[i] = ch[i] * dryLevel + apOut[i] * wetLevel;
      }
      return out;
    });

    return { output: audioRefFromWav(encodeWav(result.samples, result.sampleRate, result.numChannels)) };
  }
}

// ── PitchShift (soundtouchjs) ────────────────────────────────────

export class PitchShiftNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.PitchShift";
  static readonly title = "Pitch Shift";
  static readonly description =
    "Shifts the pitch of an audio file without changing its duration.";

  defaults() {
    return { audio: {}, semitones: 0.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const semitones = Number(inputs.semitones ?? this._props.semitones ?? 0);

    if (!audio.data) return { output: audio };
    if (semitones === 0) {
      return { output: audio };
    }

    const { SoundTouch } = await import("soundtouchjs");
    const wav = decodeWav(audio);
    const { samples, sampleRate, numChannels } = wav;
    const frameSamples = Math.floor(samples.length / numChannels);

    // SoundTouch works with stereo interleaved samples
    // Convert to stereo interleaved if mono
    let stereoInput: Float32Array;
    if (numChannels === 1) {
      stereoInput = new Float32Array(frameSamples * 2);
      for (let i = 0; i < frameSamples; i++) {
        stereoInput[i * 2] = samples[i];
        stereoInput[i * 2 + 1] = samples[i];
      }
    } else if (numChannels === 2) {
      stereoInput = samples;
    } else {
      // For >2 channels, just take first two
      stereoInput = new Float32Array(frameSamples * 2);
      for (let i = 0; i < frameSamples; i++) {
        stereoInput[i * 2] = samples[i * numChannels];
        stereoInput[i * 2 + 1] = samples[i * numChannels + 1];
      }
    }

    const st = new SoundTouch();
    st.sampleRate = sampleRate;
    st.pitchSemitones = semitones;

    // Feed in chunks
    const chunkSize = 4096;
    for (let offset = 0; offset < frameSamples; offset += chunkSize) {
      const end = Math.min(offset + chunkSize, frameSamples);
      const chunk = stereoInput.slice(offset * 2, end * 2);
      st.inputBuffer.putSamples(chunk, 0, end - offset);
      st.process();
    }

    // Flush remaining
    st.inputBuffer.putSamples(new Float32Array(0), 0, 0);
    st.process();

    const available = st.outputBuffer.frameCount;
    const stereoOutput = new Float32Array(available * 2);
    st.outputBuffer.receiveSamples(stereoOutput, available);

    // Convert back to original channel count
    let outSamples: Float32Array;
    if (numChannels === 1) {
      outSamples = new Float32Array(available);
      for (let i = 0; i < available; i++) {
        outSamples[i] = (stereoOutput[i * 2] + stereoOutput[i * 2 + 1]) / 2;
      }
    } else {
      outSamples = new Float32Array(available * numChannels);
      for (let i = 0; i < available; i++) {
        outSamples[i * numChannels] = stereoOutput[i * 2];
        outSamples[i * numChannels + 1] = stereoOutput[i * 2 + 1];
        for (let ch = 2; ch < numChannels; ch++) {
          outSamples[i * numChannels + ch] = 0;
        }
      }
    }

    return { output: audioRefFromWav(encodeWav(outSamples, sampleRate, numChannels)) };
  }
}

// ── TimeStretch (soundtouchjs) ───────────────────────────────────

export class TimeStretchNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.TimeStretch";
  static readonly title = "Time Stretch";
  static readonly description =
    "Changes the speed of an audio file without altering its pitch.";

  defaults() {
    return { audio: {}, rate: 1.0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const audio = (inputs.audio ?? this._props.audio ?? {}) as Record<string, unknown>;
    const rate = Number(inputs.rate ?? this._props.rate ?? 1.0);

    if (!audio.data) return { output: audio };
    if (rate === 1.0) {
      return { output: audio };
    }

    const { SoundTouch } = await import("soundtouchjs");
    const wav = decodeWav(audio);
    const { samples, sampleRate, numChannels } = wav;
    const frameSamples = Math.floor(samples.length / numChannels);

    let stereoInput: Float32Array;
    if (numChannels === 1) {
      stereoInput = new Float32Array(frameSamples * 2);
      for (let i = 0; i < frameSamples; i++) {
        stereoInput[i * 2] = samples[i];
        stereoInput[i * 2 + 1] = samples[i];
      }
    } else if (numChannels === 2) {
      stereoInput = samples;
    } else {
      stereoInput = new Float32Array(frameSamples * 2);
      for (let i = 0; i < frameSamples; i++) {
        stereoInput[i * 2] = samples[i * numChannels];
        stereoInput[i * 2 + 1] = samples[i * numChannels + 1];
      }
    }

    const st = new SoundTouch();
    st.sampleRate = sampleRate;
    st.tempo = rate;

    const chunkSize = 4096;
    for (let offset = 0; offset < frameSamples; offset += chunkSize) {
      const end = Math.min(offset + chunkSize, frameSamples);
      const chunk = stereoInput.slice(offset * 2, end * 2);
      st.inputBuffer.putSamples(chunk, 0, end - offset);
      st.process();
    }

    st.inputBuffer.putSamples(new Float32Array(0), 0, 0);
    st.process();

    const available = st.outputBuffer.frameCount;
    const stereoOutput = new Float32Array(available * 2);
    st.outputBuffer.receiveSamples(stereoOutput, available);

    let outSamples: Float32Array;
    if (numChannels === 1) {
      outSamples = new Float32Array(available);
      for (let i = 0; i < available; i++) {
        outSamples[i] = (stereoOutput[i * 2] + stereoOutput[i * 2 + 1]) / 2;
      }
    } else {
      outSamples = new Float32Array(available * numChannels);
      for (let i = 0; i < available; i++) {
        outSamples[i * numChannels] = stereoOutput[i * 2];
        outSamples[i * numChannels + 1] = stereoOutput[i * 2 + 1];
        for (let ch = 2; ch < numChannels; ch++) {
          outSamples[i * numChannels + ch] = 0;
        }
      }
    }

    return { output: audioRefFromWav(encodeWav(outSamples, sampleRate, numChannels)) };
  }
}

// ── NoiseGate (stub) ─────────────────────────────────────────────

export class NoiseGateNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.NoiseGate";
  static readonly title = "Noise Gate";
  static readonly description =
    "Applies a noise gate effect to an audio file.";

  defaults() {
    return { audio: {}, threshold_db: -50.0, attack_ms: 1.0, release_ms: 100.0 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    throw new Error("lib.pedalboard.NoiseGate: not yet implemented in TypeScript. Use Python bridge.");
  }
}

// ── Phaser (stub) ────────────────────────────────────────────────

export class PhaserNode extends BaseNode {
  static readonly nodeType = "lib.pedalboard.Phaser";
  static readonly title = "Phaser";
  static readonly description =
    "Applies a phaser effect to an audio file.";

  defaults() {
    return { audio: {}, rate_hz: 1.0, depth: 0.5, centre_frequency_hz: 1300.0, feedback: 0.0, mix: 0.5 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    throw new Error("lib.pedalboard.Phaser: not yet implemented in TypeScript. Use Python bridge.");
  }
}

// ── Export ────────────────────────────────────────────────────────

export const LIB_PEDALBOARD_EXTRA_NODES = [
  BitcrushNode,
  CompressNode,
  DistortionNode,
  LimiterNode,
  ReverbNode,
  PitchShiftNode,
  TimeStretchNode,
  NoiseGateNode,
  PhaserNode,
] as const;
