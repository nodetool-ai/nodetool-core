import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";
import {
  getApiKey,
  kieExecuteTask,
  kieExecuteSunoTask,
  uploadAudioInput,
  isRefSet,
} from "./kie-base.js";

// ── Suno-backed nodes ────────────────────────────────────────────────────────

export class GenerateMusicNode extends BaseNode {
  static readonly nodeType = "kie.audio.GenerateMusic";
  static readonly title = "Generate Music";
  static readonly description =
    "Generate music tracks using the Suno AI model via the Kie.ai API. " +
    "Supports both simple prompt mode and custom mode with fine-grained style, " +
    "vocal gender, and weight controls.";

  defaults() {
    return {
      custom_mode: false,
      prompt: "",
      style: "",
      title: "",
      instrumental: false,
      model: "V4_5PLUS",
      negative_tags: "",
      vocal_gender: "",
      style_weight: 0,
      weirdness_constraint: 0,
      audio_weight: 0,
      persona_id: "",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const customMode = Boolean(inputs.custom_mode ?? this._props.custom_mode ?? false);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const model = String(inputs.model ?? this._props.model ?? "V4_5PLUS");
    const instrumental = Boolean(inputs.instrumental ?? this._props.instrumental ?? false);

    const payload: Record<string, unknown> = {
      customMode,
      instrumental,
      callBackUrl: "https://example.com/callback",
      model,
      prompt,
    };

    if (customMode) {
      const style = String(inputs.style ?? this._props.style ?? "");
      const title = String(inputs.title ?? this._props.title ?? "");
      if (!style) throw new Error("style is required in custom mode");
      if (!title) throw new Error("title is required in custom mode");
      if (!instrumental && !prompt) throw new Error("prompt required in custom mode with vocals");
      payload.style = style;
      payload.title = title;
      const neg = String(inputs.negative_tags ?? this._props.negative_tags ?? "");
      if (neg) payload.negativeTags = neg;
      const vg = String(inputs.vocal_gender ?? this._props.vocal_gender ?? "");
      if (vg) payload.vocalGender = vg;
      const sw = Number(inputs.style_weight ?? this._props.style_weight ?? 0);
      if (sw) payload.styleWeight = sw;
      const wc = Number(inputs.weirdness_constraint ?? this._props.weirdness_constraint ?? 0);
      if (wc) payload.weirdnessConstraint = wc;
      const aw = Number(inputs.audio_weight ?? this._props.audio_weight ?? 0);
      if (aw) payload.audioWeight = aw;
      const pid = String(inputs.persona_id ?? this._props.persona_id ?? "");
      if (pid) payload.personaId = pid;
    } else {
      if (!prompt) throw new Error("prompt is required");
    }

    const result = await kieExecuteSunoTask(apiKey, payload, 4000, 120);
    return { output: { data: result.data } };
  }
}

export class ExtendMusicNode extends BaseNode {
  static readonly nodeType = "kie.audio.ExtendMusic";
  static readonly title = "Extend Music";
  static readonly description =
    "Extend an existing audio track using the Suno AI model via the Kie.ai API. " +
    "Upload a source audio clip and specify a continuation prompt and style to " +
    "generate an extended version of the track.";

  defaults() {
    return {
      audio: null,
      prompt: "",
      style: "",
      continue_at: 0,
      model: "V4_5PLUS",
      instrumental: false,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const audio = inputs.audio ?? this._props.audio;
    if (!isRefSet(audio)) throw new Error("audio is required");

    const audioUrl = await uploadAudioInput(apiKey, audio);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const style = String(inputs.style ?? this._props.style ?? "");
    const continueAt = Number(inputs.continue_at ?? this._props.continue_at ?? 0);
    const model = String(inputs.model ?? this._props.model ?? "V4_5PLUS");
    const instrumental = Boolean(inputs.instrumental ?? this._props.instrumental ?? false);

    const payload: Record<string, unknown> = {
      customMode: true,
      prompt,
      style,
      instrumental,
      model,
      continue_at: continueAt,
      audio_url: audioUrl,
      continue: true,
      callBackUrl: "https://example.com/callback",
    };

    const result = await kieExecuteSunoTask(apiKey, payload, 4000, 120);
    return { output: { data: result.data } };
  }
}

export class CoverAudioNode extends BaseNode {
  static readonly nodeType = "kie.audio.CoverAudio";
  static readonly title = "Cover Audio";
  static readonly description =
    "Generate a cover version of an audio track using the Suno AI model via the Kie.ai API. " +
    "Upload a source audio file and supply a style prompt to produce a stylistically " +
    "transformed cover of the original.";

  defaults() {
    return {
      audio: null,
      prompt: "",
      style: "",
      model: "V4_5PLUS",
      instrumental: false,
      vocal_gender: "",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const audio = inputs.audio ?? this._props.audio;
    if (!isRefSet(audio)) throw new Error("audio is required");

    const audioUrl = await uploadAudioInput(apiKey, audio);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const style = String(inputs.style ?? this._props.style ?? "");
    const model = String(inputs.model ?? this._props.model ?? "V4_5PLUS");
    const instrumental = Boolean(inputs.instrumental ?? this._props.instrumental ?? false);
    const vocalGender = String(inputs.vocal_gender ?? this._props.vocal_gender ?? "");

    const payload: Record<string, unknown> = {
      customMode: true,
      prompt,
      style,
      instrumental,
      model,
      audio_url: audioUrl,
      cover: true,
      callBackUrl: "https://example.com/callback",
    };

    if (vocalGender) payload.vocalGender = vocalGender;

    const result = await kieExecuteSunoTask(apiKey, payload, 4000, 120);
    return { output: { data: result.data } };
  }
}

export class AddInstrumentalNode extends BaseNode {
  static readonly nodeType = "kie.audio.AddInstrumental";
  static readonly title = "Add Instrumental";
  static readonly description =
    "Add or overlay an AI-generated instrumental track onto an existing audio file " +
    "using the Suno AI model via the Kie.ai API. Supply a prompt and style to " +
    "control the character of the added instrumentation.";

  defaults() {
    return {
      audio: null,
      prompt: "",
      style: "",
      model: "V4_5PLUS",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const audio = inputs.audio ?? this._props.audio;
    if (!isRefSet(audio)) throw new Error("audio is required");

    const audioUrl = await uploadAudioInput(apiKey, audio);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const style = String(inputs.style ?? this._props.style ?? "");
    const model = String(inputs.model ?? this._props.model ?? "V4_5PLUS");

    const payload: Record<string, unknown> = {
      customMode: true,
      prompt,
      style,
      instrumental: true,
      model,
      audio_url: audioUrl,
      add_instrumental: true,
      callBackUrl: "https://example.com/callback",
    };

    const result = await kieExecuteSunoTask(apiKey, payload, 4000, 120);
    return { output: { data: result.data } };
  }
}

export class AddVocalsNode extends BaseNode {
  static readonly nodeType = "kie.audio.AddVocals";
  static readonly title = "Add Vocals";
  static readonly description =
    "Add AI-generated vocals to an existing audio track using the Suno AI model " +
    "via the Kie.ai API. Provide a prompt, style, and optional vocal gender preference " +
    "to shape the generated voice performance.";

  defaults() {
    return {
      audio: null,
      prompt: "",
      style: "",
      model: "V4_5PLUS",
      vocal_gender: "",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const audio = inputs.audio ?? this._props.audio;
    if (!isRefSet(audio)) throw new Error("audio is required");

    const audioUrl = await uploadAudioInput(apiKey, audio);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const style = String(inputs.style ?? this._props.style ?? "");
    const model = String(inputs.model ?? this._props.model ?? "V4_5PLUS");
    const vocalGender = String(inputs.vocal_gender ?? this._props.vocal_gender ?? "");

    const payload: Record<string, unknown> = {
      customMode: true,
      prompt,
      style,
      instrumental: false,
      model,
      audio_url: audioUrl,
      add_vocals: true,
      callBackUrl: "https://example.com/callback",
    };

    if (vocalGender) payload.vocalGender = vocalGender;

    const result = await kieExecuteSunoTask(apiKey, payload, 4000, 120);
    return { output: { data: result.data } };
  }
}

export class ReplaceMusicSectionNode extends BaseNode {
  static readonly nodeType = "kie.audio.ReplaceMusicSection";
  static readonly title = "Replace Music Section";
  static readonly description =
    "Replace a specified time section of a music track with AI-generated content " +
    "using the Suno AI model via the Kie.ai API. Define the start and end times of " +
    "the section to replace and provide a prompt and style for the replacement content.";

  defaults() {
    return {
      audio: null,
      prompt: "",
      style: "",
      start_time: 0,
      end_time: 30,
      model: "V4_5PLUS",
      instrumental: false,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const audio = inputs.audio ?? this._props.audio;
    if (!isRefSet(audio)) throw new Error("audio is required");

    const audioUrl = await uploadAudioInput(apiKey, audio);
    const prompt = String(inputs.prompt ?? this._props.prompt ?? "");
    const style = String(inputs.style ?? this._props.style ?? "");
    const startTime = Number(inputs.start_time ?? this._props.start_time ?? 0);
    const endTime = Number(inputs.end_time ?? this._props.end_time ?? 30);
    const model = String(inputs.model ?? this._props.model ?? "V4_5PLUS");
    const instrumental = Boolean(inputs.instrumental ?? this._props.instrumental ?? false);

    const payload: Record<string, unknown> = {
      customMode: true,
      prompt,
      style,
      instrumental,
      model,
      audio_url: audioUrl,
      replace_section: true,
      start_time: startTime,
      end_time: endTime,
      callBackUrl: "https://example.com/callback",
    };

    const result = await kieExecuteSunoTask(apiKey, payload, 4000, 120);
    return { output: { data: result.data } };
  }
}

// ── ElevenLabs-backed nodes ──────────────────────────────────────────────────

export class ElevenLabsTextToSpeechNode extends BaseNode {
  static readonly nodeType = "kie.audio.ElevenLabsTextToSpeech";
  static readonly title = "ElevenLabs Text To Speech";
  static readonly description =
    "Convert text to speech using ElevenLabs voice synthesis via the Kie.ai API. " +
    "Select a voice ID and model to produce high-quality, natural-sounding audio " +
    "from any text input.";

  defaults() {
    return {
      text: "",
      voice_id: "",
      model_id: "eleven_multilingual_v2",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const text = String(inputs.text ?? this._props.text ?? "");
    const voiceId = String(inputs.voice_id ?? this._props.voice_id ?? "");
    const modelId = String(inputs.model_id ?? this._props.model_id ?? "eleven_multilingual_v2");

    if (!text) throw new Error("text is required");
    if (!voiceId) throw new Error("voice_id is required");

    const result = await kieExecuteTask(apiKey, "elevenlabs/text-to-speech", {
      text,
      voice_id: voiceId,
      model_id: modelId,
    });

    return { output: { data: result.data } };
  }
}

export class ElevenLabsAudioIsolationNode extends BaseNode {
  static readonly nodeType = "kie.audio.ElevenLabsAudioIsolation";
  static readonly title = "ElevenLabs Audio Isolation";
  static readonly description =
    "Isolate and separate audio elements (e.g. vocals from background music) " +
    "using ElevenLabs audio isolation via the Kie.ai API. Upload any mixed audio " +
    "file to receive a cleaned, isolated audio output.";

  defaults() {
    return {
      audio: null,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const audio = inputs.audio ?? this._props.audio;
    if (!isRefSet(audio)) throw new Error("audio is required");

    const audioUrl = await uploadAudioInput(apiKey, audio);

    const result = await kieExecuteTask(apiKey, "elevenlabs/audio-isolation", {
      audio_url: audioUrl,
    });

    return { output: { data: result.data } };
  }
}

export class ElevenLabsSoundEffectNode extends BaseNode {
  static readonly nodeType = "kie.audio.ElevenLabsSoundEffect";
  static readonly title = "ElevenLabs Sound Effect";
  static readonly description =
    "Generate sound effects from a text description using ElevenLabs via the Kie.ai API. " +
    "Control the duration of the generated effect and the degree to which the prompt " +
    "influences the output with the prompt_influence parameter.";

  defaults() {
    return {
      text: "",
      duration_seconds: 0,
      prompt_influence: 0.3,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const text = String(inputs.text ?? this._props.text ?? "");
    const durationSeconds = Number(inputs.duration_seconds ?? this._props.duration_seconds ?? 0);
    const promptInfluence = Number(inputs.prompt_influence ?? this._props.prompt_influence ?? 0.3);

    if (!text) throw new Error("text is required");

    const taskInput: Record<string, unknown> = {
      text,
      prompt_influence: promptInfluence,
    };
    if (durationSeconds > 0) taskInput.duration_seconds = durationSeconds;

    const result = await kieExecuteTask(apiKey, "elevenlabs/sound-effect", taskInput);

    return { output: { data: result.data } };
  }
}

export class ElevenLabsSpeechToTextNode extends BaseNode {
  static readonly nodeType = "kie.audio.ElevenLabsSpeechToText";
  static readonly title = "ElevenLabs Speech To Text";
  static readonly description =
    "Transcribe speech from an audio file using ElevenLabs via the Kie.ai API. " +
    "Specify a language code to improve transcription accuracy for non-English audio. " +
    "Returns the transcribed text from the audio input.";

  defaults() {
    return {
      audio: null,
      language_code: "en",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const audio = inputs.audio ?? this._props.audio;
    if (!isRefSet(audio)) throw new Error("audio is required");

    const audioUrl = await uploadAudioInput(apiKey, audio);
    const languageCode = String(inputs.language_code ?? this._props.language_code ?? "en");

    const result = await kieExecuteTask(apiKey, "elevenlabs/speech-to-text", {
      audio_url: audioUrl,
      language_code: languageCode,
    });

    return { output: { data: result.data } };
  }
}

export class ElevenLabsV3DialogueNode extends BaseNode {
  static readonly nodeType = "kie.audio.ElevenLabsV3Dialogue";
  static readonly title = "ElevenLabs V3 Dialogue";
  static readonly description =
    "Generate multi-speaker dialogue audio using ElevenLabs V3 via the Kie.ai API. " +
    "Provide a script with speaker turns and a mapping of speaker names to ElevenLabs " +
    "voice IDs to produce a complete, multi-voice dialogue audio track.";

  defaults() {
    return {
      script: "",
      voice_assignments: {} as Record<string, string>,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const script = String(inputs.script ?? this._props.script ?? "");
    const voiceAssignments = (inputs.voice_assignments ?? this._props.voice_assignments ?? {}) as Record<string, string>;

    if (!script) throw new Error("script is required");

    const result = await kieExecuteTask(apiKey, "elevenlabs/v3-dialogue", {
      script,
      voice_assignments: voiceAssignments,
    });

    return { output: { data: result.data } };
  }
}

// ── Exports ──────────────────────────────────────────────────────────────────

export const KIE_AUDIO_NODES: readonly NodeClass[] = [
  GenerateMusicNode,
  ExtendMusicNode,
  CoverAudioNode,
  AddInstrumentalNode,
  AddVocalsNode,
  ReplaceMusicSectionNode,
  ElevenLabsTextToSpeechNode,
  ElevenLabsAudioIsolationNode,
  ElevenLabsSoundEffectNode,
  ElevenLabsSpeechToTextNode,
  ElevenLabsV3DialogueNode,
];
