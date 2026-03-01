import { BaseNode } from "@nodetool/node-sdk";
abstract class SimpleInputNode extends BaseNode {
  static readonly defaultValue: unknown = null;

  defaults() {
    const cls = this.constructor as typeof SimpleInputNode;
    return { value: cls.defaultValue };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const cls = this.constructor as typeof SimpleInputNode;
    return { value: this._props.value ?? cls.defaultValue };
  }
}

export class FloatInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.FloatInput";
  static readonly title = "Float Input";
  static readonly description = "Input node for Float Input";
  static readonly defaultValue = 0;
}

export class BooleanInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.BooleanInput";
  static readonly title = "Boolean Input";
  static readonly description = "Input node for Boolean Input";
  static readonly defaultValue = false;
}

export class IntegerInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.IntegerInput";
  static readonly title = "Integer Input";
  static readonly description = "Input node for Integer Input";
  static readonly defaultValue = 0;
}

export class SelectInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.SelectInput";
  static readonly title = "Select Input";
  static readonly description = "Input node for Select Input";
  static readonly defaultValue = "";
}

export class StringListInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.StringListInput";
  static readonly title = "String List Input";
  static readonly description = "Input node for String List Input";
  static readonly defaultValue = [];
}

export class FolderPathInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.FolderPathInput";
  static readonly title = "Folder Path Input";
  static readonly description = "Input node for Folder Path Input";
  static readonly defaultValue = "";
}

export class HuggingFaceModelInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.HuggingFaceModelInput";
  static readonly title = "HuggingFace Model Input";
  static readonly description = "Input node for HuggingFace Model Input";
  static readonly defaultValue = {};
}

export class ColorInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.ColorInput";
  static readonly title = "Color Input";
  static readonly description = "Input node for Color Input";
  static readonly defaultValue = {};
}

export class ImageSizeInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.ImageSizeInput";
  static readonly title = "Image Size Input";
  static readonly description = "Input node for Image Size Input";
  static readonly defaultValue = {};
}

export class LanguageModelInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.LanguageModelInput";
  static readonly title = "Language Model Input";
  static readonly description = "Input node for Language Model Input";
  static readonly defaultValue = {};
}

export class ImageModelInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.ImageModelInput";
  static readonly title = "Image Model Input";
  static readonly description = "Input node for Image Model Input";
  static readonly defaultValue = {};
}

export class VideoModelInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.VideoModelInput";
  static readonly title = "Video Model Input";
  static readonly description = "Input node for Video Model Input";
  static readonly defaultValue = {};
}

export class TTSModelInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.TTSModelInput";
  static readonly title = "TTS Model Input";
  static readonly description = "Input node for TTS Model Input";
  static readonly defaultValue = {};
}

export class ASRModelInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.ASRModelInput";
  static readonly title = "ASR Model Input";
  static readonly description = "Input node for ASR Model Input";
  static readonly defaultValue = {};
}

export class EmbeddingModelInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.EmbeddingModelInput";
  static readonly title = "Embedding Model Input";
  static readonly description = "Input node for Embedding Model Input";
  static readonly defaultValue = {};
}

export class DataframeInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.DataframeInput";
  static readonly title = "Dataframe Input";
  static readonly description = "Input node for Dataframe Input";
  static readonly defaultValue = {};
}

export class DocumentInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.DocumentInput";
  static readonly title = "Document Input";
  static readonly description = "Input node for Document Input";
  static readonly defaultValue = {};
}

export class ImageInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.ImageInput";
  static readonly title = "Image Input";
  static readonly description = "Input node for Image Input";
  static readonly defaultValue = {};
}

export class ImageListInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.ImageListInput";
  static readonly title = "Image List Input";
  static readonly description = "Input node for Image List Input";
  static readonly defaultValue = [];
}

export class VideoListInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.VideoListInput";
  static readonly title = "Video List Input";
  static readonly description = "Input node for Video List Input";
  static readonly defaultValue = [];
}

export class AudioListInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.AudioListInput";
  static readonly title = "Audio List Input";
  static readonly description = "Input node for Audio List Input";
  static readonly defaultValue = [];
}

export class TextListInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.TextListInput";
  static readonly title = "Text List Input";
  static readonly description = "Input node for Text List Input";
  static readonly defaultValue = [];
}

export class VideoInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.VideoInput";
  static readonly title = "Video Input";
  static readonly description = "Input node for Video Input";
  static readonly defaultValue = {};
}

export class AudioInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.AudioInput";
  static readonly title = "Audio Input";
  static readonly description = "Input node for Audio Input";
  static readonly defaultValue = {};
}

export class Model3DInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.Model3DInput";
  static readonly title = "Model3D Input";
  static readonly description = "Input node for Model3D Input";
  static readonly defaultValue = {};
}

export class AssetFolderInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.AssetFolderInput";
  static readonly title = "Asset Folder Input";
  static readonly description = "Input node for Asset Folder Input";
  static readonly defaultValue = {};
}

export class FilePathInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.FilePathInput";
  static readonly title = "File Path Input";
  static readonly description = "Input node for File Path Input";
  static readonly defaultValue = "";
}

export class MessageInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.MessageInput";
  static readonly title = "Message Input";
  static readonly description = "Input node for Message Input";
  static readonly defaultValue = {};
}

export class MessageListInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.MessageListInput";
  static readonly title = "Message List Input";
  static readonly description = "Input node for Message List Input";
  static readonly defaultValue = [];
}

export class StringInputNode extends SimpleInputNode {
  static readonly nodeType = "nodetool.input.StringInput";
  static readonly title = "String Input";
  static readonly description = "String workflow input";
  static readonly defaultValue = "";

  defaults() {
    return {
      value: "",
      max_length: 0,
      line_mode: "single_line",
    };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const raw = String(this._props.value ?? "");
    const max = Number(this._props.max_length ?? 0);
    if (max > 0 && raw.length > max) {
      return { value: raw.slice(0, max) };
    }
    return { value: raw };
  }
}

export class RealtimeAudioInputNode extends BaseNode {
  static readonly nodeType = "nodetool.input.RealtimeAudioInput";
  static readonly title = "Realtime Audio Input";
  static readonly description = "Streaming audio input";
  static readonly isStreamingOutput = true;

  defaults() {
    return { value: null };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { chunk: this._props.value ?? null };
  }
}

export class DocumentFileInputNode extends BaseNode {
  static readonly nodeType = "nodetool.input.DocumentFileInput";
  static readonly title = "Document File Input";
  static readonly description = "Document path input";

  defaults() {
    return { value: "" };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = String(this._props.value ?? "");
    return { document: { uri: p ? `file://${p}` : "" }, path: p };
  }
}

export class MessageDeconstructorNode extends BaseNode {
  static readonly nodeType = "nodetool.input.MessageDeconstructor";
  static readonly title = "Message Deconstructor";
  static readonly description = "Extract fields from message object";

  defaults() {
    return { value: {} };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const msg = (this._props.value ?? {}) as Record<string, unknown>;
    const content = msg.content;
    let text = "";
    let image: unknown = null;
    let audio: unknown = null;
    if (typeof content === "string") {
      text = content;
    } else if (Array.isArray(content)) {
      for (const item of content) {
        if (!item || typeof item !== "object") continue;
        const block = item as Record<string, unknown>;
        const type = String(block.type ?? "");
        if (type === "text") text = String(block.text ?? "");
        else if (type === "image") image = block.image ?? null;
        else if (type === "audio") audio = block.audio ?? null;
      }
    }
    const provider = msg.provider;
    const modelId = msg.model;
    const model =
      typeof provider === "string" && typeof modelId === "string"
        ? { provider, id: modelId }
        : null;
    return {
      id: msg.id ?? null,
      thread_id: msg.thread_id ?? null,
      role: String(msg.role ?? ""),
      text,
      image,
      audio,
      model,
    };
  }
}

export const INPUT_NODES = [
  FloatInputNode,
  BooleanInputNode,
  IntegerInputNode,
  StringInputNode,
  SelectInputNode,
  StringListInputNode,
  FolderPathInputNode,
  HuggingFaceModelInputNode,
  ColorInputNode,
  ImageSizeInputNode,
  LanguageModelInputNode,
  ImageModelInputNode,
  VideoModelInputNode,
  TTSModelInputNode,
  ASRModelInputNode,
  EmbeddingModelInputNode,
  DataframeInputNode,
  DocumentInputNode,
  ImageInputNode,
  ImageListInputNode,
  VideoListInputNode,
  AudioListInputNode,
  TextListInputNode,
  VideoInputNode,
  AudioInputNode,
  Model3DInputNode,
  RealtimeAudioInputNode,
  AssetFolderInputNode,
  FilePathInputNode,
  DocumentFileInputNode,
  MessageInputNode,
  MessageListInputNode,
  MessageDeconstructorNode,
] as const;
