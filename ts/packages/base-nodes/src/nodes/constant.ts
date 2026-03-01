import { BaseNode } from "@nodetool/node-sdk";

abstract class ConstantNode extends BaseNode {
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("value" in inputs) {
      return { output: inputs.value };
    }
    return { output: this._props.value ?? null };
  }
}

export class ConstantBaseNode extends BaseNode {
  static readonly nodeType = "nodetool.constant.Constant";
  static readonly title = "Constant";
  static readonly description = "Base constant node";

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: null };
  }
}

export class ConstantBoolNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Bool";
  static readonly title = "Bool";
  static readonly description = "Boolean constant";

  defaults() {
    return { value: false };
  }
}

export class ConstantIntegerNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Integer";
  static readonly title = "Integer";
  static readonly description = "Integer constant";

  defaults() {
    return { value: 0 };
  }
}

export class ConstantFloatNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Float";
  static readonly title = "Float";
  static readonly description = "Float constant";

  defaults() {
    return { value: 0.0 };
  }
}

export class ConstantStringNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.String";
  static readonly title = "String";
  static readonly description = "String constant";

  defaults() {
    return { value: "" };
  }
}

export class ConstantListNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.List";
  static readonly title = "List";
  static readonly description = "List constant";

  defaults() {
    return { value: [] as unknown[] };
  }
}

export class ConstantTextListNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.TextList";
  static readonly title = "Text List";
  static readonly description = "List of text constants";

  defaults() {
    return { value: [] as string[] };
  }
}

export class ConstantDictNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Dict";
  static readonly title = "Dict";
  static readonly description = "Dictionary constant";

  defaults() {
    return { value: {} as Record<string, unknown> };
  }
}

export class ConstantAudioNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Audio";
  static readonly title = "Audio";
  static readonly description = "Audio reference constant";
  defaults() { return { value: {} }; }
}

export class ConstantImageNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Image";
  static readonly title = "Image";
  static readonly description = "Image reference constant";
  defaults() { return { value: {} }; }
}

export class ConstantVideoNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Video";
  static readonly title = "Video";
  static readonly description = "Video reference constant";
  defaults() { return { value: {} }; }
}

export class ConstantDocumentNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Document";
  static readonly title = "Document";
  static readonly description = "Document reference constant";
  defaults() { return { value: {} }; }
}

export class ConstantJSONNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.JSON";
  static readonly title = "JSON";
  static readonly description = "JSON reference constant";
  defaults() { return { value: {} }; }
}

export class ConstantModel3DNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.Model3D";
  static readonly title = "Model 3D";
  static readonly description = "3D model reference constant";
  defaults() { return { value: {} }; }
}

export class ConstantDataFrameNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.DataFrame";
  static readonly title = "Data Frame";
  static readonly description = "Data frame reference constant";
  defaults() { return { value: {} }; }
}

export class ConstantAudioListNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.AudioList";
  static readonly title = "Audio List";
  static readonly description = "Audio list constant";
  defaults() { return { value: [] as unknown[] }; }
}

export class ConstantImageListNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.ImageList";
  static readonly title = "Image List";
  static readonly description = "Image list constant";
  defaults() { return { value: [] as unknown[] }; }
}

export class ConstantVideoListNode extends ConstantNode {
  static readonly nodeType = "nodetool.constant.VideoList";
  static readonly title = "Video List";
  static readonly description = "Video list constant";
  defaults() { return { value: [] as unknown[] }; }
}

export class ConstantSelectNode extends BaseNode {
  static readonly nodeType = "nodetool.constant.Select";
  static readonly title = "Select";
  static readonly description = "Select value constant";

  defaults() {
    return { value: "", options: [] as unknown[], enum_type_name: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: inputs.value ?? this._props.value ?? "" };
  }
}

export class ConstantImageSizeNode extends BaseNode {
  static readonly nodeType = "nodetool.constant.ImageSize";
  static readonly title = "Image Size";
  static readonly description = "Image size constant with width/height";

  defaults() {
    return { value: { width: 1024, height: 1024 } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = (inputs.value ?? this._props.value ?? { width: 1024, height: 1024 }) as {
      width?: number;
      height?: number;
      [key: string]: unknown;
    };
    const width = Number(value.width ?? 0);
    const height = Number(value.height ?? 0);
    return { output: value, image_size: value, width, height };
  }
}

export class ConstantDateNode extends BaseNode {
  static readonly nodeType = "nodetool.constant.Date";
  static readonly title = "Date";
  static readonly description = "Date constant";

  defaults() {
    return { year: 2024, month: 1, day: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const year = Number(inputs.year ?? this._props.year ?? 2024);
    const month = Number(inputs.month ?? this._props.month ?? 1);
    const day = Number(inputs.day ?? this._props.day ?? 1);
    return { output: { year, month, day } };
  }
}

export class ConstantDateTimeNode extends BaseNode {
  static readonly nodeType = "nodetool.constant.DateTime";
  static readonly title = "Date Time";
  static readonly description = "Date-time constant";

  defaults() {
    return {
      year: 2024,
      month: 1,
      day: 1,
      hour: 0,
      minute: 0,
      second: 0,
      millisecond: 0,
      tzinfo: "",
      utc_offset: "",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {
      output: {
        year: Number(inputs.year ?? this._props.year ?? 2024),
        month: Number(inputs.month ?? this._props.month ?? 1),
        day: Number(inputs.day ?? this._props.day ?? 1),
        hour: Number(inputs.hour ?? this._props.hour ?? 0),
        minute: Number(inputs.minute ?? this._props.minute ?? 0),
        second: Number(inputs.second ?? this._props.second ?? 0),
        millisecond: Number(inputs.millisecond ?? this._props.millisecond ?? 0),
        tzinfo: String(inputs.tzinfo ?? this._props.tzinfo ?? ""),
        utc_offset: String(inputs.utc_offset ?? this._props.utc_offset ?? ""),
      },
    };
  }
}

class ConstantModelValueNode extends ConstantNode {
  defaults() { return { value: {} }; }
}

export class ConstantASRModelNode extends ConstantModelValueNode {
  static readonly nodeType = "nodetool.constant.ASRModelConstant";
  static readonly title = "ASRModel Constant";
  static readonly description = "ASR model constant";
}

export class ConstantEmbeddingModelNode extends ConstantModelValueNode {
  static readonly nodeType = "nodetool.constant.EmbeddingModelConstant";
  static readonly title = "Embedding Model Constant";
  static readonly description = "Embedding model constant";
}

export class ConstantImageModelNode extends ConstantModelValueNode {
  static readonly nodeType = "nodetool.constant.ImageModelConstant";
  static readonly title = "Image Model Constant";
  static readonly description = "Image model constant";
}

export class ConstantLanguageModelNode extends ConstantModelValueNode {
  static readonly nodeType = "nodetool.constant.LanguageModelConstant";
  static readonly title = "Language Model Constant";
  static readonly description = "Language model constant";
}

export class ConstantTTSModelNode extends ConstantModelValueNode {
  static readonly nodeType = "nodetool.constant.TTSModelConstant";
  static readonly title = "TTSModel Constant";
  static readonly description = "TTS model constant";
}

export class ConstantVideoModelNode extends ConstantModelValueNode {
  static readonly nodeType = "nodetool.constant.VideoModelConstant";
  static readonly title = "Video Model Constant";
  static readonly description = "Video model constant";
}

export const CONSTANT_NODES = [
  ConstantBaseNode,
  ConstantBoolNode,
  ConstantIntegerNode,
  ConstantFloatNode,
  ConstantStringNode,
  ConstantListNode,
  ConstantTextListNode,
  ConstantDictNode,
  ConstantAudioNode,
  ConstantImageNode,
  ConstantVideoNode,
  ConstantDocumentNode,
  ConstantJSONNode,
  ConstantModel3DNode,
  ConstantDataFrameNode,
  ConstantAudioListNode,
  ConstantImageListNode,
  ConstantVideoListNode,
  ConstantSelectNode,
  ConstantImageSizeNode,
  ConstantDateNode,
  ConstantDateTimeNode,
  ConstantASRModelNode,
  ConstantEmbeddingModelNode,
  ConstantImageModelNode,
  ConstantLanguageModelNode,
  ConstantTTSModelNode,
  ConstantVideoModelNode,
] as const;
