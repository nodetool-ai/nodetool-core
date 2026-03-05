import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";
import {
  getApiKey,
  kieExecuteTask,
  uploadImageInput,
  uploadAudioInput,
  uploadVideoInput,
  isRefSet,
} from "./kie-base.js";

// ---------------------------------------------------------------------------
// 1. KlingTextToVideo
// ---------------------------------------------------------------------------
export class KlingTextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.KlingTextToVideo";
  static readonly title = "Kling 2.6 Text To Video";
  static readonly description =
    "Generate videos from text using Kuaishou Kling 2.6 model via Kie.ai.";

  defaults() {
    return {
      prompt:
        "A cinematic video with smooth motion, natural lighting, and high detail.",
      aspect_ratio: "16:9",
      duration: 5,
      resolution: "768P",
      seed: -1,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "kling-2.6/text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "768P"),
        duration: String(inputs.duration ?? "5"),
        seed: Number(inputs.seed ?? -1),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 2. KlingImageToVideo
// ---------------------------------------------------------------------------
export class KlingImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.KlingImageToVideo";
  static readonly title = "Kling 2.6 Image To Video";
  static readonly description =
    "Generate videos from images using Kuaishou Kling 2.6 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      sound: false,
      duration: 5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_urls: string[] = [];
    for (const img of [inputs.image1, inputs.image2, inputs.image3]) {
      if (isRefSet(img)) image_urls.push(await uploadImageInput(apiKey, img));
    }
    if (image_urls.length === 0) throw new Error("At least one image is required");
    const result = await kieExecuteTask(
      apiKey,
      "kling-2.6/image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_urls,
        sound: Boolean(inputs.sound ?? false),
        duration: String(inputs.duration ?? "5"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 3. KlingAIAvatarStandard
// ---------------------------------------------------------------------------
export class KlingAIAvatarStandardNode extends BaseNode {
  static readonly nodeType = "kie.video.KlingAIAvatarStandard";
  static readonly title = "Kling AI Avatar Standard";
  static readonly description =
    "Generate AI avatar videos using Kling v1 standard model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      mode: "standard",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const audio_url = await uploadAudioInput(apiKey, inputs.audio);
    const result = await kieExecuteTask(
      apiKey,
      "kling/v1-avatar-standard",
      {
        image_url,
        audio_url,
        prompt: String(inputs.prompt ?? ""),
        mode: String(inputs.mode ?? "standard"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 4. KlingAIAvatarPro
// ---------------------------------------------------------------------------
export class KlingAIAvatarProNode extends BaseNode {
  static readonly nodeType = "kie.video.KlingAIAvatarPro";
  static readonly title = "Kling AI Avatar Pro";
  static readonly description =
    "Generate AI avatar videos using Kling v1 pro model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      mode: "standard",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const audio_url = await uploadAudioInput(apiKey, inputs.audio);
    const result = await kieExecuteTask(
      apiKey,
      "kling/v1-avatar-pro",
      {
        image_url,
        audio_url,
        prompt: String(inputs.prompt ?? ""),
        mode: String(inputs.mode ?? "standard"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 5. GrokImagineTextToVideo
// ---------------------------------------------------------------------------
export class GrokImagineTextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.GrokImagineTextToVideo";
  static readonly title = "Grok Imagine Text To Video";
  static readonly description =
    "Generate videos from text using Grok Imagine model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      resolution: "1080p",
      duration: "medium",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "grok-imagine/text-to-video",
      {
        prompt,
        resolution: String(inputs.resolution ?? "1080p"),
        duration: String(inputs.duration ?? "medium"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 6. GrokImagineImageToVideo
// ---------------------------------------------------------------------------
export class GrokImagineImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.GrokImagineImageToVideo";
  static readonly title = "Grok Imagine Image To Video";
  static readonly description =
    "Generate videos from images using Grok Imagine model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: "medium",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "grok-imagine/image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        duration: String(inputs.duration ?? "medium"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 7. SeedanceV1LiteTextToVideo
// ---------------------------------------------------------------------------
export class SeedanceV1LiteTextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.SeedanceV1LiteTextToVideo";
  static readonly title = "Seedance V1 Lite Text To Video";
  static readonly description =
    "Generate videos from text using Seedance V1 Lite model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      resolution: "720p",
      duration: "5",
      remove_watermark: true,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "seedance/v1-lite-text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "720p"),
        duration: String(inputs.duration ?? "5"),
        remove_watermark: Boolean(inputs.remove_watermark ?? true),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 8. SeedanceV1ProTextToVideo
// ---------------------------------------------------------------------------
export class SeedanceV1ProTextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.SeedanceV1ProTextToVideo";
  static readonly title = "Seedance V1 Pro Text To Video";
  static readonly description =
    "Generate videos from text using Seedance V1 Pro model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      resolution: "720p",
      duration: "5",
      remove_watermark: true,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "seedance/v1-pro-text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "720p"),
        duration: String(inputs.duration ?? "5"),
        remove_watermark: Boolean(inputs.remove_watermark ?? true),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 9. SeedanceV1LiteImageToVideo
// ---------------------------------------------------------------------------
export class SeedanceV1LiteImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.SeedanceV1LiteImageToVideo";
  static readonly title = "Seedance V1 Lite Image To Video";
  static readonly description =
    "Generate videos from images using Seedance V1 Lite model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      resolution: "720p",
      duration: "5",
      remove_watermark: true,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_urls: string[] = [];
    for (const img of [inputs.image1, inputs.image2, inputs.image3]) {
      if (isRefSet(img)) image_urls.push(await uploadImageInput(apiKey, img));
    }
    if (image_urls.length === 0) throw new Error("At least one image is required");
    const result = await kieExecuteTask(
      apiKey,
      "seedance/v1-lite-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_urls,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "720p"),
        duration: String(inputs.duration ?? "5"),
        remove_watermark: Boolean(inputs.remove_watermark ?? true),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 10. SeedanceV1ProImageToVideo
// ---------------------------------------------------------------------------
export class SeedanceV1ProImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.SeedanceV1ProImageToVideo";
  static readonly title = "Seedance V1 Pro Image To Video";
  static readonly description =
    "Generate videos from images using Seedance V1 Pro model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      resolution: "720p",
      duration: "5",
      remove_watermark: true,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_urls: string[] = [];
    for (const img of [inputs.image1, inputs.image2, inputs.image3]) {
      if (isRefSet(img)) image_urls.push(await uploadImageInput(apiKey, img));
    }
    if (image_urls.length === 0) throw new Error("At least one image is required");
    const result = await kieExecuteTask(
      apiKey,
      "seedance/v1-pro-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_urls,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "720p"),
        duration: String(inputs.duration ?? "5"),
        remove_watermark: Boolean(inputs.remove_watermark ?? true),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 11. SeedanceV1ProFastImageToVideo
// ---------------------------------------------------------------------------
export class SeedanceV1ProFastImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.SeedanceV1ProFastImageToVideo";
  static readonly title = "Seedance V1 Pro Fast Image To Video";
  static readonly description =
    "Generate videos from images quickly using Seedance V1 Pro Fast model via Kie.ai.";

  defaults() {
    return {
      aspect_ratio: "16:9",
      resolution: "720p",
      duration: "5",
      remove_watermark: true,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_urls: string[] = [];
    for (const img of [inputs.image1, inputs.image2, inputs.image3]) {
      if (isRefSet(img)) image_urls.push(await uploadImageInput(apiKey, img));
    }
    if (image_urls.length === 0) throw new Error("At least one image is required");
    const result = await kieExecuteTask(
      apiKey,
      "seedance/v1-pro-fast-image-to-video",
      {
        image_urls,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "720p"),
        duration: String(inputs.duration ?? "5"),
        remove_watermark: Boolean(inputs.remove_watermark ?? true),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 12. HailuoTextToVideoPro
// ---------------------------------------------------------------------------
export class HailuoTextToVideoProNode extends BaseNode {
  static readonly nodeType = "kie.video.HailuoTextToVideoPro";
  static readonly title = "Hailuo 2.3 Pro Text To Video";
  static readonly description =
    "Generate videos from text using Hailuo 2.3 Pro model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: "6",
      resolution: "768P",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const resolution = String(inputs.resolution ?? "768P");
    const duration = String(inputs.duration ?? "6");
    if (resolution === "1080P" && duration === "10") {
      throw new Error("1080P resolution with 10s duration is not supported");
    }
    const result = await kieExecuteTask(
      apiKey,
      "hailuo/2-3-text-to-video-pro",
      { prompt, duration, resolution },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 13. HailuoTextToVideoStandard
// ---------------------------------------------------------------------------
export class HailuoTextToVideoStandardNode extends BaseNode {
  static readonly nodeType = "kie.video.HailuoTextToVideoStandard";
  static readonly title = "Hailuo 2.3 Standard Text To Video";
  static readonly description =
    "Generate videos from text using Hailuo 2.3 Standard model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: "6",
      resolution: "768P",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const resolution = String(inputs.resolution ?? "768P");
    const duration = String(inputs.duration ?? "6");
    if (resolution === "1080P" && duration === "10") {
      throw new Error("1080P resolution with 10s duration is not supported");
    }
    const result = await kieExecuteTask(
      apiKey,
      "hailuo/2-3-text-to-video-standard",
      { prompt, duration, resolution },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 14. HailuoImageToVideoPro
// ---------------------------------------------------------------------------
export class HailuoImageToVideoProNode extends BaseNode {
  static readonly nodeType = "kie.video.HailuoImageToVideoPro";
  static readonly title = "Hailuo 2.3 Pro Image To Video";
  static readonly description =
    "Generate videos from images using Hailuo 2.3 Pro model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: "6",
      resolution: "768P",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const resolution = String(inputs.resolution ?? "768P");
    const duration = String(inputs.duration ?? "6");
    if (resolution === "1080P" && duration === "10") {
      throw new Error("1080P resolution with 10s duration is not supported");
    }
    const result = await kieExecuteTask(
      apiKey,
      "hailuo/2-3-image-to-video-pro",
      {
        image_url,
        prompt: String(inputs.prompt ?? ""),
        duration,
        resolution,
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 15. HailuoImageToVideoStandard
// ---------------------------------------------------------------------------
export class HailuoImageToVideoStandardNode extends BaseNode {
  static readonly nodeType = "kie.video.HailuoImageToVideoStandard";
  static readonly title = "Hailuo 2.3 Standard Image To Video";
  static readonly description =
    "Generate videos from images using Hailuo 2.3 Standard model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: "6",
      resolution: "768P",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const resolution = String(inputs.resolution ?? "768P");
    const duration = String(inputs.duration ?? "6");
    if (resolution === "1080P" && duration === "10") {
      throw new Error("1080P resolution with 10s duration is not supported");
    }
    const result = await kieExecuteTask(
      apiKey,
      "hailuo/2-3-image-to-video-standard",
      {
        image_url,
        prompt: String(inputs.prompt ?? ""),
        duration,
        resolution,
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 16. Kling25TurboTextToVideo
// ---------------------------------------------------------------------------
export class Kling25TurboTextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Kling25TurboTextToVideo";
  static readonly title = "Kling 2.5 Turbo Text To Video";
  static readonly description =
    "Generate videos from text using Kling 2.5 Turbo model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: "5",
      aspect_ratio: "16:9",
      negative_prompt: "",
      cfg_scale: 0.5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "kling/v2-5-turbo-text-to-video-pro",
      {
        prompt,
        duration: String(inputs.duration ?? "5"),
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        negative_prompt: String(inputs.negative_prompt ?? ""),
        cfg_scale: Number(inputs.cfg_scale ?? 0.5),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 17. Kling25TurboImageToVideo
// ---------------------------------------------------------------------------
export class Kling25TurboImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Kling25TurboImageToVideo";
  static readonly title = "Kling 2.5 Turbo Image To Video";
  static readonly description =
    "Generate videos from images using Kling 2.5 Turbo model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: "5",
      aspect_ratio: "16:9",
      negative_prompt: "",
      cfg_scale: 0.5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "kling/v2-5-turbo-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        duration: String(inputs.duration ?? "5"),
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        negative_prompt: String(inputs.negative_prompt ?? ""),
        cfg_scale: Number(inputs.cfg_scale ?? 0.5),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 18. Sora2ProTextToVideo
// ---------------------------------------------------------------------------
export class Sora2ProTextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Sora2ProTextToVideo";
  static readonly title = "Sora 2 Pro Text To Video";
  static readonly description =
    "Generate videos from text using Sora 2 Pro model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      n_frames: "default",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "sora-2/pro-text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        n_frames: String(inputs.n_frames ?? "default"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 19. Sora2ProImageToVideo
// ---------------------------------------------------------------------------
export class Sora2ProImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Sora2ProImageToVideo";
  static readonly title = "Sora 2 Pro Image To Video";
  static readonly description =
    "Generate videos from images using Sora 2 Pro model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      n_frames: "default",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "sora-2/pro-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        n_frames: String(inputs.n_frames ?? "default"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 20. Sora2ProStoryboard
// ---------------------------------------------------------------------------
export class Sora2ProStoryboardNode extends BaseNode {
  static readonly nodeType = "kie.video.Sora2ProStoryboard";
  static readonly title = "Sora 2 Pro Storyboard";
  static readonly description =
    "Generate storyboard videos using Sora 2 Pro model with up to 5 reference images via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      images: [],
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const rawImages = Array.isArray(inputs.images) ? inputs.images : [];
    const image_urls: string[] = [];
    for (const img of rawImages.slice(0, 5)) {
      if (isRefSet(img)) image_urls.push(await uploadImageInput(apiKey, img));
    }
    const result = await kieExecuteTask(
      apiKey,
      "sora-2/pro-storyboard",
      { prompt, image_urls },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 21. Sora2TextToVideo
// ---------------------------------------------------------------------------
export class Sora2TextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Sora2TextToVideo";
  static readonly title = "Sora 2 Text To Video";
  static readonly description =
    "Generate videos from text using Sora 2 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      n_frames: "default",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "sora-2/text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        n_frames: String(inputs.n_frames ?? "default"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 22. WanMultiShotTextToVideoPro
// ---------------------------------------------------------------------------
export class WanMultiShotTextToVideoProNode extends BaseNode {
  static readonly nodeType = "kie.video.WanMultiShotTextToVideoPro";
  static readonly title = "Wan Multi-Shot Text To Video Pro";
  static readonly description =
    "Generate multi-shot videos from text using Wan Pro model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      resolution: "720P",
      duration: 5,
      shot_count: 3,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "wan/multi-shot-text-to-video-pro",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "720P"),
        duration: String(inputs.duration ?? "5"),
        shot_count: Number(inputs.shot_count ?? 3),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 23. Wan26TextToVideo
// ---------------------------------------------------------------------------
export class Wan26TextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Wan26TextToVideo";
  static readonly title = "Wan 2.6 Text To Video";
  static readonly description =
    "Generate videos from text using Wan 2.6 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      resolution: "720P",
      duration: 5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "wan/2-6-text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "720P"),
        duration: String(inputs.duration ?? "5"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 24. Wan26ImageToVideo
// ---------------------------------------------------------------------------
export class Wan26ImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Wan26ImageToVideo";
  static readonly title = "Wan 2.6 Image To Video";
  static readonly description =
    "Generate videos from images using Wan 2.6 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      resolution: "720P",
      duration: 5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "wan/2-6-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        resolution: String(inputs.resolution ?? "720P"),
        duration: String(inputs.duration ?? "5"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 25. Wan26VideoToVideo
// ---------------------------------------------------------------------------
export class Wan26VideoToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Wan26VideoToVideo";
  static readonly title = "Wan 2.6 Video To Video";
  static readonly description =
    "Transform videos using Wan 2.6 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      resolution: "720P",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const video_url = await uploadVideoInput(apiKey, inputs.video);
    const result = await kieExecuteTask(
      apiKey,
      "wan/2-6-video-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        video_url,
        resolution: String(inputs.resolution ?? "720P"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 26. TopazVideoUpscale
// ---------------------------------------------------------------------------
export class TopazVideoUpscaleNode extends BaseNode {
  static readonly nodeType = "kie.video.TopazVideoUpscale";
  static readonly title = "Topaz Video Upscale";
  static readonly description =
    "Upscale videos using Topaz model via Kie.ai.";

  defaults() {
    return {
      scale_factor: 2,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const video_url = await uploadVideoInput(apiKey, inputs.video);
    const result = await kieExecuteTask(
      apiKey,
      "topaz/video-upscale",
      {
        video_url,
        scale_factor: Number(inputs.scale_factor ?? 2),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 27. InfinitalkV1
// ---------------------------------------------------------------------------
export class InfinitalkV1Node extends BaseNode {
  static readonly nodeType = "kie.video.InfinitalkV1";
  static readonly title = "Infinitalk V1";
  static readonly description =
    "Generate talking-head videos from image and audio using Infinitalk V1 via Kie.ai.";

  defaults() {
    return {};
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const audio_url = await uploadAudioInput(apiKey, inputs.audio);
    const result = await kieExecuteTask(
      apiKey,
      "infinitalk/v1",
      { image_url, audio_url },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 28. Veo31TextToVideo
// ---------------------------------------------------------------------------
export class Veo31TextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Veo31TextToVideo";
  static readonly title = "Veo 3.1 Text To Video";
  static readonly description =
    "Generate videos from text using Google Veo 3.1 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      duration: "8",
      generate_audio: true,
      negative_prompt: "",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "veo-3-1/text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        duration: String(inputs.duration ?? "8"),
        generate_audio: Boolean(inputs.generate_audio ?? true),
        negative_prompt: String(inputs.negative_prompt ?? ""),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 29. RunwayGen3AlphaTextToVideo
// ---------------------------------------------------------------------------
export class RunwayGen3AlphaTextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.RunwayGen3AlphaTextToVideo";
  static readonly title = "Runway Gen-3 Alpha Text To Video";
  static readonly description =
    "Generate videos from text using Runway Gen-3 Alpha model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: 5,
      aspect_ratio: "16:9",
      watermark: false,
      seed: -1,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "runway/gen3-alpha-text-to-video",
      {
        prompt,
        duration: String(inputs.duration ?? "5"),
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        watermark: Boolean(inputs.watermark ?? false),
        seed: Number(inputs.seed ?? -1),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 30. RunwayGen3AlphaImageToVideo
// ---------------------------------------------------------------------------
export class RunwayGen3AlphaImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.RunwayGen3AlphaImageToVideo";
  static readonly title = "Runway Gen-3 Alpha Image To Video";
  static readonly description =
    "Generate videos from images using Runway Gen-3 Alpha model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: 5,
      aspect_ratio: "16:9",
      watermark: false,
      seed: -1,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "runway/gen3-alpha-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        duration: String(inputs.duration ?? "5"),
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        watermark: Boolean(inputs.watermark ?? false),
        seed: Number(inputs.seed ?? -1),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 31. RunwayGen3AlphaExtendVideo
// ---------------------------------------------------------------------------
export class RunwayGen3AlphaExtendVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.RunwayGen3AlphaExtendVideo";
  static readonly title = "Runway Gen-3 Alpha Extend Video";
  static readonly description =
    "Extend videos using Runway Gen-3 Alpha model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: 5,
      watermark: false,
      seed: -1,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const video_url = await uploadVideoInput(apiKey, inputs.video);
    const result = await kieExecuteTask(
      apiKey,
      "runway/gen3-alpha-extend-video",
      {
        prompt: String(inputs.prompt ?? ""),
        video_url,
        duration: String(inputs.duration ?? "5"),
        watermark: Boolean(inputs.watermark ?? false),
        seed: Number(inputs.seed ?? -1),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 32. RunwayAlephVideo
// ---------------------------------------------------------------------------
export class RunwayAlephVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.RunwayAlephVideo";
  static readonly title = "Runway Aleph Video";
  static readonly description =
    "Generate videos using Runway Aleph model with optional image reference via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: 5,
      aspect_ratio: "16:9",
      watermark: false,
      seed: -1,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const payload: Record<string, unknown> = {
      prompt: String(inputs.prompt ?? ""),
      duration: String(inputs.duration ?? "5"),
      aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
      watermark: Boolean(inputs.watermark ?? false),
      seed: Number(inputs.seed ?? -1),
    };
    if (isRefSet(inputs.image)) {
      payload.image_url = await uploadImageInput(apiKey, inputs.image);
    }
    const result = await kieExecuteTask(apiKey, "runway/aleph-video", payload, 8000, 450);
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 33. LumaModifyVideo
// ---------------------------------------------------------------------------
export class LumaModifyVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.LumaModifyVideo";
  static readonly title = "Luma Modify Video";
  static readonly description =
    "Modify existing videos using Luma AI model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      loop: false,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const video_url = await uploadVideoInput(apiKey, inputs.video);
    const result = await kieExecuteTask(
      apiKey,
      "luma/modify-video",
      {
        prompt: String(inputs.prompt ?? ""),
        video_url,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        loop: Boolean(inputs.loop ?? false),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 34. Veo31ImageToVideo
// ---------------------------------------------------------------------------
export class Veo31ImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Veo31ImageToVideo";
  static readonly title = "Veo 3.1 Image To Video";
  static readonly description =
    "Generate videos from images using Google Veo 3.1 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      duration: "8",
      generate_audio: true,
      negative_prompt: "",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "veo-3-1/image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        duration: String(inputs.duration ?? "8"),
        generate_audio: Boolean(inputs.generate_audio ?? true),
        negative_prompt: String(inputs.negative_prompt ?? ""),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 35. Veo31ReferenceToVideo
// ---------------------------------------------------------------------------
export class Veo31ReferenceToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Veo31ReferenceToVideo";
  static readonly title = "Veo 3.1 Reference To Video";
  static readonly description =
    "Generate videos from multiple reference images using Google Veo 3.1 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      images: [],
      aspect_ratio: "16:9",
      duration: "8",
      generate_audio: true,
      negative_prompt: "",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const rawImages = Array.isArray(inputs.images) ? inputs.images : [];
    const image_urls: string[] = [];
    for (const img of rawImages) {
      if (isRefSet(img)) image_urls.push(await uploadImageInput(apiKey, img));
    }
    const result = await kieExecuteTask(
      apiKey,
      "veo-3-1/reference-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_urls,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        duration: String(inputs.duration ?? "8"),
        generate_audio: Boolean(inputs.generate_audio ?? true),
        negative_prompt: String(inputs.negative_prompt ?? ""),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 36. KlingMotionControl
// ---------------------------------------------------------------------------
export class KlingMotionControlNode extends BaseNode {
  static readonly nodeType = "kie.video.KlingMotionControl";
  static readonly title = "Kling Motion Control";
  static readonly description =
    "Generate videos with motion control using Kling model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      motion_type: "camera",
      duration: 5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "kling/motion-control",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        motion_type: String(inputs.motion_type ?? "camera"),
        duration: String(inputs.duration ?? "5"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 37. Kling21TextToVideo
// ---------------------------------------------------------------------------
export class Kling21TextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Kling21TextToVideo";
  static readonly title = "Kling 2.1 Text To Video";
  static readonly description =
    "Generate videos from text using Kling 2.1 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      duration: 5,
      negative_prompt: "",
      cfg_scale: 0.5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "kling/v2-1-text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        duration: String(inputs.duration ?? "5"),
        negative_prompt: String(inputs.negative_prompt ?? ""),
        cfg_scale: Number(inputs.cfg_scale ?? 0.5),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 38. Kling21ImageToVideo
// ---------------------------------------------------------------------------
export class Kling21ImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Kling21ImageToVideo";
  static readonly title = "Kling 2.1 Image To Video";
  static readonly description =
    "Generate videos from images using Kling 2.1 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: 5,
      negative_prompt: "",
      cfg_scale: 0.5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "kling/v2-1-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        duration: String(inputs.duration ?? "5"),
        negative_prompt: String(inputs.negative_prompt ?? ""),
        cfg_scale: Number(inputs.cfg_scale ?? 0.5),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 39. Wan25TextToVideo
// ---------------------------------------------------------------------------
export class Wan25TextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Wan25TextToVideo";
  static readonly title = "Wan 2.5 Text To Video";
  static readonly description =
    "Generate videos from text using Wan 2.5 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      resolution: "720P",
      duration: 5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "wan/2-5-text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "720P"),
        duration: String(inputs.duration ?? "5"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 40. Wan25ImageToVideo
// ---------------------------------------------------------------------------
export class Wan25ImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Wan25ImageToVideo";
  static readonly title = "Wan 2.5 Image To Video";
  static readonly description =
    "Generate videos from images using Wan 2.5 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      resolution: "720P",
      duration: 5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "wan/2-5-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        resolution: String(inputs.resolution ?? "720P"),
        duration: String(inputs.duration ?? "5"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 41. WanAnimate
// ---------------------------------------------------------------------------
export class WanAnimateNode extends BaseNode {
  static readonly nodeType = "kie.video.WanAnimate";
  static readonly title = "Wan Animate";
  static readonly description =
    "Animate images using Wan model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      resolution: "720P",
      duration: 5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "wan/animate",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        resolution: String(inputs.resolution ?? "720P"),
        duration: String(inputs.duration ?? "5"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 42. WanSpeechToVideo
// ---------------------------------------------------------------------------
export class WanSpeechToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.WanSpeechToVideo";
  static readonly title = "Wan Speech To Video";
  static readonly description =
    "Generate speech-driven videos using Wan model with image and audio inputs via Kie.ai.";

  defaults() {
    return {
      prompt: "",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const audio_url = await uploadAudioInput(apiKey, inputs.audio);
    const result = await kieExecuteTask(
      apiKey,
      "wan/speech-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        audio_url,
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 43. Wan22TextToVideo
// ---------------------------------------------------------------------------
export class Wan22TextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Wan22TextToVideo";
  static readonly title = "Wan 2.2 Text To Video";
  static readonly description =
    "Generate videos from text using Wan 2.2 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "16:9",
      resolution: "720P",
      duration: 5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "wan/2-2-text-to-video",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "16:9"),
        resolution: String(inputs.resolution ?? "720P"),
        duration: String(inputs.duration ?? "5"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 44. Wan22ImageToVideo
// ---------------------------------------------------------------------------
export class Wan22ImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Wan22ImageToVideo";
  static readonly title = "Wan 2.2 Image To Video";
  static readonly description =
    "Generate videos from images using Wan 2.2 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      resolution: "720P",
      duration: 5,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "wan/2-2-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        resolution: String(inputs.resolution ?? "720P"),
        duration: String(inputs.duration ?? "5"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 45. Hailuo02TextToVideo
// ---------------------------------------------------------------------------
export class Hailuo02TextToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Hailuo02TextToVideo";
  static readonly title = "Hailuo 0.2 Text To Video";
  static readonly description =
    "Generate videos from text using Hailuo 0.2 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: "6",
      resolution: "768P",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt is required");
    const result = await kieExecuteTask(
      apiKey,
      "hailuo/0-2-text-to-video",
      {
        prompt,
        duration: String(inputs.duration ?? "6"),
        resolution: String(inputs.resolution ?? "768P"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 46. Hailuo02ImageToVideo
// ---------------------------------------------------------------------------
export class Hailuo02ImageToVideoNode extends BaseNode {
  static readonly nodeType = "kie.video.Hailuo02ImageToVideo";
  static readonly title = "Hailuo 0.2 Image To Video";
  static readonly description =
    "Generate videos from images using Hailuo 0.2 model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      duration: "6",
      resolution: "768P",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const image_url = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "hailuo/0-2-image-to-video",
      {
        prompt: String(inputs.prompt ?? ""),
        image_url,
        duration: String(inputs.duration ?? "6"),
        resolution: String(inputs.resolution ?? "768P"),
      },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 47. Sora2WatermarkRemover
// ---------------------------------------------------------------------------
export class Sora2WatermarkRemoverNode extends BaseNode {
  static readonly nodeType = "kie.video.Sora2WatermarkRemover";
  static readonly title = "Sora 2 Watermark Remover";
  static readonly description =
    "Remove watermarks from Sora 2 generated videos via Kie.ai.";

  defaults() {
    return {};
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const video_url = await uploadVideoInput(apiKey, inputs.video);
    const result = await kieExecuteTask(
      apiKey,
      "sora-2/watermark-remover",
      { video_url },
      8000,
      450
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
export const KIE_VIDEO_NODES: readonly NodeClass[] = [
  KlingTextToVideoNode,
  KlingImageToVideoNode,
  KlingAIAvatarStandardNode,
  KlingAIAvatarProNode,
  GrokImagineTextToVideoNode,
  GrokImagineImageToVideoNode,
  SeedanceV1LiteTextToVideoNode,
  SeedanceV1ProTextToVideoNode,
  SeedanceV1LiteImageToVideoNode,
  SeedanceV1ProImageToVideoNode,
  SeedanceV1ProFastImageToVideoNode,
  HailuoTextToVideoProNode,
  HailuoTextToVideoStandardNode,
  HailuoImageToVideoProNode,
  HailuoImageToVideoStandardNode,
  Kling25TurboTextToVideoNode,
  Kling25TurboImageToVideoNode,
  Sora2ProTextToVideoNode,
  Sora2ProImageToVideoNode,
  Sora2ProStoryboardNode,
  Sora2TextToVideoNode,
  WanMultiShotTextToVideoProNode,
  Wan26TextToVideoNode,
  Wan26ImageToVideoNode,
  Wan26VideoToVideoNode,
  TopazVideoUpscaleNode,
  InfinitalkV1Node,
  Veo31TextToVideoNode,
  RunwayGen3AlphaTextToVideoNode,
  RunwayGen3AlphaImageToVideoNode,
  RunwayGen3AlphaExtendVideoNode,
  RunwayAlephVideoNode,
  LumaModifyVideoNode,
  Veo31ImageToVideoNode,
  Veo31ReferenceToVideoNode,
  KlingMotionControlNode,
  Kling21TextToVideoNode,
  Kling21ImageToVideoNode,
  Wan25TextToVideoNode,
  Wan25ImageToVideoNode,
  WanAnimateNode,
  WanSpeechToVideoNode,
  Wan22TextToVideoNode,
  Wan22ImageToVideoNode,
  Hailuo02TextToVideoNode,
  Hailuo02ImageToVideoNode,
  Sora2WatermarkRemoverNode,
];
