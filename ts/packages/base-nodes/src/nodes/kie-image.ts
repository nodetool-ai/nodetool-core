import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";
import { getApiKey, kieExecuteTask, uploadImageInput, isRefSet } from "./kie-base.js";

// ---------------------------------------------------------------------------
// 1. Flux2ProTextToImage
// ---------------------------------------------------------------------------
export class Flux2ProTextToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.Flux2ProTextToImage";
  static readonly title = "Flux 2 Pro Text To Image";
  static readonly description =
    "Generate images using Black Forest Labs Flux 2 Pro Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "flux-2/pro-text-to-image",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 2. Flux2ProImageToImage
// ---------------------------------------------------------------------------
export class Flux2ProImageToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.Flux2ProImageToImage";
  static readonly title = "Flux 2 Pro Image To Image";
  static readonly description =
    "Transform existing images using Black Forest Labs Flux 2 Pro Image-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      images: [],
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const images = (inputs.images as unknown[]) ?? [];
    const input_urls: string[] = [];
    for (const img of images) {
      if (img && typeof img === "object" && ((img as Record<string, unknown>).data || (img as Record<string, unknown>).uri)) {
        input_urls.push(await uploadImageInput(apiKey, img));
      }
    }
    const result = await kieExecuteTask(
      apiKey,
      "flux-2/pro-image-to-image",
      {
        prompt,
        input_urls,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 3. Flux2FlexTextToImage
// ---------------------------------------------------------------------------
export class Flux2FlexTextToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.Flux2FlexTextToImage";
  static readonly title = "Flux 2 Flex Text To Image";
  static readonly description =
    "Generate images using Black Forest Labs Flux 2 Flex Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "flux-2/flex-text-to-image",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 4. Flux2FlexImageToImage
// ---------------------------------------------------------------------------
export class Flux2FlexImageToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.Flux2FlexImageToImage";
  static readonly title = "Flux 2 Flex Image To Image";
  static readonly description =
    "Transform existing images using Black Forest Labs Flux 2 Flex Image-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      images: [],
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const images = (inputs.images as unknown[]) ?? [];
    const input_urls: string[] = [];
    for (const img of images) {
      if (img && typeof img === "object" && ((img as Record<string, unknown>).data || (img as Record<string, unknown>).uri)) {
        input_urls.push(await uploadImageInput(apiKey, img));
      }
    }
    const result = await kieExecuteTask(
      apiKey,
      "flux-2/flex-image-to-image",
      {
        prompt,
        input_urls,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 5. Seedream45TextToImage
// ---------------------------------------------------------------------------
export class Seedream45TextToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.Seedream45TextToImage";
  static readonly title = "Seedream 4.5 Text To Image";
  static readonly description =
    "Generate images using the Seedream 4.5 Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "seedream/4-5-text-to-image",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 6. Seedream45Edit
// ---------------------------------------------------------------------------
export class Seedream45EditNode extends BaseNode {
  static readonly nodeType = "kie.image.Seedream45Edit";
  static readonly title = "Seedream 4.5 Edit";
  static readonly description =
    "Edit an existing image using the Seedream 4.5 editing model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      image: null,
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "seedream/4-5-edit",
      {
        prompt,
        image_url: imageUrl,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 7. ZImage
// ---------------------------------------------------------------------------
export class ZImageNode extends BaseNode {
  static readonly nodeType = "kie.image.ZImage";
  static readonly title = "Z-Image Turbo";
  static readonly description =
    "Generate images using the Z-Image Turbo model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      seed: -1,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const params: Record<string, unknown> = {
      prompt,
      aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
    };
    const seed = Number(inputs.seed ?? -1);
    if (seed >= 0) params.seed = seed;
    const result = await kieExecuteTask(apiKey, "z-image/turbo", params, 1500, 200);
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 8. NanoBanana
// ---------------------------------------------------------------------------
export class NanoBananaNode extends BaseNode {
  static readonly nodeType = "kie.image.NanoBanana";
  static readonly title = "Nano Banana";
  static readonly description =
    "Generate images using the Nano Banana Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "nano-banana/text-to-image",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 9. NanoBananaPro
// ---------------------------------------------------------------------------
export class NanoBananaProNode extends BaseNode {
  static readonly nodeType = "kie.image.NanoBananaPro";
  static readonly title = "Nano Banana Pro";
  static readonly description =
    "Generate images using the Nano Banana Pro Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "nano-banana-pro/text-to-image",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 10. FluxKontext
// ---------------------------------------------------------------------------
export class FluxKontextNode extends BaseNode {
  static readonly nodeType = "kie.image.FluxKontext";
  static readonly title = "Flux Kontext";
  static readonly description =
    "Generate or edit images using the Flux Kontext model via Kie.ai, optionally with reference images.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      images: [],
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const images = (inputs.images as unknown[]) ?? [];
    const input_urls: string[] = [];
    for (const img of images) {
      if (img && typeof img === "object" && ((img as Record<string, unknown>).data || (img as Record<string, unknown>).uri)) {
        input_urls.push(await uploadImageInput(apiKey, img));
      }
    }
    const params: Record<string, unknown> = {
      prompt,
      aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
    };
    if (input_urls.length > 0) params.input_urls = input_urls;
    const result = await kieExecuteTask(apiKey, "flux-kontext/text-to-image", params, 1500, 200);
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 11. GrokImagineTextToImage
// ---------------------------------------------------------------------------
export class GrokImagineTextToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.GrokImagineTextToImage";
  static readonly title = "Grok Imagine Text To Image";
  static readonly description =
    "Generate images using the Grok Imagine Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      n: 1,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "grok-imagine/text-to-image",
      {
        prompt,
        n: Number(inputs.n ?? 1),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 12. GrokImagineUpscale
// ---------------------------------------------------------------------------
export class GrokImagineUpscaleNode extends BaseNode {
  static readonly nodeType = "kie.image.GrokImagineUpscale";
  static readonly title = "Grok Imagine Upscale";
  static readonly description =
    "Upscale an image using the Grok Imagine Upscale model via Kie.ai.";

  defaults() {
    return {
      image: null,
      scale_factor: 2,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "grok-imagine/upscale",
      {
        image_url: imageUrl,
        scale_factor: Number(inputs.scale_factor ?? 2),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 13. QwenTextToImage
// ---------------------------------------------------------------------------
export class QwenTextToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.QwenTextToImage";
  static readonly title = "Qwen Text To Image";
  static readonly description =
    "Generate images using the Qwen Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "qwen/text-to-image",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 14. QwenImageToImage
// ---------------------------------------------------------------------------
export class QwenImageToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.QwenImageToImage";
  static readonly title = "Qwen Image To Image";
  static readonly description =
    "Transform existing images using the Qwen Image-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      image: null,
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "qwen/image-to-image",
      {
        prompt,
        image_url: imageUrl,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 15. TopazImageUpscale
// ---------------------------------------------------------------------------
export class TopazImageUpscaleNode extends BaseNode {
  static readonly nodeType = "kie.image.TopazImageUpscale";
  static readonly title = "Topaz Image Upscale";
  static readonly description =
    "Upscale and enhance images using the Topaz Image Upscale model via Kie.ai.";

  defaults() {
    return {
      image: null,
      scale_factor: 2,
      model_name: "Standard V2",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "topaz/image-upscale",
      {
        image_url: imageUrl,
        scale_factor: Number(inputs.scale_factor ?? 2),
        model_name: String(inputs.model_name ?? "Standard V2"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 16. RecraftRemoveBackground
// ---------------------------------------------------------------------------
export class RecraftRemoveBackgroundNode extends BaseNode {
  static readonly nodeType = "kie.image.RecraftRemoveBackground";
  static readonly title = "Recraft Remove Background";
  static readonly description =
    "Remove the background from an image using the Recraft Remove Background model via Kie.ai.";

  defaults() {
    return {
      image: null,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "recraft/remove-background",
      { image_url: imageUrl },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 17. IdeogramCharacter
// ---------------------------------------------------------------------------
export class IdeogramCharacterNode extends BaseNode {
  static readonly nodeType = "kie.image.IdeogramCharacter";
  static readonly title = "Ideogram Character";
  static readonly description =
    "Generate character images using the Ideogram V3 Character model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      character_description: "",
      images: [],
      rendering_speed: "DEFAULT",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const images = (inputs.images as unknown[]) ?? [];
    const input_urls: string[] = [];
    for (const img of images) {
      if (img && typeof img === "object" && ((img as Record<string, unknown>).data || (img as Record<string, unknown>).uri)) {
        input_urls.push(await uploadImageInput(apiKey, img));
      }
    }
    const params: Record<string, unknown> = {
      prompt,
      aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
      rendering_speed: String(inputs.rendering_speed ?? "DEFAULT"),
    };
    const characterDescription = String(inputs.character_description ?? "");
    if (characterDescription) params.character_description = characterDescription;
    if (input_urls.length > 0) params.input_urls = input_urls;
    const result = await kieExecuteTask(apiKey, "ideogram/v3-character", params, 1500, 200);
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 18. IdeogramCharacterEdit
// ---------------------------------------------------------------------------
export class IdeogramCharacterEditNode extends BaseNode {
  static readonly nodeType = "kie.image.IdeogramCharacterEdit";
  static readonly title = "Ideogram Character Edit";
  static readonly description =
    "Edit a character image using the Ideogram V3 Character Edit model with mask support via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      image: null,
      mask: null,
      character_description: "",
      images: [],
      rendering_speed: "DEFAULT",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const params: Record<string, unknown> = {
      prompt,
      image_url: imageUrl,
      rendering_speed: String(inputs.rendering_speed ?? "DEFAULT"),
    };
    if (isRefSet(inputs.mask)) {
      params.mask_url = await uploadImageInput(apiKey, inputs.mask);
    }
    const characterDescription = String(inputs.character_description ?? "");
    if (characterDescription) params.character_description = characterDescription;
    const refImages = (inputs.images as unknown[]) ?? [];
    const ref_urls: string[] = [];
    for (const img of refImages) {
      if (img && typeof img === "object" && ((img as Record<string, unknown>).data || (img as Record<string, unknown>).uri)) {
        ref_urls.push(await uploadImageInput(apiKey, img));
      }
    }
    if (ref_urls.length > 0) params.reference_image_urls = ref_urls;
    const result = await kieExecuteTask(apiKey, "ideogram/v3-character-edit", params, 1500, 200);
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 19. IdeogramCharacterRemix
// ---------------------------------------------------------------------------
export class IdeogramCharacterRemixNode extends BaseNode {
  static readonly nodeType = "kie.image.IdeogramCharacterRemix";
  static readonly title = "Ideogram Character Remix";
  static readonly description =
    "Remix a character image using the Ideogram V3 Character Remix model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      image: null,
      character_description: "",
      images: [],
      rendering_speed: "DEFAULT",
      style_type: "AUTO",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const params: Record<string, unknown> = {
      prompt,
      image_url: imageUrl,
      rendering_speed: String(inputs.rendering_speed ?? "DEFAULT"),
      style_type: String(inputs.style_type ?? "AUTO"),
    };
    const characterDescription = String(inputs.character_description ?? "");
    if (characterDescription) params.character_description = characterDescription;
    const refImages = (inputs.images as unknown[]) ?? [];
    const ref_urls: string[] = [];
    for (const img of refImages) {
      if (img && typeof img === "object" && ((img as Record<string, unknown>).data || (img as Record<string, unknown>).uri)) {
        ref_urls.push(await uploadImageInput(apiKey, img));
      }
    }
    if (ref_urls.length > 0) params.reference_image_urls = ref_urls;
    const result = await kieExecuteTask(apiKey, "ideogram/v3-character-remix", params, 1500, 200);
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 20. IdeogramV3Reframe
// ---------------------------------------------------------------------------
export class IdeogramV3ReframeNode extends BaseNode {
  static readonly nodeType = "kie.image.IdeogramV3Reframe";
  static readonly title = "Ideogram V3 Reframe";
  static readonly description =
    "Reframe an image to a different resolution using the Ideogram V3 Reframe model via Kie.ai.";

  defaults() {
    return {
      image: null,
      resolution: "AUTO",
      rendering_speed: "DEFAULT",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "ideogram/v3-reframe",
      {
        image_url: imageUrl,
        resolution: String(inputs.resolution ?? "AUTO"),
        rendering_speed: String(inputs.rendering_speed ?? "DEFAULT"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 21. RecraftCrispUpscale
// ---------------------------------------------------------------------------
export class RecraftCrispUpscaleNode extends BaseNode {
  static readonly nodeType = "kie.image.RecraftCrispUpscale";
  static readonly title = "Recraft Crisp Upscale";
  static readonly description =
    "Upscale images with crisp detail preservation using the Recraft Crisp Upscale model via Kie.ai.";

  defaults() {
    return {
      image: null,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "recraft/crisp-upscale",
      { image_url: imageUrl },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 22. Imagen4Fast
// ---------------------------------------------------------------------------
export class Imagen4FastNode extends BaseNode {
  static readonly nodeType = "kie.image.Imagen4Fast";
  static readonly title = "Imagen 4 Fast";
  static readonly description =
    "Generate images quickly using Google Imagen 4 Fast model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "imagen-4/fast",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 23. Imagen4Ultra
// ---------------------------------------------------------------------------
export class Imagen4UltraNode extends BaseNode {
  static readonly nodeType = "kie.image.Imagen4Ultra";
  static readonly title = "Imagen 4 Ultra";
  static readonly description =
    "Generate high-quality images using Google Imagen 4 Ultra model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "imagen-4/ultra",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 24. Imagen4
// ---------------------------------------------------------------------------
export class Imagen4Node extends BaseNode {
  static readonly nodeType = "kie.image.Imagen4";
  static readonly title = "Imagen 4";
  static readonly description =
    "Generate images using Google Imagen 4 Standard model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "imagen-4/standard",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 25. NanoBananaEdit
// ---------------------------------------------------------------------------
export class NanoBananaEditNode extends BaseNode {
  static readonly nodeType = "kie.image.NanoBananaEdit";
  static readonly title = "Nano Banana Edit";
  static readonly description =
    "Edit an image using the Nano Banana Edit model with optional mask support via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      image: null,
      mask: null,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const params: Record<string, unknown> = {
      prompt,
      image_url: imageUrl,
    };
    if (isRefSet(inputs.mask)) {
      params.mask_url = await uploadImageInput(apiKey, inputs.mask);
    }
    const result = await kieExecuteTask(apiKey, "nano-banana/edit", params, 1500, 200);
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 26. GPTImage4oTextToImage
// ---------------------------------------------------------------------------
export class GPTImage4oTextToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.GPTImage4oTextToImage";
  static readonly title = "GPT Image 4o Text To Image";
  static readonly description =
    "Generate images using the GPT Image 4o Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      quality: "standard",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "gpt-image-4o/text-to-image",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        quality: String(inputs.quality ?? "standard"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 27. GPTImage4oImageToImage
// ---------------------------------------------------------------------------
export class GPTImage4oImageToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.GPTImage4oImageToImage";
  static readonly title = "GPT Image 4o Image To Image";
  static readonly description =
    "Transform existing images using the GPT Image 4o Image-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      images: [],
      quality: "standard",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const images = (inputs.images as unknown[]) ?? [];
    const input_urls: string[] = [];
    for (const img of images) {
      if (img && typeof img === "object" && ((img as Record<string, unknown>).data || (img as Record<string, unknown>).uri)) {
        input_urls.push(await uploadImageInput(apiKey, img));
      }
    }
    const result = await kieExecuteTask(
      apiKey,
      "gpt-image-4o/image-to-image",
      {
        prompt,
        input_urls,
        quality: String(inputs.quality ?? "standard"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 28. GPTImage15TextToImage
// ---------------------------------------------------------------------------
export class GPTImage15TextToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.GPTImage15TextToImage";
  static readonly title = "GPT Image 1.5 Text To Image";
  static readonly description =
    "Generate images using the GPT Image 1.5 Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      quality: "standard",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "gpt-image-1-5/text-to-image",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        quality: String(inputs.quality ?? "standard"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 29. GPTImage15ImageToImage
// ---------------------------------------------------------------------------
export class GPTImage15ImageToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.GPTImage15ImageToImage";
  static readonly title = "GPT Image 1.5 Image To Image";
  static readonly description =
    "Transform existing images using the GPT Image 1.5 Image-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      images: [],
      quality: "standard",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const images = (inputs.images as unknown[]) ?? [];
    const input_urls: string[] = [];
    for (const img of images) {
      if (img && typeof img === "object" && ((img as Record<string, unknown>).data || (img as Record<string, unknown>).uri)) {
        input_urls.push(await uploadImageInput(apiKey, img));
      }
    }
    const result = await kieExecuteTask(
      apiKey,
      "gpt-image-1-5/image-to-image",
      {
        prompt,
        input_urls,
        quality: String(inputs.quality ?? "standard"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 30. IdeogramV3TextToImage
// ---------------------------------------------------------------------------
export class IdeogramV3TextToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.IdeogramV3TextToImage";
  static readonly title = "Ideogram V3 Text To Image";
  static readonly description =
    "Generate images using the Ideogram V3 Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      rendering_speed: "DEFAULT",
      style_type: "AUTO",
      negative_prompt: "",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const params: Record<string, unknown> = {
      prompt,
      aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
      rendering_speed: String(inputs.rendering_speed ?? "DEFAULT"),
      style_type: String(inputs.style_type ?? "AUTO"),
    };
    const negativePrompt = String(inputs.negative_prompt ?? "");
    if (negativePrompt) params.negative_prompt = negativePrompt;
    const result = await kieExecuteTask(apiKey, "ideogram/v3-text-to-image", params, 1500, 200);
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 31. IdeogramV3ImageToImage
// ---------------------------------------------------------------------------
export class IdeogramV3ImageToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.IdeogramV3ImageToImage";
  static readonly title = "Ideogram V3 Image To Image";
  static readonly description =
    "Transform existing images using the Ideogram V3 Image-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      image: null,
      aspect_ratio: "1:1",
      rendering_speed: "DEFAULT",
      style_type: "AUTO",
      image_weight: 50,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "ideogram/v3-image-to-image",
      {
        prompt,
        image_url: imageUrl,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        rendering_speed: String(inputs.rendering_speed ?? "DEFAULT"),
        style_type: String(inputs.style_type ?? "AUTO"),
        image_weight: Number(inputs.image_weight ?? 50),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 32. Seedream40TextToImage
// ---------------------------------------------------------------------------
export class Seedream40TextToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.Seedream40TextToImage";
  static readonly title = "Seedream 4.0 Text To Image";
  static readonly description =
    "Generate images using the Seedream 4.0 Text-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const result = await kieExecuteTask(
      apiKey,
      "seedream/4-0-text-to-image",
      {
        prompt,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// 33. Seedream40ImageToImage
// ---------------------------------------------------------------------------
export class Seedream40ImageToImageNode extends BaseNode {
  static readonly nodeType = "kie.image.Seedream40ImageToImage";
  static readonly title = "Seedream 4.0 Image To Image";
  static readonly description =
    "Transform existing images using the Seedream 4.0 Image-to-Image model via Kie.ai.";

  defaults() {
    return {
      prompt: "",
      image: null,
      aspect_ratio: "1:1",
      resolution: "1K",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const apiKey = getApiKey(inputs);
    const prompt = String(inputs.prompt ?? "");
    if (!prompt) throw new Error("Prompt cannot be empty");
    const imageUrl = await uploadImageInput(apiKey, inputs.image);
    const result = await kieExecuteTask(
      apiKey,
      "seedream/4-0-image-to-image",
      {
        prompt,
        image_url: imageUrl,
        aspect_ratio: String(inputs.aspect_ratio ?? "1:1"),
        resolution: String(inputs.resolution ?? "1K"),
      },
      1500,
      200
    );
    return { output: { data: result.data } };
  }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
export const KIE_IMAGE_NODES: readonly NodeClass[] = [
  Flux2ProTextToImageNode,
  Flux2ProImageToImageNode,
  Flux2FlexTextToImageNode,
  Flux2FlexImageToImageNode,
  Seedream45TextToImageNode,
  Seedream45EditNode,
  ZImageNode,
  NanoBananaNode,
  NanoBananaProNode,
  FluxKontextNode,
  GrokImagineTextToImageNode,
  GrokImagineUpscaleNode,
  QwenTextToImageNode,
  QwenImageToImageNode,
  TopazImageUpscaleNode,
  RecraftRemoveBackgroundNode,
  IdeogramCharacterNode,
  IdeogramCharacterEditNode,
  IdeogramCharacterRemixNode,
  IdeogramV3ReframeNode,
  RecraftCrispUpscaleNode,
  Imagen4FastNode,
  Imagen4UltraNode,
  Imagen4Node,
  NanoBananaEditNode,
  GPTImage4oTextToImageNode,
  GPTImage4oImageToImageNode,
  GPTImage15TextToImageNode,
  GPTImage15ImageToImageNode,
  IdeogramV3TextToImageNode,
  IdeogramV3ImageToImageNode,
  Seedream40TextToImageNode,
  Seedream40ImageToImageNode,
];
