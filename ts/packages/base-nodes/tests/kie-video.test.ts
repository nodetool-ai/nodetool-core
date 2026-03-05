import { vi, describe, it, expect, beforeEach } from "vitest";

vi.mock("../src/nodes/kie-base.js", () => ({
  getApiKey: vi.fn(() => "test-api-key"),
  kieExecuteTask: vi.fn(async () => ({
    data: "dmlkZW9kYXRh",
    taskId: "task_456",
  })),
  uploadImageInput: vi.fn(
    async () => "https://uploaded.example.com/image.png"
  ),
  uploadAudioInput: vi.fn(
    async () => "https://uploaded.example.com/audio.mp3"
  ),
  uploadVideoInput: vi.fn(
    async () => "https://uploaded.example.com/video.mp4"
  ),
  isRefSet: vi.fn((ref: unknown) => {
    if (!ref || typeof ref !== "object") return false;
    const r = ref as Record<string, unknown>;
    return !!(r.data || r.uri);
  }),
}));

import {
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
} from "../src/nodes/kie-video.js";

import {
  kieExecuteTask,
  uploadImageInput,
  uploadAudioInput,
  uploadVideoInput,
} from "../src/nodes/kie-base.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const SECRETS = { KIE_API_KEY: "test" };
const IMG_REF = { data: "imgdata", uri: "" };
const AUDIO_REF = { data: "audiodata", uri: "" };
const VIDEO_REF = { data: "videodata", uri: "" };
const EXPECTED_OUTPUT = { output: { data: "dmlkZW9kYXRh" } };

beforeEach(() => {
  vi.clearAllMocks();
});

// ===========================================================================
// 1. KlingTextToVideoNode
// ===========================================================================
describe("KlingTextToVideoNode", () => {
  it("metadata", () => {
    expect(KlingTextToVideoNode.nodeType).toBe("kie.video.KlingTextToVideo");
    expect(KlingTextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (KlingTextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toContain("cinematic");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.duration).toBe(5);
    expect(d.resolution).toBe("768P");
    expect(d.seed).toBe(-1);
  });

  it("process succeeds with valid prompt", async () => {
    const n = new (KlingTextToVideoNode as any)();
    const result = await n.process({
      prompt: "A flying eagle",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(kieExecuteTask).toHaveBeenCalled();
  });

  it("throws on empty prompt", async () => {
    const n = new (KlingTextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 2. KlingImageToVideoNode
// ===========================================================================
describe("KlingImageToVideoNode", () => {
  it("metadata", () => {
    expect(KlingImageToVideoNode.nodeType).toBe("kie.video.KlingImageToVideo");
    expect(KlingImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (KlingImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.sound).toBe(false);
    expect(d.duration).toBe(5);
  });

  it("process with image", async () => {
    const n = new (KlingImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate this",
      image1: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });

  it("throws when no images provided", async () => {
    const n = new (KlingImageToVideoNode as any)();
    await expect(
      n.process({ prompt: "Animate", _secrets: SECRETS })
    ).rejects.toThrow("At least one image is required");
  });
});

// ===========================================================================
// 3. KlingAIAvatarStandardNode
// ===========================================================================
describe("KlingAIAvatarStandardNode", () => {
  it("metadata", () => {
    expect(KlingAIAvatarStandardNode.nodeType).toBe(
      "kie.video.KlingAIAvatarStandard"
    );
    expect(KlingAIAvatarStandardNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (KlingAIAvatarStandardNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.mode).toBe("standard");
  });

  it("process with image and audio", async () => {
    const n = new (KlingAIAvatarStandardNode as any)();
    const result = await n.process({
      image: IMG_REF,
      audio: AUDIO_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
    expect(uploadAudioInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 4. KlingAIAvatarProNode
// ===========================================================================
describe("KlingAIAvatarProNode", () => {
  it("metadata", () => {
    expect(KlingAIAvatarProNode.nodeType).toBe("kie.video.KlingAIAvatarPro");
    expect(KlingAIAvatarProNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (KlingAIAvatarProNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.mode).toBe("standard");
  });

  it("process with image and audio", async () => {
    const n = new (KlingAIAvatarProNode as any)();
    const result = await n.process({
      image: IMG_REF,
      audio: AUDIO_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
    expect(uploadAudioInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 5. GrokImagineTextToVideoNode
// ===========================================================================
describe("GrokImagineTextToVideoNode", () => {
  it("metadata", () => {
    expect(GrokImagineTextToVideoNode.nodeType).toBe(
      "kie.video.GrokImagineTextToVideo"
    );
    expect(GrokImagineTextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (GrokImagineTextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.resolution).toBe("1080p");
    expect(d.duration).toBe("medium");
  });

  it("process succeeds", async () => {
    const n = new (GrokImagineTextToVideoNode as any)();
    const result = await n.process({
      prompt: "A sunset",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (GrokImagineTextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 6. GrokImagineImageToVideoNode
// ===========================================================================
describe("GrokImagineImageToVideoNode", () => {
  it("metadata", () => {
    expect(GrokImagineImageToVideoNode.nodeType).toBe(
      "kie.video.GrokImagineImageToVideo"
    );
    expect(GrokImagineImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (GrokImagineImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe("medium");
  });

  it("process with image", async () => {
    const n = new (GrokImagineImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 7. SeedanceV1LiteTextToVideoNode
// ===========================================================================
describe("SeedanceV1LiteTextToVideoNode", () => {
  it("metadata", () => {
    expect(SeedanceV1LiteTextToVideoNode.nodeType).toBe(
      "kie.video.SeedanceV1LiteTextToVideo"
    );
    expect(SeedanceV1LiteTextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (SeedanceV1LiteTextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.resolution).toBe("720p");
    expect(d.duration).toBe("5");
    expect(d.remove_watermark).toBe(true);
  });

  it("process succeeds", async () => {
    const n = new (SeedanceV1LiteTextToVideoNode as any)();
    const result = await n.process({
      prompt: "A river",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (SeedanceV1LiteTextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 8. SeedanceV1ProTextToVideoNode
// ===========================================================================
describe("SeedanceV1ProTextToVideoNode", () => {
  it("metadata", () => {
    expect(SeedanceV1ProTextToVideoNode.nodeType).toBe(
      "kie.video.SeedanceV1ProTextToVideo"
    );
    expect(SeedanceV1ProTextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (SeedanceV1ProTextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.resolution).toBe("720p");
    expect(d.remove_watermark).toBe(true);
  });

  it("process succeeds", async () => {
    const n = new (SeedanceV1ProTextToVideoNode as any)();
    const result = await n.process({
      prompt: "A mountain",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (SeedanceV1ProTextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 9. SeedanceV1LiteImageToVideoNode
// ===========================================================================
describe("SeedanceV1LiteImageToVideoNode", () => {
  it("metadata", () => {
    expect(SeedanceV1LiteImageToVideoNode.nodeType).toBe(
      "kie.video.SeedanceV1LiteImageToVideo"
    );
    expect(SeedanceV1LiteImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (SeedanceV1LiteImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.resolution).toBe("720p");
    expect(d.remove_watermark).toBe(true);
  });

  it("process with image", async () => {
    const n = new (SeedanceV1LiteImageToVideoNode as any)();
    const result = await n.process({
      image1: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });

  it("throws when no images provided", async () => {
    const n = new (SeedanceV1LiteImageToVideoNode as any)();
    await expect(
      n.process({ _secrets: SECRETS })
    ).rejects.toThrow("At least one image is required");
  });
});

// ===========================================================================
// 10. SeedanceV1ProImageToVideoNode
// ===========================================================================
describe("SeedanceV1ProImageToVideoNode", () => {
  it("metadata", () => {
    expect(SeedanceV1ProImageToVideoNode.nodeType).toBe(
      "kie.video.SeedanceV1ProImageToVideo"
    );
    expect(SeedanceV1ProImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (SeedanceV1ProImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.resolution).toBe("720p");
    expect(d.remove_watermark).toBe(true);
  });

  it("process with image", async () => {
    const n = new (SeedanceV1ProImageToVideoNode as any)();
    const result = await n.process({
      image1: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });

  it("throws when no images provided", async () => {
    const n = new (SeedanceV1ProImageToVideoNode as any)();
    await expect(
      n.process({ _secrets: SECRETS })
    ).rejects.toThrow("At least one image is required");
  });
});

// ===========================================================================
// 11. SeedanceV1ProFastImageToVideoNode
// ===========================================================================
describe("SeedanceV1ProFastImageToVideoNode", () => {
  it("metadata", () => {
    expect(SeedanceV1ProFastImageToVideoNode.nodeType).toBe(
      "kie.video.SeedanceV1ProFastImageToVideo"
    );
    expect(SeedanceV1ProFastImageToVideoNode.title).toBeTruthy();
  });

  it("defaults — no prompt field", () => {
    const n = new (SeedanceV1ProFastImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.resolution).toBe("720p");
    expect(d.remove_watermark).toBe(true);
    expect(d).not.toHaveProperty("prompt");
  });

  it("process with image", async () => {
    const n = new (SeedanceV1ProFastImageToVideoNode as any)();
    const result = await n.process({
      image1: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });

  it("throws when no images provided", async () => {
    const n = new (SeedanceV1ProFastImageToVideoNode as any)();
    await expect(
      n.process({ _secrets: SECRETS })
    ).rejects.toThrow("At least one image is required");
  });
});

// ===========================================================================
// 12. HailuoTextToVideoProNode
// ===========================================================================
describe("HailuoTextToVideoProNode", () => {
  it("metadata", () => {
    expect(HailuoTextToVideoProNode.nodeType).toBe(
      "kie.video.HailuoTextToVideoPro"
    );
    expect(HailuoTextToVideoProNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (HailuoTextToVideoProNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe("6");
    expect(d.resolution).toBe("768P");
  });

  it("process succeeds", async () => {
    const n = new (HailuoTextToVideoProNode as any)();
    const result = await n.process({
      prompt: "A city at night",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (HailuoTextToVideoProNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });

  it("throws on 1080P with duration 10", async () => {
    const n = new (HailuoTextToVideoProNode as any)();
    await expect(
      n.process({
        prompt: "Test",
        resolution: "1080P",
        duration: "10",
        _secrets: SECRETS,
      })
    ).rejects.toThrow("1080P resolution with 10s duration is not supported");
  });
});

// ===========================================================================
// 13. HailuoTextToVideoStandardNode
// ===========================================================================
describe("HailuoTextToVideoStandardNode", () => {
  it("metadata", () => {
    expect(HailuoTextToVideoStandardNode.nodeType).toBe(
      "kie.video.HailuoTextToVideoStandard"
    );
    expect(HailuoTextToVideoStandardNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (HailuoTextToVideoStandardNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe("6");
    expect(d.resolution).toBe("768P");
  });

  it("process succeeds", async () => {
    const n = new (HailuoTextToVideoStandardNode as any)();
    const result = await n.process({
      prompt: "A forest",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (HailuoTextToVideoStandardNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });

  it("throws on 1080P with duration 10", async () => {
    const n = new (HailuoTextToVideoStandardNode as any)();
    await expect(
      n.process({
        prompt: "Test",
        resolution: "1080P",
        duration: "10",
        _secrets: SECRETS,
      })
    ).rejects.toThrow("1080P resolution with 10s duration is not supported");
  });
});

// ===========================================================================
// 14. HailuoImageToVideoProNode
// ===========================================================================
describe("HailuoImageToVideoProNode", () => {
  it("metadata", () => {
    expect(HailuoImageToVideoProNode.nodeType).toBe(
      "kie.video.HailuoImageToVideoPro"
    );
    expect(HailuoImageToVideoProNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (HailuoImageToVideoProNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe("6");
    expect(d.resolution).toBe("768P");
  });

  it("process with image", async () => {
    const n = new (HailuoImageToVideoProNode as any)();
    const result = await n.process({
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });

  it("throws on 1080P with duration 10", async () => {
    const n = new (HailuoImageToVideoProNode as any)();
    await expect(
      n.process({
        image: IMG_REF,
        resolution: "1080P",
        duration: "10",
        _secrets: SECRETS,
      })
    ).rejects.toThrow("1080P resolution with 10s duration is not supported");
  });
});

// ===========================================================================
// 15. HailuoImageToVideoStandardNode
// ===========================================================================
describe("HailuoImageToVideoStandardNode", () => {
  it("metadata", () => {
    expect(HailuoImageToVideoStandardNode.nodeType).toBe(
      "kie.video.HailuoImageToVideoStandard"
    );
    expect(HailuoImageToVideoStandardNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (HailuoImageToVideoStandardNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe("6");
    expect(d.resolution).toBe("768P");
  });

  it("process with image", async () => {
    const n = new (HailuoImageToVideoStandardNode as any)();
    const result = await n.process({
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });

  it("throws on 1080P with duration 10", async () => {
    const n = new (HailuoImageToVideoStandardNode as any)();
    await expect(
      n.process({
        image: IMG_REF,
        resolution: "1080P",
        duration: "10",
        _secrets: SECRETS,
      })
    ).rejects.toThrow("1080P resolution with 10s duration is not supported");
  });
});

// ===========================================================================
// 16. Kling25TurboTextToVideoNode
// ===========================================================================
describe("Kling25TurboTextToVideoNode", () => {
  it("metadata", () => {
    expect(Kling25TurboTextToVideoNode.nodeType).toBe(
      "kie.video.Kling25TurboTextToVideo"
    );
    expect(Kling25TurboTextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Kling25TurboTextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe("5");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.negative_prompt).toBe("");
    expect(d.cfg_scale).toBe(0.5);
  });

  it("process succeeds", async () => {
    const n = new (Kling25TurboTextToVideoNode as any)();
    const result = await n.process({
      prompt: "A car racing",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Kling25TurboTextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 17. Kling25TurboImageToVideoNode
// ===========================================================================
describe("Kling25TurboImageToVideoNode", () => {
  it("metadata", () => {
    expect(Kling25TurboImageToVideoNode.nodeType).toBe(
      "kie.video.Kling25TurboImageToVideo"
    );
    expect(Kling25TurboImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Kling25TurboImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe("5");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.negative_prompt).toBe("");
    expect(d.cfg_scale).toBe(0.5);
  });

  it("process with image", async () => {
    const n = new (Kling25TurboImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 18. Sora2ProTextToVideoNode
// ===========================================================================
describe("Sora2ProTextToVideoNode", () => {
  it("metadata", () => {
    expect(Sora2ProTextToVideoNode.nodeType).toBe(
      "kie.video.Sora2ProTextToVideo"
    );
    expect(Sora2ProTextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Sora2ProTextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.n_frames).toBe("default");
  });

  it("process succeeds", async () => {
    const n = new (Sora2ProTextToVideoNode as any)();
    const result = await n.process({
      prompt: "A whale breaching",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Sora2ProTextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 19. Sora2ProImageToVideoNode
// ===========================================================================
describe("Sora2ProImageToVideoNode", () => {
  it("metadata", () => {
    expect(Sora2ProImageToVideoNode.nodeType).toBe(
      "kie.video.Sora2ProImageToVideo"
    );
    expect(Sora2ProImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Sora2ProImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.n_frames).toBe("default");
  });

  it("process with image", async () => {
    const n = new (Sora2ProImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 20. Sora2ProStoryboardNode
// ===========================================================================
describe("Sora2ProStoryboardNode", () => {
  it("metadata", () => {
    expect(Sora2ProStoryboardNode.nodeType).toBe(
      "kie.video.Sora2ProStoryboard"
    );
    expect(Sora2ProStoryboardNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Sora2ProStoryboardNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.images).toEqual([]);
  });

  it("process succeeds with prompt", async () => {
    const n = new (Sora2ProStoryboardNode as any)();
    const result = await n.process({
      prompt: "A story",
      images: [IMG_REF],
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Sora2ProStoryboardNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 21. Sora2TextToVideoNode
// ===========================================================================
describe("Sora2TextToVideoNode", () => {
  it("metadata", () => {
    expect(Sora2TextToVideoNode.nodeType).toBe("kie.video.Sora2TextToVideo");
    expect(Sora2TextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Sora2TextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.n_frames).toBe("default");
  });

  it("process succeeds", async () => {
    const n = new (Sora2TextToVideoNode as any)();
    const result = await n.process({
      prompt: "A spaceship launch",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Sora2TextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 22. WanMultiShotTextToVideoProNode
// ===========================================================================
describe("WanMultiShotTextToVideoProNode", () => {
  it("metadata", () => {
    expect(WanMultiShotTextToVideoProNode.nodeType).toBe(
      "kie.video.WanMultiShotTextToVideoPro"
    );
    expect(WanMultiShotTextToVideoProNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (WanMultiShotTextToVideoProNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.resolution).toBe("720P");
    expect(d.duration).toBe(5);
    expect(d.shot_count).toBe(3);
  });

  it("process succeeds", async () => {
    const n = new (WanMultiShotTextToVideoProNode as any)();
    const result = await n.process({
      prompt: "A multi-shot video",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (WanMultiShotTextToVideoProNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 23. Wan26TextToVideoNode
// ===========================================================================
describe("Wan26TextToVideoNode", () => {
  it("metadata", () => {
    expect(Wan26TextToVideoNode.nodeType).toBe("kie.video.Wan26TextToVideo");
    expect(Wan26TextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Wan26TextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.resolution).toBe("720P");
    expect(d.duration).toBe(5);
  });

  it("process succeeds", async () => {
    const n = new (Wan26TextToVideoNode as any)();
    const result = await n.process({
      prompt: "A waterfall",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Wan26TextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 24. Wan26ImageToVideoNode
// ===========================================================================
describe("Wan26ImageToVideoNode", () => {
  it("metadata", () => {
    expect(Wan26ImageToVideoNode.nodeType).toBe("kie.video.Wan26ImageToVideo");
    expect(Wan26ImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Wan26ImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.resolution).toBe("720P");
    expect(d.duration).toBe(5);
  });

  it("process with image", async () => {
    const n = new (Wan26ImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 25. Wan26VideoToVideoNode
// ===========================================================================
describe("Wan26VideoToVideoNode", () => {
  it("metadata", () => {
    expect(Wan26VideoToVideoNode.nodeType).toBe("kie.video.Wan26VideoToVideo");
    expect(Wan26VideoToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Wan26VideoToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.resolution).toBe("720P");
  });

  it("process with video", async () => {
    const n = new (Wan26VideoToVideoNode as any)();
    const result = await n.process({
      prompt: "Transform",
      video: VIDEO_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadVideoInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 26. TopazVideoUpscaleNode
// ===========================================================================
describe("TopazVideoUpscaleNode", () => {
  it("metadata", () => {
    expect(TopazVideoUpscaleNode.nodeType).toBe("kie.video.TopazVideoUpscale");
    expect(TopazVideoUpscaleNode.title).toBeTruthy();
  });

  it("defaults — no prompt", () => {
    const n = new (TopazVideoUpscaleNode as any)();
    const d = n.defaults();
    expect(d.scale_factor).toBe(2);
    expect(d).not.toHaveProperty("prompt");
  });

  it("process with video", async () => {
    const n = new (TopazVideoUpscaleNode as any)();
    const result = await n.process({
      video: VIDEO_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadVideoInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 27. InfinitalkV1Node
// ===========================================================================
describe("InfinitalkV1Node", () => {
  it("metadata", () => {
    expect(InfinitalkV1Node.nodeType).toBe("kie.video.InfinitalkV1");
    expect(InfinitalkV1Node.title).toBeTruthy();
  });

  it("defaults — empty", () => {
    const n = new (InfinitalkV1Node as any)();
    const d = n.defaults();
    expect(Object.keys(d).length).toBe(0);
  });

  it("process with image and audio", async () => {
    const n = new (InfinitalkV1Node as any)();
    const result = await n.process({
      image: IMG_REF,
      audio: AUDIO_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
    expect(uploadAudioInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 28. Veo31TextToVideoNode
// ===========================================================================
describe("Veo31TextToVideoNode", () => {
  it("metadata", () => {
    expect(Veo31TextToVideoNode.nodeType).toBe("kie.video.Veo31TextToVideo");
    expect(Veo31TextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Veo31TextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.duration).toBe("8");
    expect(d.generate_audio).toBe(true);
    expect(d.negative_prompt).toBe("");
  });

  it("process succeeds", async () => {
    const n = new (Veo31TextToVideoNode as any)();
    const result = await n.process({
      prompt: "A galaxy",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Veo31TextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 29. RunwayGen3AlphaTextToVideoNode
// ===========================================================================
describe("RunwayGen3AlphaTextToVideoNode", () => {
  it("metadata", () => {
    expect(RunwayGen3AlphaTextToVideoNode.nodeType).toBe(
      "kie.video.RunwayGen3AlphaTextToVideo"
    );
    expect(RunwayGen3AlphaTextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (RunwayGen3AlphaTextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe(5);
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.watermark).toBe(false);
    expect(d.seed).toBe(-1);
  });

  it("process succeeds", async () => {
    const n = new (RunwayGen3AlphaTextToVideoNode as any)();
    const result = await n.process({
      prompt: "A robot walking",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (RunwayGen3AlphaTextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 30. RunwayGen3AlphaImageToVideoNode
// ===========================================================================
describe("RunwayGen3AlphaImageToVideoNode", () => {
  it("metadata", () => {
    expect(RunwayGen3AlphaImageToVideoNode.nodeType).toBe(
      "kie.video.RunwayGen3AlphaImageToVideo"
    );
    expect(RunwayGen3AlphaImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (RunwayGen3AlphaImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe(5);
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.watermark).toBe(false);
    expect(d.seed).toBe(-1);
  });

  it("process with image", async () => {
    const n = new (RunwayGen3AlphaImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 31. RunwayGen3AlphaExtendVideoNode
// ===========================================================================
describe("RunwayGen3AlphaExtendVideoNode", () => {
  it("metadata", () => {
    expect(RunwayGen3AlphaExtendVideoNode.nodeType).toBe(
      "kie.video.RunwayGen3AlphaExtendVideo"
    );
    expect(RunwayGen3AlphaExtendVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (RunwayGen3AlphaExtendVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe(5);
    expect(d.watermark).toBe(false);
    expect(d.seed).toBe(-1);
  });

  it("process with video", async () => {
    const n = new (RunwayGen3AlphaExtendVideoNode as any)();
    const result = await n.process({
      prompt: "Continue",
      video: VIDEO_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadVideoInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 32. RunwayAlephVideoNode
// ===========================================================================
describe("RunwayAlephVideoNode", () => {
  it("metadata", () => {
    expect(RunwayAlephVideoNode.nodeType).toBe("kie.video.RunwayAlephVideo");
    expect(RunwayAlephVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (RunwayAlephVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe(5);
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.watermark).toBe(false);
    expect(d.seed).toBe(-1);
  });

  it("process without image", async () => {
    const n = new (RunwayAlephVideoNode as any)();
    const result = await n.process({
      prompt: "A scene",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).not.toHaveBeenCalled();
  });

  it("process with optional image", async () => {
    const n = new (RunwayAlephVideoNode as any)();
    const result = await n.process({
      prompt: "A scene",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 33. LumaModifyVideoNode
// ===========================================================================
describe("LumaModifyVideoNode", () => {
  it("metadata", () => {
    expect(LumaModifyVideoNode.nodeType).toBe("kie.video.LumaModifyVideo");
    expect(LumaModifyVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (LumaModifyVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.loop).toBe(false);
  });

  it("process with video", async () => {
    const n = new (LumaModifyVideoNode as any)();
    const result = await n.process({
      prompt: "Add effects",
      video: VIDEO_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadVideoInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 34. Veo31ImageToVideoNode
// ===========================================================================
describe("Veo31ImageToVideoNode", () => {
  it("metadata", () => {
    expect(Veo31ImageToVideoNode.nodeType).toBe("kie.video.Veo31ImageToVideo");
    expect(Veo31ImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Veo31ImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.duration).toBe("8");
    expect(d.generate_audio).toBe(true);
    expect(d.negative_prompt).toBe("");
  });

  it("process with image", async () => {
    const n = new (Veo31ImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 35. Veo31ReferenceToVideoNode
// ===========================================================================
describe("Veo31ReferenceToVideoNode", () => {
  it("metadata", () => {
    expect(Veo31ReferenceToVideoNode.nodeType).toBe(
      "kie.video.Veo31ReferenceToVideo"
    );
    expect(Veo31ReferenceToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Veo31ReferenceToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.images).toEqual([]);
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.duration).toBe("8");
    expect(d.generate_audio).toBe(true);
    expect(d.negative_prompt).toBe("");
  });

  it("process with reference images", async () => {
    const n = new (Veo31ReferenceToVideoNode as any)();
    const result = await n.process({
      prompt: "Generate",
      images: [IMG_REF, IMG_REF],
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalledTimes(2);
  });
});

// ===========================================================================
// 36. KlingMotionControlNode
// ===========================================================================
describe("KlingMotionControlNode", () => {
  it("metadata", () => {
    expect(KlingMotionControlNode.nodeType).toBe(
      "kie.video.KlingMotionControl"
    );
    expect(KlingMotionControlNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (KlingMotionControlNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.motion_type).toBe("camera");
    expect(d.duration).toBe(5);
  });

  it("process with image", async () => {
    const n = new (KlingMotionControlNode as any)();
    const result = await n.process({
      prompt: "Pan left",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 37. Kling21TextToVideoNode
// ===========================================================================
describe("Kling21TextToVideoNode", () => {
  it("metadata", () => {
    expect(Kling21TextToVideoNode.nodeType).toBe(
      "kie.video.Kling21TextToVideo"
    );
    expect(Kling21TextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Kling21TextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.duration).toBe(5);
    expect(d.negative_prompt).toBe("");
    expect(d.cfg_scale).toBe(0.5);
  });

  it("process succeeds", async () => {
    const n = new (Kling21TextToVideoNode as any)();
    const result = await n.process({
      prompt: "A dolphin jumping",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Kling21TextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 38. Kling21ImageToVideoNode
// ===========================================================================
describe("Kling21ImageToVideoNode", () => {
  it("metadata", () => {
    expect(Kling21ImageToVideoNode.nodeType).toBe(
      "kie.video.Kling21ImageToVideo"
    );
    expect(Kling21ImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Kling21ImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe(5);
    expect(d.negative_prompt).toBe("");
    expect(d.cfg_scale).toBe(0.5);
  });

  it("process with image", async () => {
    const n = new (Kling21ImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 39. Wan25TextToVideoNode
// ===========================================================================
describe("Wan25TextToVideoNode", () => {
  it("metadata", () => {
    expect(Wan25TextToVideoNode.nodeType).toBe("kie.video.Wan25TextToVideo");
    expect(Wan25TextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Wan25TextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.resolution).toBe("720P");
    expect(d.duration).toBe(5);
  });

  it("process succeeds", async () => {
    const n = new (Wan25TextToVideoNode as any)();
    const result = await n.process({
      prompt: "A sunset over the ocean",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Wan25TextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 40. Wan25ImageToVideoNode
// ===========================================================================
describe("Wan25ImageToVideoNode", () => {
  it("metadata", () => {
    expect(Wan25ImageToVideoNode.nodeType).toBe("kie.video.Wan25ImageToVideo");
    expect(Wan25ImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Wan25ImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.resolution).toBe("720P");
    expect(d.duration).toBe(5);
  });

  it("process with image", async () => {
    const n = new (Wan25ImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 41. WanAnimateNode
// ===========================================================================
describe("WanAnimateNode", () => {
  it("metadata", () => {
    expect(WanAnimateNode.nodeType).toBe("kie.video.WanAnimate");
    expect(WanAnimateNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (WanAnimateNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.resolution).toBe("720P");
    expect(d.duration).toBe(5);
  });

  it("process with image", async () => {
    const n = new (WanAnimateNode as any)();
    const result = await n.process({
      prompt: "Bring to life",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 42. WanSpeechToVideoNode
// ===========================================================================
describe("WanSpeechToVideoNode", () => {
  it("metadata", () => {
    expect(WanSpeechToVideoNode.nodeType).toBe("kie.video.WanSpeechToVideo");
    expect(WanSpeechToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (WanSpeechToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
  });

  it("process with image and audio", async () => {
    const n = new (WanSpeechToVideoNode as any)();
    const result = await n.process({
      prompt: "Talk",
      image: IMG_REF,
      audio: AUDIO_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
    expect(uploadAudioInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 43. Wan22TextToVideoNode
// ===========================================================================
describe("Wan22TextToVideoNode", () => {
  it("metadata", () => {
    expect(Wan22TextToVideoNode.nodeType).toBe("kie.video.Wan22TextToVideo");
    expect(Wan22TextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Wan22TextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.aspect_ratio).toBe("16:9");
    expect(d.resolution).toBe("720P");
    expect(d.duration).toBe(5);
  });

  it("process succeeds", async () => {
    const n = new (Wan22TextToVideoNode as any)();
    const result = await n.process({
      prompt: "A snow scene",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Wan22TextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 44. Wan22ImageToVideoNode
// ===========================================================================
describe("Wan22ImageToVideoNode", () => {
  it("metadata", () => {
    expect(Wan22ImageToVideoNode.nodeType).toBe("kie.video.Wan22ImageToVideo");
    expect(Wan22ImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Wan22ImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.resolution).toBe("720P");
    expect(d.duration).toBe(5);
  });

  it("process with image", async () => {
    const n = new (Wan22ImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 45. Hailuo02TextToVideoNode
// ===========================================================================
describe("Hailuo02TextToVideoNode", () => {
  it("metadata", () => {
    expect(Hailuo02TextToVideoNode.nodeType).toBe(
      "kie.video.Hailuo02TextToVideo"
    );
    expect(Hailuo02TextToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Hailuo02TextToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe("6");
    expect(d.resolution).toBe("768P");
  });

  it("process succeeds", async () => {
    const n = new (Hailuo02TextToVideoNode as any)();
    const result = await n.process({
      prompt: "A dance",
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
  });

  it("throws on empty prompt", async () => {
    const n = new (Hailuo02TextToVideoNode as any)();
    await expect(
      n.process({ prompt: "", _secrets: SECRETS })
    ).rejects.toThrow("Prompt is required");
  });
});

// ===========================================================================
// 46. Hailuo02ImageToVideoNode
// ===========================================================================
describe("Hailuo02ImageToVideoNode", () => {
  it("metadata", () => {
    expect(Hailuo02ImageToVideoNode.nodeType).toBe(
      "kie.video.Hailuo02ImageToVideo"
    );
    expect(Hailuo02ImageToVideoNode.title).toBeTruthy();
  });

  it("defaults", () => {
    const n = new (Hailuo02ImageToVideoNode as any)();
    const d = n.defaults();
    expect(d.prompt).toBe("");
    expect(d.duration).toBe("6");
    expect(d.resolution).toBe("768P");
  });

  it("process with image", async () => {
    const n = new (Hailuo02ImageToVideoNode as any)();
    const result = await n.process({
      prompt: "Animate",
      image: IMG_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadImageInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// 47. Sora2WatermarkRemoverNode
// ===========================================================================
describe("Sora2WatermarkRemoverNode", () => {
  it("metadata", () => {
    expect(Sora2WatermarkRemoverNode.nodeType).toBe(
      "kie.video.Sora2WatermarkRemover"
    );
    expect(Sora2WatermarkRemoverNode.title).toBeTruthy();
  });

  it("defaults — empty", () => {
    const n = new (Sora2WatermarkRemoverNode as any)();
    const d = n.defaults();
    expect(Object.keys(d).length).toBe(0);
  });

  it("process with video", async () => {
    const n = new (Sora2WatermarkRemoverNode as any)();
    const result = await n.process({
      video: VIDEO_REF,
      _secrets: SECRETS,
    });
    expect(result).toEqual(EXPECTED_OUTPUT);
    expect(uploadVideoInput).toHaveBeenCalled();
  });
});

// ===========================================================================
// Cross-cutting: all 47 nodes have kie.video.* nodeType prefix
// ===========================================================================
describe("All KIE video nodes", () => {
  const allNodeClasses = [
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
  ] as any[];

  it("has exactly 47 node classes", () => {
    expect(allNodeClasses.length).toBe(47);
  });

  it.each(allNodeClasses.map((c) => [c.nodeType, c]))(
    "%s starts with kie.video.",
    (_type, cls) => {
      expect(cls.nodeType).toMatch(/^kie\.video\./);
    }
  );

  it.each(allNodeClasses.map((c) => [c.nodeType, c]))(
    "%s has a non-empty title",
    (_type, cls) => {
      expect(typeof cls.title).toBe("string");
      expect(cls.title.length).toBeGreaterThan(0);
    }
  );

  it.each(allNodeClasses.map((c) => [c.nodeType, c]))(
    "%s defaults() returns an object",
    (_type, cls) => {
      const n = new cls();
      const d = n.defaults();
      expect(typeof d).toBe("object");
    }
  );
});
