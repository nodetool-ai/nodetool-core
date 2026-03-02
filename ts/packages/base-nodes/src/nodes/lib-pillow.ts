import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";
import sharp from "sharp";

type Desc = { nodeType: string; title: string; description: string };
type ImageRefLike = { data?: string | Uint8Array; uri?: string; [k: string]: unknown };

function decodeImage(ref: unknown): Buffer | null {
  if (!ref || typeof ref !== "object") return null;
  const data = (ref as ImageRefLike).data;
  if (!data) return null;
  if (data instanceof Uint8Array) return Buffer.from(data);
  if (typeof data === "string") return Buffer.from(data, "base64");
  return null;
}

function toRef(buf: Buffer, base?: unknown): Record<string, unknown> {
  const seed = base && typeof base === "object" ? (base as Record<string, unknown>) : {};
  return {
    ...seed,
    data: buf.toString("base64"),
  };
}

function pickImage(inputs: Record<string, unknown>, props: Record<string, unknown>): unknown {
  const keys = [
    "image",
    "input",
    "source",
    "foreground",
    "background",
    "image1",
    "image2",
    "base_image",
    "mask",
  ];
  for (const key of keys) {
    if (key in inputs) return inputs[key];
  }
  for (const key of keys) {
    if (key in props) return props[key];
  }
  return null;
}

function createPillowNode(desc: Desc): NodeClass {
  const C = class extends BaseNode {
    static readonly nodeType = desc.nodeType;
    static readonly title = desc.title;
    static readonly description = desc.description;

    defaults(): Record<string, unknown> {
      return {
        image: {},
        image1: {},
        image2: {},
        foreground: {},
        background: {},
        width: 512,
        height: 512,
        text: "",
        alpha: 0.5,
        amount: 1,
        sigma: 1.2,
        threshold: 128,
      };
    }

    async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
      const t = desc.nodeType;

      if (t === "lib.pillow.draw.Background") {
        const width = Number(inputs.width ?? this._props.width ?? 512);
        const height = Number(inputs.height ?? this._props.height ?? 512);
        const color = String((inputs.color ?? this._props.color ?? "#000000") as string);
        const buf = await sharp({
          create: { width: Math.max(1, width), height: Math.max(1, height), channels: 4, background: color },
        })
          .png()
          .toBuffer();
        return { output: { data: buf.toString("base64") } };
      }

      const baseObj = pickImage(inputs, this._props);
      const baseBytes = decodeImage(baseObj);
      if (!baseBytes) {
        return { output: baseObj ?? {} };
      }

      let img = sharp(baseBytes, { failOn: "none" });

      if (t === "lib.pillow.__init__.Blend") {
        const other = decodeImage(inputs.image2 ?? this._props.image2);
        if (other) {
          const alpha = Number(inputs.alpha ?? this._props.alpha ?? 0.5);
          const adjusted = await sharp(other)
            .ensureAlpha(Math.max(0, Math.min(1, alpha)))
            .png()
            .toBuffer();
          const mixed = await sharp(baseBytes)
            .composite([{ input: adjusted, blend: "over" }])
            .png()
            .toBuffer();
          return { output: toRef(mixed, baseObj) };
        }
      }

      if (t === "lib.pillow.__init__.Composite") {
        const fg = decodeImage(inputs.foreground ?? this._props.foreground ?? inputs.image1 ?? this._props.image1);
        if (fg) {
          const mixed = await sharp(baseBytes).composite([{ input: fg, blend: "over" }]).png().toBuffer();
          return { output: toRef(mixed, baseObj) };
        }
      }

      if (t.includes(".filter.Blur")) img = img.blur(Number(inputs.sigma ?? this._props.sigma ?? 1.2));
      else if (t.includes(".filter.Invert")) img = img.negate();
      else if (t.includes(".filter.ConvertToGrayscale")) img = img.grayscale();
      else if (t.includes(".filter.Solarize")) img = img.threshold(Number(inputs.threshold ?? this._props.threshold ?? 128));
      else if (t.includes(".filter.Smooth")) img = img.median(3);
      else if (t.includes(".filter.Emboss")) img = img.convolve({ width: 3, height: 3, kernel: [-2, -1, 0, -1, 1, 1, 0, 1, 2] });
      else if (t.includes(".filter.FindEdges") || t.includes(".filter.Canny") || t.includes(".filter.Contour"))
        img = img.convolve({ width: 3, height: 3, kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1] });
      else if (t.includes(".filter.Posterize")) img = img.png({ palette: true, colors: 16 });
      else if (t.includes(".filter.GetChannel")) {
        const channel = String(inputs.channel ?? this._props.channel ?? "red").toLowerCase();
        const idx = channel === "green" ? 1 : channel === "blue" ? 2 : 0;
        img = img.extractChannel(idx).toColourspace("b-w");
      } else if (t.includes(".filter.Expand")) {
        const border = Number(inputs.border ?? this._props.border ?? 10);
        const color = String(inputs.color ?? this._props.color ?? "black");
        img = img.extend({ top: border, left: border, right: border, bottom: border, background: color });
      } else if (t.includes(".enhance.Sharpen") || t.includes(".enhance.UnsharpMask") || t.includes(".enhance.Sharpness"))
        img = img.sharpen();
      else if (t.includes(".enhance.Equalize") || t.includes(".enhance.AutoContrast") || t.includes(".enhance.AdaptiveContrast"))
        img = img.normalize();
      else if (t.includes(".enhance.Brightness")) {
        const amount = Number(inputs.amount ?? this._props.amount ?? 1);
        img = img.modulate({ brightness: amount });
      } else if (t.includes(".enhance.Contrast") || t.includes(".enhance.Detail") || t.includes(".enhance.EdgeEnhance"))
        img = img.linear(1.15, -(128 * 0.15));
      else if (t.includes(".enhance.Color")) {
        const amount = Number(inputs.amount ?? this._props.amount ?? 1);
        img = img.modulate({ saturation: amount });
      } else if (t.includes(".enhance.RankFilter")) img = img.median(3);
      else if (t.includes(".draw.GaussianNoise")) {
        const md = await img.metadata();
        const w = md.width ?? 512;
        const h = md.height ?? 512;
        const noiseRaw = Buffer.alloc(w * h * 3);
        for (let i = 0; i < noiseRaw.length; i += 1) {
          noiseRaw[i] = Math.floor(Math.random() * 256);
        }
        const noise = await sharp(noiseRaw, { raw: { width: w, height: h, channels: 3 } })
          .png()
          .toBuffer();
        img = sharp(
          await sharp(baseBytes)
            .composite([{ input: noise, blend: "soft-light" }])
            .png()
            .toBuffer()
        );
      } else if (t.includes(".draw.RenderText")) {
        const text = String(inputs.text ?? this._props.text ?? "");
        if (text) {
          const svg = `<svg xmlns="http://www.w3.org/2000/svg"><text x="10" y="40" font-size="32" fill="white">${text
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")}</text></svg>`;
          img = sharp(await sharp(baseBytes).composite([{ input: Buffer.from(svg) }]).png().toBuffer());
        }
      } else if (t.includes(".color_grading.")) {
        if (t.endsWith("SaturationVibrance")) img = img.modulate({ saturation: 1.2 });
        else if (t.endsWith("Exposure")) img = img.modulate({ brightness: 1.1, saturation: 1.05 });
        else if (t.endsWith("ColorBalance")) img = img.tint("#f2f2ff");
        else if (t.endsWith("Vignette")) {
          const md = await img.metadata();
          const w = md.width ?? 512;
          const h = md.height ?? 512;
          const overlaySvg = `<svg xmlns='http://www.w3.org/2000/svg' width='${w}' height='${h}'><defs><radialGradient id='g'><stop offset='55%' stop-color='black' stop-opacity='0'/><stop offset='100%' stop-color='black' stop-opacity='0.35'/></radialGradient></defs><rect width='100%' height='100%' fill='url(#g)'/></svg>`;
          img = sharp(await sharp(baseBytes).composite([{ input: Buffer.from(overlaySvg), blend: "multiply" }]).png().toBuffer());
        } else if (t.endsWith("FilmLook")) img = img.modulate({ saturation: 0.9, brightness: 1.02 }).tint("#f7e8d0");
        else if (t.endsWith("SplitToning")) img = img.tint("#e6d8ff");
        else if (t.endsWith("HSLAdjust")) img = img.modulate({ saturation: 1.1, hue: 8 });
        else if (t.endsWith("LiftGammaGain")) img = img.gamma(1.1);
        else if (t.endsWith("Curves")) img = img.gamma(1.2);
        else if (t.endsWith("CDL")) img = img.linear(1.05, 0);
      }

      const out = await img.png().toBuffer();
      return { output: toRef(out, baseObj) };
    }
  };

  return C as NodeClass;
}

const DESCRIPTORS: readonly Desc[] = [
  { nodeType: "lib.pillow.__init__.Blend", title: "Blend", description: "Blend two images with adjustable alpha mixing." },
  { nodeType: "lib.pillow.__init__.Composite", title: "Composite", description: "Combine two images using a mask for advanced compositing." },
  { nodeType: "lib.pillow.color_grading.CDL", title: "CDL", description: "ASC CDL (Color Decision List) color correction." },
  { nodeType: "lib.pillow.color_grading.ColorBalance", title: "Color Balance", description: "Adjust color temperature and tint for white balance correction." },
  { nodeType: "lib.pillow.color_grading.Curves", title: "Curves", description: "RGB curves adjustment with control points for precise tonal control." },
  { nodeType: "lib.pillow.color_grading.Exposure", title: "Exposure", description: "Comprehensive tonal exposure controls similar to Lightroom/Camera Raw." },
  { nodeType: "lib.pillow.color_grading.FilmLook", title: "Film Look", description: "Apply preset cinematic film looks with adjustable intensity." },
  { nodeType: "lib.pillow.color_grading.HSLAdjust", title: "HSLAdjust", description: "Adjust hue, saturation, and luminance for specific color ranges." },
  { nodeType: "lib.pillow.color_grading.LiftGammaGain", title: "Lift Gamma Gain", description: "Three-way color corrector for shadows, midtones, and highlights." },
  { nodeType: "lib.pillow.color_grading.SaturationVibrance", title: "Saturation Vibrance", description: "Adjust color saturation with vibrance protection for skin tones." },
  { nodeType: "lib.pillow.color_grading.SplitToning", title: "Split Toning", description: "Apply different color tints to shadows and highlights." },
  { nodeType: "lib.pillow.color_grading.Vignette", title: "Vignette", description: "Apply cinematic vignette effect to darken or lighten image edges." },
  { nodeType: "lib.pillow.draw.Background", title: "Background", description: "The Background Node creates a blank background." },
  { nodeType: "lib.pillow.draw.GaussianNoise", title: "Gaussian Noise", description: "This node creates and adds Gaussian noise to an image." },
  { nodeType: "lib.pillow.draw.RenderText", title: "Render Text", description: "This node allows you to add text to images using system fonts or web fonts." },
  { nodeType: "lib.pillow.enhance.AdaptiveContrast", title: "Adaptive Contrast", description: "Applies localized contrast enhancement using adaptive techniques." },
  { nodeType: "lib.pillow.enhance.AutoContrast", title: "Auto Contrast", description: "Automatically adjusts image contrast for enhanced visual quality." },
  { nodeType: "lib.pillow.enhance.Brightness", title: "Brightness", description: "Adjusts overall image brightness to lighten or darken." },
  { nodeType: "lib.pillow.enhance.Color", title: "Color", description: "Adjusts color intensity of an image." },
  { nodeType: "lib.pillow.enhance.Contrast", title: "Contrast", description: "Adjusts image contrast to modify light-dark differences." },
  { nodeType: "lib.pillow.enhance.Detail", title: "Detail", description: "Enhances fine details in images." },
  { nodeType: "lib.pillow.enhance.EdgeEnhance", title: "Edge Enhance", description: "Enhances edge visibility by increasing contrast along boundaries." },
  { nodeType: "lib.pillow.enhance.Equalize", title: "Equalize", description: "Enhances image contrast by equalizing intensity distribution." },
  { nodeType: "lib.pillow.enhance.RankFilter", title: "Rank Filter", description: "Applies rank-based filtering to enhance or smooth image features." },
  { nodeType: "lib.pillow.enhance.Sharpen", title: "Sharpen", description: "Enhances image detail by intensifying local pixel contrast." },
  { nodeType: "lib.pillow.enhance.Sharpness", title: "Sharpness", description: "Adjusts image sharpness to enhance or reduce detail clarity." },
  { nodeType: "lib.pillow.enhance.UnsharpMask", title: "Unsharp Mask", description: "Sharpens images using the unsharp mask technique." },
  { nodeType: "lib.pillow.filter.Blur", title: "Blur", description: "Apply a Gaussian blur effect to an image." },
  { nodeType: "lib.pillow.filter.Canny", title: "Canny", description: "Apply Canny edge detection to an image." },
  { nodeType: "lib.pillow.filter.Contour", title: "Contour", description: "Apply a contour filter to highlight image edges." },
  { nodeType: "lib.pillow.filter.ConvertToGrayscale", title: "Convert To Grayscale", description: "Convert an image to grayscale." },
  { nodeType: "lib.pillow.filter.Emboss", title: "Emboss", description: "Apply an emboss filter for a 3D raised effect." },
  { nodeType: "lib.pillow.filter.Expand", title: "Expand", description: "Add a border around an image to increase its size." },
  { nodeType: "lib.pillow.filter.FindEdges", title: "Find Edges", description: "Detect and highlight edges in an image." },
  { nodeType: "lib.pillow.filter.GetChannel", title: "Get Channel", description: "Extract a specific color channel from an image." },
  { nodeType: "lib.pillow.filter.Invert", title: "Invert", description: "Invert the colors of an image." },
  { nodeType: "lib.pillow.filter.Posterize", title: "Posterize", description: "Reduce the number of colors in an image for a poster-like effect." },
  { nodeType: "lib.pillow.filter.Smooth", title: "Smooth", description: "Apply smoothing to reduce image noise and detail." },
  { nodeType: "lib.pillow.filter.Solarize", title: "Solarize", description: "Apply a solarize effect to partially invert image tones." },
] as const;

export const LIB_PILLOW_NODES: readonly NodeClass[] = DESCRIPTORS.map(createPillowNode);
