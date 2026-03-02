import { BaseNode } from "@nodetool/node-sdk";

type SvgElementLike = {
  name: string;
  attributes?: Record<string, string>;
  children?: SvgElementLike[];
  content?: string;
};

function asColor(value: unknown, fallback: string): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && "value" in (value as object)) {
    return String((value as { value?: unknown }).value ?? fallback);
  }
  return fallback;
}

function elementToString(el: SvgElementLike): string {
  const attrs = Object.entries(el.attributes ?? {})
    .map(([k, v]) => `${k}="${String(v).replaceAll('"', '&quot;')}"`)
    .join(" ");
  const open = attrs ? `<${el.name} ${attrs}>` : `<${el.name}>`;
  const children = (el.children ?? []).map(elementToString).join("");
  const content = el.content ?? "";
  return `${open}${content}${children}</${el.name}>`;
}

function svgDocument(content: string, width: number, height: number, viewBox: string): string {
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${viewBox}">${content}</svg>`;
}

function normalizeContent(content: unknown): string {
  if (Array.isArray(content)) {
    return content.map((c) => normalizeContent(c)).join("\n");
  }
  if (content && typeof content === "object" && "name" in (content as object)) {
    return elementToString(content as SvgElementLike);
  }
  return String(content ?? "");
}

export class RectLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Rect";
  static readonly title = "Rectangle";
  static readonly description = "Generate SVG rectangle element with customizable position, size, and styling.";

  defaults() { return { x: 0, y: 0, width: 100, height: 100, fill: { value: "#000000" }, stroke: { value: "none" }, stroke_width: 1 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: { name: "rect", attributes: { x: String(inputs.x ?? this._props.x ?? 0), y: String(inputs.y ?? this._props.y ?? 0), width: String(inputs.width ?? this._props.width ?? 100), height: String(inputs.height ?? this._props.height ?? 100), fill: asColor(inputs.fill ?? this._props.fill, "#000000"), stroke: asColor(inputs.stroke ?? this._props.stroke, "none"), "stroke-width": String(inputs.stroke_width ?? this._props.stroke_width ?? 1) } } };
  }
}

export class CircleLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Circle";
  static readonly title = "Circle";
  static readonly description = "Generate SVG circle element with customizable position, radius, and styling.";

  defaults() { return { cx: 0, cy: 0, radius: 50, fill: { value: "#000000" }, stroke: { value: "none" }, stroke_width: 1 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: { name: "circle", attributes: { cx: String(inputs.cx ?? this._props.cx ?? 0), cy: String(inputs.cy ?? this._props.cy ?? 0), r: String(inputs.radius ?? this._props.radius ?? 50), fill: asColor(inputs.fill ?? this._props.fill, "#000000"), stroke: asColor(inputs.stroke ?? this._props.stroke, "none"), "stroke-width": String(inputs.stroke_width ?? this._props.stroke_width ?? 1) } } };
  }
}

export class EllipseLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Ellipse";
  static readonly title = "Ellipse";
  static readonly description = "Generate SVG ellipse element with customizable position, radii, and styling.";

  defaults() { return { cx: 0, cy: 0, rx: 100, ry: 50, fill: { value: "#000000" }, stroke: { value: "none" }, stroke_width: 1 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: { name: "ellipse", attributes: { cx: String(inputs.cx ?? this._props.cx ?? 0), cy: String(inputs.cy ?? this._props.cy ?? 0), rx: String(inputs.rx ?? this._props.rx ?? 100), ry: String(inputs.ry ?? this._props.ry ?? 50), fill: asColor(inputs.fill ?? this._props.fill, "#000000"), stroke: asColor(inputs.stroke ?? this._props.stroke, "none"), "stroke-width": String(inputs.stroke_width ?? this._props.stroke_width ?? 1) } } };
  }
}

export class LineLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Line";
  static readonly title = "Line";
  static readonly description = "Generate SVG line element with customizable endpoints and styling.";

  defaults() { return { x1: 0, y1: 0, x2: 100, y2: 100, stroke: { value: "#000000" }, stroke_width: 1 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: { name: "line", attributes: { x1: String(inputs.x1 ?? this._props.x1 ?? 0), y1: String(inputs.y1 ?? this._props.y1 ?? 0), x2: String(inputs.x2 ?? this._props.x2 ?? 100), y2: String(inputs.y2 ?? this._props.y2 ?? 100), stroke: asColor(inputs.stroke ?? this._props.stroke, "#000000"), "stroke-width": String(inputs.stroke_width ?? this._props.stroke_width ?? 1) } } };
  }
}

export class PolygonLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Polygon";
  static readonly title = "Polygon";
  static readonly description = "Generate SVG polygon element with multiple vertices.";

  defaults() { return { points: "", fill: { value: "#000000" }, stroke: { value: "none" }, stroke_width: 1 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: { name: "polygon", attributes: { points: String(inputs.points ?? this._props.points ?? ""), fill: asColor(inputs.fill ?? this._props.fill, "#000000"), stroke: asColor(inputs.stroke ?? this._props.stroke, "none"), "stroke-width": String(inputs.stroke_width ?? this._props.stroke_width ?? 1) } } };
  }
}

export class PathLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Path";
  static readonly title = "Path";
  static readonly description = "Generate SVG path element using path data commands.";

  defaults() { return { path_data: "", fill: { value: "#000000" }, stroke: { value: "none" }, stroke_width: 1 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: { name: "path", attributes: { d: String(inputs.path_data ?? this._props.path_data ?? ""), fill: asColor(inputs.fill ?? this._props.fill, "#000000"), stroke: asColor(inputs.stroke ?? this._props.stroke, "none"), "stroke-width": String(inputs.stroke_width ?? this._props.stroke_width ?? 1) } } };
  }
}

export class TextLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Text";
  static readonly title = "Text";
  static readonly description = "Add text elements to SVG.";

  defaults() { return { text: "", x: 0, y: 0, font_family: "Arial", font_size: 16, fill: { value: "#000000" }, text_anchor: "start" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: { name: "text", attributes: { x: String(inputs.x ?? this._props.x ?? 0), y: String(inputs.y ?? this._props.y ?? 0), "font-family": String(inputs.font_family ?? this._props.font_family ?? "Arial"), "font-size": String(inputs.font_size ?? this._props.font_size ?? 16), fill: asColor(inputs.fill ?? this._props.fill, "#000000"), "text-anchor": String(inputs.text_anchor ?? this._props.text_anchor ?? "start") }, content: String(inputs.text ?? this._props.text ?? "") } };
  }
}

export class GaussianBlurLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.GaussianBlur";
  static readonly title = "Gaussian Blur";
  static readonly description = "Apply Gaussian blur filter effect to SVG elements.";

  defaults() { return { std_deviation: 3 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: { name: "filter", attributes: { id: "filter_gaussian_blur" }, children: [{ name: "feGaussianBlur", attributes: { stdDeviation: String(inputs.std_deviation ?? this._props.std_deviation ?? 3) } }] } };
  }
}

export class DropShadowLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.DropShadow";
  static readonly title = "Drop Shadow";
  static readonly description = "Apply drop shadow filter effect to SVG elements for depth.";

  defaults() { return { std_deviation: 3, dx: 2, dy: 2, color: { value: "#000000" } }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {
      output: {
        name: "filter",
        attributes: { id: "filter_drop_shadow" },
        children: [
          { name: "feGaussianBlur", attributes: { in: "SourceAlpha", stdDeviation: String(inputs.std_deviation ?? this._props.std_deviation ?? 3) } },
          { name: "feOffset", attributes: { dx: String(inputs.dx ?? this._props.dx ?? 2), dy: String(inputs.dy ?? this._props.dy ?? 2) } },
          { name: "feFlood", attributes: { "flood-color": asColor(inputs.color ?? this._props.color, "#000000") } },
          { name: "feComposite", attributes: { operator: "in", in2: "SourceAlpha" } },
          { name: "feMerge", children: [{ name: "feMergeNode" }, { name: "feMergeNode", attributes: { in: "SourceGraphic" } }] },
        ],
      },
    };
  }
}

export class DocumentLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Document";
  static readonly title = "SVG Document";
  static readonly description = "Combine SVG elements into a complete SVG document.";

  defaults() { return { content: [], width: 800, height: 600, viewBox: "0 0 800 600" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const content = normalizeContent(inputs.content ?? this._props.content ?? []);
    const width = Number(inputs.width ?? this._props.width ?? 800);
    const height = Number(inputs.height ?? this._props.height ?? 600);
    const viewBox = String(inputs.viewBox ?? this._props.viewBox ?? "0 0 800 600");
    const doc = svgDocument(content, width, height, viewBox);
    return { output: { data: Buffer.from(doc, "utf-8").toString("base64") } };
  }
}

export class SVGToImageLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.SVGToImage";
  static readonly title = "SVG to Image";
  static readonly description = "Create an SVG document and convert it to a raster image in one step.";

  defaults() { return { content: [], width: 800, height: 600, viewBox: "0 0 800 600", scale: 1 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const content = normalizeContent(inputs.content ?? this._props.content ?? []);
    const width = Number(inputs.width ?? this._props.width ?? 800);
    const height = Number(inputs.height ?? this._props.height ?? 600);
    const viewBox = String(inputs.viewBox ?? this._props.viewBox ?? "0 0 800 600");
    const doc = svgDocument(content, width, height, viewBox);
    return { output: { data: Buffer.from(doc, "utf-8").toString("base64"), mimeType: "image/svg+xml", width, height } };
  }
}

export class GradientLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Gradient";
  static readonly title = "Gradient";
  static readonly description = "Create linear or radial gradients for SVG elements.";

  defaults() { return { gradient_type: "linearGradient", x1: 0, y1: 0, x2: 100, y2: 100, color1: { value: "#000000" }, color2: { value: "#FFFFFF" } }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const gradientType = String(inputs.gradient_type ?? this._props.gradient_type ?? "linearGradient");
    const attrs: Record<string, string> = { id: `gradient_${gradientType}` };
    if (gradientType === "linearGradient") {
      attrs.x1 = `${String(inputs.x1 ?? this._props.x1 ?? 0)}%`;
      attrs.y1 = `${String(inputs.y1 ?? this._props.y1 ?? 0)}%`;
      attrs.x2 = `${String(inputs.x2 ?? this._props.x2 ?? 100)}%`;
      attrs.y2 = `${String(inputs.y2 ?? this._props.y2 ?? 100)}%`;
    } else {
      attrs.cx = `${String(inputs.x1 ?? this._props.x1 ?? 0)}%`;
      attrs.cy = `${String(inputs.y1 ?? this._props.y1 ?? 0)}%`;
      attrs.r = `${String(inputs.x2 ?? this._props.x2 ?? 100)}%`;
    }
    return {
      output: {
        name: gradientType,
        attributes: attrs,
        children: [
          { name: "stop", attributes: { offset: "0%", style: `stop-color:${asColor(inputs.color1 ?? this._props.color1, "#000000")};stop-opacity:1` } },
          { name: "stop", attributes: { offset: "100%", style: `stop-color:${asColor(inputs.color2 ?? this._props.color2, "#FFFFFF")};stop-opacity:1` } },
        ],
      },
    };
  }
}

export class TransformLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.Transform";
  static readonly title = "Transform";
  static readonly description = "Apply transformations to SVG elements.";

  defaults() { return { content: {}, translate_x: 0, translate_y: 0, rotate: 0, scale_x: 1, scale_y: 1 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const content = { ...((inputs.content ?? this._props.content ?? {}) as SvgElementLike) };
    if (!content || typeof content !== "object" || !("name" in content)) {
      return { output: { name: "g", attributes: {}, children: [] } };
    }
    const transforms: string[] = [];
    const tx = Number(inputs.translate_x ?? this._props.translate_x ?? 0);
    const ty = Number(inputs.translate_y ?? this._props.translate_y ?? 0);
    const rotate = Number(inputs.rotate ?? this._props.rotate ?? 0);
    const sx = Number(inputs.scale_x ?? this._props.scale_x ?? 1);
    const sy = Number(inputs.scale_y ?? this._props.scale_y ?? 1);
    if (tx !== 0 || ty !== 0) transforms.push(`translate(${tx},${ty})`);
    if (rotate !== 0) transforms.push(`rotate(${rotate})`);
    if (sx !== 1 || sy !== 1) transforms.push(`scale(${sx},${sy})`);

    if (transforms.length > 0) {
      content.attributes = { ...(content.attributes ?? {}), transform: transforms.join(" ") };
    }
    return { output: content };
  }
}

export class ClipPathLibNode extends BaseNode {
  static readonly nodeType = "lib.svg.ClipPath";
  static readonly title = "Clip Path";
  static readonly description = "Create clipping paths for SVG elements.";

  defaults() { return { clip_content: {}, content: {} }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const clipContent = (inputs.clip_content ?? this._props.clip_content ?? {}) as SvgElementLike;
    const content = { ...((inputs.content ?? this._props.content ?? {}) as SvgElementLike) };
    if (!clipContent || !content || !clipContent.name || !content.name) {
      return { output: { name: "g", attributes: {}, children: [] } };
    }
    const clipId = `clip_path_${Date.now()}`;
    content.attributes = { ...(content.attributes ?? {}), "clip-path": `url(#${clipId})` };
    return { output: { name: "g", children: [{ name: "clipPath", attributes: { id: clipId }, children: [clipContent] }, content] } };
  }
}

export const LIB_SVG_NODES = [
  RectLibNode,
  CircleLibNode,
  EllipseLibNode,
  LineLibNode,
  PolygonLibNode,
  PathLibNode,
  TextLibNode,
  GaussianBlurLibNode,
  DropShadowLibNode,
  DocumentLibNode,
  SVGToImageLibNode,
  GradientLibNode,
  TransformLibNode,
  ClipPathLibNode,
] as const;
