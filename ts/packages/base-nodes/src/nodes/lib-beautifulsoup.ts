import { BaseNode } from "@nodetool/node-sdk";
import * as cheerio from "cheerio";
import { convert } from "html-to-text";
import { Readability } from "@mozilla/readability";
import { JSDOM } from "jsdom";

export class BaseUrlLibNode extends BaseNode {
  static readonly nodeType = "lib.beautifulsoup.BaseUrl";
  static readonly title = "Base Url";
  static readonly description = "Extract the base URL from a given URL.";

  defaults() {
    return { url: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = String(inputs.url ?? this._props.url ?? "");
    if (!url) {
      throw new Error("URL must not be empty");
    }
    const parsed = new URL(url);
    return { output: `${parsed.protocol}//${parsed.host}` };
  }
}

export class ExtractLinksLibNode extends BaseNode {
  static readonly nodeType = "lib.beautifulsoup.ExtractLinks";
  static readonly title = "Extract Links";
  static readonly description = "Extract all links from HTML content with type classification.";

  defaults() {
    return { html: "", base_url: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const html = String(inputs.html ?? this._props.html ?? "");
    const baseUrl = String(inputs.base_url ?? this._props.base_url ?? "");
    const $ = cheerio.load(html);
    const rows: string[][] = [];

    $("a[href]").each((_, el) => {
      const href = $(el).attr("href") ?? "";
      const text = $(el).text().trim();
      const linkType =
        href.startsWith(baseUrl) || href.startsWith("/")
          ? "internal"
          : "external";
      rows.push([href, text, linkType]);
    });

    return {
      output: {
        columns: [
          { name: "href", data_type: "string" },
          { name: "text", data_type: "string" },
          { name: "type", data_type: "string" },
        ],
        data: rows,
      },
    };
  }
}

export class ExtractImagesLibNode extends BaseNode {
  static readonly nodeType = "lib.beautifulsoup.ExtractImages";
  static readonly title = "Extract Images";
  static readonly description = "Extract images from HTML content.";

  defaults() {
    return { html: "", base_url: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const html = String(inputs.html ?? this._props.html ?? "");
    const baseUrl = String(inputs.base_url ?? this._props.base_url ?? "");
    const $ = cheerio.load(html);
    const images: Array<{ uri: string; type: string }> = [];

    $("img[src]").each((_, el) => {
      const src = $(el).attr("src") ?? "";
      const fullUrl = new URL(src, baseUrl || undefined).href;
      images.push({ uri: fullUrl, type: "image" });
    });

    return { output: images };
  }
}

export class ExtractAudioLibNode extends BaseNode {
  static readonly nodeType = "lib.beautifulsoup.ExtractAudio";
  static readonly title = "Extract Audio";
  static readonly description = "Extract audio elements from HTML content.";

  defaults() {
    return { html: "", base_url: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const html = String(inputs.html ?? this._props.html ?? "");
    const baseUrl = String(inputs.base_url ?? this._props.base_url ?? "");
    const $ = cheerio.load(html);
    const audioList: Array<{ uri: string; type: string }> = [];

    $("audio, audio source").each((_, el) => {
      const src = $(el).attr("src");
      if (src) {
        const fullUrl = new URL(src, baseUrl || undefined).href;
        audioList.push({ uri: fullUrl, type: "audio" });
      }
    });

    return { output: audioList };
  }
}

export class ExtractVideosLibNode extends BaseNode {
  static readonly nodeType = "lib.beautifulsoup.ExtractVideos";
  static readonly title = "Extract Videos";
  static readonly description = "Extract videos from HTML content.";

  defaults() {
    return { html: "", base_url: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const html = String(inputs.html ?? this._props.html ?? "");
    const baseUrl = String(inputs.base_url ?? this._props.base_url ?? "");
    const $ = cheerio.load(html);
    const videos: Array<{ uri: string; type: string }> = [];

    $("video, video source, iframe").each((_, el) => {
      const src = $(el).attr("src");
      if (src) {
        const fullUrl = new URL(src, baseUrl || undefined).href;
        videos.push({ uri: fullUrl, type: "video" });
      }
    });

    return { output: videos };
  }
}

export class ExtractMetadataLibNode extends BaseNode {
  static readonly nodeType = "lib.beautifulsoup.ExtractMetadata";
  static readonly title = "Extract Metadata";
  static readonly description = "Extract metadata from HTML content.";

  defaults() {
    return { html: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const html = String(inputs.html ?? this._props.html ?? "");
    const $ = cheerio.load(html);

    const title = $("title").first().text() || null;
    const description =
      $('meta[name="description"]').attr("content") ?? null;
    const keywords =
      $('meta[name="keywords"]').attr("content") ?? null;

    return { title, description, keywords };
  }
}

export class HTMLToTextLibNode extends BaseNode {
  static readonly nodeType = "lib.beautifulsoup.HTMLToText";
  static readonly title = "Convert HTML to Text";
  static readonly description =
    "Converts HTML to plain text by removing tags and decoding entities using BeautifulSoup.";

  defaults() {
    return { text: "", preserve_linebreaks: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const html = String(inputs.text ?? this._props.text ?? "");
    const preserveLinebreaks = Boolean(
      inputs.preserve_linebreaks ?? this._props.preserve_linebreaks ?? true
    );

    const text = convert(html, {
      wordwrap: preserveLinebreaks ? 130 : false,
      preserveNewlines: preserveLinebreaks,
    });

    return { output: text };
  }
}

export class WebsiteContentExtractorLibNode extends BaseNode {
  static readonly nodeType = "lib.beautifulsoup.WebsiteContentExtractor";
  static readonly title = "Website Content Extractor";
  static readonly description =
    "Extract main content from a website, removing navigation, ads, and other non-essential elements.";

  defaults() {
    return { html_content: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const htmlContent = String(
      inputs.html_content ?? this._props.html_content ?? ""
    );

    const dom = new JSDOM(htmlContent);
    const reader = new Readability(dom.window.document);
    const article = reader.parse();

    if (article) {
      return { output: (article.textContent ?? "").replace(/\s+/g, " ").trim() };
    }

    // Fallback: strip tags manually like the Python version
    const $ = cheerio.load(htmlContent);
    $("script, style, nav, sidebar, footer, header").remove();

    const main =
      $("article").first().text() ||
      $("main").first().text() ||
      $('[id*="content"]').first().text() ||
      $('[class*="content"]').first().text() ||
      $("body").text();

    return { output: (main || "No main content found").replace(/\s+/g, " ").trim() };
  }
}

export const LIB_BEAUTIFULSOUP_NODES = [
  BaseUrlLibNode,
  ExtractLinksLibNode,
  ExtractImagesLibNode,
  ExtractAudioLibNode,
  ExtractVideosLibNode,
  ExtractMetadataLibNode,
  HTMLToTextLibNode,
  WebsiteContentExtractorLibNode,
] as const;
