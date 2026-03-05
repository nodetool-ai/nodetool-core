/**
 * Agent-backed skill nodes for prompt-driven task execution.
 *
 * These are stub implementations that capture all configuration fields from the
 * Python originals but cannot run without the full agent runtime stack. Each
 * concrete skill sets its own nodeType, title, description, and default overrides.
 */

import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

// ---------------------------------------------------------------------------
// Base skill node
// ---------------------------------------------------------------------------

class SkillNode extends BaseNode {
  static readonly description: string =
    "Base skill node (not directly usable).";

  defaults(): Record<string, unknown> {
    return {
      model: { provider: "", id: "" },
      prompt: "",
      timeout_seconds: 180,
      max_output_chars: 200000,
    };
  }

  async process(
    _inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    throw new Error(
      "Skills require the agent runtime which is not yet available in the TS environment"
    );
  }
}

// ---------------------------------------------------------------------------
// ShellAgentSkill
// ---------------------------------------------------------------------------

export class ShellAgentSkillNode extends SkillNode {
  static readonly nodeType = "skills._shell_agent.ShellAgentSkill";
  static readonly title = "Shell Agent Skill";
  static readonly description =
    "Reusable prompt-driven skill backed by execute_bash. skills, shell, agent, bash";
}

// ---------------------------------------------------------------------------
// BrowserSkill
// ---------------------------------------------------------------------------

export class BrowserSkillNode extends SkillNode {
  static readonly nodeType = "skills.browser.BrowserSkill";
  static readonly title = "Browser Skill";
  static readonly description =
    "Prompt-driven browser skill with bounded tool validation and schema outputs. Supports extraction and browser automation workflows. skills, browser, scrape, extraction, automation";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      timeout_seconds: 150,
      max_output_chars: 180000,
    };
  }
}

// ---------------------------------------------------------------------------
// SQLiteSkill
// ---------------------------------------------------------------------------

export class SQLiteSkillNode extends SkillNode {
  static readonly nodeType = "skills.data.SQLiteSkill";
  static readonly title = "SQLite Skill";
  static readonly description =
    "Prompt-driven SQLite skill with guarded query execution. skills, data, sqlite, query";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      timeout_seconds: 120,
      db_path: "memory.db",
      allow_mutation: false,
    };
  }
}

// ---------------------------------------------------------------------------
// SupabaseSkill
// ---------------------------------------------------------------------------

export class SupabaseSkillNode extends SkillNode {
  static readonly nodeType = "skills.data.SupabaseSkill";
  static readonly title = "Supabase Skill";
  static readonly description =
    "Prompt-driven Supabase skill with guarded SELECT execution. skills, data, supabase, query";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      timeout_seconds: 120,
    };
  }
}

// ---------------------------------------------------------------------------
// DocumentSkill
// ---------------------------------------------------------------------------

export class DocumentSkillNode extends SkillNode {
  static readonly nodeType = "skills.document.DocumentSkill";
  static readonly title = "Document Skill";
  static readonly description =
    "Prompt-driven document skill for model-based document analysis. skills, document, extraction, conversion, markdown";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      timeout_seconds: 120,
      max_output_chars: 150000,
    };
  }
}

// ---------------------------------------------------------------------------
// DocxSkill
// ---------------------------------------------------------------------------

export class DocxSkillNode extends SkillNode {
  static readonly nodeType = "skills.docx.DocxSkill";
  static readonly title = "DOCX Skill";
  static readonly description =
    "Prompt-driven DOCX creation skill. skills, docx, word, document creation, docx-js";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      timeout_seconds: 300,
      max_output_chars: 220000,
    };
  }
}

// ---------------------------------------------------------------------------
// EmailSkill
// ---------------------------------------------------------------------------

export class EmailSkillNode extends SkillNode {
  static readonly nodeType = "skills.email.EmailSkill";
  static readonly title = "Email Skill";
  static readonly description =
    "Prompt-driven email skill for IMAP/SMTP and message processing tasks. skills, email, imap, smtp, messaging";
}

// ---------------------------------------------------------------------------
// FfmpegSkill
// ---------------------------------------------------------------------------

export class FfmpegSkillNode extends SkillNode {
  static readonly nodeType = "skills.ffmpeg.FfmpegSkill";
  static readonly title = "FFmpeg Skill";
  static readonly description =
    "Prompt-driven FFmpeg skill for audio/video editing, conversion, and packaging. skills, ffmpeg, media, video, audio, transcode, remux";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      audio: { type: "audio", uri: "", asset_id: null, data: null },
      video: { type: "video", uri: "", asset_id: null, data: null },
    };
  }
}

// ---------------------------------------------------------------------------
// FilesystemSkill
// ---------------------------------------------------------------------------

export class FilesystemSkillNode extends SkillNode {
  static readonly nodeType = "skills.filesystem.FilesystemSkill";
  static readonly title = "Filesystem Skill";
  static readonly description =
    "Prompt-driven filesystem skill for file inspection and transformations. skills, filesystem, files, directories, io";
}

// ---------------------------------------------------------------------------
// GitSkill
// ---------------------------------------------------------------------------

export class GitSkillNode extends SkillNode {
  static readonly nodeType = "skills.git.GitSkill";
  static readonly title = "Git Skill";
  static readonly description =
    "Prompt-driven Git skill for repository inspection and change management. skills, git, repository, version-control";
}

// ---------------------------------------------------------------------------
// HtmlSkill
// ---------------------------------------------------------------------------

export class HtmlSkillNode extends SkillNode {
  static readonly nodeType = "skills.html.HtmlSkill";
  static readonly title = "HTML Skill";
  static readonly description =
    "Prompt-driven HTML creation skill. skills, html, web, template, static-site";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      max_output_chars: 180000,
    };
  }
}

// ---------------------------------------------------------------------------
// HttpApiSkill
// ---------------------------------------------------------------------------

export class HttpApiSkillNode extends SkillNode {
  static readonly nodeType = "skills.httpapi.HttpApiSkill";
  static readonly title = "HTTP API Skill";
  static readonly description =
    "Prompt-driven HTTP API skill for calling REST/GraphQL endpoints. skills, http, api, rest, graphql";
}

// ---------------------------------------------------------------------------
// ImageSkill
// ---------------------------------------------------------------------------

export class ImageSkillNode extends SkillNode {
  static readonly nodeType = "skills.image.ImageSkill";
  static readonly title = "Image Skill";
  static readonly description =
    "Prompt-driven image skill for model-based image reasoning. skills, image, agent, transform, extraction";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      image: { type: "image", uri: "", asset_id: null, data: null },
      timeout_seconds: 90,
      max_output_chars: 120000,
    };
  }
}

// ---------------------------------------------------------------------------
// MediaSkill
// ---------------------------------------------------------------------------

export class MediaSkillNode extends SkillNode {
  static readonly nodeType = "skills.media.MediaSkill";
  static readonly title = "Media Skill";
  static readonly description =
    "Prompt-driven media skill for model-based audio/video reasoning. skills, media, audio, video, agent";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      audio: { type: "audio", uri: "", asset_id: null, data: null },
      video: { type: "video", uri: "", asset_id: null, data: null },
    };
  }
}

// ---------------------------------------------------------------------------
// PdfLibSkill
// ---------------------------------------------------------------------------

export class PdfLibSkillNode extends SkillNode {
  static readonly nodeType = "skills.pdf_lib.PdfLibSkill";
  static readonly title = "PDF Skill";
  static readonly description =
    "Prompt-driven PDF processing skill with pdf-lib and complementary tooling. skills, pdf, pdf-lib, qpdf, poppler, pdfjs, pypdfium2";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      document: { type: "document", uri: "", asset_id: null, data: null },
      timeout_seconds: 300,
      max_output_chars: 220000,
    };
  }
}

// ---------------------------------------------------------------------------
// PptxSkill
// ---------------------------------------------------------------------------

export class PptxSkillNode extends SkillNode {
  static readonly nodeType = "skills.pptx.PptxSkill";
  static readonly title = "PPTX Skill";
  static readonly description =
    "Prompt-driven PowerPoint generation skill with PptxGenJS. skills, pptx, powerpoint, pptxgenjs, slides";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      document: { type: "document", uri: "", asset_id: null, data: null },
      timeout_seconds: 300,
      max_output_chars: 220000,
    };
  }
}

// ---------------------------------------------------------------------------
// SpreadsheetSkill
// ---------------------------------------------------------------------------

export class SpreadsheetSkillNode extends SkillNode {
  static readonly nodeType = "skills.spreadsheet.SpreadsheetSkill";
  static readonly title = "Spreadsheet Skill";
  static readonly description =
    "Prompt-driven spreadsheet skill for CSV/XLSX processing. skills, spreadsheet, csv, xlsx, tabular";
}

// ---------------------------------------------------------------------------
// VectorStoreSkill
// ---------------------------------------------------------------------------

export class VectorStoreSkillNode extends SkillNode {
  static readonly nodeType = "skills.vectorstore.VectorStoreSkill";
  static readonly title = "Vector Store Skill";
  static readonly description =
    "Prompt-driven vector store skill for indexing and similarity search workflows. skills, vectorstore, embeddings, rag, retrieval";
}

// ---------------------------------------------------------------------------
// YtDlpDownloaderSkill
// ---------------------------------------------------------------------------

export class YtDlpDownloaderSkillNode extends SkillNode {
  static readonly nodeType = "skills.ytdlp.YtDlpDownloaderSkill";
  static readonly title = "YouTube Downloader Skill";
  static readonly description =
    "Download videos from YouTube/Bilibili/Twitter and other sites via yt-dlp. skills, media, yt-dlp, downloader, youtube, bilibili, twitter";

  defaults(): Record<string, unknown> {
    return {
      ...super.defaults(),
      url: "",
      output_dir: "downloads/yt-dlp",
      timeout_seconds: 300,
      max_output_chars: 220000,
    };
  }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export const SKILLS_NODES: readonly NodeClass[] = [
  ShellAgentSkillNode,
  BrowserSkillNode,
  SQLiteSkillNode,
  SupabaseSkillNode,
  DocumentSkillNode,
  DocxSkillNode,
  EmailSkillNode,
  FfmpegSkillNode,
  FilesystemSkillNode,
  GitSkillNode,
  HtmlSkillNode,
  HttpApiSkillNode,
  ImageSkillNode,
  MediaSkillNode,
  PdfLibSkillNode,
  PptxSkillNode,
  SpreadsheetSkillNode,
  VectorStoreSkillNode,
  YtDlpDownloaderSkillNode,
];
