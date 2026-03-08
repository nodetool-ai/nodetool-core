import { BaseNode } from "@nodetool/node-sdk";

type DocumentRefLike = {
  uri?: string;
  data?: Uint8Array | string;
};

function asBytes(data: Uint8Array | string | undefined): Uint8Array {
  if (!data) return new Uint8Array();
  if (data instanceof Uint8Array) return data;
  return Uint8Array.from(Buffer.from(data, "base64"));
}

async function loadPdfDocument(pdf: DocumentRefLike) {
  const pdfjsLib = await import("pdfjs-dist/legacy/build/pdf.mjs");
  let pdfData: Uint8Array;
  if (pdf.data) {
    pdfData = asBytes(pdf.data);
  } else if (pdf.uri) {
    const { promises: fs } = await import("node:fs");
    const uri = pdf.uri.startsWith("file://") ? pdf.uri.slice(7) : pdf.uri;
    pdfData = new Uint8Array(await fs.readFile(uri));
  } else {
    throw new Error("No PDF data or URI provided");
  }
  const loadingTask = pdfjsLib.getDocument({ data: pdfData, useSystemFonts: true });
  return await loadingTask.promise;
}

function resolvePageRange(
  startPage: number,
  endPage: number,
  totalPages: number
): [number, number] {
  const start = Math.max(0, startPage);
  const end = endPage === -1 ? totalPages - 1 : Math.min(endPage, totalPages - 1);
  return [start, end];
}

// ── pdfplumber nodes ──────────────────────────────────────────────

export class GetPageCountPdfPlumberNode extends BaseNode {
  static readonly nodeType = "lib.pdfplumber.GetPageCount";
  static readonly title = "Get Page Count";
  static readonly description = "Get the total number of pages in a PDF file.";

  defaults() {
    return { pdf: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const pdf = (inputs.pdf ?? this._props.pdf ?? {}) as DocumentRefLike;
    const doc = await loadPdfDocument(pdf);
    const count = doc.numPages;
    void doc.destroy();
    return { output: count };
  }
}

export class ExtractTextPdfPlumberNode extends BaseNode {
  static readonly nodeType = "lib.pdfplumber.ExtractText";
  static readonly title = "Extract Text";
  static readonly description = "Extract text content from a PDF file.";

  defaults() {
    return { pdf: {}, start_page: 0, end_page: 4 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const pdf = (inputs.pdf ?? this._props.pdf ?? {}) as DocumentRefLike;
    const startPage = Number(inputs.start_page ?? this._props.start_page ?? 0);
    const endPage = Number(inputs.end_page ?? this._props.end_page ?? 4);

    const doc = await loadPdfDocument(pdf);
    const [start, end] = resolvePageRange(startPage, endPage, doc.numPages);
    const textParts: string[] = [];

    for (let i = start; i <= end; i++) {
      const page = await doc.getPage(i + 1); // pdfjs uses 1-based pages
      const content = await page.getTextContent();
      const pageText = content.items
        .map((item: any) => item.str)
        .join(" ");
      textParts.push(pageText);
    }

    void doc.destroy();
    return { output: textParts.join("\n\n") };
  }
}

export class ExtractPageMetadataPdfPlumberNode extends BaseNode {
  static readonly nodeType = "lib.pdfplumber.ExtractPageMetadata";
  static readonly title = "Extract Page Metadata";
  static readonly description =
    "Extract metadata from PDF pages like dimensions, rotation, etc.";

  defaults() {
    return { pdf: {}, start_page: 0, end_page: 4 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const pdf = (inputs.pdf ?? this._props.pdf ?? {}) as DocumentRefLike;
    const startPage = Number(inputs.start_page ?? this._props.start_page ?? 0);
    const endPage = Number(inputs.end_page ?? this._props.end_page ?? 4);

    const doc = await loadPdfDocument(pdf);
    const [start, end] = resolvePageRange(startPage, endPage, doc.numPages);
    const metadata: Record<string, unknown>[] = [];

    for (let i = start; i <= end; i++) {
      const page = await doc.getPage(i + 1);
      const viewport = page.getViewport({ scale: 1.0 });
      metadata.push({
        page_number: i + 1,
        width: viewport.width,
        height: viewport.height,
        rotation: page.rotate,
        bbox: [0, 0, viewport.width, viewport.height],
      });
    }

    void doc.destroy();
    return { output: metadata };
  }
}

export class ExtractTablesPdfPlumberNode extends BaseNode {
  static readonly nodeType = "lib.pdfplumber.ExtractTables";
  static readonly title = "Extract Tables";
  static readonly description =
    "Extract tables from a PDF file. Best-effort table extraction from text layout.";

  defaults() {
    return { pdf: {}, start_page: 0, end_page: 4 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const pdf = (inputs.pdf ?? this._props.pdf ?? {}) as DocumentRefLike;
    const startPage = Number(inputs.start_page ?? this._props.start_page ?? 0);
    const endPage = Number(inputs.end_page ?? this._props.end_page ?? 4);

    const doc = await loadPdfDocument(pdf);
    const [start, end] = resolvePageRange(startPage, endPage, doc.numPages);
    const tables: Record<string, unknown>[] = [];

    for (let pageIdx = start; pageIdx <= end; pageIdx++) {
      const page = await doc.getPage(pageIdx + 1);
      const content = await page.getTextContent();

      // Group items by Y position (rows) with tolerance
      const yTolerance = 3;
      const rows: Map<number, { x: number; str: string }[]> = new Map();

      for (const item of content.items as any[]) {
        if (!item.str || !item.str.trim()) continue;
        const y = Math.round(item.transform[5] / yTolerance) * yTolerance;
        if (!rows.has(y)) rows.set(y, []);
        rows.get(y)!.push({ x: item.transform[4], str: item.str.trim() });
      }

      // Sort rows by Y (descending, since PDF Y is bottom-up)
      const sortedRows = Array.from(rows.entries())
        .sort((a, b) => b[0] - a[0])
        .map(([, cells]) => cells.sort((a, b) => a.x - b.x).map((c) => c.str));

      // Only consider it a table if we have multiple rows with consistent column counts
      if (sortedRows.length >= 2) {
        const colCounts = sortedRows.map((r) => r.length);
        const mostCommon = colCounts
          .sort(
            (a, b) =>
              colCounts.filter((v) => v === b).length -
              colCounts.filter((v) => v === a).length
          )[0];

        if (mostCommon >= 2) {
          const tableRows = sortedRows.filter((r) => r.length === mostCommon);
          if (tableRows.length >= 2) {
            tables.push({
              page: pageIdx,
              header: tableRows[0],
              rows: tableRows.slice(1),
              columns: mostCommon,
            });
          }
        }
      }
    }

    void doc.destroy();
    return { output: tables };
  }
}

export class ExtractImagesPdfPlumberNode extends BaseNode {
  static readonly nodeType = "lib.pdfplumber.ExtractImages";
  static readonly title = "Extract Images";
  static readonly description =
    "Extract images from a PDF file. Note: pdfjs-dist has limited image extraction support; returns empty list for most PDFs.";

  defaults() {
    return { pdf: {}, start_page: 0, end_page: 4 };
  }

  async process(
    _inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    // pdfjs-dist does not expose raw image streams like pdfplumber does.
    // Return empty list with a note.
    return { output: [] };
  }
}

// ── pymupdf nodes ─────────────────────────────────────────────────

export class ExtractTextPyMuPdfNode extends BaseNode {
  static readonly nodeType = "lib.pymupdf.ExtractText";
  static readonly title = "Extract Text";
  static readonly description =
    "Extract plain text from a PDF document using PyMuPDF.";

  defaults() {
    return { pdf: {}, start_page: 0, end_page: -1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const pdf = (inputs.pdf ?? this._props.pdf ?? {}) as DocumentRefLike;
    const startPage = Number(inputs.start_page ?? this._props.start_page ?? 0);
    const endPage = Number(inputs.end_page ?? this._props.end_page ?? -1);

    const doc = await loadPdfDocument(pdf);
    const [start, end] = resolvePageRange(startPage, endPage, doc.numPages);
    let text = "";

    for (let i = start; i <= end; i++) {
      const page = await doc.getPage(i + 1);
      const content = await page.getTextContent();
      // Reconstruct text with line breaks based on Y positions
      let lastY: number | null = null;
      for (const item of content.items as any[]) {
        const y = item.transform[5];
        if (lastY !== null && Math.abs(y - lastY) > 2) {
          text += "\n";
        }
        text += item.str;
        lastY = y;
      }
      text += "\n\n";
    }

    void doc.destroy();
    return { output: text.trim() };
  }
}

export class ExtractMarkdownPyMuPdfNode extends BaseNode {
  static readonly nodeType = "lib.pymupdf.ExtractMarkdown";
  static readonly title = "Extract Markdown";
  static readonly description =
    "Convert PDF to Markdown format with basic formatting from font sizes.";

  defaults() {
    return { pdf: {}, start_page: 0, end_page: -1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const pdf = (inputs.pdf ?? this._props.pdf ?? {}) as DocumentRefLike;
    const startPage = Number(inputs.start_page ?? this._props.start_page ?? 0);
    const endPage = Number(inputs.end_page ?? this._props.end_page ?? -1);

    const doc = await loadPdfDocument(pdf);
    const [start, end] = resolvePageRange(startPage, endPage, doc.numPages);
    const mdParts: string[] = [];

    for (let i = start; i <= end; i++) {
      const page = await doc.getPage(i + 1);
      const content = await page.getTextContent();

      // Collect font sizes to determine header thresholds
      const sizes: number[] = [];
      for (const item of content.items as any[]) {
        if (item.str && item.str.trim()) {
          const h = item.height || item.transform?.[0] || 12;
          sizes.push(h);
        }
      }

      const avgSize = sizes.length > 0 ? sizes.reduce((a, b) => a + b, 0) / sizes.length : 12;
      const lines: string[] = [];
      let lastY: number | null = null;
      let currentLine = "";
      let currentSize = 0;

      for (const item of content.items as any[]) {
        const y = item.transform[5];
        const h = item.height || item.transform?.[0] || 12;

        if (lastY !== null && Math.abs(y - lastY) > 2) {
          // Flush current line
          if (currentLine.trim()) {
            if (currentSize > avgSize * 1.5) {
              lines.push(`# ${currentLine.trim()}`);
            } else if (currentSize > avgSize * 1.2) {
              lines.push(`## ${currentLine.trim()}`);
            } else {
              lines.push(currentLine.trim());
            }
          } else {
            lines.push("");
          }
          currentLine = "";
          currentSize = 0;
        }

        currentLine += item.str;
        currentSize = Math.max(currentSize, h);
        lastY = y;
      }

      // Flush last line
      if (currentLine.trim()) {
        if (currentSize > avgSize * 1.5) {
          lines.push(`# ${currentLine.trim()}`);
        } else if (currentSize > avgSize * 1.2) {
          lines.push(`## ${currentLine.trim()}`);
        } else {
          lines.push(currentLine.trim());
        }
      }

      mdParts.push(lines.join("\n"));
    }

    void doc.destroy();
    return { output: mdParts.join("\n\n---\n\n") };
  }
}

export class ExtractTextBlocksPyMuPdfNode extends BaseNode {
  static readonly nodeType = "lib.pymupdf.ExtractTextBlocks";
  static readonly title = "Extract Text Blocks";
  static readonly description =
    "Extract text blocks with their bounding boxes from a PDF.";

  defaults() {
    return { pdf: {}, start_page: 0, end_page: -1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const pdf = (inputs.pdf ?? this._props.pdf ?? {}) as DocumentRefLike;
    const startPage = Number(inputs.start_page ?? this._props.start_page ?? 0);
    const endPage = Number(inputs.end_page ?? this._props.end_page ?? -1);

    const doc = await loadPdfDocument(pdf);
    const [start, end] = resolvePageRange(startPage, endPage, doc.numPages);
    const blocks: Record<string, unknown>[] = [];

    for (let i = start; i <= end; i++) {
      const page = await doc.getPage(i + 1);
      const content = await page.getTextContent();
      const viewport = page.getViewport({ scale: 1.0 });

      // Group text items into blocks by proximity
      let currentBlock: {
        text: string;
        x0: number;
        y0: number;
        x1: number;
        y1: number;
      } | null = null;
      let lastY: number | null = null;

      for (const item of content.items as any[]) {
        const x = item.transform[4];
        const y = item.transform[5];
        const w = item.width || 0;
        const h = item.height || item.transform[0] || 12;

        if (lastY !== null && Math.abs(y - lastY) > h * 1.5) {
          if (currentBlock && currentBlock.text.trim()) {
            blocks.push({
              page: i,
              text: currentBlock.text.trim(),
              bbox: {
                x0: currentBlock.x0,
                y0: viewport.height - currentBlock.y1,
                x1: currentBlock.x1,
                y1: viewport.height - currentBlock.y0,
              },
            });
          }
          currentBlock = null;
        }

        if (!currentBlock) {
          currentBlock = { text: "", x0: x, y0: y, x1: x + w, y1: y + h };
        }

        currentBlock.text += item.str;
        currentBlock.x0 = Math.min(currentBlock.x0, x);
        currentBlock.y0 = Math.min(currentBlock.y0, y);
        currentBlock.x1 = Math.max(currentBlock.x1, x + w);
        currentBlock.y1 = Math.max(currentBlock.y1, y + h);
        lastY = y;
      }

      if (currentBlock && currentBlock.text.trim()) {
        blocks.push({
          page: i,
          text: currentBlock.text.trim(),
          bbox: {
            x0: currentBlock.x0,
            y0: viewport.height - currentBlock.y1,
            x1: currentBlock.x1,
            y1: viewport.height - currentBlock.y0,
          },
        });
      }
    }

    void doc.destroy();
    return { output: blocks };
  }
}

export class ExtractTextWithStylePyMuPdfNode extends BaseNode {
  static readonly nodeType = "lib.pymupdf.ExtractTextWithStyle";
  static readonly title = "Extract Text With Style";
  static readonly description =
    "Extract text with style information (font, size, color) from a PDF.";

  defaults() {
    return { pdf: {}, start_page: 0, end_page: -1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const pdf = (inputs.pdf ?? this._props.pdf ?? {}) as DocumentRefLike;
    const startPage = Number(inputs.start_page ?? this._props.start_page ?? 0);
    const endPage = Number(inputs.end_page ?? this._props.end_page ?? -1);

    const doc = await loadPdfDocument(pdf);
    const [start, end] = resolvePageRange(startPage, endPage, doc.numPages);
    const styledText: Record<string, unknown>[] = [];

    for (let i = start; i <= end; i++) {
      const page = await doc.getPage(i + 1);
      const content = await page.getTextContent();

      for (const item of content.items as any[]) {
        if (!item.str || !item.str.trim()) continue;
        const x = item.transform[4];
        const y = item.transform[5];
        const w = item.width || 0;
        const h = item.height || item.transform[0] || 12;

        styledText.push({
          text: item.str,
          font: item.fontName || "unknown",
          size: h,
          color: 0, // pdfjs doesn't easily expose color
          bbox: {
            x0: x,
            y0: y,
            x1: x + w,
            y1: y + h,
          },
        });
      }
    }

    void doc.destroy();
    return { output: styledText };
  }
}

export class ExtractTablesPyMuPdfNode extends BaseNode {
  static readonly nodeType = "lib.pymupdf.ExtractTables";
  static readonly title = "Extract Tables";
  static readonly description =
    "Extract tables from a PDF document. Best-effort text-layout based extraction.";

  defaults() {
    return { pdf: {}, start_page: 0, end_page: -1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const pdf = (inputs.pdf ?? this._props.pdf ?? {}) as DocumentRefLike;
    const startPage = Number(inputs.start_page ?? this._props.start_page ?? 0);
    const endPage = Number(inputs.end_page ?? this._props.end_page ?? -1);

    const doc = await loadPdfDocument(pdf);
    const [start, end] = resolvePageRange(startPage, endPage, doc.numPages);
    const allTables: Record<string, unknown>[] = [];

    for (let pageIdx = start; pageIdx <= end; pageIdx++) {
      const page = await doc.getPage(pageIdx + 1);
      const content = await page.getTextContent();

      const yTolerance = 3;
      const rows: Map<number, { x: number; str: string }[]> = new Map();

      for (const item of content.items as any[]) {
        if (!item.str || !item.str.trim()) continue;
        const y = Math.round(item.transform[5] / yTolerance) * yTolerance;
        if (!rows.has(y)) rows.set(y, []);
        rows.get(y)!.push({ x: item.transform[4], str: item.str.trim() });
      }

      const sortedRows = Array.from(rows.entries())
        .sort((a, b) => b[0] - a[0])
        .map(([, cells]) => cells.sort((a, b) => a.x - b.x).map((c) => c.str));

      if (sortedRows.length >= 2) {
        const colCounts = sortedRows.map((r) => r.length);
        const mostCommon = colCounts
          .sort(
            (a, b) =>
              colCounts.filter((v) => v === b).length -
              colCounts.filter((v) => v === a).length
          )[0];

        if (mostCommon >= 2) {
          const tableRows = sortedRows.filter((r) => r.length === mostCommon);
          if (tableRows.length >= 2) {
            const viewport = page.getViewport({ scale: 1.0 });
            allTables.push({
              page: pageIdx,
              bbox: { x0: 0, y0: 0, x1: viewport.width, y1: viewport.height },
              rows: tableRows.length,
              columns: mostCommon,
              header: { names: tableRows[0], external: false },
              content: tableRows,
            });
          }
        }
      }
    }

    void doc.destroy();
    return { output: allTables };
  }
}

export const LIB_PDF_NODES = [
  GetPageCountPdfPlumberNode,
  ExtractTextPdfPlumberNode,
  ExtractPageMetadataPdfPlumberNode,
  ExtractTablesPdfPlumberNode,
  ExtractImagesPdfPlumberNode,
  ExtractTextPyMuPdfNode,
  ExtractMarkdownPyMuPdfNode,
  ExtractTextBlocksPyMuPdfNode,
  ExtractTextWithStylePyMuPdfNode,
  ExtractTablesPyMuPdfNode,
] as const;
