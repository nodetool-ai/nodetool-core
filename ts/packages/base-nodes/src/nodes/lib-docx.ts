import { BaseNode } from "@nodetool/node-sdk";
import {
  Document,
  Packer,
  Paragraph,
  HeadingLevel,
  TextRun,
  ImageRun,
  Table,
  TableRow,
  TableCell,
  PageBreak,
  AlignmentType,
  WidthType,
} from "docx";
import mammoth from "mammoth";
import { readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

// Document state: an array of element descriptors that gets built up incrementally.
// SaveDocument renders them into an actual .docx file.
type ElementDescriptor =
  | { type: "paragraph"; text: string; alignment: string; bold: boolean; italic: boolean; font_size: number }
  | { type: "heading"; text: string; level: number }
  | { type: "image"; image_data: Buffer; width: number; height: number }
  | { type: "table"; data: string[][] }
  | { type: "page_break" };

type DocState = {
  elements: ElementDescriptor[];
  properties?: { title?: string; author?: string; subject?: string; keywords?: string };
};

function getDocState(input: unknown): DocState {
  if (input && typeof input === "object" && "elements" in input) {
    return input as DocState;
  }
  return { elements: [] };
}

function expandUser(p: string): string {
  if (!p) return p;
  if (p === "~") return os.homedir();
  if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2));
  return p;
}

function formatDate(template: string): string {
  const now = new Date();
  return template
    .replace(/%Y/g, String(now.getFullYear()))
    .replace(/%m/g, String(now.getMonth() + 1).padStart(2, "0"))
    .replace(/%d/g, String(now.getDate()).padStart(2, "0"))
    .replace(/%H/g, String(now.getHours()).padStart(2, "0"))
    .replace(/%M/g, String(now.getMinutes()).padStart(2, "0"))
    .replace(/%S/g, String(now.getSeconds()).padStart(2, "0"));
}

const ALIGNMENT_MAP: Record<string, (typeof AlignmentType)[keyof typeof AlignmentType]> = {
  LEFT: AlignmentType.LEFT,
  CENTER: AlignmentType.CENTER,
  RIGHT: AlignmentType.RIGHT,
  JUSTIFY: AlignmentType.JUSTIFIED,
};

const HEADING_MAP: Record<number, (typeof HeadingLevel)[keyof typeof HeadingLevel]> = {
  1: HeadingLevel.HEADING_1,
  2: HeadingLevel.HEADING_2,
  3: HeadingLevel.HEADING_3,
  4: HeadingLevel.HEADING_4,
  5: HeadingLevel.HEADING_5,
  6: HeadingLevel.HEADING_6,
};

export class CreateDocumentLibNode extends BaseNode {
  static readonly nodeType = "lib.docx.CreateDocument";
  static readonly title = "Create Document";
  static readonly description = "Creates a new Word document";

  defaults() {
    return {};
  }

  async process(): Promise<Record<string, unknown>> {
    return { output: { elements: [], properties: {} } as DocState };
  }
}

export class LoadWordDocumentLibNode extends BaseNode {
  static readonly nodeType = "lib.docx.LoadWordDocument";
  static readonly title = "Load Word Document";
  static readonly description = "Loads a Word document from disk";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const filePath = String(inputs.path ?? this._props.path ?? "");
    if (!filePath.trim()) throw new Error("path cannot be empty");
    const expanded = expandUser(filePath);
    const result = await mammoth.extractRawText({ path: expanded });
    return { output: result.value };
  }
}

export class AddHeadingLibNode extends BaseNode {
  static readonly nodeType = "lib.docx.AddHeading";
  static readonly title = "Add Heading";
  static readonly description = "Adds a heading to the document";

  defaults() {
    return { document: { elements: [] }, text: "", level: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const doc = getDocState(inputs.document ?? this._props.document);
    const text = String(inputs.text ?? this._props.text ?? "");
    const level = Number(inputs.level ?? this._props.level ?? 1);
    const newDoc: DocState = {
      ...doc,
      elements: [...doc.elements, { type: "heading", text, level }],
    };
    return { output: newDoc };
  }
}

export class AddParagraphLibNode extends BaseNode {
  static readonly nodeType = "lib.docx.AddParagraph";
  static readonly title = "Add Paragraph";
  static readonly description = "Adds a paragraph of text to the document";

  defaults() {
    return {
      document: { elements: [] },
      text: "",
      alignment: "LEFT",
      bold: false,
      italic: false,
      font_size: 12,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const doc = getDocState(inputs.document ?? this._props.document);
    const text = String(inputs.text ?? this._props.text ?? "");
    const alignment = String(inputs.alignment ?? this._props.alignment ?? "LEFT");
    const bold = Boolean(inputs.bold ?? this._props.bold ?? false);
    const italic = Boolean(inputs.italic ?? this._props.italic ?? false);
    const fontSize = Number(inputs.font_size ?? this._props.font_size ?? 12);
    const newDoc: DocState = {
      ...doc,
      elements: [
        ...doc.elements,
        { type: "paragraph", text, alignment, bold, italic, font_size: fontSize },
      ],
    };
    return { output: newDoc };
  }
}

export class AddTableLibNode extends BaseNode {
  static readonly nodeType = "lib.docx.AddTable";
  static readonly title = "Add Table";
  static readonly description = "Adds a table to the document";

  defaults() {
    return { document: { elements: [] }, data: { rows: [], columns: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const doc = getDocState(inputs.document ?? this._props.document);
    const dataInput = inputs.data ?? this._props.data ?? {};
    // Accept DataframeRef-like { data: string[][], columns: string[] } or { rows: Row[] }
    let tableData: string[][] = [];
    if (dataInput && typeof dataInput === "object") {
      const d = dataInput as { data?: unknown[][]; columns?: string[]; rows?: Record<string, unknown>[] };
      if (Array.isArray(d.data)) {
        tableData = d.data.map((row) => row.map((cell) => String(cell ?? "")));
      } else if (Array.isArray(d.rows)) {
        for (const row of d.rows) {
          tableData.push(Object.values(row).map((v) => String(v ?? "")));
        }
      }
    }
    const newDoc: DocState = {
      ...doc,
      elements: [...doc.elements, { type: "table", data: tableData }],
    };
    return { output: newDoc };
  }
}

export class AddImageLibNode extends BaseNode {
  static readonly nodeType = "lib.docx.AddImage";
  static readonly title = "Add Image";
  static readonly description = "Adds an image to the document";

  defaults() {
    return { document: { elements: [] }, image: {}, width: 0, height: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const doc = getDocState(inputs.document ?? this._props.document);
    const imageInput = inputs.image ?? this._props.image ?? {};
    const width = Number(inputs.width ?? this._props.width ?? 0);
    const height = Number(inputs.height ?? this._props.height ?? 0);

    // Read image data from path or uri
    let imageData: Buffer;
    if (typeof imageInput === "string") {
      imageData = readFileSync(expandUser(imageInput));
    } else if (imageInput && typeof imageInput === "object") {
      const img = imageInput as { uri?: string; path?: string; data?: Buffer | Uint8Array | string };
      if (img.data) {
        imageData = Buffer.isBuffer(img.data) ? img.data : Buffer.from(img.data as Uint8Array);
      } else {
        const imgPath = img.uri?.replace("file://", "") ?? img.path ?? "";
        if (!imgPath) throw new Error("Image path is not set");
        imageData = readFileSync(expandUser(imgPath));
      }
    } else {
      throw new Error("Invalid image input");
    }

    const newDoc: DocState = {
      ...doc,
      elements: [...doc.elements, { type: "image", image_data: imageData, width, height }],
    };
    return { output: newDoc };
  }
}

export class AddPageBreakLibNode extends BaseNode {
  static readonly nodeType = "lib.docx.AddPageBreak";
  static readonly title = "Add Page Break";
  static readonly description = "Adds a page break to the document";

  defaults() {
    return { document: { elements: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const doc = getDocState(inputs.document ?? this._props.document);
    const newDoc: DocState = {
      ...doc,
      elements: [...doc.elements, { type: "page_break" }],
    };
    return { output: newDoc };
  }
}

export class SetDocumentPropertiesLibNode extends BaseNode {
  static readonly nodeType = "lib.docx.SetDocumentProperties";
  static readonly title = "Set Document Properties";
  static readonly description = "Sets document metadata properties";

  defaults() {
    return { document: { elements: [] }, title: "", author: "", subject: "", keywords: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const doc = getDocState(inputs.document ?? this._props.document);
    const title = String(inputs.title ?? this._props.title ?? "");
    const author = String(inputs.author ?? this._props.author ?? "");
    const subject = String(inputs.subject ?? this._props.subject ?? "");
    const keywords = String(inputs.keywords ?? this._props.keywords ?? "");

    const newDoc: DocState = {
      ...doc,
      properties: {
        ...(doc.properties ?? {}),
        ...(title ? { title } : {}),
        ...(author ? { author } : {}),
        ...(subject ? { subject } : {}),
        ...(keywords ? { keywords } : {}),
      },
    };
    return { output: newDoc };
  }
}

export class SaveDocumentLibNode extends BaseNode {
  static readonly nodeType = "lib.docx.SaveDocument";
  static readonly title = "Save Document";
  static readonly description = "Writes the document to a file";

  defaults() {
    return { document: { elements: [] }, path: { path: "" }, filename: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const doc = getDocState(inputs.document ?? this._props.document);
    const pathInput = inputs.path ?? this._props.path;
    const folderPath =
      typeof pathInput === "string"
        ? pathInput
        : (pathInput as { path?: string })?.path ?? "";
    if (!folderPath) throw new Error("Path is not set");

    const filenameTemplate = String(inputs.filename ?? this._props.filename ?? "");
    const filename = formatDate(filenameTemplate);
    const fullPath = expandUser(path.join(folderPath, filename));

    // Build document elements
    const children = doc.elements.map((el) => {
      switch (el.type) {
        case "heading":
          return new Paragraph({
            children: [new TextRun({ text: el.text })],
            heading: HEADING_MAP[el.level] ?? HeadingLevel.HEADING_1,
          });
        case "paragraph":
          return new Paragraph({
            children: [
              new TextRun({
                text: el.text,
                bold: el.bold,
                italics: el.italic,
                size: el.font_size * 2, // docx uses half-points
              }),
            ],
            alignment: ALIGNMENT_MAP[el.alignment] ?? AlignmentType.LEFT,
          });
        case "image": {
          const imgRunOpts: ConstructorParameters<typeof ImageRun>[0] = {
            data: el.image_data,
            transformation: {
              width: el.width ? el.width * 96 : 200, // inches to pixels (approx)
              height: el.height ? el.height * 96 : 200,
            },
            type: "buf" as never,
          };
          return new Paragraph({
            children: [new ImageRun(imgRunOpts)],
          });
        }
        case "table":
          return new Table({
            rows: el.data.map(
              (rowData) =>
                new TableRow({
                  children: rowData.map(
                    (cellText) =>
                      new TableCell({
                        children: [new Paragraph({ children: [new TextRun(cellText)] })],
                        width: { size: 100, type: WidthType.AUTO },
                      })
                  ),
                })
            ),
          });
        case "page_break":
          return new Paragraph({
            children: [new PageBreak()],
          });
      }
    });

    const docObj = new Document({
      ...(doc.properties?.title ? { title: doc.properties.title } : {}),
      ...(doc.properties?.author
        ? { creator: doc.properties.author }
        : {}),
      ...(doc.properties?.subject
        ? { subject: doc.properties.subject }
        : {}),
      ...(doc.properties?.keywords
        ? { keywords: doc.properties.keywords }
        : {}),
      sections: [{ children }],
    });

    const buffer = await Packer.toBuffer(docObj);
    writeFileSync(fullPath, buffer);
    return { output: fullPath };
  }
}

export const LIB_DOCX_NODES = [
  CreateDocumentLibNode,
  LoadWordDocumentLibNode,
  AddHeadingLibNode,
  AddParagraphLibNode,
  AddTableLibNode,
  AddImageLibNode,
  AddPageBreakLibNode,
  SetDocumentPropertiesLibNode,
  SaveDocumentLibNode,
];
