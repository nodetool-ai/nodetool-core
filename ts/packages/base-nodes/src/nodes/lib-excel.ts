import { BaseNode } from "@nodetool/node-sdk";
import ExcelJS from "exceljs";
import os from "node:os";
import path from "node:path";

type Row = Record<string, unknown>;

function asRows(value: unknown): Row[] {
  if (Array.isArray(value)) {
    return value
      .filter((x): x is Row => !!x && typeof x === "object" && !Array.isArray(x))
      .map((x) => ({ ...x }));
  }
  if (value && typeof value === "object") {
    const obj = value as { rows?: unknown; data?: unknown };
    if (Array.isArray(obj.rows)) return asRows(obj.rows);
    if (Array.isArray(obj.data)) return asRows(obj.data);
  }
  return [];
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

// Workbooks are passed between nodes as ExcelJS.Workbook instances stored on a wrapper object
type WorkbookRef = { data: ExcelJS.Workbook };

function getWorkbook(input: unknown): ExcelJS.Workbook {
  if (input && typeof input === "object" && "data" in input) {
    const ref = input as WorkbookRef;
    if (ref.data instanceof ExcelJS.Workbook) return ref.data;
  }
  if (input instanceof ExcelJS.Workbook) return input;
  throw new Error("Workbook is not connected");
}

export class CreateWorkbookLibNode extends BaseNode {
  static readonly nodeType = "lib.excel.CreateWorkbook";
  static readonly title = "Create Workbook";
  static readonly description = "Creates a new Excel workbook.";

  defaults() {
    return { sheet_name: "Sheet1" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const sheetName = String(inputs.sheet_name ?? this._props.sheet_name ?? "Sheet1");
    const wb = new ExcelJS.Workbook();
    wb.addWorksheet(sheetName);
    return { output: { data: wb } };
  }
}

export class ExcelToDataFrameLibNode extends BaseNode {
  static readonly nodeType = "lib.excel.ExcelToDataFrame";
  static readonly title = "Excel To Data Frame";
  static readonly description = "Reads an Excel worksheet into a list of row dicts.";

  defaults() {
    return { workbook: {}, sheet_name: "Sheet1", has_header: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const wb = getWorkbook(inputs.workbook ?? this._props.workbook);
    const sheetName = String(inputs.sheet_name ?? this._props.sheet_name ?? "Sheet1");
    const hasHeader = inputs.has_header ?? this._props.has_header ?? true;
    const ws = wb.getWorksheet(sheetName);
    if (!ws) throw new Error(`Worksheet '${sheetName}' not found`);

    const rows: Row[] = [];
    let headers: string[] = [];
    let firstRow = true;

    ws.eachRow((row) => {
      const values = (row.values as unknown[]).slice(1); // exceljs row.values is 1-indexed
      if (firstRow && hasHeader) {
        headers = values.map((v) => String(v ?? ""));
        firstRow = false;
        return;
      }
      firstRow = false;
      if (hasHeader) {
        const rowObj: Row = {};
        for (let i = 0; i < headers.length; i++) {
          rowObj[headers[i]] = values[i] ?? null;
        }
        rows.push(rowObj);
      } else {
        const rowObj: Row = {};
        for (let i = 0; i < values.length; i++) {
          rowObj[String(i)] = values[i] ?? null;
        }
        rows.push(rowObj);
      }
    });

    return { output: { rows } };
  }
}

export class DataFrameToExcelLibNode extends BaseNode {
  static readonly nodeType = "lib.excel.DataFrameToExcel";
  static readonly title = "Data Frame To Excel";
  static readonly description = "Writes a DataFrame to an Excel worksheet.";

  defaults() {
    return {
      workbook: {},
      dataframe: { rows: [] },
      sheet_name: "Sheet1",
      start_cell: "A1",
      include_header: true,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const wb = getWorkbook(inputs.workbook ?? this._props.workbook);
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const sheetName = String(inputs.sheet_name ?? this._props.sheet_name ?? "Sheet1");
    const includeHeader = inputs.include_header ?? this._props.include_header ?? true;

    let ws = wb.getWorksheet(sheetName);
    if (!ws) {
      ws = wb.addWorksheet(sheetName);
    }

    if (rows.length === 0) return { output: { data: wb } };

    const headers = [...new Set(rows.flatMap((r) => Object.keys(r)))];
    let rowIdx = 1;

    if (includeHeader) {
      for (let col = 0; col < headers.length; col++) {
        ws.getCell(rowIdx, col + 1).value = headers[col];
      }
      rowIdx++;
    }

    for (const row of rows) {
      for (let col = 0; col < headers.length; col++) {
        const val = row[headers[col]];
        ws.getCell(rowIdx, col + 1).value = val as ExcelJS.CellValue;
      }
      rowIdx++;
    }

    return { output: { data: wb } };
  }
}

export class FormatCellsLibNode extends BaseNode {
  static readonly nodeType = "lib.excel.FormatCells";
  static readonly title = "Format Cells";
  static readonly description = "Applies formatting to a range of cells.";

  defaults() {
    return {
      workbook: {},
      sheet_name: "Sheet1",
      cell_range: "A1:B10",
      bold: false,
      background_color: "FFFF00",
      text_color: "000000",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const wb = getWorkbook(inputs.workbook ?? this._props.workbook);
    const sheetName = String(inputs.sheet_name ?? this._props.sheet_name ?? "Sheet1");
    const cellRange = String(inputs.cell_range ?? this._props.cell_range ?? "A1:B10");
    const bold = Boolean(inputs.bold ?? this._props.bold ?? false);
    const bgColor = String(inputs.background_color ?? this._props.background_color ?? "FFFF00");
    const textColor = String(inputs.text_color ?? this._props.text_color ?? "000000");

    const ws = wb.getWorksheet(sheetName);
    if (!ws) throw new Error(`Worksheet '${sheetName}' not found`);

    // Parse range like "A1:B10"
    const [start, end] = cellRange.split(":");
    const startCol = columnNameToNumber(start.replace(/[0-9]/g, ""));
    const startRow = parseInt(start.replace(/[A-Za-z]/g, ""), 10);
    const endCol = end ? columnNameToNumber(end.replace(/[0-9]/g, "")) : startCol;
    const endRow = end ? parseInt(end.replace(/[A-Za-z]/g, ""), 10) : startRow;

    for (let r = startRow; r <= endRow; r++) {
      for (let c = startCol; c <= endCol; c++) {
        const cell = ws.getCell(r, c);
        cell.font = { bold, color: { argb: "FF" + textColor } };
        if (bgColor) {
          cell.fill = {
            type: "pattern",
            pattern: "solid",
            fgColor: { argb: "FF" + bgColor },
          };
        }
      }
    }

    return { output: { data: wb } };
  }
}

function columnNameToNumber(name: string): number {
  let result = 0;
  for (let i = 0; i < name.length; i++) {
    result = result * 26 + (name.charCodeAt(i) - 64);
  }
  return result;
}

export class AutoFitColumnsLibNode extends BaseNode {
  static readonly nodeType = "lib.excel.AutoFitColumns";
  static readonly title = "Auto Fit Columns";
  static readonly description = "Automatically adjusts column widths to fit content.";

  defaults() {
    return { workbook: {}, sheet_name: "Sheet1" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const wb = getWorkbook(inputs.workbook ?? this._props.workbook);
    const sheetName = String(inputs.sheet_name ?? this._props.sheet_name ?? "Sheet1");

    const ws = wb.getWorksheet(sheetName);
    if (!ws) throw new Error(`Worksheet '${sheetName}' not found`);

    ws.columns.forEach((col) => {
      let maxLen = 0;
      col.eachCell?.({ includeEmpty: false }, (cell) => {
        const len = cell.value ? String(cell.value).length : 0;
        if (len > maxLen) maxLen = len;
      });
      col.width = maxLen + 2;
    });

    return { output: { data: wb } };
  }
}

export class SaveWorkbookLibNode extends BaseNode {
  static readonly nodeType = "lib.excel.SaveWorkbook";
  static readonly title = "Save Workbook";
  static readonly description = "Saves an Excel workbook to disk.";

  defaults() {
    return { workbook: {}, folder: { path: "" }, filename: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const wb = getWorkbook(inputs.workbook ?? this._props.workbook);
    const folderInput = inputs.folder ?? this._props.folder;
    const folderPath =
      typeof folderInput === "string"
        ? folderInput
        : (folderInput as { path?: string })?.path ?? "";
    if (!folderPath) throw new Error("Path is not set");

    const filenameTemplate = String(inputs.filename ?? this._props.filename ?? "");
    const filename = formatDate(filenameTemplate);
    const fullPath = expandUser(path.join(folderPath, filename));

    await wb.xlsx.writeFile(fullPath);
    return { output: fullPath };
  }
}

export const LIB_EXCEL_NODES = [
  CreateWorkbookLibNode,
  ExcelToDataFrameLibNode,
  DataFrameToExcelLibNode,
  FormatCellsLibNode,
  AutoFitColumnsLibNode,
  SaveWorkbookLibNode,
];
