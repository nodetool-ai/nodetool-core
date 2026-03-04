import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";
import type { ProcessingContext } from "@nodetool/runtime";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

type LibCompatDescriptor = {
  nodeType: string;
  title: string;
  description: string;
};

const PYTHON_BRIDGE_SCRIPT = String.raw`
import asyncio
import base64
import dataclasses
import importlib
import json
import sys
from enum import Enum
from pathlib import Path

def to_json(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {k: to_json(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json(v) for v in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return to_json(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return to_json(value.dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {k: to_json(v) for k, v in value.__dict__.items() if not k.startswith("_")}
    return str(value)

async def main():
    payload = json.loads(sys.stdin.read() or "{}")
    for p in payload.get("python_paths", []):
        if p and Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)

    node_type = payload["node_type"]
    module_path = "nodetool.nodes." + ".".join(node_type.split(".")[:-1])
    class_name = node_type.split(".")[-1]

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    props = payload.get("props", {})

    from nodetool.workflows.processing_context import ProcessingContext
    context = ProcessingContext(user_id="ts-runtime", auth_token="")
    node = cls(**props)
    result = await node.process(context)

    sys.stdout.write(json.dumps({"ok": True, "result": to_json(result)}))

try:
    asyncio.run(main())
except Exception as exc:
    sys.stdout.write(json.dumps({"ok": False, "error": str(exc), "type": type(exc).__name__}))
`;

async function runPythonBridge(nodeType: string, props: Record<string, unknown>): Promise<unknown> {
  const candidates = [
    process.env.NODETOOL_BASE_SRC,
    resolve(process.cwd(), "../nodetool-base/src"),
    resolve(process.cwd(), "../../nodetool-base/src"),
    resolve(process.cwd(), "../../../nodetool-base/src"),
    "/Users/mg/workspace/nodetool-base/src",
  ].filter((p): p is string => Boolean(p));

  const pythonPaths = candidates.filter((p, i, arr) => arr.indexOf(p) === i && existsSync(p));

  return await new Promise((resolvePromise, rejectPromise) => {
    const child = spawn("python3", ["-c", PYTHON_BRIDGE_SCRIPT], {
      stdio: "pipe",
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
      },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (d) => { stdout += String(d); });
    child.stderr.on("data", (d) => { stderr += String(d); });
    child.on("error", (err) => rejectPromise(err));
    child.on("close", (code) => {
      if (code !== 0) {
        rejectPromise(new Error(`Python bridge failed for ${nodeType} (exit ${code}): ${stderr || stdout}`));
        return;
      }
      try {
        const parsed = JSON.parse(stdout || "{}");
        if (!parsed.ok) {
          rejectPromise(new Error(`Python node ${nodeType} failed: ${String(parsed.error ?? "unknown error")}`));
          return;
        }
        resolvePromise(parsed.result);
      } catch (error) {
        rejectPromise(new Error(`Invalid Python bridge response for ${nodeType}: ${String(error)} :: ${stdout || stderr}`));
      }
    });

    child.stdin.write(JSON.stringify({ node_type: nodeType, props, python_paths: pythonPaths }));
    child.stdin.end();
  });
}

function createLibCompatNode(descriptor: LibCompatDescriptor): NodeClass {
  const LibCompatPythonNode = class extends BaseNode {
    static readonly nodeType = descriptor.nodeType;
    static readonly title = descriptor.title;
    static readonly description = descriptor.description;

    async process(
      inputs: Record<string, unknown>,
      _context?: ProcessingContext
    ): Promise<Record<string, unknown>> {
      const props = { ...this._props, ...inputs };
      delete (props as Record<string, unknown>).__node_id;
      delete (props as Record<string, unknown>).__node_name;
      const result = await runPythonBridge(descriptor.nodeType, props);
      if (result && typeof result === "object" && !Array.isArray(result)) {
        return result as Record<string, unknown>;
      }
      return { output: result };
    }
  };

  return LibCompatPythonNode as NodeClass;
}

const LIB_COMPAT_DESCRIPTORS = [
  { nodeType: "lib.beautifulsoup.BaseUrl", title: "Base Url", description: "Extract the base URL from a given URL." },
  { nodeType: "lib.beautifulsoup.ExtractAudio", title: "Extract Audio", description: "Extract audio elements from HTML content." },
  { nodeType: "lib.beautifulsoup.ExtractImages", title: "Extract Images", description: "Extract images from HTML content." },
  { nodeType: "lib.beautifulsoup.ExtractLinks", title: "Extract Links", description: "Extract all links from HTML content with type classification." },
  { nodeType: "lib.beautifulsoup.ExtractMetadata", title: "Extract Metadata", description: "Extract metadata from HTML content." },
  { nodeType: "lib.beautifulsoup.ExtractVideos", title: "Extract Videos", description: "Extract videos from HTML content." },
  { nodeType: "lib.beautifulsoup.HTMLToText", title: "Convert HTML to Text", description: "Converts HTML to plain text by removing tags and decoding entities using BeautifulSoup." },
  { nodeType: "lib.beautifulsoup.WebsiteContentExtractor", title: "Website Content Extractor", description: "Extract main content from a website, removing navigation, ads, and other non-essential elements." },
  { nodeType: "lib.browser.Browser", title: "Browser", description: "Fetches content from a web page using a headless browser." },
  { nodeType: "lib.browser.BrowserNavigation", title: "Browser Navigation", description: "Navigates and interacts with web pages in a browser session." },
  { nodeType: "lib.browser.BrowserUse", title: "Browser Use", description: "Browser agent tool that uses browser_use under the hood." },
  { nodeType: "lib.browser.DownloadFile", title: "Download File", description: "Downloads a file from a URL and saves it to disk." },
  { nodeType: "lib.browser.Screenshot", title: "Screenshot", description: "Takes a screenshot of a web page or specific element." },
  { nodeType: "lib.browser.SpiderCrawl", title: "Spider Crawl", description: "Crawls websites following links and emitting URLs with optional HTML content." },
  { nodeType: "lib.browser.WebFetch", title: "Web Fetch", description: "Fetches HTML content from a URL and converts it to text." },
  { nodeType: "lib.docx.AddHeading", title: "Add Heading", description: "Adds a heading to the document" },
  { nodeType: "lib.docx.AddImage", title: "Add Image", description: "Adds an image to the document" },
  { nodeType: "lib.docx.AddPageBreak", title: "Add Page Break", description: "Adds a page break to the document" },
  { nodeType: "lib.docx.AddParagraph", title: "Add Paragraph", description: "Adds a paragraph of text to the document" },
  { nodeType: "lib.docx.AddTable", title: "Add Table", description: "Adds a table to the document" },
  { nodeType: "lib.docx.CreateDocument", title: "Create Document", description: "Creates a new Word document" },
  { nodeType: "lib.docx.LoadWordDocument", title: "Load Word Document", description: "Loads a Word document from disk" },
  { nodeType: "lib.docx.SaveDocument", title: "Save Document", description: "Writes the document to a file" },
  { nodeType: "lib.docx.SetDocumentProperties", title: "Set Document Properties", description: "Sets document metadata properties" },
  { nodeType: "lib.excel.AutoFitColumns", title: "Auto Fit Columns", description: "Automatically adjusts column widths to fit content." },
  { nodeType: "lib.excel.CreateWorkbook", title: "Create Workbook", description: "Creates a new Excel workbook." },
  { nodeType: "lib.excel.DataFrameToExcel", title: "Data Frame To Excel", description: "Writes a DataFrame to an Excel worksheet." },
  { nodeType: "lib.excel.ExcelToDataFrame", title: "Excel To Data Frame", description: "Reads an Excel worksheet into a pandas DataFrame." },
  { nodeType: "lib.excel.FormatCells", title: "Format Cells", description: "Applies formatting to a range of cells." },
  { nodeType: "lib.excel.SaveWorkbook", title: "Save Workbook", description: "Saves an Excel workbook to disk." },
  { nodeType: "lib.mail.AddLabel", title: "Add Label", description: "Adds a label to a Gmail message." },
  { nodeType: "lib.mail.GmailSearch", title: "Gmail Search", description: "Searches Gmail using Gmail-specific search operators and yields matching emails." },
  { nodeType: "lib.mail.MoveToArchive", title: "Move To Archive", description: "Moves specified emails to Gmail archive." },
  { nodeType: "lib.mail.SendEmail", title: "Send Email", description: "Send a plain text email via SMTP." },
  { nodeType: "lib.markitdown.ConvertToMarkdown", title: "Convert To Markdown", description: "Converts various document formats to markdown using MarkItDown." },
  { nodeType: "lib.numpy.arithmetic.AddArray", title: "Add Array", description: "Performs addition on two arrays." },
  { nodeType: "lib.numpy.arithmetic.DivideArray", title: "Divide Array", description: "Divides the first array by the second." },
  { nodeType: "lib.numpy.arithmetic.ModulusArray", title: "Modulus Array", description: "Calculates the element-wise remainder of division." },
  { nodeType: "lib.numpy.arithmetic.MultiplyArray", title: "Multiply Array", description: "Multiplies two arrays." },
  { nodeType: "lib.numpy.arithmetic.SubtractArray", title: "Subtract Array", description: "Subtracts the second array from the first." },
  { nodeType: "lib.numpy.conversion.ArrayToList", title: "Array To List", description: "Convert a array to a nested list structure." },
  { nodeType: "lib.numpy.conversion.ArrayToScalar", title: "Array To Scalar", description: "Convert a single-element array to a scalar value." },
  { nodeType: "lib.numpy.conversion.ConvertToArray", title: "Convert To Array", description: "Convert PIL Image to normalized tensor representation." },
  { nodeType: "lib.numpy.conversion.ConvertToAudio", title: "Convert To Audio", description: "Converts a array object back to an audio file." },
  { nodeType: "lib.numpy.conversion.ConvertToImage", title: "Convert To Image", description: "Convert array data to PIL Image format." },
  { nodeType: "lib.numpy.conversion.ListToArray", title: "List To Array", description: "Convert a list of values to a array." },
  { nodeType: "lib.numpy.conversion.ScalarToArray", title: "Scalar To Array", description: "Convert a scalar value to a single-element array." },
  { nodeType: "lib.numpy.io.SaveArray", title: "Save Array", description: "Save a numpy array to a file in the specified folder." },
  { nodeType: "lib.numpy.manipulation.IndexArray", title: "Index Array", description: "Select specific indices from an array along a specified axis." },
  { nodeType: "lib.numpy.manipulation.MatMul", title: "Mat Mul", description: "Perform matrix multiplication on two input arrays." },
  { nodeType: "lib.numpy.manipulation.SliceArray", title: "Slice Array", description: "Extract a slice of an array along a specified axis." },
  { nodeType: "lib.numpy.manipulation.SplitArray", title: "Split Array", description: "Split an array into multiple sub-arrays along a specified axis." },
  { nodeType: "lib.numpy.manipulation.Stack", title: "Stack", description: "Stack multiple arrays along a specified axis." },
  { nodeType: "lib.numpy.manipulation.TransposeArray", title: "Transpose Array", description: "Transpose the dimensions of the input array." },
  { nodeType: "lib.numpy.math.AbsArray", title: "Abs Array", description: "Compute the absolute value of each element in a array." },
  { nodeType: "lib.numpy.math.CosineArray", title: "Cosine Array", description: "Computes the cosine of input angles in radians." },
  { nodeType: "lib.numpy.math.ExpArray", title: "Exp Array", description: "Calculate the exponential of each element in a array." },
  { nodeType: "lib.numpy.math.LogArray", title: "Log Array", description: "Calculate the natural logarithm of each element in a array." },
  { nodeType: "lib.numpy.math.PowerArray", title: "Power Array", description: "Raises the base array to the power of the exponent element-wise." },
  { nodeType: "lib.numpy.math.SineArray", title: "Sine Array", description: "Computes the sine of input angles in radians." },
  { nodeType: "lib.numpy.math.SqrtArray", title: "Sqrt Array", description: "Calculates the square root of the input array element-wise." },
  { nodeType: "lib.numpy.reshaping.Reshape1D", title: "Reshape 1D", description: "Reshape an array to a 1D shape without changing its data." },
  { nodeType: "lib.numpy.reshaping.Reshape2D", title: "Reshape 2D", description: "Reshape an array to a new shape without changing its data." },
  { nodeType: "lib.numpy.reshaping.Reshape3D", title: "Reshape 3D", description: "Reshape an array to a 3D shape without changing its data." },
  { nodeType: "lib.numpy.reshaping.Reshape4D", title: "Reshape 4D", description: "Reshape an array to a 4D shape without changing its data." },
  { nodeType: "lib.numpy.statistics.ArgMaxArray", title: "Arg Max Array", description: "Find indices of maximum values along a specified axis of a array." },
  { nodeType: "lib.numpy.statistics.ArgMinArray", title: "Arg Min Array", description: "Find indices of minimum values along a specified axis of a array." },
  { nodeType: "lib.numpy.statistics.MaxArray", title: "Max Array", description: "Compute the maximum value along a specified axis of a array." },
  { nodeType: "lib.numpy.statistics.MeanArray", title: "Mean Array", description: "Compute the mean value along a specified axis of a array." },
  { nodeType: "lib.numpy.statistics.MinArray", title: "Min Array", description: "Calculate the minimum value along a specified axis of a array." },
  { nodeType: "lib.numpy.statistics.SumArray", title: "Sum Array", description: "Calculate the sum of values along a specified axis of a array." },
  { nodeType: "lib.numpy.utils.BinaryOperation", title: "Binary Operation", description: "" },
  { nodeType: "lib.numpy.visualization.PlotArray", title: "Plot Array", description: "Create a plot visualization of array data." },
  { nodeType: "lib.ocr.PaddleOCR", title: "Paddle OCR", description: "Performs Optical Character Recognition (OCR) on images using PaddleOCR." },
  { nodeType: "lib.pdfplumber.ExtractImages", title: "Extract Images", description: "Extract images from a PDF file." },
  { nodeType: "lib.pdfplumber.ExtractPageMetadata", title: "Extract Page Metadata", description: "Extract metadata from PDF pages like dimensions, rotation, etc." },
  { nodeType: "lib.pdfplumber.ExtractTables", title: "Extract Tables", description: "Extract tables from a PDF file into dataframes." },
  { nodeType: "lib.pdfplumber.ExtractText", title: "Extract Text", description: "Extract text content from a PDF file." },
  { nodeType: "lib.pdfplumber.GetPageCount", title: "Get Page Count", description: "Get the total number of pages in a PDF file." },
  { nodeType: "lib.pymupdf.ExtractMarkdown", title: "Extract Markdown", description: "Convert PDF to Markdown format using pymupdf4llm." },
  { nodeType: "lib.pymupdf.ExtractTables", title: "Extract Tables", description: "Extract tables from a PDF document using PyMuPDF." },
  { nodeType: "lib.pymupdf.ExtractText", title: "Extract Text", description: "Extract plain text from a PDF document using PyMuPDF." },
  { nodeType: "lib.pymupdf.ExtractTextBlocks", title: "Extract Text Blocks", description: "Extract text blocks with their bounding boxes from a PDF." },
  { nodeType: "lib.pymupdf.ExtractTextWithStyle", title: "Extract Text With Style", description: "Extract text with style information (font, size, color) from a PDF." },
  { nodeType: "lib.seaborn.ChartRenderer", title: "Chart Renderer", description: "Node responsible for rendering chart configurations into image format using seaborn." },
  { nodeType: "lib.sqlite.CreateTable", title: "Create Table", description: "Create a new SQLite table with specified columns." },
  { nodeType: "lib.sqlite.Delete", title: "Delete", description: "Delete records from a SQLite table." },
  { nodeType: "lib.sqlite.ExecuteSQL", title: "Execute SQL", description: "Execute arbitrary SQL statements for advanced operations." },
  { nodeType: "lib.sqlite.GetDatabasePath", title: "Get Database Path", description: "Get the full path to a SQLite database file." },
  { nodeType: "lib.sqlite.Insert", title: "Insert", description: "Insert a record into a SQLite table." },
  { nodeType: "lib.sqlite.Query", title: "Query", description: "Query records from a SQLite table." },
  { nodeType: "lib.sqlite.Update", title: "Update", description: "Update records in a SQLite table." },
  { nodeType: "lib.supabase.Delete", title: "Delete", description: "Delete records from a Supabase table." },
  { nodeType: "lib.supabase.Insert", title: "Insert", description: "Insert record(s) into a Supabase table." },
  { nodeType: "lib.supabase.RPC", title: "RPC", description: "Call a PostgreSQL function via Supabase RPC." },
  { nodeType: "lib.supabase.Select", title: "Select", description: "Query records from a Supabase table." },
  { nodeType: "lib.supabase.Update", title: "Update", description: "Update records in a Supabase table." },
  { nodeType: "lib.supabase.Upsert", title: "Upsert", description: "Insert or update (upsert) records in a Supabase table." },
] as const satisfies readonly LibCompatDescriptor[];

export const LIB_COMPAT_PY_NODES: readonly NodeClass[] = LIB_COMPAT_DESCRIPTORS.map((d) =>
  createLibCompatNode(d)
);
