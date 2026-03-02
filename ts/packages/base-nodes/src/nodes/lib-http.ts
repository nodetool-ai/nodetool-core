import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";
import path from "node:path";

async function fetchResponse(
  url: string,
  init?: RequestInit
): Promise<Response> {
  const res = await fetch(url, init);
  return res;
}

function ensureOk(res: Response): void {
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText}`);
  }
}

function base64FromBytes(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString("base64");
}

function imageRefFromBytes(bytes: Uint8Array, uri = ""): Record<string, unknown> {
  return { data: base64FromBytes(bytes), uri };
}

function documentRefFromBytes(bytes: Uint8Array, uri = ""): Record<string, unknown> {
  return { data: base64FromBytes(bytes), uri };
}

function castValue(value: unknown, targetType: string): unknown {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" && value === "" && targetType !== "string" && targetType !== "object") {
    return null;
  }

  try {
    if (targetType === "int") return Math.trunc(Number(value));
    if (targetType === "float") return Number(value);
    if (targetType === "string") return String(value);
    if (targetType === "datetime") {
      if (typeof value === "number") return new Date(value * 1000).toISOString();
      const parsed = new Date(String(value));
      return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toISOString();
    }
    return value;
  } catch {
    return null;
  }
}

abstract class HTTPBaseLibNode extends BaseNode {
  defaults() {
    return { url: "" };
  }

  protected readUrl(inputs: Record<string, unknown>): string {
    return String(inputs.url ?? this._props.url ?? "");
  }
}

export class GetRequestLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.GetRequest";
  static readonly title = "GET Request";
  static readonly description = "Perform an HTTP GET request to retrieve data from a specified URL.";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetchResponse(this.readUrl(inputs));
    const text = await res.text();
    return { output: text };
  }
}

export class PostRequestLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.PostRequest";
  static readonly title = "POST Request";
  static readonly description = "Send data to a server using an HTTP POST request.";

  defaults() {
    return { url: "", data: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = String(inputs.data ?? this._props.data ?? "");
    const res = await fetchResponse(this.readUrl(inputs), { method: "POST", body: data });
    return { output: await res.text() };
  }
}

export class PutRequestLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.PutRequest";
  static readonly title = "PUT Request";
  static readonly description = "Update existing resources on a server using an HTTP PUT request.";

  defaults() {
    return { url: "", data: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = String(inputs.data ?? this._props.data ?? "");
    const res = await fetchResponse(this.readUrl(inputs), { method: "PUT", body: data });
    return { output: await res.text() };
  }
}

export class DeleteRequestLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.DeleteRequest";
  static readonly title = "DELETE Request";
  static readonly description = "Remove a resource from a server using an HTTP DELETE request.";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetchResponse(this.readUrl(inputs), { method: "DELETE" });
    return { output: await res.text() };
  }
}

export class HeadRequestLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.HeadRequest";
  static readonly title = "HEAD Request";
  static readonly description = "Retrieve headers from a resource using an HTTP HEAD request.";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetchResponse(this.readUrl(inputs), { method: "HEAD", redirect: "follow" });
    const out: Record<string, string> = {};
    res.headers.forEach((v, k) => {
      out[k] = v;
    });
    return { output: out };
  }
}

export class FetchPageLibNode extends BaseNode {
  static readonly nodeType = "lib.http.FetchPage";
  static readonly title = "Fetch Page";
  static readonly description = "Fetch a web page using Selenium and return its content.";

  defaults() {
    return { url: "", wait_time: 10 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = String(inputs.url ?? this._props.url ?? "");
    try {
      const res = await fetchResponse(url);
      return { html: await res.text(), success: true, error_message: null };
    } catch (error) {
      return { html: "", success: false, error_message: String(error) };
    }
  }
}

export class ImageDownloaderLibNode extends BaseNode {
  static readonly nodeType = "lib.http.ImageDownloader";
  static readonly title = "Image Downloader";
  static readonly description = "Download images from list of URLs and return a list of ImageRefs.";

  defaults() {
    return { images: [] as string[], base_url: "", max_concurrent_downloads: 10 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const images = Array.isArray(inputs.images ?? this._props.images)
      ? ((inputs.images ?? this._props.images ?? []) as unknown[]).map(String)
      : [];
    const baseUrl = String(inputs.base_url ?? this._props.base_url ?? "");
    const urls = images.map((u) => new URL(u, baseUrl || undefined).toString());

    const downloaded: Record<string, unknown>[] = [];
    const failedUrls: string[] = [];

    await Promise.all(
      urls.map(async (url) => {
        try {
          const res = await fetchResponse(url);
          if (!res.ok) {
            failedUrls.push(url);
            return;
          }
          const bytes = new Uint8Array(await res.arrayBuffer());
          downloaded.push(imageRefFromBytes(bytes, url));
        } catch {
          failedUrls.push(url);
        }
      })
    );

    return { images: downloaded, failed_urls: failedUrls };
  }
}

export class GetRequestBinaryLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.GetRequestBinary";
  static readonly title = "GET Binary";
  static readonly description = "Perform an HTTP GET request and return raw binary data.";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetchResponse(this.readUrl(inputs));
    return { output: base64FromBytes(new Uint8Array(await res.arrayBuffer())) };
  }
}

export class GetRequestDocumentLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.GetRequestDocument";
  static readonly title = "GET Document";
  static readonly description = "Perform an HTTP GET request and return a document";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = this.readUrl(inputs);
    const res = await fetchResponse(url);
    const bytes = new Uint8Array(await res.arrayBuffer());
    return { output: documentRefFromBytes(bytes, url) };
  }
}

export class PostRequestBinaryLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.PostRequestBinary";
  static readonly title = "POST Binary";
  static readonly description = "Send data using an HTTP POST request and return raw binary data.";

  defaults() {
    return { url: "", data: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = inputs.data ?? this._props.data ?? "";
    const body = typeof data === "string" ? data : JSON.stringify(data);
    const res = await fetchResponse(this.readUrl(inputs), { method: "POST", body });
    const bytes = new Uint8Array(await res.arrayBuffer());
    return { output: base64FromBytes(bytes) };
  }
}

export class DownloadDataframeLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.DownloadDataframe";
  static readonly title = "Download Dataframe";
  static readonly description = "Download data from a URL and return as a dataframe.";

  defaults() {
    return {
      url: "",
      file_format: "csv",
      columns: { columns: [] as Array<{ name: string; data_type: string }> },
      encoding: "utf-8",
      delimiter: ",",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = this.readUrl(inputs);
    const fileFormat = String(inputs.file_format ?? this._props.file_format ?? "csv");
    const delimiter = String(inputs.delimiter ?? this._props.delimiter ?? ",");
    const columnsObj = (inputs.columns ?? this._props.columns ?? { columns: [] }) as {
      columns?: Array<{ name: string; data_type: string }>;
    };
    const targetColumns = columnsObj.columns ?? [];

    const res = await fetchResponse(url);
    ensureOk(res);
    const content = await res.text();

    if (targetColumns.length === 0) {
      return { output: { rows: [] as Array<Record<string, unknown>> } };
    }

    const mapRows = (headers: string[], rows: unknown[][]): Array<Record<string, unknown>> => {
      const idx = new Map(headers.map((h, i) => [h, i]));
      return rows.map((row) => {
        const out: Record<string, unknown> = {};
        for (const col of targetColumns) {
          const i = idx.get(col.name);
          const raw = i === undefined ? null : row[i];
          out[col.name] = castValue(raw, col.data_type);
        }
        return out;
      });
    };

    if (fileFormat === "csv" || fileFormat === "tsv") {
      const d = fileFormat === "tsv" ? "\t" : delimiter;
      const lines = content.split(/\r?\n/).filter((line) => line.length > 0);
      if (lines.length === 0) throw new Error(`No data found in ${fileFormat.toUpperCase()}`);
      const headers = lines[0].split(d);
      const rows = lines.slice(1).map((line) => line.split(d));
      return { output: { rows: mapRows(headers, rows) } };
    }

    if (fileFormat === "json") {
      const parsed = JSON.parse(content) as unknown;
      if (!Array.isArray(parsed) || parsed.length === 0) {
        throw new Error("No data found or data is not a list of records in JSON");
      }
      if (typeof parsed[0] === "object" && parsed[0] !== null && !Array.isArray(parsed[0])) {
        const headers = Object.keys(parsed[0] as Record<string, unknown>);
        const rows = parsed.map((item) => headers.map((h) => (item as Record<string, unknown>)[h]));
        return { output: { rows: mapRows(headers, rows) } };
      }
      if (Array.isArray(parsed[0])) {
        const headers = (parsed[0] as unknown[]).map(String);
        const rows = (parsed as unknown[]).slice(1) as unknown[][];
        return { output: { rows: mapRows(headers, rows) } };
      }
      throw new Error("JSON data is a list, but items are not dictionaries or lists.");
    }

    throw new Error(`Unsupported file format: ${fileFormat}`);
  }
}

export class FilterValidURLsLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.FilterValidURLs";
  static readonly title = "Filter Valid URLs";
  static readonly description = "Filter a list of URLs by checking their validity using HEAD requests.";

  defaults() {
    return { url: "", urls: [] as string[], max_concurrent_requests: 10 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const urls = Array.isArray(inputs.urls ?? this._props.urls)
      ? ((inputs.urls ?? this._props.urls ?? []) as unknown[]).map(String)
      : [];

    const valid: string[] = [];
    await Promise.all(
      urls.map(async (url) => {
        try {
          const res = await fetchResponse(url, { method: "HEAD", redirect: "follow" });
          if (res.status >= 200 && res.status < 400) valid.push(url);
        } catch {
          // ignore invalid url
        }
      })
    );

    return { output: valid };
  }
}

export class DownloadFilesLibNode extends BaseNode {
  static readonly nodeType = "lib.http.DownloadFiles";
  static readonly title = "Download Files";
  static readonly description = "Download files from a list of URLs into a local folder.";

  defaults() {
    return { urls: [] as string[], output_folder: "downloads", max_concurrent_downloads: 5 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const urls = Array.isArray(inputs.urls ?? this._props.urls)
      ? ((inputs.urls ?? this._props.urls ?? []) as unknown[]).map(String)
      : [];
    const outputFolder = String(inputs.output_folder ?? this._props.output_folder ?? "downloads");
    const expandedFolder = outputFolder.startsWith("~/")
      ? path.join(process.env.HOME ?? "", outputFolder.slice(2))
      : outputFolder;
    await fs.mkdir(expandedFolder, { recursive: true });

    const success: string[] = [];
    const failed: string[] = [];

    await Promise.all(
      urls.map(async (url) => {
        try {
          const res = await fetchResponse(url);
          if (!res.ok) {
            failed.push(url);
            return;
          }
          const contentDisposition = res.headers.get("content-disposition") ?? "";
          let filename = "";
          const idx = contentDisposition.toLowerCase().indexOf("filename=");
          if (idx >= 0) {
            filename = contentDisposition.slice(idx + "filename=".length).trim().replace(/^"|"$/g, "");
          }
          if (!filename) {
            filename = url.split("/").pop() || "unnamed_file";
          }
          const full = path.join(expandedFolder, filename);
          const bytes = new Uint8Array(await res.arrayBuffer());
          await fs.writeFile(full, bytes);
          success.push(full);
        } catch {
          failed.push(url);
        }
      })
    );

    return { success, failed };
  }
}

abstract class JSONRequestBaseLibNode extends HTTPBaseLibNode {
  defaults() {
    return { url: "", data: {} };
  }

  protected data(inputs: Record<string, unknown>): Record<string, unknown> {
    const data = inputs.data ?? this._props.data ?? {};
    return data && typeof data === "object" && !Array.isArray(data)
      ? (data as Record<string, unknown>)
      : {};
  }
}

export class JSONPostRequestLibNode extends JSONRequestBaseLibNode {
  static readonly nodeType = "lib.http.JSONPostRequest";
  static readonly title = "POST JSON";
  static readonly description = "Send JSON data to a server using an HTTP POST request.";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetchResponse(this.readUrl(inputs), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.data(inputs)),
    });
    return { output: (await res.json()) as Record<string, unknown> };
  }
}

export class JSONPutRequestLibNode extends JSONRequestBaseLibNode {
  static readonly nodeType = "lib.http.JSONPutRequest";
  static readonly title = "PUT JSON";
  static readonly description = "Update resources with JSON data using an HTTP PUT request.";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetchResponse(this.readUrl(inputs), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.data(inputs)),
    });
    return { output: (await res.json()) as Record<string, unknown> };
  }
}

export class JSONPatchRequestLibNode extends JSONRequestBaseLibNode {
  static readonly nodeType = "lib.http.JSONPatchRequest";
  static readonly title = "PATCH JSON";
  static readonly description = "Partially update resources with JSON data using an HTTP PATCH request.";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetchResponse(this.readUrl(inputs), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.data(inputs)),
    });
    return { output: (await res.json()) as Record<string, unknown> };
  }
}

export class JSONGetRequestLibNode extends HTTPBaseLibNode {
  static readonly nodeType = "lib.http.JSONGetRequest";
  static readonly title = "GET JSON";
  static readonly description = "Perform an HTTP GET request and parse the response as JSON.";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const res = await fetchResponse(this.readUrl(inputs), {
      headers: { Accept: "application/json" },
    });
    return { output: (await res.json()) as Record<string, unknown> };
  }
}

export const LIB_HTTP_NODES = [
  GetRequestLibNode,
  PostRequestLibNode,
  PutRequestLibNode,
  DeleteRequestLibNode,
  HeadRequestLibNode,
  FetchPageLibNode,
  ImageDownloaderLibNode,
  GetRequestBinaryLibNode,
  GetRequestDocumentLibNode,
  PostRequestBinaryLibNode,
  DownloadDataframeLibNode,
  FilterValidURLsLibNode,
  DownloadFilesLibNode,
  JSONPostRequestLibNode,
  JSONPutRequestLibNode,
  JSONPatchRequestLibNode,
  JSONGetRequestLibNode,
] as const;
