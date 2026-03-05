import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SERPAPI_BASE = "https://serpapi.com/search.json";

function getSerpApiKey(inputs: Record<string, unknown>): string {
  const key =
    (inputs._secrets as Record<string, string>)?.SERPAPI_API_KEY ||
    process.env.SERPAPI_API_KEY ||
    "";
  if (!key) throw new Error("SERPAPI_API_KEY is required");
  return key;
}

async function serpRequest(
  apiKey: string,
  params: Record<string, string | number>
): Promise<Record<string, unknown>> {
  const url = new URL(SERPAPI_BASE);
  url.searchParams.set("api_key", apiKey);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") {
      url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error(`SerpAPI HTTP error: ${res.status} ${res.statusText}`);
  }
  const data = (await res.json()) as Record<string, unknown>;
  const meta = data.search_metadata as Record<string, unknown> | undefined;
  if (meta?.status === "Error" || typeof data.error === "string") {
    throw new Error(
      (data.error as string) ?? `SerpAPI returned an error: ${JSON.stringify(data)}`
    );
  }
  return data;
}

// ---------------------------------------------------------------------------
// 1. GoogleSearch
// ---------------------------------------------------------------------------
export class GoogleSearchNode extends BaseNode {
  static readonly nodeType = "search.google.GoogleSearch";
  static readonly title = "Google Search";
  static readonly description =
    "Search Google to retrieve organic search results from the web.";

  defaults() {
    return { keyword: "", num_results: 10 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const keyword = String(inputs.keyword ?? "");
    if (!keyword) throw new Error("Keyword is required");
    const numResults = Number(inputs.num_results ?? 10);
    const apiKey = getSerpApiKey(inputs);

    const data = await serpRequest(apiKey, {
      engine: "google_light",
      q: keyword,
      num: numResults,
      gl: "us",
      hl: "en",
    });

    const results = (data.organic_results as unknown[]) ?? [];
    return { output: results };
  }
}

// ---------------------------------------------------------------------------
// 2. GoogleNews
// ---------------------------------------------------------------------------
export class GoogleNewsNode extends BaseNode {
  static readonly nodeType = "search.google.GoogleNews";
  static readonly title = "Google News";
  static readonly description =
    "Search Google News to retrieve current news articles and headlines.";

  defaults() {
    return { keyword: "", num_results: 10 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const keyword = String(inputs.keyword ?? "");
    if (!keyword) throw new Error("Keyword is required");
    const numResults = Number(inputs.num_results ?? 10);
    const apiKey = getSerpApiKey(inputs);

    const data = await serpRequest(apiKey, {
      engine: "google_news",
      q: keyword,
      num: numResults,
      gl: "us",
      hl: "en",
    });

    const results = (data.news_results as unknown[]) ?? [];
    return { output: results };
  }
}

// ---------------------------------------------------------------------------
// 3. GoogleImages
// ---------------------------------------------------------------------------
export class GoogleImagesNode extends BaseNode {
  static readonly nodeType = "search.google.GoogleImages";
  static readonly title = "Google Images";
  static readonly description =
    "Search Google Images to find visual content or perform reverse image search.";

  defaults() {
    return { keyword: "", image_url: "", num_results: 20 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const keyword = String(inputs.keyword ?? "");
    const imageUrl = String(inputs.image_url ?? "");
    if (!keyword && !imageUrl)
      throw new Error("One of 'keyword' or 'image_url' is required.");
    const numResults = Number(inputs.num_results ?? 20);
    const apiKey = getSerpApiKey(inputs);

    const params: Record<string, string | number> = {
      num: numResults,
      gl: "us",
      hl: "en",
    };

    if (imageUrl) {
      params.engine = "google_reverse_image";
      params.image_url = imageUrl;
    } else {
      params.engine = "google_images";
      params.q = keyword;
    }

    const data = await serpRequest(apiKey, params);

    const images = (data.images_results as Array<Record<string, unknown>>) ?? [];
    const output = images.map((img) => ({
      uri: String(img.original ?? ""),
    }));
    return { output };
  }
}

// ---------------------------------------------------------------------------
// 4. GoogleFinance
// ---------------------------------------------------------------------------
export class GoogleFinanceNode extends BaseNode {
  static readonly nodeType = "search.google.GoogleFinance";
  static readonly title = "Google Finance";
  static readonly description =
    "Retrieve financial market data and stock information from Google Finance.";

  defaults() {
    return { query: "", window: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const query = String(inputs.query ?? "");
    if (!query) return { output: { error: "Query is required for Google Finance search." } };
    const window = String(inputs.window ?? "");
    const apiKey = getSerpApiKey(inputs);

    const params: Record<string, string | number> = {
      engine: "google_finance",
      q: query,
      gl: "us",
      hl: "en",
    };
    if (window) params.window = window;

    const data = await serpRequest(apiKey, params);
    return { output: { success: true, results: data } };
  }
}

// ---------------------------------------------------------------------------
// 5. GoogleJobs
// ---------------------------------------------------------------------------
export class GoogleJobsNode extends BaseNode {
  static readonly nodeType = "search.google.GoogleJobs";
  static readonly title = "Google Jobs";
  static readonly description =
    "Search Google Jobs for employment opportunities and job listings.";

  defaults() {
    return { query: "", location: "", num_results: 10 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const query = String(inputs.query ?? "");
    if (!query) throw new Error("Query is required for Google Jobs search.");
    const location = String(inputs.location ?? "");
    const apiKey = getSerpApiKey(inputs);

    const params: Record<string, string | number> = {
      engine: "google_jobs",
      q: query,
      gl: "us",
      hl: "en",
    };
    if (location) params.location = location;

    const data = await serpRequest(apiKey, params);
    const results = (data.jobs_results as unknown[]) ?? [];
    return { output: results };
  }
}

// ---------------------------------------------------------------------------
// 6. GoogleLens
// ---------------------------------------------------------------------------
export class GoogleLensNode extends BaseNode {
  static readonly nodeType = "search.google.GoogleLens";
  static readonly title = "Google Lens";
  static readonly description =
    "Analyze images using Google Lens to find visual matches and related content.";

  defaults() {
    return { image_url: "", num_results: 10 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const imageUrl = String(inputs.image_url ?? "");
    if (!imageUrl) throw new Error("Image URL is required for Google Lens search.");
    const apiKey = getSerpApiKey(inputs);

    const data = await serpRequest(apiKey, {
      engine: "google_lens",
      url: imageUrl,
      hl: "en",
    });

    const matches =
      (data.visual_matches as Array<Record<string, unknown>>) ?? [];
    const images = matches.map((m) => ({
      uri: String(m.image ?? m.thumbnail ?? ""),
    }));
    return { output: { results: matches, images } };
  }
}

// ---------------------------------------------------------------------------
// 7. GoogleMaps
// ---------------------------------------------------------------------------
export class GoogleMapsNode extends BaseNode {
  static readonly nodeType = "search.google.GoogleMaps";
  static readonly title = "Google Maps";
  static readonly description =
    "Search Google Maps for places, businesses, and get location details.";

  defaults() {
    return { query: "", num_results: 10 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const query = String(inputs.query ?? "");
    if (!query) throw new Error("Query is required for map search.");
    const apiKey = getSerpApiKey(inputs);

    const data = await serpRequest(apiKey, {
      engine: "google_maps",
      q: query,
      type: "search",
      gl: "us",
      hl: "en",
    });

    const localResults =
      (data.local_results as Array<Record<string, unknown>>) ?? [];
    // Rename "type" -> "place_type" to match Python model
    const results = localResults.map((r) => {
      const { type: placeType, ...rest } = r;
      return { ...rest, place_type: placeType ?? "" };
    });
    return { output: results };
  }
}

// ---------------------------------------------------------------------------
// 8. GoogleShopping
// ---------------------------------------------------------------------------
export class GoogleShoppingNode extends BaseNode {
  static readonly nodeType = "search.google.GoogleShopping";
  static readonly title = "Google Shopping";
  static readonly description =
    "Search Google Shopping for products with filters and pricing information.";

  defaults() {
    return {
      query: "",
      country: "us",
      min_price: 0,
      max_price: 0,
      condition: "",
      sort_by: "",
      num_results: 10,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const query = String(inputs.query ?? "");
    if (!query) throw new Error("Query is required for Google Shopping search.");
    const country = String(inputs.country ?? "us") || "us";
    const minPrice = Number(inputs.min_price ?? 0);
    const maxPrice = Number(inputs.max_price ?? 0);
    const condition = String(inputs.condition ?? "");
    const sortBy = String(inputs.sort_by ?? "");
    const apiKey = getSerpApiKey(inputs);

    const params: Record<string, string | number> = {
      engine: "google_shopping",
      q: query,
      gl: country,
      hl: "en",
    };

    // Build tbs filter string
    const tbsParts: string[] = [];
    if (minPrice > 0) {
      if (!tbsParts.includes("mr:1")) tbsParts.push("mr:1");
      if (!tbsParts.includes("price:1")) tbsParts.push("price:1");
      tbsParts.push(`ppr_min:${minPrice}`);
    }
    if (maxPrice > 0) {
      if (!tbsParts.includes("mr:1")) tbsParts.push("mr:1");
      if (!tbsParts.includes("price:1")) tbsParts.push("price:1");
      tbsParts.push(`ppr_max:${maxPrice}`);
    }
    if (condition) {
      const c = condition.toLowerCase();
      if (c === "new") tbsParts.push("p_cond:new");
      else if (c === "used" || c === "refurbished") tbsParts.push("p_cond:used");
    }
    if (sortBy) {
      tbsParts.push(`sort:${sortBy}`);
    }
    if (tbsParts.length > 0) {
      params.tbs = tbsParts.join(",");
    }

    const data = await serpRequest(apiKey, params);
    const results = (data.shopping_results as unknown[]) ?? [];
    return { output: results };
  }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
export const SEARCH_NODES: readonly NodeClass[] = [
  GoogleSearchNode,
  GoogleNewsNode,
  GoogleImagesNode,
  GoogleFinanceNode,
  GoogleJobsNode,
  GoogleLensNode,
  GoogleMapsNode,
  GoogleShoppingNode,
];
