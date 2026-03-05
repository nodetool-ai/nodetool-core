import { describe, it, expect, vi, beforeEach, afterAll } from "vitest";
import {
  ApifyWebScraperNode,
  ApifyGoogleSearchScraperNode,
  ApifyInstagramScraperNode,
  ApifyAmazonScraperNode,
  ApifyYouTubeScraperNode,
  ApifyTwitterScraperNode,
  ApifyLinkedInScraperNode,
  APIFY_NODES,
} from "../src/nodes/apify.js";

const originalFetch = global.fetch;
const mockFetch = vi.fn();
global.fetch = mockFetch;

afterAll(() => {
  global.fetch = originalFetch;
});

/** Helper: mock a successful actor run + dataset fetch */
function mockActorRun(datasetItems: Record<string, unknown>[]) {
  // First call: POST to run actor
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      data: { id: "run-1", defaultDatasetId: "ds-1", status: "SUCCEEDED" },
    }),
  });
  // Second call: GET dataset items
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: async () => datasetItems,
  });
}

// ---------------------------------------------------------------------------
// ApifyWebScraperNode
// ---------------------------------------------------------------------------
describe("ApifyWebScraperNode", () => {
  beforeEach(() => mockFetch.mockReset());

  it("has correct metadata", () => {
    expect(ApifyWebScraperNode.nodeType).toBe("apify.scraping.ApifyWebScraper");
    expect(ApifyWebScraperNode.title).toBe("Apify Web Scraper");
  });

  it("returns expected defaults", () => {
    const node = new ApifyWebScraperNode();
    const d = node.defaults();
    expect(d.start_urls).toEqual([]);
    expect(d.link_selector).toBe("a[href]");
    expect(d.max_pages).toBe(10);
  });

  it("throws without API key", async () => {
    const node = new ApifyWebScraperNode();
    await expect(
      node.process({ start_urls: ["https://example.com"] })
    ).rejects.toThrow(/APIFY_API_KEY/i);
  });

  it("throws when start_urls is empty", async () => {
    const node = new ApifyWebScraperNode();
    await expect(
      node.process({
        start_urls: [],
        _secrets: { APIFY_API_KEY: "k" },
      })
    ).rejects.toThrow(/start_urls is required/i);
  });

  it("calls actor and returns dataset items", async () => {
    mockActorRun([{ url: "https://example.com", title: "Example" }]);

    const node = new ApifyWebScraperNode();
    const result = await node.process({
      start_urls: ["https://example.com"],
      _secrets: { APIFY_API_KEY: "test-key" },
    });

    expect(mockFetch).toHaveBeenCalledTimes(2);
    const [runUrl, runOpts] = mockFetch.mock.calls[0];
    expect(runUrl).toContain("apify~web-scraper");
    expect(runOpts.headers.Authorization).toBe("Bearer test-key");
    expect(result.output).toEqual([
      { url: "https://example.com", title: "Example" },
    ]);
  });

  it("handles actor run API error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 403,
      text: async () => "Forbidden",
    });

    const node = new ApifyWebScraperNode();
    await expect(
      node.process({
        start_urls: ["https://example.com"],
        _secrets: { APIFY_API_KEY: "bad-key" },
      })
    ).rejects.toThrow(/Apify API error.*403/);
  });

  it("returns empty array when no dataset id", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ data: { id: "run-1" } }),
    });

    const node = new ApifyWebScraperNode();
    const result = await node.process({
      start_urls: ["https://example.com"],
      _secrets: { APIFY_API_KEY: "test-key" },
    });

    expect(result.output).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// ApifyGoogleSearchScraperNode
// ---------------------------------------------------------------------------
describe("ApifyGoogleSearchScraperNode", () => {
  beforeEach(() => mockFetch.mockReset());

  it("has correct metadata", () => {
    expect(ApifyGoogleSearchScraperNode.nodeType).toBe(
      "apify.scraping.ApifyGoogleSearchScraper"
    );
    expect(ApifyGoogleSearchScraperNode.title).toBe(
      "Apify Google Search Scraper"
    );
  });

  it("returns expected defaults", () => {
    const d = new ApifyGoogleSearchScraperNode().defaults();
    expect(d.queries).toEqual([]);
    expect(d.country_code).toBe("us");
    expect(d.language_code).toBe("en");
    expect(d.results_per_page).toBe(100);
  });

  it("throws when queries is empty", async () => {
    const node = new ApifyGoogleSearchScraperNode();
    await expect(
      node.process({ queries: [], _secrets: { APIFY_API_KEY: "k" } })
    ).rejects.toThrow(/queries is required/i);
  });

  it("clamps results_per_page to valid range", async () => {
    mockActorRun([]);

    const node = new ApifyGoogleSearchScraperNode();
    await node.process({
      queries: ["test"],
      results_per_page: 5, // below MIN_RESULTS_PER_PAGE (10)
      _secrets: { APIFY_API_KEY: "k" },
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.resultsPerPage).toBe(10);
  });

  it("calls google-search-scraper actor", async () => {
    mockActorRun([{ title: "Result" }]);

    const node = new ApifyGoogleSearchScraperNode();
    await node.process({
      queries: ["test query"],
      _secrets: { APIFY_API_KEY: "k" },
    });

    const [url] = mockFetch.mock.calls[0];
    expect(url).toContain("apify~google-search-scraper");
  });
});

// ---------------------------------------------------------------------------
// ApifyInstagramScraperNode
// ---------------------------------------------------------------------------
describe("ApifyInstagramScraperNode", () => {
  beforeEach(() => mockFetch.mockReset());

  it("has correct metadata", () => {
    expect(ApifyInstagramScraperNode.nodeType).toBe(
      "apify.scraping.ApifyInstagramScraper"
    );
  });

  it("returns expected defaults", () => {
    const d = new ApifyInstagramScraperNode().defaults();
    expect(d.usernames).toEqual([]);
    expect(d.hashtags).toEqual([]);
    expect(d.results_limit).toBe(50);
    expect(d.scrape_comments).toBe(false);
  });

  it("throws when both usernames and hashtags are empty", async () => {
    const node = new ApifyInstagramScraperNode();
    await expect(
      node.process({
        usernames: [],
        hashtags: [],
        _secrets: { APIFY_API_KEY: "k" },
      })
    ).rejects.toThrow(/usernames or hashtags/i);
  });

  it("calls instagram-scraper actor with usernames", async () => {
    mockActorRun([]);

    const node = new ApifyInstagramScraperNode();
    await node.process({
      usernames: ["testuser"],
      _secrets: { APIFY_API_KEY: "k" },
    });

    const [url] = mockFetch.mock.calls[0];
    expect(url).toContain("apify~instagram-scraper");
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.usernames).toEqual(["testuser"]);
  });
});

// ---------------------------------------------------------------------------
// ApifyAmazonScraperNode
// ---------------------------------------------------------------------------
describe("ApifyAmazonScraperNode", () => {
  beforeEach(() => mockFetch.mockReset());

  it("has correct metadata", () => {
    expect(ApifyAmazonScraperNode.nodeType).toBe(
      "apify.scraping.ApifyAmazonScraper"
    );
  });

  it("returns expected defaults", () => {
    const d = new ApifyAmazonScraperNode().defaults();
    expect(d.search_queries).toEqual([]);
    expect(d.product_urls).toEqual([]);
    expect(d.country_code).toBe("US");
    expect(d.max_items).toBe(20);
  });

  it("throws when both queries and urls are empty", async () => {
    const node = new ApifyAmazonScraperNode();
    await expect(
      node.process({
        search_queries: [],
        product_urls: [],
        _secrets: { APIFY_API_KEY: "k" },
      })
    ).rejects.toThrow(/search_queries or product_urls/i);
  });

  it("calls amazon-product-scraper actor", async () => {
    mockActorRun([]);

    const node = new ApifyAmazonScraperNode();
    await node.process({
      search_queries: ["laptop"],
      _secrets: { APIFY_API_KEY: "k" },
    });

    const [url] = mockFetch.mock.calls[0];
    expect(url).toContain("apify~amazon-product-scraper");
  });
});

// ---------------------------------------------------------------------------
// ApifyYouTubeScraperNode
// ---------------------------------------------------------------------------
describe("ApifyYouTubeScraperNode", () => {
  beforeEach(() => mockFetch.mockReset());

  it("has correct metadata", () => {
    expect(ApifyYouTubeScraperNode.nodeType).toBe(
      "apify.scraping.ApifyYouTubeScraper"
    );
  });

  it("returns expected defaults", () => {
    const d = new ApifyYouTubeScraperNode().defaults();
    expect(d.search_queries).toEqual([]);
    expect(d.max_results).toBe(50);
  });

  it("throws when all input arrays are empty", async () => {
    const node = new ApifyYouTubeScraperNode();
    await expect(
      node.process({
        search_queries: [],
        video_urls: [],
        channel_urls: [],
        _secrets: { APIFY_API_KEY: "k" },
      })
    ).rejects.toThrow(/search_queries, video_urls, or channel_urls/i);
  });

  it("builds YouTube search URLs from queries", async () => {
    mockActorRun([]);

    const node = new ApifyYouTubeScraperNode();
    await node.process({
      search_queries: ["cats"],
      _secrets: { APIFY_API_KEY: "k" },
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.startUrls[0].url).toContain("youtube.com/results");
    expect(body.startUrls[0].url).toContain("cats");
  });
});

// ---------------------------------------------------------------------------
// ApifyTwitterScraperNode
// ---------------------------------------------------------------------------
describe("ApifyTwitterScraperNode", () => {
  beforeEach(() => mockFetch.mockReset());

  it("has correct metadata", () => {
    expect(ApifyTwitterScraperNode.nodeType).toBe(
      "apify.scraping.ApifyTwitterScraper"
    );
  });

  it("returns expected defaults", () => {
    const d = new ApifyTwitterScraperNode().defaults();
    expect(d.max_tweets).toBe(100);
  });

  it("throws when all inputs are empty", async () => {
    const node = new ApifyTwitterScraperNode();
    await expect(
      node.process({
        search_terms: [],
        usernames: [],
        tweet_urls: [],
        _secrets: { APIFY_API_KEY: "k" },
      })
    ).rejects.toThrow(/search_terms, usernames, or tweet_urls/i);
  });

  it("builds Twitter search URLs from terms and usernames", async () => {
    mockActorRun([]);

    const node = new ApifyTwitterScraperNode();
    await node.process({
      search_terms: ["AI"],
      usernames: ["elonmusk"],
      _secrets: { APIFY_API_KEY: "k" },
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.startUrls).toContain(
      "https://twitter.com/search?q=AI"
    );
    expect(body.startUrls).toContain("https://twitter.com/elonmusk");
  });
});

// ---------------------------------------------------------------------------
// ApifyLinkedInScraperNode
// ---------------------------------------------------------------------------
describe("ApifyLinkedInScraperNode", () => {
  beforeEach(() => mockFetch.mockReset());

  it("has correct metadata", () => {
    expect(ApifyLinkedInScraperNode.nodeType).toBe(
      "apify.scraping.ApifyLinkedInScraper"
    );
  });

  it("returns expected defaults", () => {
    const d = new ApifyLinkedInScraperNode().defaults();
    expect(d.profile_urls).toEqual([]);
    expect(d.max_results).toBe(50);
  });

  it("throws when all URL arrays are empty", async () => {
    const node = new ApifyLinkedInScraperNode();
    await expect(
      node.process({
        profile_urls: [],
        company_urls: [],
        job_search_urls: [],
        _secrets: { APIFY_API_KEY: "k" },
      })
    ).rejects.toThrow(/profile_urls, company_urls, or job_search_urls/i);
  });

  it("calls linkedin-profile-scraper actor", async () => {
    mockActorRun([]);

    const node = new ApifyLinkedInScraperNode();
    await node.process({
      profile_urls: ["https://linkedin.com/in/test"],
      _secrets: { APIFY_API_KEY: "k" },
    });

    const [url] = mockFetch.mock.calls[0];
    expect(url).toContain("apify~linkedin-profile-scraper");
  });
});

// ---------------------------------------------------------------------------
// APIFY_NODES export
// ---------------------------------------------------------------------------
describe("APIFY_NODES", () => {
  it("exports all 7 nodes", () => {
    expect(APIFY_NODES).toHaveLength(7);
    const types = APIFY_NODES.map((n) => n.nodeType);
    expect(types).toContain("apify.scraping.ApifyWebScraper");
    expect(types).toContain("apify.scraping.ApifyGoogleSearchScraper");
    expect(types).toContain("apify.scraping.ApifyInstagramScraper");
    expect(types).toContain("apify.scraping.ApifyAmazonScraper");
    expect(types).toContain("apify.scraping.ApifyYouTubeScraper");
    expect(types).toContain("apify.scraping.ApifyTwitterScraper");
    expect(types).toContain("apify.scraping.ApifyLinkedInScraper");
  });
});
