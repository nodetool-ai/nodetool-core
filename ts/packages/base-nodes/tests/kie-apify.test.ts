import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  ApifyWebScraperNode,
  ApifyGoogleSearchScraperNode,
  ApifyInstagramScraperNode,
  ApifyAmazonScraperNode,
  ApifyYouTubeScraperNode,
  ApifyTwitterScraperNode,
  ApifyLinkedInScraperNode,
} from "../src/nodes/apify.js";

const originalFetch = globalThis.fetch;
let mockFetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockFetch = vi.fn();
  globalThis.fetch = mockFetch;
  delete process.env.APIFY_API_KEY;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  delete process.env.APIFY_API_KEY;
});

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

const secrets = { _secrets: { APIFY_API_KEY: "test-key" } };

/** Mock a successful runActor flow: POST actor run -> GET dataset items */
function mockRunActor(items: Record<string, unknown>[]) {
  mockFetch
    .mockResolvedValueOnce(
      jsonResponse({ data: { id: "run1", defaultDatasetId: "ds1" } })
    )
    .mockResolvedValueOnce(jsonResponse(items));
}

// ── ApifyWebScraperNode ────────────────────────────────────────────────────

describe("ApifyWebScraperNode", () => {
  it("returns scraped items", async () => {
    const node = new ApifyWebScraperNode();
    const items = [{ url: "https://example.com", title: "Example" }];
    mockRunActor(items);
    const result = await node.process({
      start_urls: ["https://example.com"],
      ...secrets,
    });
    expect(result.output).toEqual(items);
  });

  it("throws when start_urls is empty", async () => {
    const node = new ApifyWebScraperNode();
    await expect(
      node.process({ start_urls: [], ...secrets })
    ).rejects.toThrow("start_urls is required");
  });

  it("throws when API key missing", async () => {
    const node = new ApifyWebScraperNode();
    await expect(
      node.process({ start_urls: ["https://example.com"] })
    ).rejects.toThrow("APIFY_API_KEY not configured");
  });

  it("throws on API error", async () => {
    const node = new ApifyWebScraperNode();
    mockFetch.mockResolvedValueOnce(jsonResponse("forbidden", 403));
    await expect(
      node.process({ start_urls: ["https://example.com"], ...secrets })
    ).rejects.toThrow("Apify API error (403)");
  });

  it("returns empty array when no datasetId", async () => {
    const node = new ApifyWebScraperNode();
    mockFetch.mockResolvedValueOnce(jsonResponse({ data: { id: "run1" } }));
    const result = await node.process({
      start_urls: ["https://example.com"],
      ...secrets,
    });
    expect(result.output).toEqual([]);
  });

  it("returns empty array when dataset fetch fails", async () => {
    const node = new ApifyWebScraperNode();
    mockFetch
      .mockResolvedValueOnce(
        jsonResponse({ data: { id: "run1", defaultDatasetId: "ds1" } })
      )
      .mockResolvedValueOnce(jsonResponse("not found", 404));
    const result = await node.process({
      start_urls: ["https://example.com"],
      ...secrets,
    });
    expect(result.output).toEqual([]);
  });

  it("uses env var API key", async () => {
    process.env.APIFY_API_KEY = "env-key";
    const node = new ApifyWebScraperNode();
    mockRunActor([]);
    const result = await node.process({
      start_urls: ["https://example.com"],
    });
    expect(result.output).toEqual([]);
  });
});

// ── ApifyGoogleSearchScraperNode ───────────────────────────────────────────

describe("ApifyGoogleSearchScraperNode", () => {
  it("returns search results", async () => {
    const node = new ApifyGoogleSearchScraperNode();
    const items = [{ title: "Result 1", url: "https://example.com" }];
    mockRunActor(items);
    const result = await node.process({
      queries: ["test query"],
      ...secrets,
    });
    expect(result.output).toEqual(items);
  });

  it("throws when queries empty", async () => {
    const node = new ApifyGoogleSearchScraperNode();
    await expect(
      node.process({ queries: [], ...secrets })
    ).rejects.toThrow("queries is required");
  });

  it("clamps results_per_page to valid range", async () => {
    const node = new ApifyGoogleSearchScraperNode();
    mockRunActor([]);
    await node.process({
      queries: ["test"],
      results_per_page: 5, // below MIN_RESULTS_PER_PAGE
      ...secrets,
    });
    // Verify the fetch was called (no error)
    expect(mockFetch).toHaveBeenCalled();
  });
});

// ── ApifyInstagramScraperNode ──────────────────────────────────────────────

describe("ApifyInstagramScraperNode", () => {
  it("returns items with usernames", async () => {
    const node = new ApifyInstagramScraperNode();
    const items = [{ username: "test_user", posts: 10 }];
    mockRunActor(items);
    const result = await node.process({
      usernames: ["test_user"],
      ...secrets,
    });
    expect(result.output).toEqual(items);
  });

  it("returns items with hashtags", async () => {
    const node = new ApifyInstagramScraperNode();
    mockRunActor([{ tag: "coding" }]);
    const result = await node.process({
      hashtags: ["coding"],
      ...secrets,
    });
    expect(result.output).toEqual([{ tag: "coding" }]);
  });

  it("throws when both usernames and hashtags empty", async () => {
    const node = new ApifyInstagramScraperNode();
    await expect(
      node.process({ usernames: [], hashtags: [], ...secrets })
    ).rejects.toThrow("Either usernames or hashtags is required");
  });
});

// ── ApifyAmazonScraperNode ─────────────────────────────────────────────────

describe("ApifyAmazonScraperNode", () => {
  it("returns products from search queries", async () => {
    const node = new ApifyAmazonScraperNode();
    const items = [{ title: "Product", price: "$10" }];
    mockRunActor(items);
    const result = await node.process({
      search_queries: ["laptop"],
      ...secrets,
    });
    expect(result.output).toEqual(items);
  });

  it("returns products from product URLs", async () => {
    const node = new ApifyAmazonScraperNode();
    mockRunActor([{ title: "Item" }]);
    const result = await node.process({
      product_urls: ["https://amazon.com/dp/123"],
      ...secrets,
    });
    expect(result.output).toEqual([{ title: "Item" }]);
  });

  it("throws when both search_queries and product_urls empty", async () => {
    const node = new ApifyAmazonScraperNode();
    await expect(
      node.process({
        search_queries: [],
        product_urls: [],
        ...secrets,
      })
    ).rejects.toThrow("Either search_queries or product_urls is required");
  });
});

// ── ApifyYouTubeScraperNode ────────────────────────────────────────────────

describe("ApifyYouTubeScraperNode", () => {
  it("returns results from search queries", async () => {
    const node = new ApifyYouTubeScraperNode();
    mockRunActor([{ title: "Video 1" }]);
    const result = await node.process({
      search_queries: ["cats"],
      ...secrets,
    });
    expect(result.output).toEqual([{ title: "Video 1" }]);
  });

  it("returns results from video URLs", async () => {
    const node = new ApifyYouTubeScraperNode();
    mockRunActor([{ title: "Video" }]);
    const result = await node.process({
      video_urls: ["https://youtube.com/watch?v=abc"],
      ...secrets,
    });
    expect(result.output).toEqual([{ title: "Video" }]);
  });

  it("returns results from channel URLs", async () => {
    const node = new ApifyYouTubeScraperNode();
    mockRunActor([{ channel: "test" }]);
    const result = await node.process({
      channel_urls: ["https://youtube.com/@testchannel"],
      ...secrets,
    });
    expect(result.output).toEqual([{ channel: "test" }]);
  });

  it("throws when all input arrays empty", async () => {
    const node = new ApifyYouTubeScraperNode();
    await expect(
      node.process({
        search_queries: [],
        video_urls: [],
        channel_urls: [],
        ...secrets,
      })
    ).rejects.toThrow(
      "At least one of search_queries, video_urls, or channel_urls is required"
    );
  });
});

// ── ApifyTwitterScraperNode ────────────────────────────────────────────────

describe("ApifyTwitterScraperNode", () => {
  it("returns tweets from search terms", async () => {
    const node = new ApifyTwitterScraperNode();
    mockRunActor([{ text: "tweet content" }]);
    const result = await node.process({
      search_terms: ["AI"],
      ...secrets,
    });
    expect(result.output).toEqual([{ text: "tweet content" }]);
  });

  it("returns tweets from usernames", async () => {
    const node = new ApifyTwitterScraperNode();
    mockRunActor([{ text: "user tweet" }]);
    const result = await node.process({
      usernames: ["elonmusk"],
      ...secrets,
    });
    expect(result.output).toEqual([{ text: "user tweet" }]);
  });

  it("returns tweets from tweet URLs", async () => {
    const node = new ApifyTwitterScraperNode();
    mockRunActor([{ text: "specific tweet" }]);
    const result = await node.process({
      tweet_urls: ["https://twitter.com/user/status/123"],
      ...secrets,
    });
    expect(result.output).toEqual([{ text: "specific tweet" }]);
  });

  it("throws when all inputs empty", async () => {
    const node = new ApifyTwitterScraperNode();
    await expect(
      node.process({
        search_terms: [],
        usernames: [],
        tweet_urls: [],
        ...secrets,
      })
    ).rejects.toThrow(
      "At least one of search_terms, usernames, or tweet_urls is required"
    );
  });
});

// ── ApifyLinkedInScraperNode ───────────────────────────────────────────────

describe("ApifyLinkedInScraperNode", () => {
  it("returns profiles from profile URLs", async () => {
    const node = new ApifyLinkedInScraperNode();
    mockRunActor([{ name: "John" }]);
    const result = await node.process({
      profile_urls: ["https://linkedin.com/in/john"],
      ...secrets,
    });
    expect(result.output).toEqual([{ name: "John" }]);
  });

  it("returns results from company URLs", async () => {
    const node = new ApifyLinkedInScraperNode();
    mockRunActor([{ company: "Acme" }]);
    const result = await node.process({
      company_urls: ["https://linkedin.com/company/acme"],
      ...secrets,
    });
    expect(result.output).toEqual([{ company: "Acme" }]);
  });

  it("returns results from job search URLs", async () => {
    const node = new ApifyLinkedInScraperNode();
    mockRunActor([{ job: "Engineer" }]);
    const result = await node.process({
      job_search_urls: ["https://linkedin.com/jobs/search?q=engineer"],
      ...secrets,
    });
    expect(result.output).toEqual([{ job: "Engineer" }]);
  });

  it("throws when all inputs empty", async () => {
    const node = new ApifyLinkedInScraperNode();
    await expect(
      node.process({
        profile_urls: [],
        company_urls: [],
        job_search_urls: [],
        ...secrets,
      })
    ).rejects.toThrow(
      "At least one of profile_urls, company_urls, or job_search_urls is required"
    );
  });
});

// ── Defaults coverage ────────────────────────────────────────────────────

describe("Node defaults coverage", () => {
  it("ApifyWebScraperNode defaults", () => {
    const node = new ApifyWebScraperNode();
    const d = node.defaults();
    expect(d.start_urls).toEqual([]);
    expect(d.link_selector).toBe("a[href]");
    expect(d.page_function).toBe("");
    expect(d.max_pages).toBe(10);
    expect(d.wait_for_finish).toBe(300);
  });

  it("ApifyGoogleSearchScraperNode defaults", () => {
    const node = new ApifyGoogleSearchScraperNode();
    const d = node.defaults();
    expect(d.queries).toEqual([]);
    expect(d.country_code).toBe("us");
    expect(d.language_code).toBe("en");
    expect(d.max_pages).toBe(1);
    expect(d.results_per_page).toBe(100);
    expect(d.wait_for_finish).toBe(300);
  });

  it("ApifyInstagramScraperNode defaults", () => {
    const node = new ApifyInstagramScraperNode();
    const d = node.defaults();
    expect(d.usernames).toEqual([]);
    expect(d.hashtags).toEqual([]);
    expect(d.results_limit).toBe(50);
    expect(d.scrape_comments).toBe(false);
    expect(d.scrape_likes).toBe(false);
    expect(d.wait_for_finish).toBe(600);
  });

  it("ApifyAmazonScraperNode defaults", () => {
    const node = new ApifyAmazonScraperNode();
    const d = node.defaults();
    expect(d.search_queries).toEqual([]);
    expect(d.product_urls).toEqual([]);
    expect(d.country_code).toBe("US");
    expect(d.max_items).toBe(20);
    expect(d.scrape_reviews).toBe(false);
    expect(d.wait_for_finish).toBe(600);
  });

  it("ApifyYouTubeScraperNode defaults", () => {
    const node = new ApifyYouTubeScraperNode();
    const d = node.defaults();
    expect(d.search_queries).toEqual([]);
    expect(d.video_urls).toEqual([]);
    expect(d.channel_urls).toEqual([]);
    expect(d.max_results).toBe(50);
    expect(d.scrape_comments).toBe(false);
    expect(d.wait_for_finish).toBe(600);
  });

  it("ApifyTwitterScraperNode defaults", () => {
    const node = new ApifyTwitterScraperNode();
    const d = node.defaults();
    expect(d.search_terms).toEqual([]);
    expect(d.usernames).toEqual([]);
    expect(d.tweet_urls).toEqual([]);
    expect(d.max_tweets).toBe(100);
    expect(d.wait_for_finish).toBe(600);
  });

  it("ApifyLinkedInScraperNode defaults", () => {
    const node = new ApifyLinkedInScraperNode();
    const d = node.defaults();
    expect(d.profile_urls).toEqual([]);
    expect(d.company_urls).toEqual([]);
    expect(d.job_search_urls).toEqual([]);
    expect(d.max_results).toBe(50);
    expect(d.wait_for_finish).toBe(600);
  });
});
