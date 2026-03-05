import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

// Constants
const DEFAULT_PAGE_FUNCTION =
  "async function pageFunction(context) { return context.request.loadedUrl; }";
const MIN_RESULTS_PER_PAGE = 10;
const MAX_RESULTS_PER_PAGE = 100;

const APIFY_API_BASE = "https://api.apify.com/v2";

function getApifyApiKey(inputs: Record<string, unknown>): string {
  const key =
    (inputs._secrets as Record<string, string>)?.APIFY_API_KEY ||
    process.env.APIFY_API_KEY;
  if (!key) throw new Error("APIFY_API_KEY not configured");
  return key;
}

interface ApifyRun {
  data?: {
    id?: string;
    defaultDatasetId?: string;
    status?: string;
  };
}

async function runActor(
  apiKey: string,
  actorId: string,
  input: Record<string, unknown>,
  waitSecs: number
): Promise<Record<string, unknown>[]> {
  const encodedActorId = actorId.replace("/", "~");
  const url = `${APIFY_API_BASE}/acts/${encodedActorId}/runs?waitForFinish=${waitSecs}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Apify API error (${response.status}): ${text}`);
  }

  const run = (await response.json()) as ApifyRun;
  const datasetId = run.data?.defaultDatasetId;
  if (!datasetId) return [];

  const datasetUrl = `${APIFY_API_BASE}/datasets/${datasetId}/items?format=json`;
  const datasetResponse = await fetch(datasetUrl, {
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  if (!datasetResponse.ok) return [];
  return (await datasetResponse.json()) as Record<string, unknown>[];
}

// ---------------------------------------------------------------------------
// 1. ApifyWebScraper
// ---------------------------------------------------------------------------
export class ApifyWebScraperNode extends BaseNode {
  static readonly nodeType = "apify.scraping.ApifyWebScraper";
  static readonly title = "Apify Web Scraper";
  static readonly description =
    "Scrape websites using Apify's Web Scraper actor. Extracts data from web pages using CSS selectors or custom JavaScript.";

  defaults() {
    return {
      start_urls: [],
      link_selector: "a[href]",
      page_function: "",
      max_pages: 10,
      wait_for_finish: 300,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApifyApiKey(inputs);
    const startUrls = (inputs.start_urls as string[]) ?? [];
    if (startUrls.length === 0) throw new Error("start_urls is required");

    const pageFunction =
      String(inputs.page_function ?? "") || DEFAULT_PAGE_FUNCTION;

    const runInput = {
      startUrls: startUrls.map((url) => ({ url })),
      linkSelector: String(inputs.link_selector ?? "a[href]"),
      pageFunction,
      maxPagesPerCrawl: Number(inputs.max_pages ?? 10),
    };

    const items = await runActor(
      apiKey,
      "apify/web-scraper",
      runInput,
      Number(inputs.wait_for_finish ?? 300)
    );
    return { output: items };
  }
}

// ---------------------------------------------------------------------------
// 2. ApifyGoogleSearchScraper
// ---------------------------------------------------------------------------
export class ApifyGoogleSearchScraperNode extends BaseNode {
  static readonly nodeType = "apify.scraping.ApifyGoogleSearchScraper";
  static readonly title = "Apify Google Search Scraper";
  static readonly description =
    "Scrape Google Search results using Apify's Google Search Scraper. Extract organic results, ads, related searches, and more.";

  defaults() {
    return {
      queries: [],
      country_code: "us",
      language_code: "en",
      max_pages: 1,
      results_per_page: 100,
      wait_for_finish: 300,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApifyApiKey(inputs);
    const queries = (inputs.queries as string[]) ?? [];
    if (queries.length === 0) throw new Error("queries is required");

    const resultsPerPage = Math.min(
      Math.max(MIN_RESULTS_PER_PAGE, Number(inputs.results_per_page ?? 100)),
      MAX_RESULTS_PER_PAGE
    );

    const runInput = {
      queries: queries.join("\n"),
      countryCode: String(inputs.country_code ?? "us"),
      languageCode: String(inputs.language_code ?? "en"),
      maxPagesPerQuery: Number(inputs.max_pages ?? 1),
      resultsPerPage,
    };

    const items = await runActor(
      apiKey,
      "apify/google-search-scraper",
      runInput,
      Number(inputs.wait_for_finish ?? 300)
    );
    return { output: items };
  }
}

// ---------------------------------------------------------------------------
// 3. ApifyInstagramScraper
// ---------------------------------------------------------------------------
export class ApifyInstagramScraperNode extends BaseNode {
  static readonly nodeType = "apify.scraping.ApifyInstagramScraper";
  static readonly title = "Apify Instagram Scraper";
  static readonly description =
    "Scrape Instagram profiles, posts, comments, and hashtags. Extract user data, post details, engagement metrics, and more.";

  defaults() {
    return {
      usernames: [],
      hashtags: [],
      results_limit: 50,
      scrape_comments: false,
      scrape_likes: false,
      wait_for_finish: 600,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApifyApiKey(inputs);
    const usernames = (inputs.usernames as string[]) ?? [];
    const hashtags = (inputs.hashtags as string[]) ?? [];
    if (usernames.length === 0 && hashtags.length === 0) {
      throw new Error("Either usernames or hashtags is required");
    }

    const runInput: Record<string, unknown> = {
      resultsLimit: Number(inputs.results_limit ?? 50),
      scrapeComments: Boolean(inputs.scrape_comments ?? false),
      scrapeLikes: Boolean(inputs.scrape_likes ?? false),
    };

    if (usernames.length > 0) runInput.usernames = usernames;
    if (hashtags.length > 0) runInput.hashtags = hashtags;

    const items = await runActor(
      apiKey,
      "apify/instagram-scraper",
      runInput,
      Number(inputs.wait_for_finish ?? 600)
    );
    return { output: items };
  }
}

// ---------------------------------------------------------------------------
// 4. ApifyAmazonScraper
// ---------------------------------------------------------------------------
export class ApifyAmazonScraperNode extends BaseNode {
  static readonly nodeType = "apify.scraping.ApifyAmazonScraper";
  static readonly title = "Apify Amazon Scraper";
  static readonly description =
    "Scrape Amazon product data including prices, reviews, and ratings. Extract product details, seller information, and customer reviews.";

  defaults() {
    return {
      search_queries: [],
      product_urls: [],
      country_code: "US",
      max_items: 20,
      scrape_reviews: false,
      wait_for_finish: 600,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApifyApiKey(inputs);
    const searchQueries = (inputs.search_queries as string[]) ?? [];
    const productUrls = (inputs.product_urls as string[]) ?? [];
    if (searchQueries.length === 0 && productUrls.length === 0) {
      throw new Error("Either search_queries or product_urls is required");
    }

    const runInput: Record<string, unknown> = {
      countryCode: String(inputs.country_code ?? "US"),
      maxItems: Number(inputs.max_items ?? 20),
      scrapeReviews: Boolean(inputs.scrape_reviews ?? false),
    };

    if (searchQueries.length > 0) runInput.searchQueries = searchQueries;
    if (productUrls.length > 0) runInput.productUrls = productUrls;

    const items = await runActor(
      apiKey,
      "apify/amazon-product-scraper",
      runInput,
      Number(inputs.wait_for_finish ?? 600)
    );
    return { output: items };
  }
}

// ---------------------------------------------------------------------------
// 5. ApifyYouTubeScraper
// ---------------------------------------------------------------------------
export class ApifyYouTubeScraperNode extends BaseNode {
  static readonly nodeType = "apify.scraping.ApifyYouTubeScraper";
  static readonly title = "Apify YouTube Scraper";
  static readonly description =
    "Scrape YouTube videos, channels, and playlists. Extract video metadata, comments, channel info, and statistics.";

  defaults() {
    return {
      search_queries: [],
      video_urls: [],
      channel_urls: [],
      max_results: 50,
      scrape_comments: false,
      wait_for_finish: 600,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApifyApiKey(inputs);
    const searchQueries = (inputs.search_queries as string[]) ?? [];
    const videoUrls = (inputs.video_urls as string[]) ?? [];
    const channelUrls = (inputs.channel_urls as string[]) ?? [];
    if (
      searchQueries.length === 0 &&
      videoUrls.length === 0 &&
      channelUrls.length === 0
    ) {
      throw new Error(
        "At least one of search_queries, video_urls, or channel_urls is required"
      );
    }

    const startUrls: { url: string }[] = [];
    for (const query of searchQueries) {
      startUrls.push({
        url: `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}`,
      });
    }
    for (const url of videoUrls) {
      startUrls.push({ url });
    }
    for (const url of channelUrls) {
      startUrls.push({ url });
    }

    const runInput = {
      startUrls,
      maxResults: Number(inputs.max_results ?? 50),
      scrapeComments: Boolean(inputs.scrape_comments ?? false),
    };

    const items = await runActor(
      apiKey,
      "apify/youtube-scraper",
      runInput,
      Number(inputs.wait_for_finish ?? 600)
    );
    return { output: items };
  }
}

// ---------------------------------------------------------------------------
// 6. ApifyTwitterScraper
// ---------------------------------------------------------------------------
export class ApifyTwitterScraperNode extends BaseNode {
  static readonly nodeType = "apify.scraping.ApifyTwitterScraper";
  static readonly title = "Apify Twitter Scraper";
  static readonly description =
    "Scrape Twitter/X posts, profiles, and followers. Extract tweets, user information, and engagement metrics.";

  defaults() {
    return {
      search_terms: [],
      usernames: [],
      tweet_urls: [],
      max_tweets: 100,
      wait_for_finish: 600,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApifyApiKey(inputs);
    const searchTerms = (inputs.search_terms as string[]) ?? [];
    const usernames = (inputs.usernames as string[]) ?? [];
    const tweetUrls = (inputs.tweet_urls as string[]) ?? [];
    if (
      searchTerms.length === 0 &&
      usernames.length === 0 &&
      tweetUrls.length === 0
    ) {
      throw new Error(
        "At least one of search_terms, usernames, or tweet_urls is required"
      );
    }

    const startUrls: string[] = [];
    for (const term of searchTerms) {
      startUrls.push(
        `https://twitter.com/search?q=${encodeURIComponent(term)}`
      );
    }
    for (const username of usernames) {
      startUrls.push(`https://twitter.com/${username}`);
    }
    startUrls.push(...tweetUrls);

    const runInput = {
      startUrls,
      maxItems: Number(inputs.max_tweets ?? 100),
    };

    const items = await runActor(
      apiKey,
      "apify/twitter-scraper",
      runInput,
      Number(inputs.wait_for_finish ?? 600)
    );
    return { output: items };
  }
}

// ---------------------------------------------------------------------------
// 7. ApifyLinkedInScraper
// ---------------------------------------------------------------------------
export class ApifyLinkedInScraperNode extends BaseNode {
  static readonly nodeType = "apify.scraping.ApifyLinkedInScraper";
  static readonly title = "Apify LinkedIn Scraper";
  static readonly description =
    "Scrape LinkedIn profiles, company pages, and job postings. Extract professional information, connections, and company data.";

  defaults() {
    return {
      profile_urls: [],
      company_urls: [],
      job_search_urls: [],
      max_results: 50,
      wait_for_finish: 600,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const apiKey = getApifyApiKey(inputs);
    const profileUrls = (inputs.profile_urls as string[]) ?? [];
    const companyUrls = (inputs.company_urls as string[]) ?? [];
    const jobSearchUrls = (inputs.job_search_urls as string[]) ?? [];
    if (
      profileUrls.length === 0 &&
      companyUrls.length === 0 &&
      jobSearchUrls.length === 0
    ) {
      throw new Error(
        "At least one of profile_urls, company_urls, or job_search_urls is required"
      );
    }

    const allUrls = [...profileUrls, ...companyUrls, ...jobSearchUrls];

    const runInput = {
      startUrls: allUrls.map((url) => ({ url })),
      maxResults: Number(inputs.max_results ?? 50),
    };

    const items = await runActor(
      apiKey,
      "apify/linkedin-profile-scraper",
      runInput,
      Number(inputs.wait_for_finish ?? 600)
    );
    return { output: items };
  }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
export const APIFY_NODES: readonly NodeClass[] = [
  ApifyWebScraperNode,
  ApifyGoogleSearchScraperNode,
  ApifyInstagramScraperNode,
  ApifyAmazonScraperNode,
  ApifyYouTubeScraperNode,
  ApifyTwitterScraperNode,
  ApifyLinkedInScraperNode,
] as const;
